"""CLI — ``python3 -m tools.dataset <verb>``.

The verbs are one pipeline, each reading what the previous wrote:

    ingest    logs + frozen files -> corpus.jsonl (+ ingest_report.json)
    validate  corpus.jsonl        -> exit 1 with every schema error
    split     corpus.jsonl        -> splits.jsonl
    export    splits.jsonl        -> sft.jsonl, manifest.jsonl, labels.json, eval_*.json
    stats     splits.jsonl        -> counts, label distribution, band coverage

Paths come from ``TI_DATASET_LOGS`` / ``TI_DATASET_OUT`` with repo-relative
defaults, so nothing here names a machine.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from . import criteria as C
from . import export as E
from . import schema, split as S, stats as ST
from .ingest import ingest
from .paths import logs_dir, out_dir

CORPUS = "corpus.jsonl"
SPLITS = "splits.jsonl"
REPORT = "ingest_report.json"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="tools.dataset", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="derived-data dir (default: $TI_DATASET_OUT "
                                               "or var/dataset)")
    sub = ap.add_subparsers(dest="verb", required=True)

    p = sub.add_parser("ingest", help="read the logs and frozen files into corpus.jsonl")
    p.add_argument("--logs", default=None,
                   help="log dir holding hops.jsonl/live.jsonl (default: $TI_DATASET_LOGS "
                        "or service/logs)")
    p.add_argument("--data", default=None, help="dir with the frozen tests/data files")

    sub.add_parser("validate", help="schema-check corpus.jsonl; exit 1 when invalid")

    p = sub.add_parser("split", help="leakage-aware train/val/test split")
    p.add_argument("--seed", type=int, default=S.DEFAULT_SEED)

    p = sub.add_parser("export", help="write SFT chat JSONL, label legend and eval files")
    p.add_argument("--criteria", default=C.ROUTING_OPS, choices=sorted(C.CRITERIA))
    p.add_argument("--include-observed", action="store_true",
                   help="also export records whose label is an executed op, not a vetted one")

    sub.add_parser("stats", help="per-split counts, label distribution, band coverage")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = out_dir(args.out)
    if args.verb == "ingest":
        return _ingest(args, out)
    if args.verb == "validate":
        return _validate(out)
    if args.verb == "split":
        return _split(args, out)
    if args.verb == "export":
        return _export(args, out)
    if args.verb == "stats":
        return _stats(out)
    return 2


def _ingest(args: argparse.Namespace, out: Path) -> int:
    data = Path(args.data) if args.data else None
    records, report = ingest(logs_dir(args.logs), data)
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / CORPUS, records)
    (out / REPORT).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"ingested {len(records)} records -> {out / CORPUS}")
    print(ST.render_ingest(report))
    return 0


def _validate(out: Path) -> int:
    records = _load_jsonl(out / CORPUS)
    if not records:
        print(f"no records at {out / CORPUS} — run `ingest` first", file=sys.stderr)
        return 2
    errors = schema.check_corpus(records)
    if errors:
        print(f"INVALID: {len(errors)} problem(s) in {len(records)} records",
              file=sys.stderr)
        for msg in errors:
            print(f"  {msg}", file=sys.stderr)
        return 1
    print(f"ok: {len(records)} records pass the schema")
    return 0


def _split(args: argparse.Namespace, out: Path) -> int:
    records = _load_jsonl(out / CORPUS)
    if not records:
        print(f"no records at {out / CORPUS} — run `ingest` first", file=sys.stderr)
        return 2
    splits = S.split_records(records, seed=args.seed)
    problems = S.check_leakage(splits)
    if problems:
        print("LEAKAGE:", file=sys.stderr)
        for msg in problems:
            print(f"  {msg}", file=sys.stderr)
        return 1
    rows = [row for name in S.SPLITS for row in splits[name]]
    _write_jsonl(out / SPLITS, rows)
    sizes = ", ".join(f"{name}={len(splits[name])}" for name in S.SPLITS)
    print(f"split (seed={args.seed}) {sizes} -> {out / SPLITS}")
    return 0


def _export(args: argparse.Namespace, out: Path) -> int:
    rows = _load_jsonl(out / SPLITS) or _load_jsonl(out / CORPUS)
    if not rows:
        print(f"nothing to export at {out} — run `ingest` and `split` first", file=sys.stderr)
        return 2
    splits: dict[str, list[dict[str, Any]]] = {name: [] for name in S.SPLITS}
    for row in rows:
        splits.setdefault(row.get("split") or "train", []).append(row)
    written = E.write_export(splits, out, criteria_id=args.criteria,
                             include_observed=args.include_observed)
    evals = ", ".join(f"{k}={v}" for k, v in written["eval"].items()) or "none"
    print(f"sft={written['sft']} rows, skipped={written['skipped']} "
          f"({args.criteria}, observed={'in' if args.include_observed else 'out'})")
    print(f"eval: {evals}")
    print(f"wrote {out / E.SFT_NAME}, {out / E.MANIFEST_NAME}, {out / E.LABELS_NAME}")
    return 0


def _stats(out: Path) -> int:
    rows = _load_jsonl(out / SPLITS) or _load_jsonl(out / CORPUS)
    if not rows:
        print(f"no records at {out} — run `ingest` first", file=sys.stderr)
        return 2
    splits: dict[str, list[dict[str, Any]]] = {name: [] for name in S.SPLITS}
    for row in rows:
        splits.setdefault(row.get("split") or "train", []).append(row)
    ingests = _load_json(out / REPORT)
    print(ST.render(ST.collect(splits), ingests=ingests))
    return 0


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                rows.append(rec)
    return rows


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
