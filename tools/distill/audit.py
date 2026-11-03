#!/usr/bin/env python3
"""audit — the human-check sample: what the teacher said, laid out to be argued with.

Teacher labels are not ground truth. 01 §5 puts label error at 5–15% before
verification, and a distillation corpus that trusts the teacher trains the student
on exactly that error. So this verb lifts a stratified sample out of the labelled
corpus and prints it three ways for a human:

* the **utterance** the model will see as the request;
* the **state** — the bounded context the decision was actually made on;
* the teacher's **choice and confidence**, beside every candidate it chose between,
  so the reading it rejected is visible at a glance.

It writes a CSV with empty ``verdict`` / ``correct_label`` / ``notes`` columns. Fill
them in (``right`` / ``wrong``) and ``stats.py --audit`` turns them into the
teacher's measured calibration on the audited slice — the only place a confidence
number ever meets an outcome.

Sampling is deterministic (``--seed``), stratified by band and spread across labels,
so re-running the audit on the same corpus gives the same rows to review.

    python3 tools/distill/audit.py --in var/distill/labels.jsonl \\
        --out var/distill/audit.csv --per-band 6
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import random
import sys
from typing import Any

if __package__ in (None, ""):  # run as a script: put tools/ on the path for `distill.*`
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from distill import corpus  # noqa: E402

COLUMNS = ("row_id", "band", "tier_target", "source", "split", "utterance",
           "candidates", "teacher_choice", "teacher_conf", "intended",
           "verdict", "correct_label", "notes")


def sample(rows: list[dict[str, Any]], per_band: int, seed: str = "audit"
           ) -> list[dict[str, Any]]:
    """Per band, round-robin across the labels the teacher chose — deterministically.

    Round-robin over labels rather than a flat draw: a flat draw over a corpus whose
    rows are mostly one or two tools would hand back a sample that cannot show which
    op the teacher is wrong about.
    """
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for band in corpus.BANDS:
        pool = [r for r in rows if r["band"] == band and r.get("target")]
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in pool:
            buckets.setdefault(str(row["target"]["tool"]), []).append(row)
        keys = sorted(buckets)
        rng.shuffle(keys)
        for bucket in buckets.values():
            rng.shuffle(bucket)
        taken = 0
        while taken < per_band and any(buckets[k] for k in keys):
            for key in keys:
                if buckets[key] and taken < per_band:
                    out.append(buckets[key].pop(0))
                    taken += 1
    return out


def _candidate_lines(row: dict[str, Any]) -> str:
    chosen = str(row["target"]["tool"])
    parts = []
    for name in sorted(row["candidates"]):
        mark = "  <== TEACHER" if name == chosen else ""
        parts.append(f"{name}{mark}")
    return " | ".join(parts)


def _print_block(row: dict[str, Any], index: int, total: int) -> None:
    target = row["target"]
    rival = row.get("tier_target")
    print(f"\n[{index}/{total}] {row['id']}  band={row['band']}"
          + (f"  rival={rival}" if rival else "")
          + f"  source={row['source']}  split={row['split']}")
    print(f"  utterance      {row['utterance']}")
    state = row.get("state") or {}
    digest = (state.get("digest") or {}) if isinstance(state, dict) else {}
    ctx = (state.get("params_context") or {}) if isinstance(state, dict) else {}
    if ctx or digest:
        print(f"  state          ctx={ctx} digest={digest}")
    print(f"  candidates     {_candidate_lines(row)}")
    print(f"  teacher        {target['tool']}  conf={float(target['conf']):.3f}"
          f"  ({row.get('label_method')})")
    if row.get("intended"):
        agree = "agrees" if target["tool"] == row["intended"] else "DISAGREES"
        print(f"  frame intent   {agree} with {row['intended']}")
    if rival:
        verdict = "chose the rival" if target["tool"] == rival else "chose the label"
        print(f"  boundary       {verdict}")
    print(f"  raw reply      {(row.get('teacher') or {}).get('raw', '')[:160]!r}")


def write_csv(rows: list[dict[str, Any]], path: pathlib.Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "row_id": row["id"], "band": row["band"],
                "tier_target": row.get("tier_target") or "",
                "source": row["source"], "split": row["split"],
                "utterance": row["utterance"],
                "candidates": _candidate_lines(row),
                "teacher_choice": row["target"]["tool"],
                "teacher_conf": f"{float(row['target']['conf']):.3f}",
                "intended": row.get("intended") or "",
                "verdict": "", "correct_label": "", "notes": "",
            })
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="source", default=str(corpus.OUT_DIR / "labels.jsonl"))
    ap.add_argument("--out", dest="target", default=str(corpus.OUT_DIR / "audit.csv"))
    ap.add_argument("--per-band", type=int, default=6,
                    help="rows to review per band")
    ap.add_argument("--seed", default="audit", help="deterministic sample seed")
    ap.add_argument("--quiet", action="store_true", help="write the CSV without printing")
    ap.add_argument("--dry-run", action="store_true",
                    help="accepted for pipeline uniformity; this verb never calls out")
    args = ap.parse_args(argv)

    src = pathlib.Path(args.source).expanduser()
    rows = corpus.read_jsonl(src)
    if not rows:
        raise SystemExit(f"no labelled rows at {src} — run label.py first")
    for row in rows:
        corpus.assert_labelled(row)

    picked = sample(rows, args.per_band, seed=args.seed)
    written = write_csv(picked, pathlib.Path(args.target).expanduser())
    if not args.quiet:
        print(f"audit sample: {written} rows from {len(rows)} labelled "
              f"({args.per_band} per band, seed={args.seed})")
        for index, row in enumerate(picked, start=1):
            _print_block(row, index, written)
        print(f"\nCSV -> {pathlib.Path(args.target).expanduser()}")
        print("fill in verdict (right/wrong) and correct_label where the teacher was "
              "wrong; stats.py --audit reads it back")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
