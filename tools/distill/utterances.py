#!/usr/bin/env python3
"""utterances — generate the candidate routing rows the teacher will label.

Two sources, one gate:

* **templates** — hand-authored frames expanded over slot values, sized per band
  and per op so every op gets equal head weight and t3 (the boundary pairs) gets
  the largest share;
* **the session corpus** — real operator phrasings mined through ``pi-sessions``
  (see ``mine.py``), which buys register variety the frames cannot.

Both pass the same ``corpus.Gate``: dedup by normalised similarity (see
``similarity.METHOD``) and a hard refusal against the frozen eval set. The gate is
one instance for the whole run, so a mined row is checked against the template
rows generated before it.

Session rows are written to ``candidates_session.jsonl`` and stay **out** of the
corpus unless ``--include-mined`` is passed. The measured reason is in the manifest:
operator turns are specifications and pasted output, so the survivors of the
request-shape filter are usually not routing requests at all, and a teacher label on
a non-request is noise where the corpus has to be clean. Review the quarantine file,
then opt in.

Nothing here is labelled. ``label.py`` attaches teacher targets to these rows;
this file only decides *what gets asked*, never what the answer is.

    python3 tools/distill/utterances.py                 # templates + session mining
    python3 tools/distill/utterances.py --dry-run        # templates only, no subprocess
    python3 tools/distill/utterances.py --out-dir var/distill
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

if __package__ in (None, ""):  # run as a script: put tools/ on the path for `distill.*`
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from distill import catalog, corpus, mine, similarity, templates  # noqa: E402


def _counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {"band": {}, "source": {}, "op": {}, "split": {}}
    for row in rows:
        for key, value in (("band", row["band"]), ("source", row["source"]),
                           ("op", row["intended"] or "?"), ("split", row["split"])):
            out[key][str(value)] = out[key].get(str(value), 0) + 1
    return out


def generate(out_dir: pathlib.Path, use_mining: bool, include_mined: bool = False,
             limit: int = 40, snippet: int = 400) -> dict[str, Any]:
    """Write candidates.jsonl + manifest.json; return the manifest.

    Mined rows are written to ``candidates_session.jsonl`` and are **not** part of
    the training corpus unless ``include_mined`` is set. On this machine that is the
    right default and the numbers say why: operator turns are specifications and
    pasted output, so the mechanically-valid survivors are mostly not routing
    requests at all, and a teacher label on a non-request is noise in exactly the
    place the corpus is supposed to be clean. The miner stays wired in — the count,
    the filter reasons and the quarantine file all ship — so a corpus that does
    contain request-shaped turns is one flag away, and the operator can review the
    quarantined rows before enabling it.
    """
    gate = corpus.Gate()
    rows, template_report = templates.generate(gate)

    mined_rows: list[dict[str, Any]] = []
    if use_mining:
        mined, mine_report = mine.mine(limit=limit, snippet=snippet)
        for item in mined:
            if gate.offer(item.text):
                continue
            mined_rows.append(templates.build_row(
                band=item.band, source="session", op=item.op, rival=item.rival,
                family=item.family, text=item.text, index=0))
        mine_stats = mine_report.as_dict()
        mine_stats["accepted"] = len(mined_rows)
    else:
        mine_stats = {"skipped": "mining disabled (--dry-run / --no-mine)"}

    corpus.write_jsonl(out_dir / "candidates_session.jsonl", mined_rows)
    corpus.write_jsonl(out_dir / "dropped.jsonl", gate.rejected)
    if include_mined:
        for offset, row in enumerate(mined_rows, start=len(rows) + 1):
            row["id"] = corpus.row_id(offset)
            rows.append(row)
    mine_stats["merged"] = bool(include_mined)

    for row in rows:  # belt and braces: every row is validated and re-checked
        corpus.assert_row(row)
        corpus.assert_not_frozen(row["utterance"], gate.frozen)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = corpus.write_jsonl(out_dir / "candidates.jsonl", rows)

    leak = corpus.leakage_report(rows, gate.frozen)
    manifest = {
        "generated_at": corpus.now(),
        "schema_ver": catalog.SCHEMA_VER,
        "catalog_sha": catalog.catalog_sha(),
        "tools": sorted(catalog.catalog()),
        "rows": written,
        "counts": _counts(rows),
        "similarity": similarity.METHOD,
        "leakage": leak,
        "dropped": gate.dropped,
        "templates": {k: v for k, v in template_report.items()
                      if k != "dropped_examples"},
        "mining": mine_stats,
        "paths": {"candidates": "candidates.jsonl", "dropped": "dropped.jsonl"},
        "frozen": {"path": corpus.FROZEN_PATH.relative_to(corpus.ROOT).as_posix(),
                   "rows": len(gate.frozen), "sha": gate.frozen.sha},
    }
    corpus.write_json(out_dir / "manifest.json", manifest)
    return manifest


def _print(manifest: dict[str, Any], out_dir: pathlib.Path) -> None:
    counts = manifest["counts"]
    print(f"wrote {manifest['rows']} rows -> {out_dir / 'candidates.jsonl'}")
    for key in ("band", "source", "split"):
        table = ", ".join(f"{k}={v}" for k, v in sorted(counts[key].items()))
        print(f"  by {key:6s} {table}")
    print("  by op     " + ", ".join(f"{k}={v}" for k, v in sorted(counts["op"].items())))
    print(f"  dropped   {manifest['dropped']}")
    templ = manifest.get("templates") or {}
    if templ.get("per_band"):
        print(f"  targets   {templ.get('targets')}")
    mining = manifest.get("mining") or {}
    if mining.get("skipped"):
        print(f"  mining    skipped: {mining['skipped']}")
    else:
        state = "merged" if mining.get("merged") else "quarantined (not in the corpus)"
        print(f"  mining    {mining.get('accepted')} rows from {mining.get('searched')} "
              f"searches / {mining.get('hits')} hits / {mining.get('units')} units "
              f"{mining.get('rejected')} -> {state}")
    leak = manifest["leakage"]
    print(f"  leakage   {'OK' if leak['ok'] else 'COLLISION'} — {leak['checked_rows']} "
          f"rows vs {leak['frozen_rows']} frozen, max ratio {leak['max_ratio']}")
    print(f"  similarity {manifest['similarity']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=str(corpus.OUT_DIR),
                    help="where candidates.jsonl + manifest.json land")
    ap.add_argument("--dry-run", action="store_true",
                    help="no external process and no network: template rows only")
    ap.add_argument("--no-mine", action="store_true", help="skip session mining")
    ap.add_argument("--include-mined", action="store_true",
                    help="merge the session rows into the corpus (default: quarantine "
                         "them to candidates_session.jsonl for review)")
    ap.add_argument("--limit", type=int, default=40, help="hits per mining seed")
    ap.add_argument("--snippet", type=int, default=400, help="excerpt width per hit")
    args = ap.parse_args(argv)

    out_dir = pathlib.Path(args.out_dir).expanduser()
    use_mining = not (args.dry_run or args.no_mine)
    manifest = generate(out_dir, use_mining=use_mining,
                        include_mined=args.include_mined, limit=args.limit,
                        snippet=args.snippet)
    _print(manifest, out_dir)
    return 0 if manifest["leakage"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
