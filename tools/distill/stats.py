#!/usr/bin/env python3
"""stats — what the corpus actually is, and how far its confidence has been checked.

Per band: how many rows, which tools the teacher chose, and how confident it was. Then
the two numbers that decide whether the corpus is usable for training:

**Teacher acceptance** — how often the teacher's choice matches the op the frame was
built to elicit. A t1 frame could not be clearer, so a low agreement there is a bug in
the frame, the description or the prompt, not a hard example. On t3 it is the opposite
reading of the same number: the teacher picking the rival means the boundary is real.

**Calibration** — agreement between the teacher's confidence and a human's verdict on
the audited slice. This is the one place a confidence meets an outcome, so with no
annotated ``audit.csv`` the honest answer is that calibration is *unknown*, and the
bins are printed with their row counts and no accuracy column rather than filled in
from the teacher's own numbers. ECE over the audited slice is reported with n, and
flagged as too small to trust below 30 annotated rows (03 §2 wants far more for a
threshold; this is a smoke test, not a calibration curve).

    python3 tools/distill/stats.py --in var/distill/labels.jsonl \\
        --audit var/distill/audit.csv
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import statistics
import sys
from typing import Any

if __package__ in (None, ""):  # run as a script: put tools/ on the path for `distill.*`
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from distill import corpus, cost  # noqa: E402

BINS = ((0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.01))
CONFIDENCE_FLOOR = 30  # annotated rows below this: report the bins, distrust the ECE


def _band_bin(conf: float) -> str:
    for low, high in BINS:
        if low <= conf < high:
            return f"{low:.2f}-{min(high, 1.0):.2f}"
    return "?"


def _conf_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    confs = [float(r["target"]["conf"]) for r in rows]
    if not confs:
        return {}
    counts: dict[str, int] = {}
    for conf in confs:
        key = _band_bin(conf)
        counts[key] = counts.get(key, 0) + 1
    return {"mean": round(statistics.fmean(confs), 3),
            "median": round(statistics.median(confs), 3),
            "min": round(min(confs), 3), "max": round(max(confs), 3),
            "bins": {k: counts[k] for k in sorted(counts)}}


def summarise(rows: list[dict[str, Any]], reference: Any | None = None) -> dict[str, Any]:
    """Everything the report prints, computed once."""
    out: dict[str, Any] = {
        "rows": len(rows),
        "by_band": {}, "by_source": {}, "by_split": {},
        "by_label": {}, "by_band_label": {}, "confidence": {},
        "acceptance": {}, "boundary": {}, "leakage": corpus.leakage_report(rows, reference),
        "cost": cost.summarise(rows),
    }
    for row in rows:
        tool = str(row["target"]["tool"])
        out["by_band"][row["band"]] = out["by_band"].get(row["band"], 0) + 1
        out["by_source"][row["source"]] = out["by_source"].get(row["source"], 0) + 1
        out["by_split"][row["split"]] = out["by_split"].get(row["split"], 0) + 1
        out["by_label"][tool] = out["by_label"].get(tool, 0) + 1
        key = f"{row['band']}/{tool}"
        out["by_band_label"][key] = out["by_band_label"].get(key, 0) + 1

    for band in corpus.BANDS:
        subset = [r for r in rows if r["band"] == band]
        if subset:
            out["confidence"][band] = _conf_block(subset)
        intended = [r for r in subset if r.get("intended")]
        if intended:
            agreed = sum(1 for r in intended if r["target"]["tool"] == r["intended"])
            out["acceptance"][band] = {
                "rows": len(intended), "agreed": agreed,
                "rate": round(agreed / len(intended), 3),
                "note": ("the frame's own op: a miss here is a bug, not a hard row"
                         if band != "t3" else
                         "the frame's op: a miss means the teacher read the rival")}
        boundary = [r for r in subset if r.get("tier_target")]
        if boundary:
            took_rival = sum(1 for r in boundary if r["target"]["tool"] == r["tier_target"])
            out["boundary"][band] = {
                "rows": len(boundary), "chose_rival": took_rival,
                "rate": round(took_rival / len(boundary), 3)}
    return out


def calibration(audit_csv: pathlib.Path) -> dict[str, Any]:
    """Confidence-band agreement on the human-annotated slice, or an honest unknown."""
    if not audit_csv.exists():
        return {"known": False,
                "reason": (f"no annotated audit slice at {audit_csv.name} — the "
                           "teacher's confidence has never met an outcome, so its "
                           "calibration is UNKNOWN (not zero, not 1.0)")}
    annotated: list[dict[str, str]] = []
    with audit_csv.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            verdict = (row.get("verdict") or "").strip().lower()
            if verdict in ("right", "wrong", "y", "n", "yes", "no", "correct",
                           "incorrect", "true", "false", "1", "0"):
                annotated.append(row)
    if not annotated:
        return {"known": False,
                "reason": (f"{audit_csv.name} has no filled-in verdict column — "
                           "calibration is UNKNOWN until a human annotates it")}
    bins: dict[str, dict[str, Any]] = {}
    for row in annotated:
        try:
            conf = float(row["teacher_conf"])
        except (KeyError, TypeError, ValueError):
            continue
        verdict = (row.get("verdict") or "").strip().lower()
        correct = verdict in ("right", "y", "yes", "correct", "true", "1")
        bucket = bins.setdefault(_band_bin(conf), {"n": 0, "correct": 0, "conf": []})
        bucket["n"] += 1
        bucket["correct"] += int(correct)
        bucket["conf"].append(conf)
    total = sum(b["n"] for b in bins.values())
    ece = 0.0
    table: dict[str, Any] = {}
    for key in sorted(bins):
        b = bins[key]
        acc = b["correct"] / b["n"]
        mean_conf = statistics.fmean(b["conf"])
        ece += (b["n"] / total) * abs(acc - mean_conf)
        table[key] = {"n": b["n"], "correct": b["correct"],
                      "accuracy": round(acc, 3), "mean_conf": round(mean_conf, 3),
                      "gap": round(acc - mean_conf, 3)}
    return {"known": True, "annotated": total, "ece": round(ece, 4),
            "bins": table,
            "caveat": (f"n={total} annotated rows — an ECE this small is a smoke test, "
                       "not a calibration curve (03 §2 asks for hundreds per bin)"
                       if total < CONFIDENCE_FLOOR else
                       f"n={total} annotated rows")}


def _bar(count: int, scale: int = 40) -> str:
    return "#" * max(1, round(count / max(scale, 1) * 20)) if count else ""


def report(stats: dict[str, Any], cal: dict[str, Any]) -> None:
    print(f"rows {stats['rows']}   "
          f"bands {stats['by_band']}   sources {stats['by_source']}   "
          f"splits {stats['by_split']}")
    leak = stats["leakage"]
    print(f"leakage    {'OK' if leak['ok'] else 'COLLISION'} — {leak['checked_rows']} rows "
          f"vs {leak['frozen_rows']} frozen, max ratio {leak['max_ratio']} "
          f"(row {leak['max_ratio_row']})")
    if leak["collisions"]:
        for hit in leak["collisions"]:
            print(f"  COLLISION {hit['id']}: {hit['utterance']!r} ~ {hit['matched']!r}")

    print("\nlabel distribution")
    for tool, count in sorted(stats["by_label"].items(), key=lambda kv: -kv[1]):
        bands = ", ".join(f"{b}={stats['by_band_label'].get(f'{b}/{tool}', 0)}"
                          for b in corpus.BANDS
                          if stats["by_band_label"].get(f"{b}/{tool}"))
        print(f"  {tool:16s} {count:4d}  {_bar(count)}  {bands}")

    print("\nconfidence per band")
    for band, block in stats["confidence"].items():
        print(f"  {band}  n={stats['by_band'][band]:4d}  mean={block['mean']:.3f}  "
              f"median={block['median']:.3f}  min={block['min']:.3f}  "
              f"max={block['max']:.3f}")
        print(f"      bins {block['bins']}")

    print("\nteacher agreement with the frame's op")
    for band, block in stats["acceptance"].items():
        print(f"  {band}  {block['agreed']}/{block['rows']} = {block['rate']:.3f}  "
              f"({block['note']})")
    for band, block in stats.get("boundary", {}).items():
        print(f"  {band}  chose the rival in {block['chose_rival']}/{block['rows']} "
              f"= {block['rate']:.3f}")

    print("\ncalibration (teacher confidence vs human verdict)")
    if not cal.get("known"):
        print(f"  UNKNOWN — {cal['reason']}")
    else:
        print(f"  n={cal['annotated']} annotated, ECE={cal['ece']}  ({cal['caveat']})")
        for key, block in cal["bins"].items():
            print(f"  conf {key}  n={block['n']:3d}  acc={block['accuracy']:.3f}  "
                  f"mean_conf={block['mean_conf']:.3f}  gap={block['gap']:+.3f}")

    summary = stats["cost"]
    print(f"\ncost       ${summary['usd']:.6f} over {summary['rows']} rows "
          f"({summary['method']}); by method {summary['by_method']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="source", default=str(corpus.OUT_DIR / "labels.jsonl"))
    ap.add_argument("--audit", default=str(corpus.OUT_DIR / "audit.csv"),
                    help="annotated audit CSV; absent -> calibration reported unknown")
    ap.add_argument("--json", action="store_true", help="machine-readable dump")
    ap.add_argument("--dry-run", action="store_true",
                    help="accepted for pipeline uniformity; this verb never calls out")
    args = ap.parse_args(argv)

    rows = corpus.read_jsonl(pathlib.Path(args.source).expanduser())
    if not rows:
        raise SystemExit(f"no labelled rows at {args.source} — run label.py first")
    for row in rows:
        corpus.assert_labelled(row)
    stats = summarise(rows)
    cal = calibration(pathlib.Path(args.audit).expanduser())
    if args.json:
        import json

        print(json.dumps({"stats": stats, "calibration": cal}, indent=2, sort_keys=True))
        return 0
    report(stats, cal)
    return 0 if stats["leakage"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
