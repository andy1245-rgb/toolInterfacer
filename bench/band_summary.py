#!/usr/bin/env python3
"""Per-band (t1/t2/t3) breakdown of a backend sweep over utterances_v2.

Joins each result file's per_row entries to the band map in the utterances
file (by idx) and reports accuracy + latency per band, pooled across runs.

Usage: python bench/band_summary.py [--utterances tests/data/utterances_v2.json]
       [--results-dir bench/results/dell] [--glob 'v2-*.json']
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANDS = ("t1", "t2", "t3")


def percentile(vals: list[float], pct: float) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    idx = (len(vals) - 1) * pct
    lo, hi = int(idx), min(int(idx) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (idx - lo)


def lane_of(path: Path) -> str:
    return path.name[:-5].replace("-run2", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--utterances", default=str(ROOT / "tests/data/utterances_v2.json"))
    ap.add_argument("--results-dir", default=str(ROOT / "bench/results/dell"))
    ap.add_argument("--glob", default="v2-*.json")
    args = ap.parse_args()

    bands = [r["band"] for r in json.loads(Path(args.utterances).read_text())]
    files = sorted(Path(args.results_dir).glob(args.glob))

    lanes: dict[str, dict] = {}
    for f in files:
        d = json.loads(f.read_text())
        rows = d["per_row"]
        if len(rows) != len(bands):
            raise SystemExit(f"{f.name}: {len(rows)} rows != {len(bands)} bands")
        lane = lanes.setdefault(lane_of(f), {"backend": d["backend"], "runs": []})
        lane["runs"].append((f.name, rows))

    print(f"# per-band sweep — {Path(args.utterances).name} "
          f"({len(bands)} rows: "
          + ", ".join(f"{b}={bands.count(b)}" for b in BANDS) + ")")
    print(f"# {len(files)} result files, {len(lanes)} lanes\n")

    hdr = f"{'lane':<22}{'backend':<18}{'run':<6}"
    for b in BANDS:
        hdr += f" | {b+' acc':>8} {b+' p50':>9}"
    hdr += f" | {'overall':>8} {'p50':>8}"
    print(hdr)
    print("-" * len(hdr))

    for lane, info in lanes.items():
        pooled = {b: {"n": 0, "correct": 0, "ms": []} for b in BANDS}
        for name, rows in info["runs"]:
            run = "run2" if "-run2" in name else "run1"
            cell, tot_n, tot_c, all_ms = f"{lane:<22}{info['backend']:<18}{run:<6}", 0, 0, []
            for b in BANDS:
                sel = [(r, bands[i]) for i, r in enumerate(rows) if bands[i] == b]
                n = len(sel)
                c = sum(int(r["correct"]) for r, _ in sel)
                ms = [r["ms"] for r, _ in sel]
                cell += f" | {c}/{n} {c/n:5.0%} {percentile(ms, 0.5):8.0f}"
                pooled[b]["n"] += n
                pooled[b]["correct"] += c
                pooled[b]["ms"] += ms
                tot_n += n
                tot_c += c
                all_ms += ms
            cell += f" | {tot_c/tot_n:7.0%} {percentile(all_ms, 0.5):7.0f}"
            print(cell)
        pl = f"{lane:<22}{info['backend']:<18}{'POOL':<6}"
        for b in BANDS:
            p = pooled[b]
            pl += f" | {p['correct']}/{p['n']} {p['correct']/p['n']:5.0%} {percentile(p['ms'], 0.5):8.0f}"
        print(pl)
        print()


if __name__ == "__main__":
    main()
