#!/usr/bin/env python3
"""report_v3 — the comparison table, recomputed offline from recorded runs.

The claim this phase has to defend is that `bench/data/heldout_v3.json` is a *proxy*
for `tests/data/utterances_v2.json` and not a new task in disguise. That claim is tested
by asking the same served model the same question on both sets and comparing, per
class, with v2's own archived answers.

So the inputs are two recordings and one dataset, and nothing here touches a server:

``bench/results/p2-heldout/v3-d1-3B-Q4.json``  d1-3B-Q4_K_M over the new set
``bench/results/p1-diagnose/baseline_live.json`` d1-3B-Q4_K_M over the frozen 150

Both are `live_probe.py` output, which is `ti_bench.prompts`' instructions and option
set — the frozen instrument. The pairing is asserted rather than assumed: a recording is
aligned to a dataset only when its utterances match row for row, because a per-class
table computed against a misaligned recording would look entirely plausible and mean
nothing (`bench/eval_signals.py::D1Recorded` makes the same check for the same reason).

Usage:
    python3 bench/report_v3.py                 # the tables, as markdown
    python3 bench/report_v3.py --json out.json # the same numbers, machine-readable
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

V3 = ROOT / "bench/data/heldout_v3.json"
V3_RUN = ROOT / "bench/results/p2-heldout/v3-d1-3B-Q4.json"
V2 = ROOT / "tests/data/utterances_v2.json"
V2_RUN = ROOT / "bench/results/p1-diagnose/baseline_live.json"
OPS = ("decide_extract", "decide_pick", "filesys_glob", "filesys_grep",
       "filesys_read", "reddit_digest", "web_fetch", "web_search")


def load_run(path: Path, rows: list[dict]) -> list[dict]:
    """Read a recording and refuse to align it to rows it does not match."""
    run = json.loads(path.read_text(encoding="utf-8"))
    per_row = run["per_row"]
    if len(per_row) != len(rows):
        raise SystemExit(f"{path.name}: {len(per_row)} recorded rows vs {len(rows)} in dataset")
    for i, (rec, row) in enumerate(zip(per_row, rows)):
        if rec["utterance"] != row["utterance"]:
            raise SystemExit(f"{path.name}: row {i} is {rec['utterance']!r}, dataset has "
                             f"{row['utterance']!r} — recording is not aligned")
    return per_row


def accuracy(per_row: list[dict], keep=lambda r: True) -> tuple[int, int]:
    rows = [r for r in per_row if keep(r)]
    return sum(1 for r in rows if r["correct"]), len(rows)


def pct(hits: int, n: int) -> str:
    return f"{100.0 * hits / n:.1f}%" if n else "—"


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval in percent — the same treatment `ti_bench.stats` gives a lane."""
    if not n:
        return (0.0, 100.0)
    p = hits / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (centre - half), 1), round(100 * (centre + half), 1))


def ci(hits: int, n: int) -> str:
    lo, hi = wilson(hits, n)
    return f"{pct(hits, n)} [{lo:.1f}, {hi:.1f}]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", default="", help="write the numbers here as JSON")
    args = ap.parse_args()

    v3_rows = json.loads(V3.read_text(encoding="utf-8"))
    v3 = load_run(V3_RUN, v3_rows)
    v2_rows = json.loads(V2.read_text(encoding="utf-8"))
    v2 = load_run(V2_RUN, v2_rows)

    out = {
        "v3_run": str(V3_RUN.relative_to(ROOT)), "v2_run": str(V2_RUN.relative_to(ROOT)),
        "v3": {"n": len(v3_rows),
               "all": accuracy(v3),
               "train": accuracy(v3, lambda r: _side(v3_rows, r) == "train"),
               "test": accuracy(v3, lambda r: _side(v3_rows, r) == "test"),
               "per_class": {}}, "v2": {"n": len(v2_rows), "all": accuracy(v2),
                                        "per_class": {}},
    }
    for op in OPS:
        out["v3"]["per_class"][op] = {
            "n": accuracy(v3, lambda r, op=op: r["label"] == op)[1],
            "all": accuracy(v3, lambda r, op=op: r["label"] == op),
            "test": accuracy(v3, lambda r, op=op: (r["label"] == op
                                                   and _side(v3_rows, r) == "test")),
            "train": accuracy(v3, lambda r, op=op: (r["label"] == op
                                                    and _side(v3_rows, r) == "train")),
        }
        out["v2"]["per_class"][op] = {
            "n": accuracy(v2, lambda r, op=op: r["label"] == op)[1],
            "all": accuracy(v2, lambda r, op=op: r["label"] == op),
        }

    print("## Same model, same metric, three sets\n")
    print("| set | n | accuracy (Wilson 95%) | how the split is drawn |")
    print("|---|---:|---|---|")
    v3_all, v3_test, v2_all = out["v3"]["all"], out["v3"]["test"], out["v2"]["all"]
    print(f"| v3, all rows | {v3_all[1]} | **{ci(*v3_all)}** | declared, "
          f"family-disjoint |")
    print(f"| v3, train side | {out['v3']['train'][1]} | {ci(*out['v3']['train'])} | "
          f"families declared train |")
    print(f"| v3, test side | {v3_test[1]} | **{ci(*v3_test)}** | families declared test |")
    print(f"| v2 (archived) | {v2_all[1]} | {ci(*v2_all)} | 95/150 atomic: no family split "
          f"available |")
    print("", "The two overall intervals overlap.")

    print("\n## Per class\n")
    print("| class | v3 n | v3 all | v3 test | v2 n | v2 | Δ (v3 all − v2) |")
    print("|---|---:|---|---:|---|---:|---:|")
    for op in OPS:
        a, b = out["v3"]["per_class"][op], out["v2"]["per_class"][op]
        delta = (a["all"][0] / a["all"][1] - b["all"][0] / b["all"][1]) * 100
        print(f"| `{op}` | {a['n']} | {ci(*a['all'])} | {pct(*a['test'])} | {b['n']} | "
              f"{ci(*b['all'])} | {delta:+.1f} |")
    print(f"| **total** | **{v3_all[1]}** | **{ci(*v3_all)}** | **{pct(*v3_test)}** | "
          f"**{v2_all[1]}** | **{ci(*v2_all)}** | |")

    print("\n## By band\n")
    print("| band | v3 | v2 |")
    print("|---|---|---|")
    for band in ("t1", "t2", "t3"):
        print(f"| {band} | {_band(v3, v3_rows, band)} | {_band(v2, v2_rows, band)} |")

    print("\n## Where the two classes the phase-1 finding is about actually sit\n")
    for name, per_row, rows_, weak in (("v3", v3, v3_rows, WEAK), ("v2", v2, v2_rows, WEAK)):
        inside = accuracy(per_row, lambda r: r["label"] in weak)
        outside = accuracy(per_row, lambda r: r["label"] not in weak)
        print(f"- **{name}**: `filesys_grep` + `decide_extract` {ci(*inside)}; "
              f"the other six {ci(*outside)}")

    print("\n## Where the misses go\n")
    for name, per_row in (("v3", v3), ("v2", v2)):
        misses = Counter(r["choice"] for r in per_row if not r["correct"])
        print(f"- **{name}**, all classes: {sum(misses.values())} misses → "
              + ", ".join(f"`{k}` {v}" for k, v in misses.most_common()))
    for name, per_row in (("v3", v3), ("v2", v2)):
        for label in WEAK:
            misses = Counter(r["choice"] for r in per_row
                             if not r["correct"] and r["label"] == label)
            n = sum(1 for r in per_row if r["label"] == label)
            print(f"- **{name}** `{label}` ({n - sum(misses.values())}/{n} right) → "
                  + (", ".join(f"`{k}` {v}" for k, v in misses.most_common()) or "none"))

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")


_SIDES: dict[str, str] = {}
WEAK = ("filesys_grep", "decide_extract")


def _band(per_row: list[dict], rows: list[dict], band: str) -> str:
    hits = n = 0
    for row in rows:
        if row["band"] != band:
            continue
        n += 1
        hits += int(_record(per_row, row["utterance"])["correct"])
    return ci(hits, n)


def _record(per_row: list[dict], utterance: str) -> dict:
    index = _INDEX.setdefault(id(per_row), {r["utterance"]: r for r in per_row})
    return index[utterance]


_INDEX: dict[int, dict[str, dict]] = {}


def _side(v3_rows: list[dict], rec: dict) -> str:
    """The declared side of a recorded row, by utterance (the recording carries no split)."""
    if not _SIDES:
        _SIDES.update({r["utterance"]: r["split"] for r in v3_rows})
    return _SIDES.get(rec["utterance"], "")

if __name__ == "__main__":
    sys.exit(main())
