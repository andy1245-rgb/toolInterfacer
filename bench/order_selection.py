#!/usr/bin/env python3
"""Select an option order on one half of the corpus, then report what it does on the other half.

Choosing the best of N option orders on the same 415 rows is overfitting with extra steps: N
orders give N chances to fit the rows you are also scoring on. So the selection is done on the
training half and the claimed gain is read off the evaluation half, and both numbers are printed.
If the selection does not survive the other half, this script says so in its own output rather
than leaving the reader to work it out.

    python3 bench/order_selection.py                       # the live artifacts, by split
    python3 bench/order_selection.py --dir <dir> --by family

Reads `<dir>/<arm>/rows.jsonl` produced by `bench/harness.run --write-rows`. No model, no server.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = "bench/results/live-orders-2026-11-10"


def _read_rows(rows_file: Path, arm: str | None = None) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for line in rows_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if arm is not None and row.get("arm") != arm:
            continue
        records[row["id"]] = row
    return records


def load_arms(directory: Path) -> dict[str, dict[str, dict]]:
    """arm -> row id -> record, for every arm directory that has a rows file."""
    arms: dict[str, dict[str, dict]] = {}
    for sub in sorted(p for p in directory.iterdir() if p.is_dir()):
        rows_file = sub / "rows.jsonl"
        if not rows_file.exists():
            continue
        records = _read_rows(rows_file)
        if records:
            arms[sub.name] = records
    return arms


def load_from_rows_file(path: Path) -> dict[str, dict[str, dict]]:
    """Every arm inside one multi-arm rows.jsonl — how a recorded permutation joins the set."""
    by_arm: dict[str, dict[str, dict]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        by_arm.setdefault(row["arm"], {})[row["id"]] = row
    return {name: recs for name, recs in by_arm.items() if recs}


def halve(records: dict[str, dict], by: str) -> tuple[set[str], set[str]]:
    """The two halves: the corpus's declared split, or the family's parity (deterministic)."""
    if by == "split":
        def is_train(row: dict) -> bool:
            # newer artifacts carry the split in meta; the 2026-11-03 records carry it at the top
            return ((row.get("meta") or {}).get("split") or row.get("split")) == "train"
        first = {i for i, r in records.items() if is_train(r)}
        return first, set(records) - first
    first = {i for i, r in records.items()
             if sum(ord(c) for c in str(r.get("group", ""))) % 2 == 0}
    return first, set(records) - first


def accuracy(records: dict[str, dict], ids: set[str]) -> tuple[int, int]:
    right = sum(1 for i in ids if records[i]["correct"])
    return right, len(ids)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--by", default="split", choices=("split", "family"),
                    help="how to halve the rows: the declared split, or family parity")
    ap.add_argument("--noun", default="order",
                    help="what is being selected (order, weight, …) — used in the verdict")
    ap.add_argument("--reference", default="",
                    help="the arm every other is compared against (default: order_frozen, else "
                         "the first arm by name)")
    ap.add_argument("--only", default="",
                    help="comma-separated arm names to keep; the rest are ignored")
    ap.add_argument("--rows", action="append", default=[],
                    help="an extra multi-arm rows.jsonl to include (repeatable); arm names come "
                         "from the file")
    args = ap.parse_args(argv)

    directory = Path(args.dir)
    if not directory.is_absolute():
        directory = ROOT / directory
    if not directory.exists():
        raise SystemExit(f"{directory} does not exist — run the live arms first")

    arms = load_arms(directory)
    for extra in args.rows:
        path = Path(extra)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise SystemExit(f"--rows {path} does not exist")
        for name, records in load_from_rows_file(path).items():
            arms.setdefault(name, records)
    if args.only:
        keep = {a.strip() for a in args.only.split(",") if a.strip()}
        arms = {name: recs for name, recs in arms.items() if name in keep}
        if not arms:
            raise SystemExit(f"--only {sorted(keep)} matched no arm")
    if len(arms) < 2:
        raise SystemExit(f"{directory} holds {len(arms)} arm(s); selection needs at least two")

    reference = (args.reference or
                 next((n for n in ("order_frozen", "w50", "floor_a") if n in arms), sorted(arms)[0]))
    any_id = next(iter(arms[reference]))
    first, second = halve(arms[reference], args.by)
    if not first or not second:
        raise SystemExit(f"halving by {args.by} produced an empty half")

    print(f"{directory.name}: {len(arms)} arms, {len(any_id and arms[reference])} rows, "
          f"halves by {args.by} = {len(first)} / {len(second)}\n")
    print(f"{'arm':<16}{'selection half':>16}{'other half':>13}{'vs frozen (other)':>20}")
    table: dict[str, tuple[float, float]] = {}
    for name, records in arms.items():
        a_right, a_n = accuracy(records, first)
        b_right, b_n = accuracy(records, second)
        a = a_right / a_n if a_n else 0.0
        b = b_right / b_n if b_n else 0.0
        table[name] = (a, b)
        ref_b = accuracy(arms[reference], second)
        delta = b_right - ref_b[0]
        print(f"{name:<16}{a_right:>8}/{a_n:<7}{b_right:>7}/{b_n:<5}{delta:>+20}")

    best = max(table.items(), key=lambda kv: (kv[1][0], kv[1][1]))
    frozen = table[reference]
    best_name, (best_a, best_b) = best
    gain_selected = best_a - frozen[0]
    gain_realised = best_b - frozen[1]
    print(f"\nselected on the {args.by} half: {best_name} "
          f"({best_a:.3f} against {reference}'s {frozen[0]:.3f}, +{gain_selected:.1%} there)")
    print(f"that {args.noun} on the other half: {best_b:.3f} against {frozen[1]:.3f} "
          f"= {gain_realised:+.1%} realised")
    if best_name == reference:
        print(f"\nverdict: {reference} won the selection too — no {args.noun} change is supported.")
        return 0
    if gain_realised <= 0:
        print(f"\nverdict: the selection did not transfer. The {args.noun} that won the {args.by} "
              f"half loses {abs(gain_realised):.1%} on the other half, so the gain was fitted to "
              f"the half it was chosen on and there is no {args.noun} result here.")
        return 0
    print(f"\nverdict: the selection transferred, at {gain_realised:+.1%} on the half it was not "
          f"chosen on. That is the number to quote, not the {gain_selected:+.1%} selection gain.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
