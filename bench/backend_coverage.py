#!/usr/bin/env python3
"""Which offline backends answer, and how often they are right.

The classical tier abstained on every request it ever received and nothing noticed, because the
only number anyone kept was accuracy — and a tier that answers nothing has a perfect accuracy on
nothing. This prints the number that would have caught it: **answered**, per backend, beside the
precision of the answers.

    python3 bench/backend_coverage.py                 # eval split only
    python3 bench/backend_coverage.py --all           # all rows, with the split marked

No server, no key, no network: a backend that needs one is listed as `needs-server` rather than
silently scoring zero. That distinction is the point — "cannot run here" and "ran and said
nothing" used to look identical.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "service"))

from bench.harness.rows import load  # noqa: E402

OFFLINE = ("heuristic_micro", "rule_extract", "fixture", "classical")
NEEDS_SERVER = ("d1_systemone", "logit_read", "openrouter")
CRITERIA_TYPES = {
    # what an op-choosing backend needs to see: the eight ops, as the router offers them
    "choice": None,
}


def _make(name: str):
    from ti.decide.primitives import make_backend

    return make_backend(name)


def measure(name: str, rows) -> dict:
    backend = _make(name)
    answered = right = errors = 0
    first = ""
    for row in rows:
        try:
            answer = backend.choice(row.text, "", {o: o for o in row.options})
        except Exception as exc:  # noqa: BLE001 - a raise is a result, not a crash
            errors += 1
            if not first:
                first = f"{type(exc).__name__}: {exc}"
            continue
        choice = getattr(answer, "choice", None)
        if not choice:
            continue
        answered += 1
        right += 1 if choice == row.label else 0
    return {"n": len(rows), "answered": answered, "right": right, "errors": errors,
            "first_error": first}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="tool_routing")
    ap.add_argument("--all", action="store_true", help="score every row, not just the eval split")
    args = ap.parse_args(argv)

    rows = list(load(args.dataset))
    if not args.all:
        rows = [r for r in rows if r.meta.get("split") == "test"]
    print(f"{args.dataset}: {len(rows)} rows "
          f"({'all' if args.all else 'eval split'})\n")
    print(f"{'backend':<16}{'answered':>10}{'abstained':>11}{'errors':>8}"
          f"{'precision':>11}  verdict")
    for name in OFFLINE:
        try:
            m = measure(name, rows)
        except Exception as exc:  # noqa: BLE001
            print(f"{name:<16}{'-':>10}{'-':>11}{'-':>8}{'-':>11}  cannot run: "
                  f"{type(exc).__name__}: {exc}")
            continue
        precision = (m["right"] / m["answered"]) if m["answered"] else None
        if m["answered"] == 0 and "extract-only" in m["first_error"]:
            verdict = "serves extract, not choice — not a choice backend"
        elif m["answered"] == 0:
            verdict = "DEAD — answered nothing, and its accuracy says nothing"
        elif precision is not None and precision < 0.5:
            verdict = "answering badly"
        else:
            verdict = "alive"
        pretty = f"{100 * precision:.1f}%" if precision is not None else "-"
        print(f"{name:<16}{m['answered']:>10}{m['n'] - m['answered']:>11}{m['errors']:>8}"
              f"{pretty:>11}  {verdict}")
    for name in NEEDS_SERVER:
        print(f"{name:<16}{'-':>10}{'-':>11}{'-':>8}{'-':>11}  needs a server or a key; "
              f"not measured here")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
