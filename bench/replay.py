#!/usr/bin/env python3
"""P1 replay bench — run one plan through the real engine and print its receipt.

It uses the **configured** backend (default `pair`), so it is model-free only
once `TI_CONF_OVERLAY` points the choice decision at `fixture` (README
Quickstart), and it makes no network call only for a plan with no web op: an L2
`research` plan runs the real `web_search` / `web_fetch` CLIs against live pages
and takes minutes. The all-offline run is `bench/demo.py`, which pins the
decisions *and* both CLIs (`tests/data/fake_cli.py`).

Usage: python bench/replay.py [--intent "read the file we discussed"]
       python bench/replay.py --plan   # dry: resolved steps only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))

from ti.conf import load  # noqa: E402
from ti.ops.op_api import import_all  # noqa: E402
from ti.service_entry import use_tool  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--intent", default="read the file we discussed")
    ap.add_argument("--cwd", default=str(ROOT))
    ap.add_argument("--plan", action="store_true", help="dry mode: resolved steps only")
    args = ap.parse_args()

    import_all()
    _ = load()  # fail fast on config problems

    body = {"intent": args.intent, "mode": "plan" if args.plan else "auto",
            "digest": {"cwd": args.cwd}}
    t0 = time.monotonic()
    out = use_tool(body)
    wall = (time.monotonic() - t0) * 1000

    print(json.dumps(out, indent=2, default=str)[:4000])
    if out["status"] == "done" and not args.plan:
        de_chops = [h["conf"] for h in out["hops"]
                    if (str(h["op"]).startswith("decide") or h["op"] == "askback")
                    and h["conf"] is not None]
        print("\n== summary ==")
        print(f"status={out['status']} wall_ms={out.get('wall_ms', wall):.1f} hops={len(out['hops'])}")
        for h in out["hops"]:
            conf = "n/a" if h["conf"] is None else f"{h['conf']:.2f}"
            print(f"  {h['op']:<14} {h['ms']:>7.1f}ms conf={conf:<5}  {str(h['artifact_summary'])[:90]}")
        if de_chops:
            print(f"decide-max conf={max(de_chops):.2f} "
                  f"(gate={load()['policy']['gate_route']})")
    # A run that failed is not a successful run. Without this the script answered
    # exit 0 for `status: error`, so a wrapper, a CI job or a `&&` chain read it as
    # a pass (found by the first-contact review,
    # docs/quickstart-stranger-2026-11-03.md §3).
    if out["status"] == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
