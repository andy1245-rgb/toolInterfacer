#!/usr/bin/env python3
"""P1 replay bench — model-free end-to-end run through the real engine.

Runs a plan (fixture decide backend, zero network) and prints hops,
confidences, and wall time.

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
        de_chops = [h for h in out["hops"] if str(h["op"]).startswith("decide") or h["op"] == "askback"]
        print("\n== summary ==")
        print(f"status={out['status']} wall_ms={out.get('wall_ms', wall):.1f} hops={len(out['hops'])}")
        for h in out["hops"]:
            print(f"  {h['op']:<14} {h['ms']:>7.1f}ms conf={h['conf']:.2f}  {str(h['artifact_summary'])[:90]}")
        if de_chops:
            print(f"decide-max conf={max(h['conf'] for h in de_chops):.2f} "
                  f"(gate={load()['policy']['gate_route']})")


if __name__ == "__main__":
    main()
