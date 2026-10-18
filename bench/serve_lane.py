#!/usr/bin/env python3
"""serve_lane.py <lane> — argv-safe llama-server launcher for one bench lane.

Builds the serve argv in-process (and serializes any chat-template-kwargs with
``json.dumps``) so no part of the command line round-trips through a shell. An
unquoted ``{"enable_thinking": false}`` is a PowerShell 5.1 parse error: it killed
the Qwen lane's serve silently while a stale server answered the probes
(docs/model-sweep-consolidated.md, CORRECTION).

Lane definitions live in ``bench/lanes.json`` (model filenames + flags, no host
paths — the model directory is supplied at runtime). Two ways to run:

  python3 bench/serve_lane.py qwen --models-dir D:/models
  python3 bench/serve_lane.py qwen --models-dir D:/models --print-argv

``serve_pyhost`` is imported from ``--serve-pyhost`` / ``$TI_PYHOST_DIR``, else
the directory of this file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_LANES = HERE / "lanes.json"


def load_lanes(path: str | Path) -> dict:
    cfg = json.loads(Path(path).read_text())
    if not cfg.get("lanes"):
        raise SystemExit(f"{path}: no lanes defined")
    return cfg


def _is_absolute(path: str) -> bool:
    # posix absolute, or a Windows drive path (C:\… / C:/…) even off-Windows
    return os.path.isabs(path) or (len(path) > 1 and path[1] == ":")


def build_argv(cfg: dict, lane: str, models_dir: str = "") -> list[str]:
    if lane not in cfg["lanes"]:
        raise SystemExit(f"unknown lane '{lane}' (known: {', '.join(sorted(cfg['lanes']))})")
    spec = cfg["lanes"][lane]
    model = spec["model"]
    if models_dir and not _is_absolute(model):
        model = os.path.join(models_dir, model)
    argv = ["-m", model] + list(cfg.get("base", [])) + list(spec.get("extra", []))
    kwargs = spec.get("chat_template_kwargs")
    if kwargs is not None:
        # serialized here, in-process: this JSON is never written to a shell line
        argv += ["--chat-template-kwargs", json.dumps(kwargs)]
    return argv


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("lane", help="lane name from bench/lanes.json")
    ap.add_argument("--lanes", default=str(DEFAULT_LANES), help="lane config JSON")
    ap.add_argument("--models-dir", default=os.environ.get("TI_MODELS_DIR", ""),
                    help="directory prepended to the lane's model filename")
    ap.add_argument("--serve-pyhost", default=os.environ.get("TI_PYHOST_DIR", str(HERE)),
                    help="directory containing serve_pyhost.py")
    ap.add_argument("--print-argv", action="store_true",
                    help="print the built argv as JSON and exit (do not serve)")
    args = ap.parse_args(argv)

    cfg = load_lanes(args.lanes)
    serve_argv = build_argv(cfg, args.lane, args.models_dir)
    if args.print_argv:
        print(json.dumps(serve_argv))
        return 0

    sys.path.insert(0, str(Path(args.serve_pyhost)))
    import serve_pyhost

    sys.argv = ["serve_pyhost.py"] + serve_argv
    serve_pyhost.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
