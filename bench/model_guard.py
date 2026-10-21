#!/usr/bin/env python3
"""Served-model assertion for bench lanes — the `/props` identity guard.

`/health` answers "ok" for *any* llama-server, so a lane that gates only on
health can silently probe a stale server squatting the port. That is exactly how
four files in `bench/results/dell/` titled "Qwen3.5-4B" came to hold d1-3B-Q4
numbers (see `docs/model-sweep-consolidated.md`, CORRECTION). This module asserts
the served model identity before any probe traffic.

`preflight` is pure (a `/props` dict in, no network) and is the single source of
truth shared by `bench/live_probe.py --expect-model` and the CLI below, which
`bench/lane.sh` calls to fail a lane fast.

Usage:
  python3 bench/model_guard.py --expect-model d1-3B-Q4_K_M.gguf \
      [--url http://127.0.0.1:8080 | --props-file PATH | --props-file -]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

NO_EXPECT_WARNING = (
    "!! " + "=" * 66 + "\n"
    "!! WARNING: --expect-model not set — NO served-model assertion made.\n"
    "!! A stale llama-server owning the port would be measured silently\n"
    "!! (see docs/model-sweep-consolidated.md, CORRECTION).\n"
    "!! Set --expect-model <gguf-basename> to guard this lane.\n"
    "!! " + "=" * 66
)


class ModelAssertionError(Exception):
    """The served model is not the model this lane intended to measure."""


def basename(path: str) -> str:
    """Basename of a model path written with either POSIX or Windows separators."""
    return (path or "").replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def served_model(props: dict) -> dict:
    """Identity fields a /props payload exposes ('' when the payload lacks them)."""
    p = props if isinstance(props, dict) else {}
    path = str(p.get("model_path") or p.get("model_alias") or "")
    return {
        "path": path,
        "basename": basename(path),
        "ftype": str(p.get("model_ftype") or ""),
        "build_info": str(p.get("build_info") or ""),
        "sha": str(p.get("model_sha") or p.get("sha") or ""),
        "size": p.get("model_size", p.get("size")),
    }


def assert_model_match(props: dict, expect_model: str, expect_sha: str = "",
                       expect_size: int | None = None, source: str = "/props") -> dict:
    """Return the served identity, or raise ModelAssertionError on any mismatch."""
    if not isinstance(props, dict) or not props:
        raise ModelAssertionError(f"{source} is missing or empty — cannot assert the served model")
    served = served_model(props)
    if not served["basename"]:
        raise ModelAssertionError(
            f"{source} carries no model_path/model_alias — cannot assert the served model")
    if served["basename"] != expect_model:
        raise ModelAssertionError(
            f"served model is '{served['basename']}' ({served['path']}), expected "
            f"'{expect_model}' — refusing to probe a different model")
    if expect_sha:
        if not served["sha"]:
            raise ModelAssertionError(f"{source} carries no sha to check against '{expect_sha}'")
        if served["sha"] != expect_sha:
            raise ModelAssertionError(f"served sha '{served['sha']}' != expected '{expect_sha}'")
    if expect_size is not None:
        if served["size"] is None:
            raise ModelAssertionError(
                f"{source} carries no model size to check against {expect_size}")
        if int(served["size"]) != int(expect_size):
            raise ModelAssertionError(f"served size {served['size']} != expected {expect_size}")
    return served


def preflight(props: dict, expect_model: str, expect_sha: str = "",
              expect_size: int | None = None, source: str = "/props") -> tuple[dict, str | None]:
    """The lane's identity gate.

    Returns ``(served_identity, warning)``. When ``expect_model`` is falsy there
    is nothing to check and ``warning`` is the no-assertion banner; otherwise
    ``warning`` is None and a mismatch raises ModelAssertionError.
    """
    if not expect_model:
        return served_model(props), NO_EXPECT_WARNING
    return assert_model_match(props, expect_model, expect_sha, expect_size, source), None


def _fetch_props(base_url: str) -> dict:
    with urllib.request.urlopen(base_url.rstrip("/") + "/props", timeout=10) as resp:
        return json.loads(resp.read())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--expect-model", required=True,
                    help="gguf basename the server must be serving")
    ap.add_argument("--expect-sha", default="", help="optional sha the served model must match")
    ap.add_argument("--expect-size", type=int, default=None,
                    help="optional byte size the served model must match")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--props-file", metavar="PATH",
                     help="read a saved /props JSON ('-' = stdin)")
    src.add_argument("--url", default="http://127.0.0.1:8080",
                     help="base url to GET /props from (default %(default)s)")
    args = ap.parse_args(argv)

    if args.props_file:
        raw = sys.stdin.read() if args.props_file == "-" else Path(args.props_file).read_text()
        source = args.props_file
    else:
        source = args.url.rstrip("/") + "/props"
        try:
            raw = _fetch_props(args.url)
        except Exception as e:  # unreachable server, HTTP error, bad JSON
            print(f"ABORT: cannot read {source}: {e}", file=sys.stderr)
            return 2
    if isinstance(raw, str):
        try:
            props = json.loads(raw.lstrip("\ufeff"))
        except json.JSONDecodeError as e:
            print(f"ABORT: {source} is not valid JSON: {e}", file=sys.stderr)
            return 2
    else:
        props = raw

    try:
        served = assert_model_match(props, args.expect_model, args.expect_sha,
                                    args.expect_size, source)
    except ModelAssertionError as e:
        print(f"ABORT: {e}", file=sys.stderr)
        return 1
    print(f"props OK: serving {served['basename']} (ftype={served['ftype']} "
          f"build={served['build_info']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
