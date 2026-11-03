#!/usr/bin/env python3
"""demo — one intent in, one answer out, with the receipt that shows what happened.

    python3 bench/demo.py                # offline: fixture decisions, stub CLIs
    python3 bench/demo.py --live         # the shipped config: real model, real CLIs

Three scenes, each a different shape of call:

  L0  the caller names the operation        -> no decision is made at all
  L1  a vague intent over real files        -> the service globs, picks and reads
  L2  a compound intent                     -> a step graph, gated, ending in a digest

The point of the demo is the **hop list**. A normal agent needs one model round trip
per tool call; here the model makes one call and the service reports every hop it
took to answer, with the confidence each one reached. That receipt is the product.

Everything is labelled: the header says which decision backend answered and whether
the CLI tools were real or the repo's own fixtures, because a demo that quietly runs
against recorded answers proves nothing.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAKE_CLI = ROOT / "tests" / "data" / "fake_cli.py"

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
GREEN, YELLOW, RED, BLUE, CYAN = "\033[32m", "\033[33m", "\033[31m", "\033[34m", "\033[36m"
USE_COLOUR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}" if USE_COLOUR else text


def _b(text: str) -> str:
    return f"{BOLD}{text}{RESET}" if USE_COLOUR else text


def _d(text: str) -> str:
    return f"{DIM}{text}{RESET}" if USE_COLOUR else text


# --------------------------------------------------------------------- service

class Service:
    """A real decision service on a free port, torn down on exit."""

    def __init__(self, *, live: bool) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="ti-demo-"))
        self.proc: subprocess.Popen | None = None
        self.port = _free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.token = secrets.token_urlsafe(24)
        self.live = live

    def start(self) -> None:
        overlay = {"http": {"port": self.port},
                   "hops_log": {"path": str(self.tmp / "hops.jsonl")}}
        if not self.live:
            # Deterministic and offline: recorded decisions, and the web CLIs
            # replaced by the repo's own fixtures so scene 3 can run anywhere.
            overlay["decide"] = {
                "backends_in_use": {"choice": "fixture", "noul": "fixture",
                                    "extract": "rule_extract"},
                "cache": {"enabled": False},
                "fixture": {"default_choice_conf": 0.93},
            }
            overlay["ops"] = {
                "web_search": {"script": str(FAKE_CLI), "backend": "free-search", "brave": "off"},
                "web_fetch": {"script": str(FAKE_CLI), "max_chars": 600},
                "reddit": {"cwd": str(self.tmp / "no-reddit"), "cwd_candidates": [],
                           "cli": ["fake"]},
                "cli_timeout_s": 20,
            }
        path = self.tmp / "overlay.json"
        path.write_text(json.dumps(overlay), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": "service",
               "TI_CONF_OVERLAY": str(path), "TI_HTTP_TOKEN": self.token}
        self.proc = subprocess.Popen([sys.executable, "-m", "ti.http.api"], cwd=str(ROOT),
                                     env=env, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, start_new_session=True)
        _wait_health(f"{self.base}/health", 25.0)

    def health(self) -> dict:
        with urllib.request.urlopen(f"{self.base}/health", timeout=10) as resp:
            return json.loads(resp.read())

    def call(self, body: dict) -> tuple[dict, float]:
        t0 = time.monotonic()
        req = urllib.request.Request(
            f"{self.base}/v1/useTool", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "X-TI-Token": self.token})
        with urllib.request.urlopen(req, timeout=120) as resp:
            out = json.loads(resp.read())
        return out, (time.monotonic() - t0) * 1000

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), 15)
            except OSError:
                self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        shutil.rmtree(self.tmp, ignore_errors=True)


def _free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_health(url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except (urllib.error.URLError, OSError):
            time.sleep(0.1)
    raise SystemExit(f"demo: the service did not answer {url} within {timeout:.0f}s")


# ----------------------------------------------------------------------- render

def _rule(width: int = 78) -> str:
    return _d("─" * width)


def _pad(text: str, width: int, *, right: bool = False) -> str:
    """Pad *before* colouring. `f"{coloured:<16}"` counts the ANSI escapes as
    characters, so a coloured column drifts by however many bytes the colour is."""
    return text.rjust(width) if right else text.ljust(width)


def _hop_table(hops: list[dict]) -> None:
    if not hops:
        print("  " + _d("(no hops)"))
        return
    print("  " + _d(_pad("step", 16) + _pad("op", 18) + _pad("ms", 9, right=True)
                   + _pad("conf", 7, right=True) + "  artifact"))
    for hop in hops:
        conf = hop.get("conf")
        shown = "  —" if conf is None else f"{conf:.2f}"
        colour = GREEN if (conf is not None and conf >= 0.7) else (
            YELLOW if conf is not None else DIM)
        summary = " ".join(str(hop.get("artifact_summary") or "").split())
        print("  " + _pad(str(hop.get("step"))[:15], 16)
              + _pad(str(hop.get("op"))[:17], 18)
              + _pad(f"{hop.get('ms', 0):.1f}", 9, right=True)
              + _c(_pad(shown, 7, right=True), colour)
              + "  " + _d(summary[:34]))


def _result_preview(out: dict, width: int = 300) -> str:
    result = out.get("result")
    if isinstance(result, dict) and "sources" in result:
        sources = result.get("sources") or []
        queries = result.get("queries") or []
        parts = [f"{len(sources)} source(s)"]
        if queries:
            parts.append("queries: " + ", ".join(str(q) for q in queries[:3]))
        if sources:
            parts.append("first: " + str(sources[0].get("url", "")))
        skipped = result.get("skipped") or []
        if skipped:
            parts.append(f"{len(skipped)} skipped")
        return " · ".join(parts)
    if isinstance(result, dict):
        text = json.dumps(result, default=str)
    elif isinstance(result, list):
        text = ", ".join(str(x) for x in result[:4]) + (" …" if len(result) > 4 else "")
    else:
        text = str(result)
    text = " ".join(text.split())
    return text[:width] + ("…" if len(text) > width else "")


def scene(number: int, title: str, intent: str, body: dict, service: Service,
          note: str) -> dict:
    print()
    print(_b(f"scene {number} — {title}"))
    print(f"  intent  {_c(repr(intent), CYAN)}")
    print(_d(f"          {note}"))
    out, wall = service.call(body)

    print()
    _hop_table(out.get("hops") or [])
    print()
    status = out.get("status", "?")
    colour = GREEN if status == "done" else (YELLOW if status == "need_input" else RED)
    conf = out.get("confidence")
    conf_txt = "absent (nothing was measured)" if conf is None else f"{conf:.2f}"
    print(f"  status {_c(status, colour)}   confidence {conf_txt}   "
          f"{_b(str(len(out.get('hops') or [])))} hops in {wall:.0f} ms wall")
    if out.get("degraded"):
        print(f"  {_c('degraded', YELLOW)} — " + "; ".join(out.get("assumptions") or []))
    if out.get("error"):
        print(f"  {_c('error', RED)} — {out['error'][:150]}")
    if out.get("question"):
        q = out["question"]
        print(f"  {_c('asks back', YELLOW)} — {q.get('question', '')[:110]}")
        for option in (q.get("options") or [])[:4]:
            print(_d(f"      · {option.get('id')}"))
    if out.get("result") is not None:
        print(f"  result {_result_preview(out)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live", action="store_true",
                    help="use the shipped config: the configured decision model and the "
                         "real web/reddit CLIs (needs a model server and configured CLIs)")
    ap.add_argument("--keep-services", action="store_true",
                    help="leave the service running and print its URL (for poking at it)")
    args = ap.parse_args()

    demo_dir = Path(tempfile.mkdtemp(prefix="ti-demo-files-"))
    (demo_dir / "notes.md").write_text(
        "# Project notes\n\nThe decision model runs one forward pass.\n", encoding="utf-8")
    (demo_dir / "budget.csv").write_text("item,qty\nbolts,12\nnuts,40\n", encoding="utf-8")
    (demo_dir / "server-out.log").write_text("[ready] :8080\n", encoding="utf-8")

    service = Service(live=args.live)
    service.start()
    # `python3 bench/demo.py | head` closes the pipe mid-print, and a
    # BrokenPipeError raised during interpreter flush skips a `finally`. The
    # service runs in its own session, so without this it outlives the demo and
    # holds its port — observed while writing this file. `stop` is idempotent.
    atexit.register(service.stop)
    atexit.register(shutil.rmtree, demo_dir, True)
    try:
        health = service.health()
        decide = health.get("decide") or {}

        print()
        print(_b("toolInterfacer") + _d("  ·  one intent in, one answer out, plus the receipt"))
        print(_rule())
        print(f"  service   {service.base}   ({len(health.get('ops') or [])} ops registered)")
        choice = decide.get("choice", "?")
        backend_note = "recorded answers — deterministic, offline" if choice == "fixture" \
            else "a real decision backend"
        print(f"  deciding  {_c(str(choice), CYAN)}  {_d('(' + backend_note + ')')}")
        if args.live:
            print(f"  tools     {_d('the configured web/reddit CLIs')}")
        else:
            print(f"  tools     {_d('the repo fixtures (tests/data/fake_cli.py) — deterministic')}")
        print(_d(f"  files     {demo_dir}"))
        print(_rule())

        if choice == "fixture":
            print(_d("  note: scene 3's picks are recorded answers, not a model's. Run with"))
            print(_d("        --live to see the configured backend actually decide."))

        scene(1, "the caller names the operation", "read notes.md",
              {"intent": "read the notes file",
               "digest": {"cwd": str(demo_dir)},
               "params": {"tool": "filesys_read",
                          "args": {"path": "notes.md"}}},
              service,
              "L0 passthrough — no decision is made, so there is nothing to reason about")

        scene(2, "a vague intent, resolved against real files", "read the notes file",
              {"intent": "read the notes file", "digest": {"cwd": str(demo_dir)}},
              service,
              "L1 slot fill — the service globs the directory, picks, then reads")

        scene(3, "a compound intent", "research lora rank for a 3b router",
              {"intent": "research lora rank for a 3b router",
               "digest": {"cwd": str(demo_dir)}},
              service,
              "L2 template — extract queries, search, pick, fetch with a relevance gate, "
              "assemble a digest")

        print()
        print(_rule())
        print(_b("  the whole point: ") +
              "the model made one call per scene, and got every hop above as evidence.")
        print(_d("  A normal agent makes one model round trip per tool call. Here the number"))
        print(_d("  of hops and the number of round trips are decoupled — which helps when a"))
        print(_d("  round trip is expensive and hurts when the local decision is. Measured:"))
        print(_d("  docs/roundtrips-2026-11-03.md."))
        print(_rule())
        print()

        if args.keep_services:
            input(f"service left running at {service.base} — press enter to stop ")
        return 0
    finally:
        service.stop()
        shutil.rmtree(demo_dir, ignore_errors=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:          # `| head` is a supported way to run a demo
        raise SystemExit(0) from None
