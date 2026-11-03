#!/usr/bin/env python3
"""live_e2e — prove `POST /v1/useTool` works end to end with a real model behind it.

The whole test suite is fixture-driven: `ti.decide.backends.fixture` replays
recorded answers and the op backends are replayed or faked, so nothing in `tests/`
proves that the *shipped* path — HTTP ingress → inspector → plan → decide wall →
ops wall → response contract — runs at all. This probe does, over the wire, and
says which decide backend answered.

What it does:

  1. Asserts the served model before any traffic (`model_guard`, same check as
     `bench/live_probe.py --expect-model`), so a stale server squatting the port
     cannot be measured silently.
  2. Reads `GET /health` for the op registry and the decide backend map: a `done`
     from the replay fixture and a `done` from d1-3B look identical in a response
     body, and only the map tells them apart. A fixture-backed choice backend
     fails the run (`--allow-fixture` accepts it for a smoke test).
  3. Posts a fixed task list and checks each response field by field (spec §3):
     `status`, `hops[{step,op,ms,conf,artifact_summary}]`, `confidence`,
     `sources`, `degraded`, `assumptions`, plus `question` for an ask-back.
  4. Checks the hops against what actually ran: every hop op must be a registered
     op (or one of the runner's own ask-back records), and a `filesys_read` hop
     must be backed by the bytes of a real file.

Attachment is explicit. Without `--serve` it talks to whatever is already
listening (and fails if nothing is); with `--serve` it starts llama-server and
the service on ports nothing else holds and tears both down in a `finally`. The
service it starts gets a bench overlay: the model's base URL, a hops log inside
the run's temp dir, and `--decide-timeout` for `decide.d1_systemone.timeout_s`
(the shipped 30 s is tight for an L2 pick on a shared box — measured 20 s and
61 s for the same request, see docs/live-e2e-2026-11-03.md).

    python3 bench/live_e2e.py --serve                       # start, probe, stop
    python3 bench/live_e2e.py --skip-l2                     # attach; skip the ~3 min L2 case
    python3 bench/live_e2e.py --service http://127.0.0.1:1115 \
        --model-url http://127.0.0.1:8099                   # attach to a running pair

Exit codes: 0 every case passed, 1 a case or contract violation failed,
2 the stack was unreachable or the probe refused to start it.

Ports, model, and server are env-overridable (`TI_E2E_SERVICE_PORT`,
`TI_E2E_MODEL_PORT`, `TI_MODEL`, `TI_SERVER`); nothing machine-specific is baked
in, and this file holds no absolute path that is not derived from its own
location or from `$HOME`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_guard import (  # noqa: E402
    NO_EXPECT_WARNING,
    ModelAssertionError,
    assert_model_match,
    served_model,
)

FIXTURE_DIR = ROOT / "bench" / "fixtures" / "live-e2e"
DEFAULT_EXPECT_MODEL = "d1-3B-Q4_K_M.gguf"
DEFAULT_MODEL_PORT = int(os.environ.get("TI_E2E_MODEL_PORT") or 8099)
DEFAULT_SERVICE_PORT = int(os.environ.get("TI_E2E_SERVICE_PORT") or 1114)

VALID_STATUS = ("done", "need_input", "error", "plan")
QUESTION_KINDS = ("choice", "confirm", "free")
HOP_FIELDS = ("step", "op", "ms", "conf", "artifact_summary")
# core/runner writes these two hops itself; they are not registered ops.
RUNNER_OPS = frozenset({"askback", "askback.resolved"})
FIXTURE_BACKENDS = frozenset({"fixture", "low_pick", "conf_degraded_pick", "cascade_live"})

# A compound intent whose first query keeps a topic term (rule_extract leads with
# the routing verb, see docs/live-e2e-2026-11-03.md), so the L2 chain has
# something to search for.
L2_INTENT = "research llama.cpp quantization options for cpu inference"
L2_TOPIC_TERMS = ("llama", "quantization")

AMBIGUOUS_FILES = ("report-draft.md", "report-final.md", "report-old.md",
                   "report-v1.md", "report-v2.md")
AMBIGUOUS_INTENT = "read the report file"
AMBIGUOUS_ANSWER = "report-v1.md"


# --------------------------------------------------------------------------- io


def http_json(url: str, payload: Any = None, timeout: float = 30.0) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw) if raw else None


def wait_for(url: str, seconds: float, name: str) -> Any:
    """Poll a URL until it answers; the last error is the failure message."""
    deadline = time.monotonic() + seconds
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return http_json(url, timeout=5.0)
        except Exception as exc:  # noqa: BLE001 — any failure means "not up yet"
            last = exc
            time.sleep(0.5)
    raise RuntimeError(f"{name} did not answer {url} within {seconds:.0f}s: {last}")


def abort(message: str) -> int:
    """Print a setup refusal and hand back exit code 2 (the stack was not probed)."""
    print(f"ABORT: {message}", file=sys.stderr)
    return 2


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


# ------------------------------------------------------------------- contract


def summary_payload(summary: Any) -> str:
    """The artifact text a hop summary carries, minus `_summarize`'s `<kind> :: ` label."""
    if not isinstance(summary, str):
        return ""
    _head, sep, rest = summary.partition(" :: ")
    return rest if sep else summary


_LIST_COUNT = re.compile(r"^[a-z_]+\[(\d+)\]")


def summary_count(summary: Any) -> int:
    """The length `_summarize` records for a list artifact (`url_list[4]  e.g. ...`)."""
    if not isinstance(summary, str):
        return -1
    match = _LIST_COUNT.match(summary)
    return int(match.group(1)) if match else -1


def check_contract(body: Any, ops: list[str]) -> list[str]:
    """Spec §3, field by field. Returns every violation; empty means conformant."""
    problems: list[str] = []
    if not isinstance(body, dict):
        return [f"response is {type(body).__name__}, not a JSON object"]

    status = body.get("status")
    if status not in VALID_STATUS:
        problems.append(f"status {status!r} not in {list(VALID_STATUS)}")

    hops = body.get("hops")
    if not isinstance(hops, list):
        problems.append("hops missing or not a list (every useTool body carries it)")
    else:
        for i, hop in enumerate(hops):
            if not isinstance(hop, dict):
                problems.append(f"hops[{i}] is {type(hop).__name__}, not an object")
                continue
            for field_name in HOP_FIELDS:
                if field_name not in hop:
                    problems.append(f"hops[{i}] has no {field_name!r}")
            value = hop.get("ms")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                problems.append(f"hops[{i}].ms is {value!r}, not a number")
            # `conf` is `float | None` by design: a rule extractor has no
            # distribution to report and says so rather than inventing one
            # (docs/confidence-sources.md). Absence is not a contract violation.
            hop_conf = hop.get("conf")
            if hop_conf is not None and (not isinstance(hop_conf, (int, float))
                                         or isinstance(hop_conf, bool)):
                problems.append(f"hops[{i}].conf is {hop_conf!r}, not a number or null")
            for text in ("step", "op", "artifact_summary"):
                if text in hop and not isinstance(hop[text], str):
                    problems.append(f"hops[{i}].{text} is {type(hop[text]).__name__}, not a string")
            if not isinstance(hop.get("conf"), (int, float)) or isinstance(hop.get("conf"), bool):
                continue
            if not 0.0 <= float(hop["conf"]) <= 1.0:
                problems.append(f"hops[{i}].conf {hop['conf']} outside [0,1]")

    if status == "done" and body.get("result") is None:
        problems.append("status 'done' with no result — nothing was executed")
    if status == "error":
        if not isinstance(body.get("error"), str) or not str(body.get("error")).strip():
            problems.append("status 'error' with no message of its own")
    elif "error" in body:
        problems.append(f"status {status!r} carries an error field")

    # `confidence` is optional for *every* status: it is present when — and only
    # when — a hop measured one (docs/confidence-sources.md). An L0 passthrough
    # makes no routing decision and an error run delivered nothing, so both omit
    # it; a probe that demanded it scored correct bodies as FAIL
    # (docs/review2-2026-11-03.md NB3).
    confidence = body.get("confidence")
    if confidence is not None:
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            problems.append(f"confidence {confidence!r} is not a number")
        elif not 0.0 <= float(confidence) <= 1.0:
            problems.append(f"confidence {confidence} outside [0,1]")

    if "degraded" in body and not isinstance(body["degraded"], bool):
        problems.append(f"degraded is {type(body['degraded']).__name__}, not a boolean")
    if "assumptions" in body:
        assumptions = body["assumptions"]
        if not isinstance(assumptions, list) or not all(isinstance(a, str) for a in assumptions):
            problems.append("assumptions is not a list of strings")
    if "sources" in body:
        sources = body["sources"]
        if not isinstance(sources, list):
            problems.append(f"sources is {type(sources).__name__}, not a list")

    question = body.get("question")
    if status == "need_input" and question is None:
        problems.append("status 'need_input' with no question — the caller cannot answer")
    if question is not None:
        if not isinstance(question, dict):
            problems.append("question is not an object")
        else:
            if question.get("kind") not in QUESTION_KINDS:
                problems.append(f"question.kind {question.get('kind')!r} not in {list(QUESTION_KINDS)}")
            if not isinstance(question.get("question"), str) or not question["question"].strip():
                problems.append("question.question is missing or empty")
            options = question.get("options")
            if options is not None:
                if not isinstance(options, list):
                    problems.append("question.options is not a list")
                else:
                    for i, option in enumerate(options):
                        if not isinstance(option, dict) or not isinstance(option.get("id"), str):
                            problems.append(f"question.options[{i}] has no string id")
                        elif not isinstance(option.get("desc"), str):
                            problems.append(f"question.options[{i}].desc is not a string")

    if isinstance(hops, list):
        known = set(ops) | RUNNER_OPS
        for i, hop in enumerate(hops):
            op = hop.get("op") if isinstance(hop, dict) else None
            if isinstance(op, str) and op not in known:
                problems.append(f"hops[{i}].op {op!r} is neither a registered op nor a runner record")
    return problems


def backend_problem(decide: dict, allow_fixture: bool) -> str:
    """Why this run cannot claim the live wall, or '' when it can.

    A `done` from the replay fixture and a `done` from d1-3B are the same body, so
    the only honest evidence of *which* backend answered is the map the service
    reports. "It passed" is not evidence the model ran.
    """
    if allow_fixture:
        return ""
    choice = str((decide or {}).get("choice") or "")
    if not choice:
        return ("the service reports no decide backend (/health carries no 'decide' map), so "
                "this run cannot say what answered — pass --allow-fixture to accept that")
    if choice in FIXTURE_BACKENDS:
        return (f"the choice backend is {choice!r} — a replay backend proves the transport, "
                "not the model; pass --allow-fixture to accept that")
    return ""


def hop_ops(body: dict) -> list[str]:
    return [str(h.get("op")) for h in body.get("hops") or [] if isinstance(h, dict)]


def hop_for(body: dict, op: str) -> dict | None:
    for hop in body.get("hops") or []:
        if isinstance(hop, dict) and hop.get("op") == op:
            return hop
    return None


def content_backing(body: dict, expected: list[tuple[str, str]]) -> tuple[list[str], str]:
    """Check a read result against real bytes: (problems, name of the file matched)."""
    result = body.get("result")
    if not isinstance(result, str):
        return ([f"result is {type(result).__name__}, not the file text"], "")
    matches = [name for name, text in expected if text == result]
    if not matches:
        return ([f"result matches no file on disk (first 60 chars: {result[:60]!r})"], "")
    name = matches[0]
    problems: list[str] = []
    read = hop_for(body, "filesys_read")
    if read is None:
        return (["no filesys_read hop, yet the result is file text"], name)
    payload = summary_payload(read.get("artifact_summary"))
    if not payload or not result.startswith(payload):
        problems.append("the filesys_read hop's artifact_summary is not this file's text")
    return (problems, name)


# ---------------------------------------------------------------------- cases


@dataclass
class Ctx:
    fixture: Path = FIXTURE_DIR
    ambiguous: Path = Path(".")
    readme: Path = ROOT / "README.md"

    def fixture_files(self) -> list[tuple[str, str]]:
        return [(p.name, p.read_text(encoding="utf-8")) for p in sorted(self.fixture.iterdir())
                if p.is_file()]

    def ambiguous_files(self) -> list[tuple[str, str]]:
        return [(p.name, p.read_text(encoding="utf-8")) for p in sorted(self.ambiguous.iterdir())
                if p.is_file()]


@dataclass
class Case:
    name: str
    requirement: str                  # what the case is here to prove, one line
    build: Callable[[Ctx], dict]
    verify: Callable[[Ctx, dict], list[str]]
    timeout: float = 120.0


def _verify_l0(ctx: Ctx, body: dict) -> list[str]:
    problems: list[str] = []
    if body.get("status") != "done":
        problems.append(f"status {body.get('status')!r}, expected 'done' (L0 needs no decision)")
    if body.get("degraded") is not False:
        problems.append(f"degraded {body.get('degraded')!r}, expected False")
    if hop_ops(body) != ["filesys_read"]:
        problems.append(f"hops {hop_ops(body)}, expected exactly ['filesys_read']")
    steps = [h.get("step") for h in body.get("hops") or []]
    if steps != ["L0"]:
        problems.append(f"hop steps {steps}, expected ['L0']")
    if body.get("confidence") != 1.0:
        problems.append(f"confidence {body.get('confidence')!r}, expected 1.0 (no decision was taken)")
    expected = [(ctx.readme.name, ctx.readme.read_text(encoding="utf-8"))]
    problems.extend(content_backing(body, expected)[0])
    return problems


def _verify_l1(ctx: Ctx, body: dict) -> list[str]:
    problems: list[str] = []
    if body.get("status") != "done":
        problems.append(f"status {body.get('status')!r}, expected 'done' "
                        "(a 3-candidate slot-fill is not ambiguous)")
    if body.get("degraded") is not False:
        problems.append(f"degraded {body.get('degraded')!r}: the slot gate did not pass "
                        f"(confidence {body.get('confidence')!r})")
    if hop_ops(body) != ["filesys_glob", "decide_pick", "filesys_read"]:
        problems.append(f"hops {hop_ops(body)}, expected glob → pick → read")
    pick = hop_for(body, "decide_pick")
    if pick is not None and not isinstance(pick.get("conf"), (int, float)):
        problems.append("the decide_pick hop carries no numeric conf")
    backing, name = content_backing(body, ctx.fixture_files())
    problems.extend(backing)
    if name and pick is not None and name not in str(pick.get("artifact_summary")):
        problems.append(f"the pick hop does not name the file that was read ({name})")
    return problems


def _verify_l2(ctx: Ctx, body: dict) -> list[str]:
    problems: list[str] = []
    if body.get("status") not in ("done", "need_input"):
        problems.append(f"status {body.get('status')!r} — the L2 plan did not run: "
                        f"{body.get('error')!r}")
    ops = hop_ops(body)
    for required in ("decide_extract", "web_search", "decide_pick", "web_fetch"):
        if required not in ops:
            problems.append(f"no {required} hop (hops: {ops})")
    if ops and ops[0] != "decide_extract":
        problems.append(f"the plan started with {ops[0]!r}, not the extract step")
    extract = hop_for(body, "decide_extract")
    if extract is not None:
        queries = summary_payload(extract.get("artifact_summary")).lower()
        if not any(term in queries for term in L2_TOPIC_TERMS):
            problems.append(f"the extract hop's queries carry no intent term: {queries[:120]!r}")
    search = hop_for(body, "web_search")
    if search is not None and summary_count(search.get("artifact_summary")) < 1:
        problems.append("the web_search hop returned no results "
                        f"({search.get('artifact_summary')!r})")
    pick = hop_for(body, "decide_pick")
    if pick is not None and "http" not in summary_payload(pick.get("artifact_summary")):
        problems.append(f"the pick hop chose no URL ({pick.get('artifact_summary')!r})")
    fetch = hop_for(body, "web_fetch")
    if fetch is not None and summary_count(fetch.get("artifact_summary")) < 1:
        problems.append("the web_fetch hop returned no page "
                        f"({fetch.get('artifact_summary')!r})")
    return problems


def _verify_ambiguous(ctx: Ctx, body: dict) -> list[str]:
    problems: list[str] = []
    if body.get("status") != "need_input":
        problems.append(f"status {body.get('status')!r}, expected 'need_input' — an ambiguous "
                        f"pick (conf {body.get('confidence')!r}) under the slot gate must ask, "
                        "not guess")
    question = body.get("question") or {}
    if question.get("kind") != "choice":
        problems.append(f"question.kind {question.get('kind')!r}, expected 'choice' "
                        "(the candidates were enumerated, so the answer is structurable)")
    options = question.get("options") or []
    if len(options) < 2:
        problems.append(f"{len(options)} option(s) listed; a choice question needs the candidates")
    if "askback" not in hop_ops(body):
        problems.append("no askback hop: the question did not come from the gate")
    return problems


def _verify_answer(ctx: Ctx, body: dict) -> list[str]:
    problems: list[str] = []
    if body.get("status") != "done":
        problems.append(f"status {body.get('status')!r}, expected 'done' — the answer must "
                        f"resolve the gate (question: {body.get('question')!r})")
    if body.get("question") is not None:
        problems.append("still asking after the answer was supplied")
    if "askback.resolved" not in hop_ops(body):
        problems.append(f"no askback.resolved hop (hops: {hop_ops(body)})")
    resolved = hop_for(body, "askback.resolved")
    confidence = body.get("confidence")
    if resolved is not None and isinstance(confidence, (int, float)):
        if abs(float(confidence) - float(resolved.get("conf", -1))) > 1e-9:
            problems.append(f"confidence {confidence} does not report the resolved answer's "
                            f"conf {resolved.get('conf')}")
    backing, _name = content_backing(body, ctx.ambiguous_files())
    problems.extend(backing)
    result = body.get("result")
    wanted = dict(ctx.ambiguous_files()).get(AMBIGUOUS_ANSWER, "")
    if isinstance(result, str) and result != wanted:
        problems.append(f"result is not {AMBIGUOUS_ANSWER} — the supplied answer was not used")
    return problems


def cases(skip_l2: bool, l2_timeout: float, only: list[str] | None = None,
          timeout: float = 120.0) -> list[Case]:
    built = [
        Case(
            name="l0_passthrough_read",
            requirement="L0: params fully specify a tool — no decision, real read",
            build=lambda ctx: {"intent": "read the known file",
                               "params": {"tool": "filesys_read",
                                          "args": {"path": str(ctx.readme)}}},
            verify=_verify_l0,
            timeout=timeout,
        ),
        Case(
            name="l1_slot_fill",
            requirement="L1: candidates from disk, one live decision, gated read",
            build=lambda ctx: {"intent": "read the notes file",
                               "digest": {"cwd": str(ctx.fixture)}},
            verify=_verify_l1,
            timeout=timeout,
        ),
        Case(
            name="l2_compound_research",
            requirement="L2: extract → search → pick → fetch through the decide wall",
            build=lambda ctx: {"intent": L2_INTENT, "digest": {"cwd": str(ctx.fixture)}},
            verify=_verify_l2,
            timeout=l2_timeout,
        ),
        Case(
            name="ambiguous_needs_input",
            requirement="gate below threshold with candidates listed → need_input + question",
            build=lambda ctx: {"intent": AMBIGUOUS_INTENT,
                               "digest": {"cwd": str(ctx.ambiguous)}},
            verify=_verify_ambiguous,
            timeout=timeout,
        ),
        Case(
            name="ambiguous_answer_resolves",
            requirement="params.answer resolves the gate the previous call asked about",
            build=lambda ctx: {"intent": AMBIGUOUS_INTENT,
                               "params": {"answer": AMBIGUOUS_ANSWER},
                               "digest": {"cwd": str(ctx.ambiguous)}},
            verify=_verify_answer,
            timeout=timeout,
        ),
    ]
    return [c for c in built
            if not (skip_l2 and c.name == "l2_compound_research")
            and (not only or c.name in only)]


# ----------------------------------------------------------------------- run


@dataclass
class Row:
    case: Case
    body: dict = field(default_factory=dict)
    wall_ms: float = 0.0
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def run_cases(chosen: list[Case], ctx: Ctx, ops: list[str],
              transport: Callable[[dict, float], dict]) -> list[Row]:
    rows: list[Row] = []
    for case in chosen:
        row = Row(case=case)
        body = case.build(ctx)
        started = time.monotonic()
        try:
            row.body = transport(body, case.timeout) or {}
        except Exception as exc:  # noqa: BLE001 — a dead transport is a failed case, not a crash
            row.problems = [f"transport failed: {type(exc).__name__}: {exc}"]
            row.wall_ms = round((time.monotonic() - started) * 1000, 1)
            rows.append(row)
            continue
        row.wall_ms = round((time.monotonic() - started) * 1000, 1)
        row.problems = check_contract(row.body, ops) + case.verify(ctx, row.body)
        rows.append(row)
    return rows


def _fmt_conf(body: dict) -> str:
    conf = body.get("confidence")
    return "-" if not isinstance(conf, (int, float)) else f"{float(conf):.2f}"


def print_table(rows: list[Row]) -> None:
    width = max([len(r.case.name) for r in rows] + [4])
    print(f"\n{'#':>2} {'case':<{width}} {'status':<10} {'conf':>5} {'hops':>4} "
          f"{'wall ms':>9}  verdict")
    print("-" * (width + 44))
    for i, row in enumerate(rows, 1):
        status = str(row.body.get("status") or "-")
        hops = len(row.body.get("hops") or [])
        print(f"{i:>2} {row.case.name:<{width}} {status:<10} {_fmt_conf(row.body):>5} {hops:>4} "
              f"{row.wall_ms:>9.1f}  {'PASS' if row.ok else 'FAIL'}")
    for row in rows:
        print(f"\n{row.case.name}: {row.case.requirement}")
        if not row.ok:
            for problem in row.problems:
                print(f"  FAIL {problem}")
            continue
        print(f"  hop ops: {' → '.join(hop_ops(row.body)) or '(none)'}")
        if row.body.get("assumptions"):
            print(f"  assumptions: {'; '.join(str(a) for a in row.body['assumptions'])[:200]}")
        if row.body.get("question"):
            question = row.body["question"]
            print(f"  question ({question.get('kind')}): {str(question.get('question'))[:160]}")
            for option in (question.get("options") or [])[:6]:
                print(f"    {option.get('id')}")
        print(f"  sources: {'present' if row.body.get('sources') is not None else 'absent'}")


# -------------------------------------------------------------------- serving


class Stack:
    """llama-server + the decision service, started only under --serve."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.tmp = Path(tempfile.mkdtemp(prefix="ti-live-e2e-"))
        self.procs: list[subprocess.Popen] = []
        self.service = args.service.rstrip("/")
        self.model_url = args.model_url.rstrip("/")

    # -- paths ------------------------------------------------------------
    def _first_existing(self, explicit: str, candidates: list[Path], what: str) -> Path:
        if explicit:
            path = Path(os.path.expanduser(explicit))
            if not path.exists():
                raise SystemExit(abort(f"{what} {path} does not exist"))
            return path
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise SystemExit(abort(
            f"no {what} found. Tried:\n  " + "\n  ".join(str(c) for c in candidates)
            + f"\nPass it explicitly or set {'TI_MODEL' if what == 'model' else 'TI_SERVER'}."))

    def server_bin(self) -> Path:
        return self._first_existing(self.args.server, [
            ROOT / "extern" / "llama.cpp" / "build" / "bin" / "llama-server",
            Path.home() / "toolInterfacer" / "extern" / "llama.cpp" / "build" / "bin" / "llama-server",
        ], "server")

    def model_path(self) -> Path:
        name = self.args.expect_model or DEFAULT_EXPECT_MODEL
        return self._first_existing(self.args.model, [
            ROOT / "models" / name,
            Path.home() / "toolInterfacer" / "models" / name,
        ], "model")

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        service_port = int(self.args.service_port)
        model_port = int(self.args.model_port)
        for port, what in ((model_port, "model"), (service_port, "service")):
            if not port_free(port):
                raise SystemExit(abort(
                    f"127.0.0.1:{port} is already in use — refusing to touch another process's "
                    f"{what}. Pick another with --{what}-port / TI_E2E_{what.upper()}_PORT."))
        server, model = self.server_bin(), self.model_path()
        print(f"serving {model.name} on 127.0.0.1:{model_port} (CPU, "
              f"{self.args.threads} threads)", flush=True)
        self.procs.append(subprocess.Popen(
            [str(server), "-m", str(model), "--host", "127.0.0.1", "--port", str(model_port),
             "-c", str(self.args.ctx), "-t", str(self.args.threads), "--parallel", "1"],
            cwd=str(server.parent), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True))
        wait_for(f"{self.model_url}/props", self.args.start_timeout, "llama-server")

        overlay = self.tmp / "overlay.json"
        overlay.write_text(json.dumps({
            # `timeout_s` is a bench knob, not a semantic one: the shipped 30 s is
            # tight for an L2 pick on a shared CPU box (measured 20 s and 61 s for
            # the same 509-token request — see docs/live-e2e-2026-11-03.md), and a
            # probe that fails on machine load teaches nothing about the code.
            "decide": {"d1_systemone": {"base_url": self.model_url,
                                        "timeout_s": float(self.args.decide_timeout)}},
            "http": {"port": service_port},
            "hops_log": {"path": str(self.tmp / "hops.jsonl")},
        }), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": "service", "TI_CONF_OVERLAY": str(overlay)}
        print(f"serving the decision service on 127.0.0.1:{service_port} (overlay "
              f"{overlay.name})", flush=True)
        self.procs.append(subprocess.Popen(
            [sys.executable, "-m", "ti.http.api"], cwd=str(ROOT), env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True))
        wait_for(f"{self.service}/health", self.args.start_timeout, "decision service")

    def stop(self) -> None:
        started = list(self.procs)
        for proc in reversed(started):
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), 15)
                except OSError:
                    proc.terminate()
        for proc in reversed(started):
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), 9)
                except OSError:
                    proc.kill()
        self.procs.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)
        if started:
            print(f"stack stopped ({len(started)} process group(s): llama-server + service)")
        else:
            print("stack: nothing was started")


# --------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="live end-to-end probe for POST /v1/useTool")
    ap.add_argument("--service", default=os.environ.get("TI_BASE_URL")
                    or f"http://127.0.0.1:{DEFAULT_SERVICE_PORT}",
                    help="decision service base URL (default %(default)s)")
    ap.add_argument("--model-url", default=f"http://127.0.0.1:{DEFAULT_MODEL_PORT}",
                    help="llama.cpp base URL the model guard reads /props from (default %(default)s)")
    ap.add_argument("--serve", action="store_true",
                    help="start llama-server + the service on --model-port/--service-port and "
                         "tear them down afterwards (default: attach to what is running)")
    ap.add_argument("--model", default=os.environ.get("TI_MODEL", ""), help="gguf path (--serve)")
    ap.add_argument("--server", default=os.environ.get("TI_SERVER", ""),
                    help="llama-server binary (--serve)")
    ap.add_argument("--model-port", default=DEFAULT_MODEL_PORT, help="port for --serve (default %(default)s)")
    ap.add_argument("--service-port", default=DEFAULT_SERVICE_PORT,
                    help="port for --serve (default %(default)s)")
    ap.add_argument("--expect-model", default=DEFAULT_EXPECT_MODEL,
                    help="gguf basename the served model must match; '' disables the guard "
                         "(default %(default)s)")
    ap.add_argument("--threads", type=int, default=8, help="llama-server -t (--serve)")
    ap.add_argument("--ctx", type=int, default=2048, help="llama-server -c (--serve)")
    ap.add_argument("--start-timeout", type=float, default=120.0,
                    help="seconds to wait for each server to answer (default %(default)s)")
    ap.add_argument("--decide-timeout", type=float, default=120.0,
                    help="decide.d1_systemone.timeout_s for the service --serve starts "
                         "(default %(default)s; the shipped value is 30, tight under load)")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="per-request timeout, excluding the L2 case (default %(default)s)")
    ap.add_argument("--l2-timeout", type=float, default=420.0,
                    help="per-request timeout for the L2 case, which drives the web CLIs "
                         "(default %(default)s)")
    ap.add_argument("--skip-l2", action="store_true",
                    help="skip the L2 case (it runs real searches and takes minutes)")
    ap.add_argument("--only", action="append", default=[], metavar="CASE",
                    help="run only this case (repeatable)")
    ap.add_argument("--allow-fixture", action="store_true",
                    help="pass even when the service's choice backend is the replay fixture "
                         "(a smoke test of the transport, not of the model)")
    ap.add_argument("--json", default="", help="write the summary (rows + health) to this path")
    return ap


def make_ambiguous_dir() -> Path:
    """Five equally plausible files, so 'the report file' has no right answer."""
    path = Path(tempfile.mkdtemp(prefix="ti-live-e2e-amb-"))
    for name in AMBIGUOUS_FILES:
        (path / name).write_text(f"# {name}\n\nbody of {name}\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ambiguous = make_ambiguous_dir()
    ctx = Ctx(fixture=FIXTURE_DIR, ambiguous=ambiguous, readme=ROOT / "README.md")
    stack = Stack(args) if args.serve else None
    try:
        if stack is not None:
            stack.start()
            service, model_url = stack.service, stack.model_url
        else:
            service, model_url = args.service.rstrip("/"), args.model_url.rstrip("/")

        # 1. The stack must be there. A probe that passes against nothing is worthless.
        try:
            health = http_json(f"{service}/health", timeout=15.0)
        except Exception as exc:  # noqa: BLE001
            print(f"ABORT: no decision service at {service}/health — {exc}", file=sys.stderr)
            return 2
        if not isinstance(health, dict) or health.get("status") != "ok":
            print(f"ABORT: {service}/health answered {health!r}", file=sys.stderr)
            return 2
        ops = [str(op) for op in health.get("ops") or []]
        decide = health.get("decide") or {}
        if not ops:
            print("ABORT: the service reports no registered ops — op modules never imported",
                  file=sys.stderr)
            return 2

        # 2. The served model, before any probe traffic.
        try:
            props = http_json(f"{model_url}/props", timeout=15.0)
        except Exception as exc:  # noqa: BLE001
            print(f"ABORT: no model server at {model_url}/props — {exc}", file=sys.stderr)
            return 2
        if args.expect_model:
            try:
                served = assert_model_match(props, args.expect_model,
                                            source=f"{model_url}/props")
            except ModelAssertionError as exc:
                print(f"ABORT: {exc}", file=sys.stderr)
                return 2
        else:
            served = served_model(props)
            print(NO_EXPECT_WARNING, file=sys.stderr)

        choice_backend = str(decide.get("choice") or "")
        print(f"service  {service}  ops={len(ops)}")
        print(f"decide   choice={choice_backend or '?'} noul={decide.get('noul') or '?'} "
              f"extract={decide.get('extract') or '?'} cascade={health.get('cascade')}")
        print(f"model    {served['basename']} ({served['ftype']}) at {model_url}")
        problem = backend_problem(decide, args.allow_fixture)
        if problem:
            print(f"\nFAIL: {problem}")
            return 1

        chosen = cases(args.skip_l2, args.l2_timeout, args.only, args.timeout)
        if not chosen:
            print(f"ABORT: no case matched --only {args.only}", file=sys.stderr)
            return 2

        def transport(body: dict, timeout: float) -> dict:
            got = http_json(f"{service}/v1/useTool", body, timeout=timeout)
            if not isinstance(got, dict):
                raise RuntimeError(f"service answered {type(got).__name__}, not an object")
            return got

        rows = run_cases(chosen, ctx, ops, transport)
        print_table(rows)
        failed = [row for row in rows if not row.ok]
        print(f"\n{len(rows) - len(failed)}/{len(rows)} cases passed "
              f"(choice backend: {choice_backend or 'unknown'})")

        if args.json:
            Path(args.json).write_text(json.dumps({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "service": service, "model_url": model_url,
                "served_model": served, "decide": decide, "ops": ops,
                "cases": [{"case": r.case.name, "status": r.body.get("status"),
                           "confidence": r.body.get("confidence"), "wall_ms": r.wall_ms,
                           "hops": hop_ops(r.body), "ok": r.ok, "problems": r.problems}
                          for r in rows],
            }, indent=2), encoding="utf-8")
            print(f"summary written to {args.json}")
        return 1 if failed else 0
    finally:
        if stack is not None:
            stack.stop()
        shutil.rmtree(ambiguous, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
