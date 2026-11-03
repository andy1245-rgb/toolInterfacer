#!/usr/bin/env python3
"""roundtrips — how many model round trips and how much wall time a task costs,
with the agent's own tool calls (arm A) and with one `useTool` call (arm B).

The claim under measurement is spec §1 / README: *the LLM pays one round trip
for a whole pipeline*. The repo has latency for decisions and accuracy for
routing; it has never had the number a user feels. This measures it.

Two arms, one fixed task list (`roundtrip_tasks.py`), one box:

**Arm A — native.** A scripted tool-caller runs the tool sequence an agent model
would issue for the task, one call per round trip, executing the same op
implementations the service executes (`service/ti/ops/...`, in process). No LLM
is involved: the round-trip *count* is a property of the shape (the model cannot
send call n+1 before seeing the result of call n), not of a particular model, and
the per-round-trip *cost* is applied separately, as a stated number.

**Arm B — useTool.** One POST /v1/useTool per task against the real service. The
hops the service makes internally are read out of the response's `hops` array and
reported as `hops`, never as round trips. A `need_input` answer is one round trip
and the answer call is a second, because that is what the LLM pays.

Round trips are counted as model passes that carry a tool call (or a question to
the human). The final pass that writes the answer is the same in both arms and is
excluded from both.

Cost model — stated, not buried. A round trip is one inference pass of the agent
model over its context. Arm A's wall is `exec_ms + A_rt * PASS_MS`; arm B's is
`wall_ms + B_rt * PASS_MS`, where `wall_ms` was measured against the live service
and already contains the service's own tool execution. PASS_MS is an assumption
on a hosted agent model (`--pass-ms`, default 2000) and this file prints the
whole table at several values plus the exact breakeven per task, so the reader can
substitute their own number instead of trusting this one.

    python3 bench/roundtrips.py --serve --json bench/results/roundtrips-<date>.json
    python3 bench/roundtrips.py --service http://127.0.0.1:1115   # attach

Exit codes: 0 the run completed (whatever the verdict), 2 the stack was not
reachable or the request list could not be built.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT / "service"))

import roundtrip_tasks as tasklist  # noqa: E402  (bench/roundtrip_tasks.py)

DEFAULT_SERVICE_PORT = int(os.environ.get("TI_RT_SERVICE_PORT") or 1116)
DEFAULT_MODEL_URL = os.environ.get("TI_RT_MODEL_URL") or "http://127.0.0.1:8080"
#: The agent-model pass, as an assumption. The spec (§1) says a round trip is
#: "seconds each"; 2 s is the middle of that, and --pass-ms / the sensitivity
#: table exist so nobody has to take it on faith.
DEFAULT_PASS_MS = 2000.0
PASS_MS_SWEEP = (250.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0)
#: One pass of a *local* 3B model emitting a tool call, measured on this box and
#: recorded beside the assumption: a floor for "an inference pass is not free",
#: not a stand-in for a hosted agent model.
ANCHOR_TOKENS = 48


# --------------------------------------------------------------------- transport


class Transport:
    """POST /v1/useTool with the service token, timing each round trip."""

    def __init__(self, base: str, token: str, timeout: float) -> None:
        self.base = base.rstrip("/")
        self.token = token
        self.timeout = timeout

    def post(self, body: dict[str, Any]) -> tuple[dict[str, Any], float]:
        request = urllib.request.Request(
            f"{self.base}/v1/useTool", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "X-TI-Token": self.token})
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as exc:            # a refusal is an answer
            payload = {"status": "error",
                       "error": f"HTTP {exc.code}: {exc.read()[:200].decode(errors='replace')}",
                       "hops": []}
        return payload, (time.monotonic() - started) * 1000.0


def health(base: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"{base.rstrip('/')}/health", timeout=15) as response:
        return json.loads(response.read() or b"{}")


def wait_for(url: str, seconds: float, what: str) -> Any:
    deadline, last = time.monotonic() + seconds, None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                return json.loads(response.read() or b"{}")
        except Exception as exc:  # noqa: BLE001 — "not up yet" is every failure here
            last = exc
            time.sleep(0.4)
    raise RuntimeError(f"{what} did not answer {url} within {seconds:.0f}s: {last}")


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def resolve_token(explicit: str) -> str:
    """The client half of `service/ti/http/auth.py`: env, then the runtime file."""
    if explicit:
        return explicit
    name = os.environ.get("TI_HTTP_TOKEN", "").strip()
    if name:
        return name
    from ti.http import auth  # service is on sys.path above
    return auth.token()


def measure_local_pass(model_url: str) -> dict[str, Any]:
    """One inference pass of whatever model server the service decides with.

    It is here to keep the per-round-trip cost from being a number that only
    exists in prose: a 3B model on this box takes this long to emit ~48 tokens,
    so any agent-model pass is at least this expensive on the same hardware.
    Failure to reach it is not an error — the field goes null.
    """
    body = {"messages": [{"role": "user", "content":
                          "Reply with a JSON object naming one tool call, 40 tokens maximum."}],
            "max_tokens": ANCHOR_TOKENS, "temperature": 0}
    request = urllib.request.Request(f"{model_url.rstrip('/')}/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.loads(response.read() or b"{}")
    except Exception as exc:  # noqa: BLE001
        return {"ms": None, "error": f"{type(exc).__name__}: {exc}"}
    usage = payload.get("usage") or {}
    return {"ms": round((time.monotonic() - started) * 1000.0, 1),
            "completion_tokens": usage.get("completion_tokens"),
            "model_url": model_url}


# -------------------------------------------------------------------------- arms


@dataclass
class NativeRun:
    """Arm A: the scripted tool-caller's run."""

    round_trips: int = 0
    exec_ms: float = 0.0
    steps: list[dict] = field(default_factory=list)
    result: Any = None
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"round_trips": self.round_trips, "exec_ms": round(self.exec_ms, 1),
                "steps": self.steps, "error": self.error,
                "result_digest": tasklist.digest_of(self.result)}


def run_native(task: tasklist.Task, cwd: Path) -> NativeRun:
    """Execute the task's native sequence in process, one round trip per call.

    Each call is timed; `ask_user` is a round trip that runs no tool (it is the
    pass that asks the human), so it counts and costs nothing in exec time.
    """
    from ti.ops.filesys import confine
    from ti.ops.op_api import get as get_op, import_all, validate_params

    import_all()
    run = NativeRun(round_trips=len(task.native))
    token = confine.set_root(str(cwd))
    prior: list[Any] = []
    try:
        for name, spec in task.native:
            params = spec(prior) if callable(spec) else dict(spec)
            started = time.monotonic()
            if name == tasklist.WAIT_FOR_HUMAN:
                run.steps.append({"tool": name, "ms": 0.0, "error": None})
                continue
            try:
                args, _soft = validate_params(name, params)
                _kind, data, conf = get_op(name).run(args, None)
                elapsed = (time.monotonic() - started) * 1000.0
                run.exec_ms += elapsed
                run.steps.append({"tool": name, "ms": round(elapsed, 1),
                                  "conf": conf, "error": None})
                run.result = data
                prior.append(data)
            except Exception as exc:  # noqa: BLE001 — a failed tool is a failed task,
                elapsed = (time.monotonic() - started) * 1000.0    # not a crashed bench
                run.exec_ms += elapsed
                run.steps.append({"tool": name, "ms": round(elapsed, 1),
                                  "error": f"{type(exc).__name__}: {exc}"})
                run.error = f"{name}: {type(exc).__name__}: {exc}"
                prior.append([])
    finally:
        confine.reset_root(token)
    return run


@dataclass
class ServiceRun:
    """Arm B: what the wire cost."""

    round_trips: int = 0
    wall_ms: float = 0.0
    hops: list[dict] = field(default_factory=list)
    status: str = ""
    result: Any = None
    error: str = ""
    asked: str = ""
    bodies: list[dict] = field(default_factory=list)

    @property
    def decide_hops(self) -> list[dict]:
        return [h for h in self.hops if str(h.get("op", "")).startswith("decide_")]

    @property
    def decide_ms(self) -> float:
        return round(sum(float(h.get("ms") or 0.0) for h in self.decide_hops), 1)

    def as_dict(self) -> dict[str, Any]:
        return {"round_trips": self.round_trips, "wall_ms": round(self.wall_ms, 1),
                "hops": len(self.hops), "hop_ops": [str(h.get("op")) for h in self.hops],
                "decide_hops": len(self.decide_hops), "decide_ms": self.decide_ms,
                "status": self.status,
                "asked": self.asked, "error": self.error,
                "result_digest": tasklist.digest_of(self.result)}


def run_service(task: tasklist.Task, cwd: Path,
                transport: Transport) -> ServiceRun:
    """One useTool call, plus the answer call when the service asks back."""
    run = ServiceRun()
    body = {"intent": task.intent, "params": dict(task.params),
            "digest": {"cwd": str(cwd)}}
    payload, elapsed = transport.post(body)
    run.round_trips = 1
    run.wall_ms += elapsed
    run.hops.extend(payload.get("hops") or [])
    run.bodies.append(payload)
    if payload.get("status") == "need_input":
        options = [o.get("id") for o in (payload.get("question") or {}).get("options") or []
                   if isinstance(o, dict)]
        answer = task.answer or (options[0] if options else "")
        run.asked = str(answer)
        if answer:
            asked = {"intent": task.intent,
                     "params": {**task.params, "answer": answer},
                     "digest": {"cwd": str(cwd)}}
            payload, elapsed = transport.post(asked)
            run.round_trips += 1
            run.wall_ms += elapsed
            run.hops.extend(payload.get("hops") or [])
            run.bodies.append(payload)
    run.status = str(payload.get("status") or "")
    run.result = payload.get("result")
    run.error = str(payload.get("error") or "")
    return run


# ------------------------------------------------------------------- cost model


def wall_ms(exec_or_service_ms: float, round_trips: int, pass_ms: float) -> float:
    """Visible wall time: the tool work plus one model pass per round trip.

    Arm B's measured `wall_ms` already includes the service's tool execution, so
    adding `round_trips * pass_ms` to it gives the same shape as arm A's
    `exec_ms + round_trips * pass_ms`: both sides then carry only the passes the
    model makes of its own accord.
    """
    return exec_or_service_ms + round_trips * pass_ms


def breakeven_pass_ms(native: NativeRun, service: ServiceRun) -> float | None:
    """The per-pass cost at which B's wall equals A's, for this task.

    None when B cannot win at any pass cost: it issues at least as many round
    trips and pays the service's own wall on top.
    """
    saved = native.round_trips - service.round_trips
    if saved <= 0:
        return None
    return max(0.0, (service.wall_ms - native.exec_ms) / saved)


# ------------------------------------------------------------------------- run


@dataclass
class Row:
    task: tasklist.Task
    native: NativeRun
    service: ServiceRun
    a_correct: bool
    a_detail: str
    b_correct: bool
    b_detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.task.id, "shape": self.task.shape, "intent": self.task.intent,
            "native": self.native.as_dict(), "service": self.service.as_dict(),
            "correct": {"a": self.a_correct, "b": self.b_correct,
                        "a_detail": self.a_detail, "b_detail": self.b_detail,
                        "agree": self.a_correct == self.b_correct},
            "note": self.task.note,
        }


def run_all(tasks: list[tasklist.Task], ctx: tasklist.Ctx,
            transport: Transport) -> list[Row]:
    rows: list[Row] = []
    for task in tasks:
        cwd = ctx.cwd(task)
        native = run_native(task, cwd)
        service = run_service(task, cwd, transport)
        a_ok, a_detail = task.check(native.result, ctx)
        b_ok, b_detail = task.check(service.result, ctx)
        rows.append(Row(task=task, native=native, service=service,
                        a_correct=a_ok, a_detail=a_detail,
                        b_correct=b_ok, b_detail=b_detail))
    return rows


def aggregate(rows: list[Row], pass_ms: float,
              project_decide_ms: float | None = None) -> dict[str, Any]:
    a_rt = sum(r.native.round_trips for r in rows)
    b_rt = sum(r.service.round_trips for r in rows)
    a_exec = sum(r.native.exec_ms for r in rows)
    b_wall = sum(r.service.wall_ms for r in rows)
    a_wall = wall_ms(a_exec, a_rt, pass_ms)
    b_wall_total = wall_ms(b_wall, b_rt, pass_ms)
    saving = a_wall - b_wall_total
    hops = sum(len(r.service.hops) for r in rows)
    deciding = sum(r.service.decide_ms for r in rows)
    decide_hops = sum(len(r.service.decide_hops) for r in rows)
    out = {
        "pass_ms": pass_ms,
        "round_trips": {"a": a_rt, "b": b_rt, "saved": a_rt - b_rt,
                        "saved_pct": round(100.0 * (a_rt - b_rt) / a_rt, 1) if a_rt else 0.0},
        "wall_ms": {"a": round(a_wall, 1), "b": round(b_wall_total, 1),
                    "saved": round(saving, 1),
                    "saved_pct": round(100.0 * saving / a_wall, 1) if a_wall else 0.0},
        "tool_exec_ms": {"a": round(a_exec, 1), "b": round(b_wall, 1)},
        "service_overhead_ms": round(b_wall - a_exec, 1),
        "decide_ms_on_b": round(deciding, 1),
        "hops_on_b": hops,
        "hops_per_round_trip": round(hops / b_rt, 2) if b_rt else 0.0,
        "correct": {"a": sum(1 for r in rows if r.a_correct),
                    "b": sum(1 for r in rows if r.b_correct), "of": len(rows)},
        "decide_hops_on_b": decide_hops,
    }
    if project_decide_ms is not None:
        # A projection, not a measurement: replace the measured decide time with
        # `project_decide_ms` per decide hop and re-price the same wall. It is
        # here because the decision model shipped today costs seconds on a CPU
        # box and the spec prices a decision at ~250 ms, so the crossover the
        # reader cares about lives between those two numbers.
        projected = b_wall - deciding + decide_hops * project_decide_ms
        proj_b = wall_ms(projected, b_rt, pass_ms)
        out["projected"] = {
            "decide_ms_per_hop": project_decide_ms,
            "service_ms": round(projected, 1),
            "wall_ms": round(proj_b, 1),
            "saved_ms": round(a_wall - proj_b, 1),
            "saved_pct": round(100.0 * (a_wall - proj_b) / a_wall, 1) if a_wall else 0.0,
        }
    return out


def sensitivity(rows: list[Row]) -> list[dict[str, Any]]:
    return [aggregate(rows, value) for value in PASS_MS_SWEEP]


def print_report(rows: list[Row], pass_ms: float) -> None:
    width = max(len(r.task.id) for r in rows)
    print(f"\npass_ms = {pass_ms:.0f} (assumption; see the sweep below)")
    print(f"{'#':>2} {'task':<{width}} {'A rt':>4} {'B rt':>4} {'A exec':>8} "
          f"{'B wall':>8} {'hops':>4} {'B dec':>7} {'A ms':>8} {'B ms':>8}  A B  breakeven")
    print("-" * (width + 84))
    total_a = total_b = total_ae = total_bw = 0.0
    for index, row in enumerate(rows, 1):
        a_wall = wall_ms(row.native.exec_ms, row.native.round_trips, pass_ms)
        b_wall = wall_ms(row.service.wall_ms, row.service.round_trips, pass_ms)
        total_a += a_wall
        total_b += b_wall
        total_ae += row.native.exec_ms
        total_bw += row.service.wall_ms
        break_even = breakeven_pass_ms(row.native, row.service)
        be_text = "never" if break_even is None else f"{break_even:>8.0f}"
        print(f"{index:>2} {row.task.id:<{width}} {row.native.round_trips:>4} "
              f"{row.service.round_trips:>4} {row.native.exec_ms:>7.0f}m "
              f"{row.service.wall_ms:>7.0f}m {len(row.service.hops):>4} "
              f"{row.service.decide_ms:>6.0f}m {a_wall:>7.0f}m {b_wall:>7.0f}m  "
              f"{'y' if row.a_correct else 'n'} {'y' if row.b_correct else 'n'}  {be_text}")
    print("-" * (width + 84))
    print(f"{'':>2} {'total':<{width}} {sum(r.native.round_trips for r in rows):>4} "
          f"{sum(r.service.round_trips for r in rows):>4} {total_ae:>7.0f}m "
          f"{total_bw:>7.0f}m {sum(len(r.service.hops) for r in rows):>4} "
          f"{sum(r.service.decide_ms for r in rows):>6.0f}m {total_a:>7.0f}m {total_b:>7.0f}m")
    print("\nper-task detail")
    for row in rows:
        verdict = f"A {'ok' if row.a_correct else 'FAIL'} / B {'ok' if row.b_correct else 'FAIL'}"
        print(f"  {row.task.id}: {verdict} — {row.task.note}")
        if not row.a_correct:
            print(f"      A expected {row.a_detail}")
        if not row.b_correct:
            print(f"      B expected {row.b_detail}")
        if row.service.asked:
            print(f"      B asked back, answered {row.service.asked!r} "
                  f"(status {row.service.status})")
        if row.service.error:
            print(f"      B error: {row.service.error[:160]}")
        if row.native.error:
            print(f"      A error: {row.native.error[:160]}")


def print_sweep(rows: list[Row]) -> None:
    print(f"\n{'pass_ms':>8} {'A rt':>5} {'B rt':>5} {'A wall':>9} {'B wall':>9} "
          f"{'saving':>9} {'%':>6} {'saved rt':>8}")
    print("-" * 68)
    for point in sensitivity(rows):
        print(f"{point['pass_ms']:>8.0f} {point['round_trips']['a']:>5} "
              f"{point['round_trips']['b']:>5} {point['wall_ms']['a']:>8.0f}m "
              f"{point['wall_ms']['b']:>8.0f}m {point['wall_ms']['saved']:>8.0f}m "
              f"{point['wall_ms']['saved_pct']:>5.1f}% {point['round_trips']['saved']:>8}")


# ---------------------------------------------------------------------- serving


class Service:
    """Start/stop the decision service under test (arm B's endpoint)."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.tmp = Path(tempfile.mkdtemp(prefix="ti-roundtrips-"))
        self.proc: subprocess.Popen | None = None
        self.base = f"http://127.0.0.1:{int(args.service_port)}"
        self.token = secrets.token_urlsafe(24)

    def start(self) -> None:
        if not port_free(int(self.args.service_port)):
            raise RuntimeError(f"127.0.0.1:{self.args.service_port} is already in use — "
                               f"pick another with --service-port")
        overlay = self.tmp / "overlay.json"
        overlay.write_text(json.dumps({
            "decide": {"d1_systemone": {"base_url": self.args.model_url,
                                        "timeout_s": float(self.args.decide_timeout)}},
            "http": {"port": int(self.args.service_port)},
            "hops_log": {"path": str(self.tmp / "hops.jsonl")},
        }), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": "service",
               "TI_CONF_OVERLAY": str(overlay), "TI_HTTP_TOKEN": self.token}
        self.proc = subprocess.Popen([sys.executable, "-m", "ti.http.api"], cwd=str(ROOT),
                                     env=env, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, start_new_session=True)
        wait_for(f"{self.base}/health", self.args.start_timeout, "decision service")

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), 15)
            except OSError:
                self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.proc.pid), 9)
                except OSError:
                    self.proc.kill()
        self.proc = None
        shutil.rmtree(self.tmp, ignore_errors=True)


# ------------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="measure model round trips and wall time, native vs useTool")
    parser.add_argument("--service", default="",
                        help="base URL of a running service (default: start one on --service-port)")
    parser.add_argument("--serve", action="store_true",
                        help="start the service on --service-port and stop it afterwards")
    parser.add_argument("--service-port", default=DEFAULT_SERVICE_PORT,
                        help="port for --serve (default %(default)s)")
    parser.add_argument("--model-url", default=DEFAULT_MODEL_URL,
                        help="decide model base URL for the --serve overlay and the pass "
                             "anchor (default %(default)s)")
    parser.add_argument("--token", default="", help="service token (default: env, then the "
                                                   "runtime token file)")
    parser.add_argument("--decide-timeout", type=float, default=120.0,
                        help="decide.d1_systemone.timeout_s for the service --serve starts "
                             "(default %(default)s; the shipped value is 30)")
    parser.add_argument("--start-timeout", type=float, default=60.0,
                        help="seconds to wait for the service to answer /health")
    parser.add_argument("--timeout", type=float, default=420.0,
                        help="per-request timeout, one task (default %(default)s)")
    parser.add_argument("--pass-ms", type=float, default=DEFAULT_PASS_MS,
                        help="assumed cost of one agent-model pass in ms (default %(default)s)")
    parser.add_argument("--project-decide-ms", type=float, default=None,
                        help="also report arm B re-priced as if each decide hop cost this "
                             "many ms (a projection on measured hops, not a measurement; "
                             "250 is the figure spec/1.0-spec.md §1 claims)")
    parser.add_argument("--only", action="append", default=[], metavar="TASK",
                        help="run only this task (repeatable)")
    parser.add_argument("--measure-anchor", action="store_true",
                        help="also time one pass of the local decide model (a floor for the "
                             "cost of an inference pass on this box)")
    parser.add_argument("--json", default="", help="write the run to this path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started_at = datetime.now(timezone.utc).astimezone()
    load_start = os.getloadavg()

    tmp = Path(tempfile.mkdtemp(prefix="ti-roundtrips-fixtures-"))
    service = None
    try:
        notes, reports = tasklist.fixtures(tmp)
        ctx = tasklist.Ctx(root=ROOT, notes=notes, reports=reports).build()
        chosen = [t for t in tasklist.tasks(ctx) if not args.only or t.id in args.only]
        if not chosen:
            print("ABORT: no task matched --only", file=sys.stderr)
            return 2

        if args.serve or not args.service:
            service = Service(args)
            service.start()
            base, token = service.base, service.token
            print(f"service: {base} (started, decide model at {args.model_url})")
        else:
            base = args.service.rstrip("/")
            token = resolve_token(args.token)
            print(f"service: {base} (attached)")
        try:
            report = health(base)
        except Exception as exc:  # noqa: BLE001
            print(f"ABORT: no service at {base}/health — {exc}", file=sys.stderr)
            return 2
        decide = report.get("decide") or {}
        print(f"decide backends: {decide or '(none reported)'}")

        transport = Transport(base, token, args.timeout)
        anchor = measure_local_pass(args.model_url) if args.measure_anchor else {"ms": None}
        rows = run_all(chosen, ctx, transport)
        load_end = os.getloadavg()
        print_report(rows, args.pass_ms)
        print_sweep(rows)

        summary = aggregate(rows, args.pass_ms, args.project_decide_ms)
        print(f"\nservice overhead on B: {summary['service_overhead_ms']:.0f} ms "
              f"(measured wall minus arm A's tool execution, over {len(rows)} tasks)")
        print(f"decide time on B: {summary['decide_ms_on_b']:.0f} ms; "
              f"hops on B: {summary['hops_on_b']} for {summary['round_trips']['b']} round trips")
        print(f"correct: A {summary['correct']['a']}/{summary['correct']['of']}, "
              f"B {summary['correct']['b']}/{summary['correct']['of']}")
        print(f"load average: {load_start[0]:.2f} at start, {os.getloadavg()[0]:.2f} at end "
              f"({os.cpu_count()} cpus)")
        if "projected" in summary:
            projection = summary["projected"]
            print(f"projected at {projection['decide_ms_per_hop']:.0f} ms/decision (the spec's "
                  f"figure, on the measured {summary['decide_hops_on_b']} decide hops): "
                  f"B wall {projection['wall_ms']:.0f} ms, saving {projection['saved_ms']:.0f} ms "
                  f"({projection['saved_pct']:+.1f}%)")
        if anchor.get("ms") is not None:
            print(f"local pass anchor: {anchor['ms']:.0f} ms for "
                  f"{anchor.get('completion_tokens')} tokens on {args.model_url}")

        out = {
            "generated": started_at.isoformat(),
            "finished": datetime.now(timezone.utc).astimezone().isoformat(),
            "commit": _git(["rev-parse", "--short", "HEAD"]),
            "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "service": {"base": base, "decide": decide, "started_by_bench": service is not None},
            "machine": {"cpus": os.cpu_count(),
                        "loadavg_start": [round(v, 2) for v in load_start],
                        "loadavg_end": [round(v, 2) for v in load_end]},
            "cost_model": {"pass_ms_assumed": args.pass_ms,
                           "project_decide_ms": args.project_decide_ms,
                           "sweep_ms": list(PASS_MS_SWEEP),
                           "local_pass_anchor": anchor,
                           "definition": "a round trip is one agent-model inference that "
                                         "issues a tool call; the final answer pass is "
                                         "excluded from both arms"},
            "tasks": [row.as_dict() for row in rows],
            "aggregate": summary,
            "sensitivity": sensitivity(rows),
        }
        if args.json:
            path = Path(args.json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
            print(f"wrote {path}")
        return 0
    finally:
        if service is not None:
            service.stop()
            print(f"service stopped; port {args.service_port} released")
        shutil.rmtree(tmp, ignore_errors=True)


def _git(argv: list[str]) -> str:
    try:
        return subprocess.run(["git", *argv], cwd=str(ROOT), capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
