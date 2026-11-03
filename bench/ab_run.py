#!/usr/bin/env python3
"""Three-arm decision A/B — pair vs hosted-only vs local-only, interleaved.

The comparison the merge (1619f35) needs: `pair` is now the shipped `choice`
backend, and nobody has measured it against either of the two arms it sits
between. What that measurement is allowed to claim depends entirely on how it is
run, so this harness is built around one rule:

  **A row is measured under all three arms back to back, never arm-by-arm.**

This box's local latency swings by ~25% over minutes (ambient load, powersave
governor, thermal), so "run A for five minutes, then B for five minutes, then
compare" produces a confident, wrong answer — the drift lands on whichever arm
ran during the spike and it is indistinguishable from a real difference. Every
row here runs arm A, B and C within a few seconds of each other, and the order of
the three rotates (row i starts with arm i mod 3), so each arm is first, second
and third equally often over 30 rows. A spike inside a row hits all three.

Arms, each selected through resolve.resolve("choice") with TI_CONF_OVERLAY
pointing at a per-arm overlay (the same knob an operator would use):

  A  pair          local d1 stage1, hosted stage2 behind tau=0.90
  B  openrouter    hosted only
  C  d1_systemone  local only

The prompt material is `live_probe.INSTRUCTIONS` + `live_probe.OP_DESCRIPTIONS`,
imported verbatim, so accuracy is comparable to the recorded v2 runs.

Three measurement hazards are neutralised on purpose, and each is reported:

  * **cache** — a decision-cache hit is a microsecond answer that measures the
    table, not the model (cache.py says so itself). Both mechanisms are used:
    the overlay sets decide.cache.enabled false so resolve() wraps no cache at
    all, and cache.clear() runs before every single call. Every row records
    resolve's `last_source` and the run asserts it is "primary" throughout with
    the shared cache empty at the end.
  * **pre-gate** — heuristic_micro commits a hop for free on URLs, reddit
    mentions and path-like tokens. It is arm-invariant, so it adds nothing to an
    arm comparison while adding microsecond rows to all three latency
    distributions; the overlay sets decide.routing "direct" and the harness
    reports separately how many of the 30 utterances the gate *would* have
    claimed, as a static fact about the sample.
  * **served model** — /health answers "ok" for any llama-server. The lane is
    guarded by model_guard.preflight against --expect-model (d1-3B-Q4_K_M.gguf)
    exactly like bench/live_probe.py, so a stale server cannot be measured
    silently (docs/model-sweep-consolidated.md, CORRECTION).

Failure paths: a prior review found zero coverage for the four degradations
pair.py's docstring promises, and the "hard fallback to hosted" is the reason the
merge was allowed. Each is asserted for real — over real sockets, against a dead
port, a stub 429 and an emptied environment — never by reading the code back.

Usage:
  python3 bench/ab_run.py [--n 30] [--seed 20260707] [--out PATH] [--limit N]
                          [--expect-model d1-3B-Q4_K_M.gguf] [--skip-failpaths]
                          [--engineer-dead-port 9]

The OpenRouter key is read from the environment variable decide.openrouter.key_env
names (TI_OPENROUTER_API_KEY) and is never printed, logged, written or committed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from live_probe import INSTRUCTIONS, OP_DESCRIPTIONS, percentile  # noqa: E402
from model_guard import ModelAssertionError, preflight  # noqa: E402
from ti import conf  # noqa: E402
from ti.decide import cache as cache_mod  # noqa: E402
from ti.decide.primitives import make_backend  # noqa: E402
from ti.decide.resolve import resolve  # noqa: E402

ARMS = ("pair", "openrouter", "d1_systemone")
TAU = 0.90
K = len(OP_DESCRIPTIONS)
DEFAULT_OUT = ROOT / "bench" / "results" / "ab-three-arm-30rows.json"
PREDICTION_FILE = ROOT / "bench" / "results" / "dell" / "v2-d1-3B-Q4-systemone-props-asserted.json"


# -- configuration ------------------------------------------------------------

def arm_overlays() -> dict[str, dict[str, Any]]:
    """One overlay per arm: the arm's backend, no cache, no pre-gate.

    routing/cache are pinned per arm rather than inherited so the three arms
    differ in exactly one key — decide.backends_in_use.choice — and a future
    defaults.json edit cannot silently give one arm a gate the others lack.
    """
    return {arm: {"decide": {"backends_in_use": {"choice": arm},
                             "routing": "direct",
                             "cache": {"enabled": False},
                             "cascade": {"enabled": False}}}
            for arm in ARMS}


def write_overlays(directory: Path) -> dict[str, str]:
    """Materialise the overlays as files; TI_CONF_OVERLAY only accepts a path."""
    paths = {}
    for arm, overlay in arm_overlays().items():
        path = directory / f"overlay-{arm}.json"
        path.write_text(json.dumps(overlay, indent=2))
        paths[arm] = str(path)
    return paths


def arm_backend(arm: str, overlay_paths: dict[str, str]):
    """A fresh resolve("choice") for `arm`, in the config that arm's overlay selects.

    Built per call on purpose: ops/decisions builds a backend per hop, and pair.py
    keeps its observability fields on the instance, so a shared instance would let
    one row's record bleed into the next. conf.reload() is outside the measured
    window (the timer starts after this returns).
    """
    os.environ["TI_CONF_OVERLAY"] = overlay_paths[arm]
    conf.reload()
    return resolve("choice")


# -- sample -------------------------------------------------------------------

def sample_indices(rows: list[dict], n: int, seed: int) -> list[int]:
    """`n` row indices, stratified across the 8 labels, deterministic per seed.

    The file is band-blocked (t1/t2/t3 runs), so a prefix would sample one band and
    a handful of labels. Quotas are floor(n/8) per label with the remainder spread
    over the first labels alphabetically, so the 30 rows carry every label and no
    label dominates.
    """
    pools: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        pools.setdefault(row["label"], []).append(i)
    labels = sorted(pools)
    rng = random.Random(seed)
    base, extra = divmod(n, len(labels))
    picked: list[int] = []
    for pos, label in enumerate(labels):
        quota = min(len(pools[label]), base + (1 if pos < extra else 0))
        picked.extend(rng.sample(pools[label], quota))
    return sorted(picked)


# -- one measured call --------------------------------------------------------

def pair_stages(backend) -> list:
    """The pair's stage backends. A bench observation: pair keeps them private."""
    return list(getattr(getattr(backend, "primary", backend), "_stages", []))


def measure(arm: str, utterance: str, overlay_paths: dict[str, str],
            started: float) -> dict[str, Any]:
    """One arm call on one utterance, timed and with its backend's own record."""
    backend = arm_backend(arm, overlay_paths)
    cache_mod.clear()  # belt and braces: the overlay already leaves resolve cacheless
    t0 = time.perf_counter()
    error, answer = None, None
    try:
        answer = backend.choice(utterance, INSTRUCTIONS, OP_DESCRIPTIONS)
    except Exception as exc:  # a raised arm is a measured outcome, not a crash
        error = f"{type(exc).__name__}: {exc}"
    wall_ms = (time.perf_counter() - t0) * 1000.0
    rec: dict[str, Any] = {
        "arm": arm,
        "t_s": round(time.perf_counter() - started, 3),
        "wall_ms": round(wall_ms, 1),
        "source": str(getattr(backend, "last_source", "")),
        "choice": None,
        "conf": 0.0,
        "probabilities": {},
        "error": error,
    }
    if answer is not None:
        rec["choice"] = answer.choice
        rec["conf"] = round(float(answer.confidence), 4)
        rec["probabilities"] = {k: round(float(v), 6)
                               for k, v in (answer.probabilities or {}).items()}
    primary = getattr(backend, "primary", backend)
    if arm == "pair":
        rec["stage"] = primary.last_stage
        rec["escalated"] = bool(primary.last_escalated)
        rec["reason"] = primary.last_reason
        rec["stage1_pmax"] = round(float(primary.last_stage1_conf), 6)
    stages = pair_stages(backend)
    hosted = primary if arm == "openrouter" else (stages[1] if len(stages) > 1 else None)
    rec["provider"] = str(getattr(hosted, "last_provider", "") or "")
    return rec


def run_trials(rows: list[dict], idx: list[int], overlay_paths: dict[str, str],
               started: float) -> list[dict]:
    """Every sampled row through every arm, back to back, order rotating per row."""
    out: list[dict] = []
    for row_no, i in enumerate(idx):
        shift = row_no % len(ARMS)
        order = list(ARMS[shift:]) + list(ARMS[:shift])
        label = rows[i]["label"]
        utterance = rows[i]["utterance"]
        for pos, arm in enumerate(order):
            rec = measure(arm, utterance, overlay_paths, started)
            rec.update(idx=i, row=row_no, label=label, utterance=utterance,
                       order_pos=pos, order="/".join(order),
                       correct=bool(rec["choice"] == label))
            out.append(rec)
        print(f"  row {row_no + 1:>2}/{len(idx)} idx={i:<3} {label:<15} "
              + " ".join(f"{r['arm'][:4]}={r['wall_ms']:>7.0f}ms"
                         f"{'!' if r['error'] else ('+' if r['correct'] else '-')}"
                         for r in out[-3:]), flush=True)
    return out


# -- failure paths ------------------------------------------------------------

class StubServer:
    """A real HTTP server that answers every POST with one canned status.

    Used to induce the two conditions the pair's fallback is supposed to survive
    without needing the real upstream to cooperate: an upstream that is not
    listening (a dead port needs no server) and a 429. Counts the requests it
    received, which is how "stage2 was never called" is proved rather than
    inferred from a log line.
    """

    def __init__(self, status: int = 429, retry_after: str = "1",
                 body: str = '{"error": {"message": "stub"}}') -> None:
        self.status = status
        self.retry_after = retry_after
        self.body = body.encode()
        self.hits = 0
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                stub.hits += 1
                self.send_response(stub.status)
                self.send_header("Content-Type", "application/json")
                if stub.retry_after:
                    self.send_header("Retry-After", stub.retry_after)
                self.send_header("Content-Length", str(len(stub.body)))
                self.end_headers()
                self.wfile.write(stub.body)

            def log_message(self, *args) -> None:  # a bench does not want its stderr
                return

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/api/v1"

    def __enter__(self) -> "StubServer":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


def merge(dst: dict, src: dict) -> dict:
    """Deep-merge `src` onto `dst` in place (the same rule conf._merge applies)."""
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            merge(dst[key], value)
        else:
            dst[key] = value
    return dst


def pair_with(patch: dict[str, Any]):
    """A fresh PairBackend, built on the shipped config with `patch` applied.

    The pair itself (make_backend("pair"), not resolve()) because these tests are
    about pair.py's policy rules; a pre-gate or a cache in front of it would be a
    second thing under test. conf.reload() first, so every path starts from
    defaults rather than from the previous path's patch.
    """
    overlay = os.environ.get("TI_CONF_OVERLAY", "")
    cfg = conf.reload()
    merge(cfg, patch)
    return make_backend("pair"), overlay


def hosted_rows(trials: list[dict]) -> list[dict]:
    """Rows where the pair actually escalated — the rows these paths need."""
    return [t for t in trials if t["arm"] == "pair" and t.get("escalated")]


def local_answer(trials: list[dict], idx: int) -> str | None:
    """Arm C's choice for one row: what rule 5 is supposed to hand back."""
    for t in trials:
        if t["arm"] == "d1_systemone" and t["idx"] == idx:
            return t["choice"]
    return None


def check(name: str, expected: str, observed: dict[str, Any]) -> dict[str, Any]:
    return {"path": name, "expected": expected,
            "observed": observed, "ok": bool(observed.pop("ok"))}


def run_failure_paths(rows: list[dict], trials: list[dict], overlay_paths: dict[str, str],
                      dead_port: int) -> list[dict]:
    """The four documented degradations plus the escalate_on_error=false rule."""
    results: list[dict] = []
    low = [t for t in hosted_rows(trials) if t["reason"] == "low_confidence"]
    high = [t for t in trials if t["arm"] == "pair" and t["reason"] == "confident"]
    if not low:
        raise SystemExit("failure paths need at least one low-confidence row")
    low_row = rows[low[0]["idx"]]
    high_row = rows[high[0]["idx"]] if high else None
    key_env = conf.load()["decide"]["openrouter"]["key_env"]
    dead = {"decide": {"d1_systemone": {"base_url": f"http://127.0.0.1:{dead_port}"},
                       "openrouter": {"base_url": f"http://127.0.0.1:{dead_port}"}}}
    os.environ["TI_CONF_OVERLAY"] = overlay_paths["pair"]

    def call(pair, row) -> tuple[Any, str | None]:
        try:
            return pair.choice(row["utterance"], INSTRUCTIONS, OP_DESCRIPTIONS), None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    # (a) no hosted key: stage2 cannot answer, so rule 5 must hand back stage1's
    #     answer instead of failing a run that had a usable (degraded) decision.
    saved_key = os.environ.pop(key_env, None)
    try:
        pair, _ = pair_with({})
        answer, error = call(pair, low_row)
        want = local_answer(trials, low[0]["idx"])
        results.append(check(
            "a1 hosted key absent, low stage1 conf",
            "stage1's answer, reason=stage2_error, no exception (rule 5)",
            {"ok": error is None and answer is not None
                   and answer.choice == want
                   and pair.last_reason == "stage2_error"
                   and pair.last_stage == "stage1"
                   and pair.last_escalated is True,
             "stage": pair.last_stage, "reason": pair.last_reason,
             "escalated": pair.last_escalated, "error": error,
             "choice": getattr(answer, "choice", None), "stage1_choice": want,
             "stage1_pmax": round(pair.last_stage1_conf, 4)}))
        if high_row is not None:
            pair, _ = pair_with({})
            answer, error = call(pair, high_row)
            results.append(check(
                "a2 hosted key absent, high stage1 conf",
                "stage1's answer, reason=confident, stage2 never called",
                {"ok": error is None and answer is not None
                       and pair.last_reason == "confident"
                       and pair.last_escalated is False,
                 "stage": pair.last_stage, "reason": pair.last_reason,
                 "escalated": pair.last_escalated, "error": error,
                 "choice": getattr(answer, "choice", None),
                 "stage1_pmax": round(pair.last_stage1_conf, 4)}))
    finally:
        if saved_key is not None:
            os.environ[key_env] = saved_key

    # (b) local stage unreachable: the hard fallback. stage1 raises, so the pair
    #     must answer from hosted even though nothing local was ever produced.
    death = {"decide": {"d1_systemone": {"base_url": dead["decide"]["d1_systemone"]["base_url"]}}}
    pair, _ = pair_with(death)
    answer, error = call(pair, low_row)
    results.append(check(
        "b local stage unreachable",
        "hosted's answer, reason=stage1_error, no exception",
        {"ok": error is None and answer is not None and bool(answer.choice)
               and pair.last_reason == "stage1_error"
               and pair.last_stage == "stage2"
               and pair.last_escalated is True,
         "stage": pair.last_stage, "reason": pair.last_reason,
         "escalated": pair.last_escalated, "error": error,
         "choice": getattr(answer, "choice", None),
         "stage1_pmax": round(pair.last_stage1_conf, 4)}))

    # (c) hosted unreachable with a low-confidence stage1: the escalation raises,
    #     rule 5 returns stage1's own answer and does not propagate the error.
    pair, _ = pair_with({"decide": {"openrouter": {"base_url": dead["decide"]["openrouter"]["base_url"]}}})
    answer, error = call(pair, low_row)
    want = local_answer(trials, low[0]["idx"])
    results.append(check(
        "c hosted unreachable, low stage1 conf",
        "stage1's answer, reason=stage2_error, no exception (rule 5)",
        {"ok": error is None and answer is not None and answer.choice == want
               and pair.last_reason == "stage2_error" and pair.last_stage == "stage1",
         "stage": pair.last_stage, "reason": pair.last_reason,
         "escalated": pair.last_escalated, "error": error,
         "choice": getattr(answer, "choice"), "stage1_choice": want}))

    # (d) a 429 from the upstream: split the transport from the policy. The
    #     backend must retry a bounded number of times and then raise, and the
    #     pair must turn that raise into stage1's answer.
    with StubServer(status=429, retry_after="1") as stub:
        pair, _ = pair_with({"decide": {"openrouter": {"base_url": stub.base_url}}})
        answer, error = call(pair, low_row)
        results.append(check(
            "d1 hosted answers 429 (stub), low stage1 conf",
            f"3 attempts at the wire, then stage1's answer + reason=stage2_error",
            {"ok": error is None and answer is not None
                   and answer.choice == local_answer(trials, low[0]["idx"])
                   and pair.last_reason == "stage2_error" and stub.hits == 3,
             "stage": pair.last_stage, "reason": pair.last_reason, "error": error,
             "choice": getattr(answer, "choice"), "stub_requests": stub.hits}))
    with StubServer(status=429, retry_after="1") as stub:
        pair, _ = pair_with({"decide": {"openrouter": {"base_url": stub.base_url}}})
        backend = pair_stages(pair)[1]
        raised, message = None, ""
        try:
            backend.choice(low_row["utterance"], INSTRUCTIONS, OP_DESCRIPTIONS)
        except Exception as exc:
            raised, message = type(exc).__name__, str(exc)[:160]
        results.append(check(
            "d2 hosted backend alone, 429 (stub)",
            "OpenRouterError raised after 3 attempts — an error status, not a decision",
            {"ok": raised == "OpenRouterError" and "429" in message and stub.hits == 3,
             "raised": raised, "message": message, "stub_requests": stub.hits}))

    # (a guard rather than a documented path) rule 1 runs before rule 2: a
    # confident stage1 must not call stage2 at all. Proved by a stub that would
    # count the call if it happened.
    if high_row is not None:
        with StubServer(status=200, retry_after="",
                        body='{"choices":[{"message":{"content":"{\\"choice\\": \\"web_fetch\\", '
                             '\\"confidence\\": 0.99}"}}],"provider":"stub"}') as stub:
            pair, _ = pair_with({"decide": {"openrouter": {"base_url": stub.base_url}}})
            answer, error = call(pair, high_row)
            results.append(check(
                "e confident stage1 is not escalated",
                "stage2 receives 0 requests, reason=confident (rule 1 before rule 2)",
                {"ok": stub.hits == 0 and pair.last_reason == "confident"
                       and error is None,
                 "stub_requests": stub.hits, "reason": pair.last_reason,
                 "stage": pair.last_stage, "error": error,
                 "choice": getattr(answer, "choice", None)}))

    # (4) escalate_on_error=false: the same local failure must propagate, which is
    #     what makes the true default a *policy* choice rather than an accident.
    merge(death, {"decide": {"pair": {"escalate_on_error": False}}})
    pair, _ = pair_with(death)
    answer, error = call(pair, low_row)
    results.append(check(
        "f escalate_on_error=false, local unreachable",
        "the stage1 error propagates (rule 4)",
        {"ok": error is not None and answer is None,
         "raised": error, "reason": pair.last_reason, "stage": pair.last_stage}))

    conf.reload()
    return results


# -- statistics ---------------------------------------------------------------

def latency_block(values: list[float]) -> dict[str, Any]:
    """Percentiles, never a mean: the operator experiences the tail, not the average."""
    ordered = sorted(values)
    return {"n": len(ordered),
            "p50": round(percentile(ordered, 0.50), 1),
            "p90": round(percentile(ordered, 0.90), 1),
            "p99": round(percentile(ordered, 0.99), 1),
            "min": round(ordered[0], 1) if ordered else 0.0,
            "max": round(ordered[-1], 1) if ordered else 0.0,
            "all_ms": [round(v, 1) for v in ordered]}


def mcnemar(a: list[dict], b: list[dict]) -> dict[str, Any]:
    """Exact McNemar on paired correctness: discordant counts and a binomial p.

    a and b are the two arms' rows for the same utterances in the same order, so
    every pair is a paired observation. The exact form is used because n is 30 and
    the asymptotic chi-square is not defensible at these discordant counts.
    """
    left = {row["idx"]: row["correct"] for row in a}
    right = {row["idx"]: row["correct"] for row in b}
    both = sum(1 for i in left if left[i] and right.get(i))
    only_a = sum(1 for i in left if left[i] and not right.get(i))
    only_b = sum(1 for i in left if not left[i] and right.get(i))
    neither = sum(1 for i in left if not left[i] and not right.get(i))
    n = only_a + only_b
    if n == 0:
        p = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(0, min(only_a, only_b) + 1)) / 2 ** n
        p = min(1.0, 2.0 * tail)
    return {"arm_a": a[0]["arm"] if a else "", "arm_b": b[0]["arm"] if b else "",
            "n_pairs": len(left), "both_correct": both, "neither_correct": neither,
            "only_a_correct": only_a, "only_b_correct": only_b,
            "delta_accuracy": round((only_a - only_b) / len(left), 4) if left else 0.0,
            "p_exact": round(p, 4)}


def arm_summary(trials: list[dict], arm: str) -> dict[str, Any]:
    rows = [t for t in trials if t["arm"] == arm]
    ok_rows = [t for t in rows if t["error"] is None]
    block: dict[str, Any] = {
        "n": len(rows),
        "errors": sum(1 for t in rows if t["error"]),
        "abstains": sum(1 for t in rows if t["error"] is None and not t["choice"]),
        "correct": sum(1 for t in rows if t["correct"]),
        "accuracy": round(sum(1 for t in rows if t["correct"]) / len(rows), 4) if rows else 0.0,
        "conf_median": round(statistics.median([t["conf"] for t in ok_rows]), 4) if ok_rows else 0.0,
        "latency_ms": latency_block([t["wall_ms"] for t in ok_rows]),
        "not_primary_source": sum(1 for t in rows if t["source"] != "primary"),
    }
    if arm == "pair":
        escalated = [t for t in rows if t.get("escalated")]
        staged1 = [t for t in rows if not t.get("escalated")]
        fast = [t["wall_ms"] for t in staged1]
        slow = [t["wall_ms"] for t in escalated]
        hosted_by_idx = {t["idx"]: t["wall_ms"] for t in trials if t["arm"] == "openrouter"}
        local_by_idx = {t["idx"]: t["wall_ms"] for t in trials if t["arm"] == "d1_systemone"}
        residuals = sorted(t["wall_ms"] - local_by_idx[t["idx"]] - hosted_by_idx[t["idx"]]
                           for t in escalated if t["idx"] in hosted_by_idx)
        block["escalation"] = {
            "n_escalated": len(escalated),
            "rate": round(len(escalated) / len(rows), 4) if rows else 0.0,
            "reasons": {reason: sum(1 for t in rows if t.get("reason") == reason)
                        for reason in ("confident", "low_confidence",
                                       "stage1_error", "stage2_error")},
            "stage1_only_wall_ms": latency_block(fast),
            "escalated_wall_ms": latency_block(slow),
            "two_populations_by_stage": bool(fast) and bool(slow),
            # Two populations by construction is not the same as two visible modes:
            # if the slowest local commit is slower than the fastest escalation, the
            # pooled distribution is one smeared tail and a percentile of it is
            # meaningless to quote per mode.
            "cleanly_separated": bool(fast) and bool(slow) and min(slow) > max(fast),
            "escalations_below_slowest_commit": sum(1 for v in slow if fast and v <= max(fast)),
            # Pair latency minus the same row's local call minus the same row's hosted
            # call: ~0 means the two stages are additive and nothing hides in the wrapper.
            "additive_residual_ms": {
                "median": round(statistics.median(residuals), 1) if residuals else 0.0,
                "min": round(residuals[0], 1) if residuals else 0.0,
                "max": round(residuals[-1], 1) if residuals else 0.0,
                "n": len(residuals),
            },
        }
        providers = sorted({t["provider"] for t in rows if t["provider"]})
        block["providers"] = providers
    return block


def predicted_escalation(path: Path) -> dict[str, Any]:
    """The offline prediction: p_max = conf*(1-1/K)+1/K, escalate when p_max < tau."""
    if not path.exists():
        return {"source": str(path.relative_to(ROOT)), "error": "not found"}
    rows = json.loads(path.read_text())["per_row"]
    pmax = [r["conf"] * (1 - 1 / K) + 1 / K for r in rows]
    escalated = sum(1 for p in pmax if p < TAU)
    return {"source": str(path.relative_to(ROOT)), "n": len(rows), "k": K, "tau": TAU,
            "escalated": escalated,
            "rate": round(escalated / len(rows), 4),
            "formula": "p_max = conf*(1-1/K)+1/K, escalate when p_max < tau"}


def observed_escalation(trials: list[dict]) -> dict[str, Any]:
    rows = [t for t in trials if t["arm"] == "pair"]
    escalated = sum(1 for t in rows if t.get("escalated"))
    return {"n": len(rows), "escalated": escalated,
            "rate": round(escalated / len(rows), 4) if rows else 0.0}


def gate_hits(utterances: list[str]) -> int:
    """How many of the sample the pre-gate would have committed (arm-invariant)."""
    gate = make_backend("heuristic_micro")
    return sum(1 for u in utterances
               if (gate.choice(u, INSTRUCTIONS, OP_DESCRIPTIONS) or None) is not None)


# -- spend --------------------------------------------------------------------

def key_usage() -> dict[str, Any]:
    """The account's usage for this key (GET /key) — the spend measurement.

    Never returns or logs the key: it travels in the Authorization header only,
    and any error text is passed through the redactor before it is kept.
    """
    cfg = conf.load()["decide"]["openrouter"]
    env_name = cfg.get("key_env") or "OPENROUTER_API_KEY"
    key = os.environ.get(env_name, "").strip()
    if not key:
        return {"error": f"{env_name} not set"}
    request = urllib.request.Request(str(cfg["base_url"]).rstrip("/") + "/key",
                                     headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read())
    except Exception as exc:
        return {"error": str(exc).replace(key, "[redacted]")[:160]}
    data = payload.get("data") or {}
    return {k: data.get(k) for k in ("label", "usage", "limit", "is_free_tier")}


def spend(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Usage delta across the run, plus what that delta cannot tell us."""
    if "usage" not in before or "usage" not in after:
        return {"error": "usage unavailable", "before": before, "after": after}
    delta = float(after["usage"]) - float(before["usage"])
    return {"usage_before_usd": before["usage"], "usage_after_usd": after["usage"],
            "delta_usd": round(delta, 6),
            "note": "account-level delta for this key: any other traffic sharing it "
                    "would be included, and OpenRouter may settle usage with a lag"}


# -- report -------------------------------------------------------------------

def print_report(summary: dict[str, Any]) -> None:
    arms = summary["arms"]
    print("\n== latency per arm (wall_ms, backends driven in-process) ==")
    print(f"{'arm':<14}{'n':>4}{'p50':>9}{'p90':>9}{'p99':>9}{'min':>9}{'max':>9}"
          f"{'acc':>8}{'err':>5}{'esc':>6}")
    for arm in ARMS:
        block = arms[arm]
        lat = block["latency_ms"]
        esc = block.get("escalation", {}).get("rate")
        print(f"{arm:<14}{block['n']:>4}{lat['p50']:>9.0f}{lat['p90']:>9.0f}"
              f"{lat['p99']:>9.0f}{lat['min']:>9.0f}{lat['max']:>9.0f}"
              f"{block['accuracy']:>8.1%}{block['errors']:>5}"
              f"{(f'{esc:.0%}' if esc is not None else '-'):>6}")

    pair = arms["pair"].get("escalation") or {}
    if pair.get("two_populations_by_stage"):
        fast, slow = pair["stage1_only_wall_ms"], pair["escalated_wall_ms"]
        verdict = ("cleanly separated: every escalation is slower than every commit"
                   if pair["cleanly_separated"] else
                   f"NOT cleanly separated: {pair['escalations_below_slowest_commit']}"
                   f"/{slow['n']} escalations land at or below the slowest local commit")
        print("\n== pair latency: two populations by stage, and whether they read as two modes ==")
        print(f"  stage1 only (n={fast['n']:>2}): min={fast['min']:>6.0f}ms "
              f"p50={fast['p50']:>6.0f}ms p90={fast['p90']:>6.0f}ms max={fast['max']:>6.0f}ms")
        print(f"  escalated   (n={slow['n']:>2}): min={slow['min']:>6.0f}ms "
              f"p50={slow['p50']:>6.0f}ms p90={slow['p90']:>6.0f}ms max={slow['max']:>6.0f}ms")
        print(f"  {verdict}")
        add = pair["additive_residual_ms"]
        print(f"  additivity: pair - local - hosted per escalated row, n={add['n']}: "
              f"median={add['median']:>+7.0f}ms range=[{add['min']:+.0f}, {add['max']:+.0f}]ms")
    print("\n  full distributions (sorted wall_ms):")
    for arm in ARMS:
        values = arms[arm]["latency_ms"]["all_ms"]
        print(f"  {arm:<14}{values}")

    for key in ("pair_vs_openrouter", "pair_vs_d1_systemone"):
        block = summary["mcnemar"][key]
        print(f"\n== McNemar {block['arm_a']} vs {block['arm_b']} (exact, "
              f"{block['n_pairs']} paired rows) ==")
        print(f"  both correct {block['both_correct']} | only {block['arm_a']} "
              f"{block['only_a_correct']} | only {block['arm_b']} "
              f"{block['only_b_correct']} | neither {block['neither_correct']}")
        print(f"  accuracy delta {block['delta_accuracy']:+.1%}, exact p={block['p_exact']}")

    predicted = summary["escalation_prediction"]
    observed = pair.get("rate", 0.0)
    print("\n== escalation rate, observed vs predicted ==")
    print(f"  observed (arm A, {pair.get('n_escalated')}/{arms['pair']['n']}): {observed:.1%}")
    pop = predicted["population"]
    if "rate" in pop:
        print(f"  predicted from the recorded 150-row d1 run ({pop['n']} rows, "
              f"{pop['escalated']} escalate): {pop['rate']:.1%}")
    self_pred = predicted["same_rows"]
    if "rate" in self_pred:
        print(f"  predicted from arm C on these same {self_pred['n']} rows: "
              f"{self_pred['rate']:.1%}")
    print(f"  predicted cost model: pair ~= local p50 + escalation_rate x hosted p50 "
          f"= {arms['d1_systemone']['latency_ms']['p50']:.0f}ms + {observed:.2f} x "
          f"{arms['openrouter']['latency_ms']['p50']:.0f}ms "
          f"= {arms['d1_systemone']['latency_ms']['p50'] + observed * arms['openrouter']['latency_ms']['p50']:.0f}ms"
          f" (observed pair p50: {arms['pair']['latency_ms']['p50']:.0f}ms)")

    print("\n== failure paths (real sockets, real dead ports, real 429) ==")
    for path in summary["failure_paths"]:
        mark = "PASS" if path["ok"] else "FAIL"
        print(f"  [{mark}] {path['path']}")
        print(f"         expected: {path['expected']}")
        detail = {k: v for k, v in path["observed"].items() if v not in (None, "", {})}
        print(f"         observed: {json.dumps(detail, default=str)[:300]}")

    print("\n== spend ==")
    print(f"  {json.dumps(summary['spend'], default=str)}")
    print(f"  hosted calls: {summary['hosted_calls']} "
          f"({arms['pair'].get('escalation', {}).get('n_escalated', 0)} escalations + "
          f"{arms['openrouter']['n']} hosted-arm rows + failure paths)")


def recommend(summary: dict[str, Any]) -> list[str]:
    """The verdict, generated from the numbers rather than asserted alongside them."""
    arms = summary["arms"]
    pair, hosted, local = arms["pair"], arms["openrouter"], arms["d1_systemone"]
    esc = pair.get("escalation", {}).get("rate", 0.0)
    lines = []
    local_p50 = local["latency_ms"]["p50"]
    hosted_p50 = hosted["latency_ms"]["p50"]
    pair_p50 = pair["latency_ms"]["p50"]
    lines.append(
        f"pair p50 {pair_p50:.0f}ms vs hosted-only {hosted_p50:.0f}ms vs local-only "
        f"{local_p50:.0f}ms (interleaved, same {pair['n']} utterances).")
    if esc * hosted_p50 > hosted_p50:
        lines.append("the escalation tail is the whole cost of the pair here.")
    best_acc = max(ARMS, key=lambda a: arms[a]["accuracy"])
    lines.append(f"highest accuracy is {best_acc} at {arms[best_acc]['accuracy']:.1%}; "
                 f"pair {pair['accuracy']:.1%}, hosted {hosted['accuracy']:.1%}, "
                 f"local {local['accuracy']:.1%}.")
    return lines


# -- main ---------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="sampled utterances (stratified)")
    ap.add_argument("--seed", type=int, default=20260707, help="sample seed (reproducibility)")
    ap.add_argument("--utterances", default=str(ROOT / "tests" / "data" / "utterances_v2.json"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=0, help="smoke-test on the first N rows")
    ap.add_argument("--expect-model", default="d1-3B-Q4_K_M.gguf",
                    help="served-model assertion via /props (\"\" disables it)")
    ap.add_argument("--base-url", default="", help="override the local backend's base_url")
    ap.add_argument("--dead-port", type=int, default=9,
                    help="closed 127.0.0.1 port the failure paths point at")
    ap.add_argument("--skip-failpaths", action="store_true")
    ap.add_argument("--summarize", default="",
                    help="re-derive the report from a recorded result file (no traffic, no spend)")
    ap.add_argument("--analysis-out", default="",
                    help="also write the derived block to this path (never the raw rows)")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize:
        raw = json.loads(Path(args.summarize).read_text())
        print(f"re-analysed {display_path(Path(args.summarize))} — "
              "no network traffic, no spend")
        emit(raw, args, measured=False)
        return
    emit(measure_phase(args), args, measured=True)


def measure_phase(args: argparse.Namespace) -> dict[str, Any]:
    """Everything that costs a network call or a wall-clock millisecond.

    Kept apart from analyze() so the derived numbers can be recomputed over a
    recorded run's per_row without spending money or re-measuring a box whose
    load has moved on since (--summarize).
    """
    started = time.perf_counter()
    rows = json.loads(Path(args.utterances).read_text())

    # served-model guard, exactly as bench/live_probe.py does it
    base_url = args.base_url or conf.load()["decide"]["d1_systemone"]["base_url"]
    with urllib.request.urlopen(base_url.rstrip("/") + "/health", timeout=10) as response:
        health = json.loads(response.read())
    if health.get("status") != "ok":
        raise SystemExit(f"local server unhealthy: {health}")
    with urllib.request.urlopen(base_url.rstrip("/") + "/props", timeout=10) as response:
        props = json.loads(response.read())
    try:
        served, warning = preflight(props, args.expect_model, "", None,
                                    base_url.rstrip("/") + "/props")
    except ModelAssertionError as exc:
        raise SystemExit(f"ABORT: {exc}")
    if warning:
        print(warning, file=sys.stderr)
    print(f"served model: {served.get('basename', '?')} "
          f"({(props.get('build_info') or '')[:40]})")

    idx = sample_indices(rows, args.n, args.seed)
    if args.limit:
        idx = idx[: args.limit]
    print(f"sample: {len(idx)} rows, seed={args.seed}, labels="
          f"{sorted({rows[i]['label'] for i in idx})}")
    print(f"idx: {idx}")

    key_env = conf.load()["decide"]["openrouter"]["key_env"]
    print(f"hosted key present: {bool(os.environ.get(key_env, '').strip())} "
          f"(variable {key_env}, value never read into any output)")
    usage_before = key_usage()

    with tempfile.TemporaryDirectory(prefix="ab-run-") as tmp:
        overlay_paths = write_overlays(Path(tmp))
        for arm in ARMS:  # warm each arm once on material outside the sample
            measure(arm, "warm up: list the files in this directory", overlay_paths, started)
        print("warm-up done (3 discarded calls, one per arm)\n")
        trials = run_trials(rows, idx, overlay_paths, started)
        failures = [] if args.skip_failpaths else run_failure_paths(
            rows, trials, overlay_paths, args.dead_port)
        cache_state = cache_mod.stats()

    usage_after = key_usage()
    hosted_calls = (sum(1 for t in trials if t["arm"] == "openrouter")
                    + sum(1 for t in trials if t["arm"] == "pair" and t.get("escalated")))

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "harness": "bench/ab_run.py",
        "git_commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                    capture_output=True, text=True).stdout.strip(),
        "design": {
            "interleaving": "all three arms per utterance, back to back; order rotates "
                            "(row i starts with arm i mod 3) so each arm is first "
                            "second and third equally often",
            "cache": "decide.cache.enabled=false in every overlay (resolve wraps no "
                     "cache) AND cache.clear() before every call; last_source asserted "
                     "primary throughout, shared cache empty at the end",
            "pre_gate": "decide.routing=direct in every overlay; pre-gate hit count "
                        "reported separately as an arm-invariant property of the sample",
            "statistic": "percentiles (p50/p90/p99) plus the full sorted distribution; "
                         "no mean is reported because the operator experiences the tail",
            "prompt_material": "bench/live_probe.py INSTRUCTIONS + OP_DESCRIPTIONS, "
                               "imported verbatim",
        },
        "sample": {
            "n": len(idx), "seed": args.seed, "idx": idx,
            "utterances_file": str(Path(args.utterances).relative_to(ROOT)),
            "labels": {label: sum(1 for i in idx if rows[i]["label"] == label)
                       for label in sorted({rows[i]["label"] for i in idx})},
            "bands": {band: sum(1 for i in idx if rows[i].get("band") == band)
                      for band in sorted({rows[i].get("band") for i in idx})},
            "pre_gate_would_commit": gate_hits([rows[i]["utterance"] for i in idx]),
        },
        "config": {
            "arms": {arm: "decide.backends_in_use.choice=" + arm for arm in ARMS},
            "tau": TAU, "k": K,
            "pair": conf.load()["decide"]["pair"],
            "openrouter_model": conf.load()["decide"]["openrouter"]["model"],
            "openrouter_provider": conf.load()["decide"]["openrouter"]["provider"],
            "openrouter_reasoning": conf.load()["decide"]["openrouter"].get("reasoning"),
            "base_url": base_url,
            "served_model": served.get("basename", ""),
            "server_threads": "as-configured (not restarted by this harness)",
        },
        "cache_after": cache_state,
        "failure_paths": failures,
        "spend": spend(usage_before, usage_after),
        "hosted_calls": hosted_calls,
        "per_row": trials,
    }


def analyze(raw: dict[str, Any]) -> dict[str, Any]:
    """Everything derived from per_row — pure, and re-runnable over a recorded file."""
    trials = raw["per_row"]
    derived: dict[str, Any] = {
        "arms": {arm: arm_summary(trials, arm) for arm in ARMS},
        "mcnemar": {
            "pair_vs_openrouter": mcnemar([t for t in trials if t["arm"] == "pair"],
                                          [t for t in trials if t["arm"] == "openrouter"]),
            "pair_vs_d1_systemone": mcnemar([t for t in trials if t["arm"] == "pair"],
                                            [t for t in trials if t["arm"] == "d1_systemone"]),
        },
        "escalation_prediction": {
            "population": predicted_escalation(PREDICTION_FILE),
            "same_rows": {
                "n": sum(1 for t in trials if t["arm"] == "d1_systemone"),
                "rate": round(sum(1 for t in trials if t["arm"] == "d1_systemone"
                                  and max(t["probabilities"].values() or [0.0]) < TAU)
                              / max(1, sum(1 for t in trials if t["arm"] == "d1_systemone")), 4),
                "basis": "max(arm C probabilities) < tau, the same criterion the pair applies",
            },
            "observed": observed_escalation(trials),
        },
    }
    pop = derived["escalation_prediction"]["population"]
    obs = derived["escalation_prediction"]["observed"]
    if "rate" in pop and obs["n"]:
        k = obs["escalated"]
        n = obs["n"]
        p = min(1.0, 2 * sum(math.comb(n, j) * pop["rate"] ** j * (1 - pop["rate"]) ** (n - j)
                             for j in range(0, k + 1)))
        derived["escalation_prediction"]["population_test"] = {
            "observed_escalations": k, "n": n, "p_predicted": pop["rate"],
            "p_two_sided": round(p, 4),
            "reading": "one binomial draw of 30 rows from the population rate; a gap this "
                       "size is sampling, not a policy difference",
        }
    derived["recommendation"] = recommend({**raw, **derived})
    return derived


def emit(raw: dict[str, Any], args: argparse.Namespace, *,
         measured: bool) -> dict[str, Any]:
    """Derive, print, and write — the only place outputs are produced.

    A measured run writes the whole summary to --out (refusing to overwrite:
    docs/testing.md wants a new suffix per run). A --summarize pass writes
    nothing unless --analysis-out names a file, because its input already is the
    record and the derived block is the only new thing there is to say.
    """
    summary = {**raw, **analyze(raw)}
    print_report(summary)
    print("\n== recommendation ==")
    for line in summary["recommendation"]:
        print(f"  {line}")
    if measured:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            raise SystemExit(f"refusing to overwrite {display_path(out_path)} "
                             "(docs/testing.md: new suffix per run)")
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {display_path(out_path)} ({len(summary['per_row'])} arm-rows)")
    if args.analysis_out:
        analysis_path = Path(args.analysis_out)
        analysis_path.parent.mkdir(parents=True, exist_ok=True)
        if analysis_path.exists():
            raise SystemExit(f"refusing to overwrite {display_path(analysis_path)}")
        derived = {k: v for k, v in summary.items()
                   if k in ("arms", "mcnemar", "escalation_prediction", "recommendation")}
        analysis_path.write_text(json.dumps(
            {"derived_from": display_path(Path(args.summarize or args.out)),
             "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             "note": "derived from that run's per_row by the committed analysis code. "
                     "Where a key appears both here and in the run file's own derived "
                     "block, this file is the later derivation of the two: the run "
                     "file's block is what the analysis code said at measurement time.",
             "analysis": derived},
            indent=2))
        print(f"wrote {display_path(analysis_path)} (derived, no new measurement)")
    return summary


def display_path(path: Path) -> str:
    """Repo-relative when it is inside the repo, absolute otherwise (--out /tmp)."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
