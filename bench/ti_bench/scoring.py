"""scoring — one arm over one frozen set, into the report dict.

Order matters and is enforced here, not by the caller:

  1. ``arms.apply`` puts the arm's conf overrides (and ``--base-url``) in place;
  2. ``guard.preflight`` asserts the endpoint serves what the arm says — a
     mismatch raises before a single probe byte is sent, and before the backend
     is even constructed;
  3. ``arms.build`` constructs the backend through ``make_backend`` directly, so
     ``FastPathBackend``'s cache and the ``heuristic_micro`` pre-gate cannot
     answer a row. Either would show up as a 0 ms sample in the latency
     distribution while looking like a very fast model;
  4. warm-up: ``warmup`` throwaway calls on the first row, then the scored pass.
     They are not rows, so they are excluded from every number by construction
     and their count is reported. This is the "one warm-up discarded per arm"
     rule docs/decision-pair-vs-hosted.md measured with; keeping the first row
     would charge it the model-page-in cost the rest of the distribution does not
     pay.

The state handed to ``choice`` is the utterance string, exactly as
``bench/live_probe.py`` passes it, so a number from here is comparable with the
historical lanes.
"""
from __future__ import annotations

import sys
import time
from typing import TextIO

from . import arms, guard, sets, stats
from .prompts import INSTRUCTIONS, OP_DESCRIPTIONS

SCHEMA = "ti-bench/1"

# How a row was answered. Only MODEL is an external call; IN_PROCESS is a
# heuristic or a replay, and the other two are not answers at all.
MODEL = "model"
IN_PROCESS = "in-process"
ABSTAIN = "abstain"
ERROR = "error"

# Backends that keep per-call telemetry on the instance (pair: last_stage,
# last_reason; openrouter: last_provider) do not put it in ChoiceAnswer — that
# dataclass is the shared primitive and does not get to grow. Reading whatever is
# there keeps the report useful for escalation analysis without coupling to a
# backend's internals.
TELEMETRY = ("last_stage", "last_escalated", "last_reason", "last_stage1_conf",
             "last_provider")


def _telemetry(backend) -> dict:
    out = {}
    for name in TELEMETRY:
        value = getattr(backend, name, None)
        if isinstance(value, (str, bool, int, float)):
            out[name] = value
    return out


def _row(row: dict, *, chosen, confidence, correct, ms, source, real_call,
         error=None, extra=None) -> dict:
    return {"id": row["id"], "uid": row["uid"], "idx": row["idx"], "band": row["band"],
            "utterance": row["utterance"], "expected": row["expected"],
            "chosen": chosen, "correct": bool(correct), "confidence": confidence,
            "latency_ms": None if ms is None else round(ms, 3),
            "real_call": bool(real_call), "source": source, "error": error,
            "extra": extra or {}}


def score(arm: arms.Arm, rows: list[dict], *, limit: int = 0, warmup: int | None = None,
          expect_model: str = "", base_url: str = "", allow_unasserted: bool = False,
          stream: TextIO | None = None) -> dict:
    """Probe ``arm`` over ``rows`` and return the report dict (nothing is written)."""
    stream = stream if stream is not None else sys.stderr
    rows = rows[:limit] if limit else rows
    if not rows:
        raise ValueError("no rows to score")
    warmup = arm.real_call if warmup is None else warmup

    arms.apply(arm, base_url)
    guard_block = guard.preflight(arm, expect_model, allow_unasserted, stream)
    backend = arms.build(arm)

    warmup_ms: list[float] = []
    for _ in range(max(0, warmup)):
        t0 = time.perf_counter()
        try:
            backend.choice(rows[0]["utterance"], INSTRUCTIONS, OP_DESCRIPTIONS)
        except Exception:  # a throwaway call: a failure here is not a measurement
            pass
        warmup_ms.append((time.perf_counter() - t0) * 1000)

    per_row = []
    for row in rows:
        t0 = time.perf_counter()
        try:
            answer = backend.choice(row["utterance"], INSTRUCTIONS, OP_DESCRIPTIONS)
        except Exception as e:  # one bad row must not lose the other 149
            per_row.append(_row(row, chosen=None, confidence=None, correct=False,
                                ms=(time.perf_counter() - t0) * 1000,
                                source=ERROR, real_call=False, error=str(e) or type(e).__name__))
            continue
        ms = (time.perf_counter() - t0) * 1000
        extra = _telemetry(backend)
        if answer is None:  # deliberate abstain (heuristic_micro: no opinion)
            per_row.append(_row(row, chosen=None, confidence=None, correct=False, ms=ms,
                                source=ABSTAIN, real_call=False, error="abstain (no opinion)",
                                extra=extra))
            continue
        chosen = getattr(answer, "choice", None)
        conf = float(getattr(answer, "confidence", 0.0) or 0.0)
        source = MODEL if arm.real_call else IN_PROCESS
        per_row.append(_row(row, chosen=chosen, confidence=conf,
                            correct=chosen == row["expected"], ms=ms, source=source,
                            real_call=arm.real_call and bool(chosen),
                            error=None if chosen else "empty choice", extra=extra))

    return _report(arm, rows, per_row, warmup_ms, guard_block)


def _report(arm: arms.Arm, rows: list[dict], per_row: list[dict], warmup_ms: list[float],
            guard_block: dict) -> dict:
    n = len(per_row)
    correct = sum(1 for r in per_row if r["correct"])
    by_band: dict[str, list[dict]] = {}
    for r in per_row:
        by_band.setdefault(r["band"], []).append(r)

    # Error rows carry a latency that is a transport failure, not this arm's
    # answer time (a refused connection is ~1 ms); live_probe.py drops them from
    # the distribution too. Abstains are kept: they were a real call.
    latencies = [r["latency_ms"] for r in per_row
                 if r["latency_ms"] is not None and r["source"] != ERROR]
    confidence_pairs = [(r["confidence"], r["correct"]) for r in per_row
                        if r["confidence"] is not None]

    return {
        "schema": SCHEMA,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "arm": arm.as_dict(),
        "set": {"name": rows[0]["id"].split(":")[0], "path": _set_path(rows),
                "n": n, "bands": _bands(rows)},
        "guard": guard_block,
        "warmup": {"calls_discarded": len(warmup_ms),
                   "latency_ms": [round(ms, 3) for ms in warmup_ms]},
        "counts": {"n": n, "correct": correct, "wrong": n - correct,
                   "abstain": sum(1 for r in per_row if r["source"] == ABSTAIN),
                   "error": sum(1 for r in per_row if r["source"] == ERROR),
                   "real_calls": sum(1 for r in per_row if r["real_call"])},
        "accuracy": stats.rate(correct, n),
        "per_band": {band: stats.rate(sum(1 for r in members if r["correct"]), len(members))
                     for band, members in sorted(by_band.items())},
        "latency": {"n_used": len(latencies),
                    "p50": round(stats.percentile(latencies, 0.50), 3),
                    "p90": round(stats.percentile(latencies, 0.90), 3),
                    "p99": round(stats.percentile(latencies, 0.99), 3),
                    "mean": round(sum(latencies) / len(latencies), 3) if latencies else None,
                    "min": round(min(latencies), 3) if latencies else None,
                    "max": round(max(latencies), 3) if latencies else None,
                    "error_rows_excluded":
                        sum(1 for r in per_row if r["source"] == ERROR)},
        "calibration": stats.calibration(confidence_pairs),
        "calibration_n": len(confidence_pairs),
        "per_row": per_row,
    }


def _set_path(rows: list[dict]) -> str:
    name = rows[0]["id"].split(":")[0]
    return sets.path_of(name)


def _bands(rows: list[dict]) -> dict[str, int]:
    return sets.bands(rows)
