"""Runner — the op → artifact → gate → next-step cycle.

A plan (from ti.planner) is a list of Steps. The runner:
  1. resolves the step's op from the ops registry,
  2. runs it with validated params (validated against the op's schema),
  3. stores its artifact,
  4. evaluates the gate's post-verifiers,
  5. advances; on GateBlocked routes into the ask-back path (policy),
  6. on budget exhaustion returns a degraded result instead of failing.

Steps may name injected params from `params` (LLM keys matching the step
schema) and source artifacts from earlier steps.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.artifacts import ArtifactStore
from ..core.budget import Budget, BudgetExceeded
from ..core.gates import Gate, GateBlocked, check_post, check_pre


@dataclass
class Step:
    id: str
    op: str
    params: dict[str, Any] = field(default_factory=dict)   # spec-side params (config-resolved)
    inject: dict[str, str] = field(default_factory=dict)   # step-key -> artifact key to feed in
    gate: Gate = field(default_factory=Gate)


@dataclass
class RunResult:
    status: str                     # "done" | "need_input" | "error"
    result: Any = None
    degraded: bool = False
    assumptions: list[str] = field(default_factory=list)
    hops: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


def run_plan(steps: list[Step], llm_params: dict[str, Any], budget: Budget,
             on_ask: Callable[..., Any], run_op: Callable[..., Any],
             run_verifier: Callable[..., float]) -> RunResult:
    """run_op(op_name, params, store) -> (kind, data, conf).

    run_verifier(name, artifact) -> float | None.
    on_ask(question_spec) -> answer (raises AskBudgetExceeded-like via budget).
    """
    raise NotImplementedError("P1")
