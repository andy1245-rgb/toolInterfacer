"""Gates — pre-conditions and post-verifiers attached to steps.

A step may declare:
  pre:  [(artifact_key, "exists"|"conf_gte"|"noul_gte"|... , threshold_or_None)]
  post: a verifier name evaluated against the step's artifact ("is_relevant"...)

Gate evaluation is pure: it reads the store and the decide primitives registry
(the caller passes the n-oul bound function). Failures raise GateBlocked, which
the runner routes into the ask-back path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


class GateBlocked(Exception):
    def __init__(self, step: str, what: str, detail: dict[str, Any]) -> None:
        super().__init__(f"gate blocked step {step!r}: {what}")
        self.step = step
        self.what = what
        self.detail = detail


@dataclass
class Gate:
    """Declarative gate spec, resolved at run time against config + artifacts."""
    pre: list[dict[str, Any]] = field(default_factory=list)   # {"artifact": key, "kind": ..., "min": ...}
    post: list[dict[str, Any]] = field(default_factory=list)  # {"verifier": name, "min": 0.75}


def check_pre(gate: Gate, step: str, artifacts: Any) -> None:
    for cond in gate.pre:
        art = artifacts.try_get(cond["artifact"])
        if art is None:
            raise GateBlocked(step, f"missing artifact {cond['artifact']!r}", cond)
        if cond.get("kind") == "conf_gte" and art.conf < float(cond.get("min", 0.0)):
            raise GateBlocked(step, f"artifact {cond['artifact']!r} conf {art.conf:.2f} below gate", cond)


def check_post(gate: Gate, step: str, artifact: Any, run_verifier: Callable[..., float]) -> None:
    for cond in gate.post:
        p = run_verifier(cond["verifier"], artifact)
        if p is not None and p < float(cond.get("min", 0.75)):
            raise GateBlocked(step, f"post check {cond['verifier']!r} = {p:.2f} below gate", cond)
