"""Ask-back — the triggers, question builders, and budget path.

Trigger list (spec §4): route conf below gate; slot conf below gate *after
candidate enumeration*; noul verification failed twice; contradictory
artifacts. Questions are always structurable: choice (candidates listed) >
confirm > free. Payload includes hops + artifacts so the LLM answers once and
the loop resumes with no redo.

on_ask(question_spec) is provided by the caller (http.api / tests).
Exhausting max_asks flips the run to degraded-done: return best-guess results
with assumptions[] — a degraded digest still has information value; an abort
does not.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Question:
    kind: str                       # "choice" | "confirm" | "free"
    question: str
    options: list[dict[str, Any]] = field(default_factory=list)  # {"id","desc"}
    hops_seen: list[dict[str, Any]] = field(default_factory=list)


class AskBack:
    def __init__(self, on_ask: Any, max_asks: int) -> None:
        self._on_ask = on_ask
        self._max = max_asks
        self.used = 0

    def ask(self, q: Question) -> Any:
        if self.used >= self._max:
            raise MaxAsksReached()
        self.used += 1
        return self._on_ask(q)

    @property
    def exhausted(self) -> bool:
        return self.used >= self._max


class MaxAsksReached(Exception):
    pass
