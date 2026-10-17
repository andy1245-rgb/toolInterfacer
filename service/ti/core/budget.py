"""Budget — hop, ask, and wall-clock ceilings for one plan run."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


class BudgetExceeded(Exception):
    def __init__(self, what: str) -> None:
        super().__init__(what)
        self.what = what


@dataclass
class Budget:
    max_hops: int
    max_asks: int
    max_seconds: float
    hops: int = 0
    asks: int = 0
    t0: float = field(default_factory=time.monotonic)

    def use_hop(self) -> None:
        self.hops += 1
        if self.hops > self.max_hops:
            raise BudgetExceeded(f"hop budget {self.max_hops} exceeded")

    def use_ask(self) -> None:
        self.asks += 1
        if self.asks > self.max_asks:
            raise BudgetExceeded(f"ask budget {self.max_asks} exceeded")

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def tick_clock(self) -> None:
        if self.elapsed() > self.max_seconds:
            raise BudgetExceeded(f"wall clock {self.max_seconds}s exceeded")

    @property
    def degraded(self) -> bool:
        return self.asks > 0
