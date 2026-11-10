"""llama_systemone — the existing /v1/systemone path, unchanged behaviour.

The wire code is `bench/routing_diagnose.Server`, the same client every recorded arm used, so
a live arm through the harness and a recorded arm through `rewrite_transfer.py` are asking the
server the identical question.

Descriptions, not bare names: the instrument's option *text* is what the model reads. The
dataset carries it in ``row.meta["descriptions"]`` when it has one, and a predictor that gets
only a tuple of names uses the names themselves.
"""
from __future__ import annotations

import sys
from pathlib import Path

from ..protocol import PredictorError, Prediction, register

ROOT = Path(__file__).resolve().parents[3]


def _server(base_url: str, timeout: float, retries: int):
    bench = str(ROOT / "bench")
    if bench not in sys.path:
        sys.path.insert(0, bench)
    try:
        from routing_diagnose import Server  # noqa: PLC0415 - optional at import time
    except Exception as exc:  # noqa: BLE001
        raise PredictorError(f"llama_systemone: cannot import the frozen probe client: {exc}")
    return Server(base_url, timeout=timeout, retries=retries)


class LlamaSystemOne:
    def __init__(self, base_url: str = "http://127.0.0.1:8080", timeout: float = 120.0,
                 retries: int = 3, instructions: str = ""):
        self.base_url = base_url
        self.server = _server(base_url, timeout, retries)
        self.instructions = instructions
        self.name = f"llama_systemone:{base_url}"

    def _ask(self, text: str, criteria: dict[str, str], options: tuple[str, ...]) -> Prediction:
        import time
        t0 = time.perf_counter()
        answer = self.server.ask(text, criteria, self.instructions or _frozen_instructions())
        latency = (time.perf_counter() - t0) * 1000
        choice = answer.get("choice")
        if choice not in options:
            raise PredictorError(
                f"llama_systemone returned {choice!r}, which is not in the option set")
        return Prediction(choice=choice, probabilities=answer.get("probabilities") or None,
                          latency_ms=latency)

    def predict_for(self, row) -> Prediction:
        descriptions = (row.meta or {}).get("descriptions") or {}
        criteria = {o: descriptions.get(o, o) for o in row.options}
        return self._ask(row.text, criteria, row.options)

    def predict(self, text: str, options: tuple[str, ...]) -> Prediction:
        return self._ask(text, {o: o for o in options}, options)


def _frozen_instructions() -> str:
    bench = str(ROOT / "bench")
    if bench not in sys.path:
        sys.path.insert(0, bench)
    from ti_bench.prompts import INSTRUCTIONS
    return INSTRUCTIONS


@register("llama_systemone")
def build(base_url: str = "http://127.0.0.1:8080", timeout: str = "120",
          retries: str = "3", **_kw) -> LlamaSystemOne:
    return LlamaSystemOne(base_url=base_url, timeout=float(timeout), retries=int(retries))
