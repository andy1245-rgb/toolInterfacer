"""logit_read — the existing letter read (`service/ti/decide/backends/logit_read.py`).

One /v1/chat/completions per row with the options as letters A..Z, max_tokens=1, logprobs on,
then the letter logprobs are mapped back onto the option set. The service backend reads its
endpoint and model from the service conf, so this wrapper builds it once and overrides the two
fields the spec names — that keeps the wire behaviour identical while letting the harness point
it at any served gguf.
"""
from __future__ import annotations

import sys
from pathlib import Path

from ..protocol import PredictorError, Prediction, register

ROOT = Path(__file__).resolve().parents[3]


class LogitRead:
    def __init__(self, base_url: str = "http://127.0.0.1:8080", model: str = "jev-style"):
        service = str(ROOT / "service")
        if service not in sys.path:
            sys.path.insert(0, service)
        try:
            from ti.decide.backends.logit_read import LogitReadBackend  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            raise PredictorError(f"logit_read: cannot import the service backend: {exc}")
        try:
            self.backend = LogitReadBackend()
        except Exception as exc:  # noqa: BLE001 - conf may not load outside the service
            raise PredictorError(f"logit_read: the service backend would not construct: {exc}")
        self.backend.base_url = base_url
        self.backend.model = model
        self.name = f"logit_read:{model}"

    def predict(self, text: str, options: tuple[str, ...]) -> Prediction:
        if len(options) > 20:
            raise PredictorError("logit_read supports at most 20 options")
        import time
        t0 = time.perf_counter()
        try:
            answer = self.backend.decide(text, list(options))
        except Exception as exc:  # noqa: BLE001
            raise PredictorError(f"logit_read: {type(exc).__name__}: {exc}")
        latency = (time.perf_counter() - t0) * 1000
        choice = getattr(answer, "choice", None) or getattr(answer, "label", None)
        probs = getattr(answer, "probabilities", None)
        if choice is None:
            raise PredictorError(f"logit_read: backend returned {answer!r}")
        return Prediction(choice=choice, probabilities=probs, latency_ms=latency)


@register("logit_read")
def build(base_url: str = "http://127.0.0.1:8080", model: str = "jev-style", **_kw) -> LogitRead:
    return LogitRead(base_url=base_url, model=model)
