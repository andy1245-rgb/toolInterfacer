"""fake — a deterministic predictor with a known error rate, for testing the harness itself.

Two modes:

* **cheat sheet** (``truth=path.json`` or rows passed directly): the predictor knows the right
  answer for each text, and is right with probability ``accuracy`` decided by a hash of the
  seed and the text. That makes the number of correct rows exactly computable in a test, and
  makes the *whole* pipeline — dataset, predictor, metrics, controls, artifacts — testable
  with no model and no network.
* **blind**: no cheat sheet, so it picks an option by hash. Useful for exercising the plumbing
  on a corpus it has never seen, and the accuracy it produces is the chance floor.

It is a fixture, not a model: it never pretends to be one, and it is the only predictor in the
registry that may knowingly read the label.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..protocol import Prediction, register
from ..rows import Row


def _unit(*parts: str) -> float:
    """Deterministic uniform in [0, 1) from the given parts."""
    h = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16) / float(16 ** 16)


class FakePredictor:
    def __init__(self, truth: dict[str, str] | None = None, accuracy: float = 1.0,
                 seed: str = "0", probabilities: bool = True,
                 latency_ms: float = 0.0):
        if not 0.0 <= accuracy <= 1.0:
            raise ValueError("accuracy must be in [0, 1]")
        self.truth = dict(truth or {})
        self.hit_rate = accuracy
        self.seed = str(seed)
        self.want_probs = probabilities
        self.latency_ms = latency_ms
        self.name = f"fake:acc={accuracy}:seed={seed}"

    def _is_right(self, text: str) -> bool:
        return _unit("fake", self.seed, text) < self.hit_rate

    def predict(self, text: str, options: tuple[str, ...]) -> Prediction:
        target = self.truth.get(text)
        if target is not None and target in options:
            if self._is_right(text):
                pick = target
            else:
                # a deterministic neighbour, so misses are stable and countable
                pick = options[(options.index(target) + 1) % len(options)]
        else:
            pick = options[int(_unit("blind", self.seed, text) * len(options)) % len(options)]
        probs = None
        if self.want_probs:
            rest = (1.0 - 0.6) / max(1, len(options) - 1)
            probs = {o: rest for o in options}
            probs[pick] = 0.6
        return Prediction(choice=pick, probabilities=probs, latency_ms=self.latency_ms)


@register("fake")
def build(truth: str = "", accuracy: str = "1.0", seed: str = "0",
          probabilities: str = "true", latency_ms: str = "0.0", **_kw) -> FakePredictor:
    table: dict[str, str] = {}
    if truth:
        p = Path(truth)
        raw = json.loads(p.read_text(encoding="utf-8"))
        table = raw if isinstance(raw, dict) else {r["text"]: r["label"] for r in raw}
    return FakePredictor(truth=table, accuracy=float(accuracy), seed=seed,
                         probabilities=str(probabilities).lower() not in ("0", "false", "no"),
                         latency_ms=float(latency_ms))


def truth_from_rows(rows: list[Row]) -> dict[str, str]:
    """The cheat sheet a test hands to a FakePredictor."""
    return {r.text: r.label for r in rows}
