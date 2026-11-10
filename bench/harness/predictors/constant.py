"""constant — always the same op. The baseline every model has to beat.

Not a joke: on the current agentFleet corpus, always answering the same option scores better
than the local engine does. A harness whose constant arm is not printed cannot tell you that.
"""
from __future__ import annotations

from ..protocol import Prediction, register


class ConstantPredictor:
    def __init__(self, choice: str = "", index: int = 0):
        self.choice = choice
        self.index = index
        self.name = f"constant:{choice or f'#{index}'}"

    def predict(self, text: str, options: tuple[str, ...]) -> Prediction:
        if self.choice:
            pick = self.choice
        else:
            pick = options[self.index % len(options)]
        return Prediction(choice=pick, probabilities=None, latency_ms=0.0)


@register("constant")
def build(choice: str = "", index: str = "0", **_kw) -> ConstantPredictor:
    return ConstantPredictor(choice=choice, index=int(index))
