"""d1_systemone — d1-3B (Liquid LFM decision model) via llama.cpp /v1/systemone.

One POST, zero generated tokens, calibrated confidence. Requires llama.cpp
>= b11500.
"""
from __future__ import annotations

import json
import urllib.request

from ...conf import load
from ..primitives import ChoiceAnswer, DecideBackend, register


@register("d1_systemone")
class D1SystemoneBackend(DecideBackend):
    name = "d1_systemone"

    def __init__(self) -> None:
        cfg = load()["decide"]["d1_systemone"]
        self.base_url = cfg["base_url"]

    def _ask(self, state, questions: dict) -> dict:
        body = json.dumps({"state": state, "questions": questions}).encode()
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/v1/systemone",
            data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=cfg_timeout()) as resp:
            return json.loads(resp.read())

    def choice(self, state, instructions: str, criteria: dict[str, str]) -> ChoiceAnswer:
        out = self._ask(state, {"route": {"type": "choice",
                                          "instructions": instructions,
                                          "criteria": dict(criteria)}})
        ans = (out.get("answers") or {}).get("route") or {}
        probs = {k: float(v) for k, v in (ans.get("probabilities") or {}).items()}
        return ChoiceAnswer(ans.get("choice"), probs, float(ans.get("confidence") or 0.0))

    def noul(self, state, question: str) -> float:
        out = self._ask(state, {"check": {"type": "noul", "question": question}})
        return float((out.get("answers") or {}).get("check", {}).get("noul", 0.0))


def cfg_timeout() -> int:
    return int(load()["decide"]["d1_systemone"].get("timeout_s", 30))
