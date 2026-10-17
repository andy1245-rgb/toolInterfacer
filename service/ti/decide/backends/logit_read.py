"""logit_read — Jev-Style logit-read decision backend for a plain llama.cpp server.

One POST /v1/chat/completions per decision: prompt ends in "Answer:" with the
options as letters A-Z, max_tokens=1, logprobs=true, top_logprobs=20. The
returned top tokens are mapped onto the option letters, softmax-renormalized,
and confidence = 1 - H(p)/ln(K). Stdlib only (urllib). Supports <= 20 options.
"""
from __future__ import annotations

import json
import math
import urllib.request

from ...conf import load
from ..primitives import ChoiceAnswer, DecideBackend, register

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
HEADER = ("You are a decision function. Read the state, then answer the question "
          "by choosing exactly one option.\n\n")
MAX_OPTIONS = 20


@register("logit_read")
class LogitReadBackend(DecideBackend):
    name = "logit_read"

    def __init__(self) -> None:
        cfg = load()["decide"]["logit_read"]
        self.base_url = cfg["base_url"]
        self.model = cfg["model"]

    def _top_logprobs(self, prompt: str) -> dict[str, float]:
        body = json.dumps({"model": self.model,
                           "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": 1, "temperature": 0,
                           "logprobs": True, "top_logprobs": 20}).encode()
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=cfg_timeout()) as resp:
            out = json.loads(resp.read())
        return {t["token"]: t["logprob"]
                for t in out["choices"][0]["logprobs"]["content"][0]["top_logprobs"]}

    @staticmethod
    def _letter_probs(n_options: int, top: dict[str, float]) -> list[float]:
        floor = min(top.values()) - 5.0  # letter outside top-20: negligible mass
        logits = [max((lp for tok, lp in top.items() if tok.strip() == LETTERS[i]),
                      default=floor)
                  for i in range(n_options)]
        z = max(logits)
        exps = [math.exp(x - z) for x in logits]
        total = sum(exps)
        return [e / total for e in exps]

    @staticmethod
    def _entropy_conf(probs: list[float]) -> float:
        h = -sum(p * math.log(p) for p in probs if p > 0)
        return float(1.0 - h / math.log(len(probs)))

    def choice(self, state, instructions: str, criteria: dict[str, str]) -> ChoiceAnswer:
        if not 2 <= len(criteria) <= MAX_OPTIONS:
            raise ValueError(f"logit_read: need 2..{MAX_OPTIONS} options, got {len(criteria)}")
        state_text = state if isinstance(state, str) else json.dumps(state, default=str)
        names = list(criteria)
        lines = "\n".join(f"{LETTERS[i]}. {criteria[n]}" for i, n in enumerate(names))
        prompt = (f"{HEADER}[State]\n{state_text}\n\n[Question]\n{instructions}\n\n"
                  f"[Options]\n{lines}\n\nAnswer:")
        probs = self._letter_probs(len(names), self._top_logprobs(prompt))
        best = max(range(len(names)), key=lambda i: probs[i])
        return ChoiceAnswer(names[best],
                            {n: round(p, 6) for n, p in zip(names, probs)},
                            round(self._entropy_conf(probs), 4))

    def noul(self, state, question: str) -> float:
        state_text = state if isinstance(state, str) else json.dumps(state, default=str)
        prompt = (f"{HEADER}[State]\n{state_text}\n\n[Question]\n{question}\n\n"
                  f"[Options]\nA. yes\nB. no\n\nAnswer:")
        probs = self._letter_probs(2, self._top_logprobs(prompt))
        return round(probs[0], 6)


def cfg_timeout() -> int:
    return int(load()["decide"]["logit_read"].get("timeout_s", 30))
