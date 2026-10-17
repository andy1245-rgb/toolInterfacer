"""heuristic_micro — pattern gate that keeps the decision model off trivial hops.

Deliberately tiny: URL present → fetch; reddit scheme present → reddit;
path-like token → filesys. Not a router. Phase 5 measures it; deleted if it
does not pay.
"""
from __future__ import annotations

import re
from typing import Any

from ..primitives import ChoiceAnswer, DecideBackend, register

_URL = re.compile(r"https?://\S+")
_REDDIT = re.compile(r"\br/\w+|\breddit\b", re.I)
_PATH = re.compile(r"(?:^|\s)(/[\w./-]+|\.\.?/[\w./-]+)", re.I)


@register("heuristic_micro")
class HeuristicMicroBackend(DecideBackend):
    name = "heuristic_micro"

    def choice(self, state, instructions: str, criteria: dict[str, str]) -> ChoiceAnswer | None:
        state_text = state if isinstance(state, str) else json_dumps(state)
        hits: dict[str, float] = {}
        if _URL.search(state_text):
            hits["web_fetch"] = 0.99
        if _REDDIT.search(state_text):
            hits["reddit_digest"] = 0.99
        if _PATH.search(state_text):
            hits["filesys"] = 0.99
        options = [c for c in hits if c in criteria]
        if not options:
            return None  # no opinion — caller must fall back to a real backend
        pick = options[0]
        others = {k: v for k, v in hits.items() if k in criteria and k != pick}
        total = hits[pick] + sum(others.values())
        probs = {pick: hits[pick] / total} | {k: v / total for k, v in others.items()}
        for missing in criteria:
            probs.setdefault(missing, 1e-6)
        return ChoiceAnswer(pick, probs, hits[pick])

    def noul(self, state, question: str) -> float | None:
        return None


def json_dumps(x: Any) -> str:
    import json
    return json.dumps(x, default=str)
