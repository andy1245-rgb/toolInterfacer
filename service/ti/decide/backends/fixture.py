"""fixture — recorded-decision replay backend (model-free tests + bench).

Replays choose/noul answers from a fixture file keyed by a stable hash of
(state, question). Any run is deterministic; the same runner code paths as
production, with an in-memory "model".
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from ..primitives import ChoiceAnswer, DecideBackend, register


def _key(state: Any, name: str, payload: Any) -> str:
    blob = json.dumps({"state": state, "q": name, "c": payload},
                      sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()  # sha1 used historically for this fixture map


@register("fixture")
class FixtureBackend(DecideBackend):
    name = "fixture"

    def __init__(self, fixture: dict[str, Any] | None = None) -> None:
        from ...conf import load
        cfg = load()["decide"]["fixture"]
        path = cfg.get("fixture_path", "")
        self.map: dict[str, Any] = fixture if fixture is not None else json.load(open(path)) if path else {}

    def _lookup(self, state: Any, name: str, payload: Any, default: Any) -> Any:
        return self.map.get(_key(state, name, payload), default)

    def choice(self, state, instructions: str, criteria: dict[str, str]) -> ChoiceAnswer:
        rec = self._lookup(state, "choice", criteria,
                           {"choice": next(iter(criteria), "__none__"),
                            "probabilities": {}, "confidence": 0.5})
        probs = rec.get("probabilities") or {k: 1.0 / max(len(criteria), 1) for k in criteria}
        return ChoiceAnswer(rec["choice"], probs, rec.get("confidence", 0.5))

    def noul(self, state, question: str) -> float:
        return float(self._lookup(state, "noul", question, 0.9))
