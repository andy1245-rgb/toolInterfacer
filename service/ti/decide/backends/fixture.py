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


def _default_pick(criteria: dict[str, str], *texts: Any) -> str:
    """Fixture-only lexical match against the intent/instruction text.
    NOT a decision-quality claim; d1 (P2) routes."""
    ids = list(criteria)
    if not ids:
        return "__none__"
    import re
    blob = " ".join(t for t in texts if isinstance(t, str))
    words = [w for w in re.findall(r"[a-z0-9_]+", blob.lower()) if len(w) > 3]
    for cid in ids:
        low = cid.lower()
        if not low.endswith(('.pyc', '/', '.git')):
            if any(w in low for w in words):
                return cid
    return ids[0]


@register("fixture")
class FixtureBackend(DecideBackend):
    name = "fixture"

    def __init__(self, fixture: dict[str, Any] | None = None) -> None:
        from ...conf import load
        cfg = load()["decide"]["fixture"]
        path = cfg.get("fixture_path", "")
        self.default_choice_conf = float(cfg.get("default_choice_conf", 0.95))
        self.default_noul = float(cfg.get("default_noul", 0.9))
        self.map: dict[str, Any] = fixture if fixture is not None else json.load(open(path)) if path else {}

    def _lookup(self, state: Any, name: str, payload: Any, default: Any) -> Any:
        return self.map.get(_key(state, name, payload), default)

    def choice(self, state, instructions: str, criteria: dict[str, str]) -> ChoiceAnswer:
        ctx = state.get("intent_context") if isinstance(state, dict) else state
        rec = self._lookup(state, "choice", criteria,
                           {"choice": _default_pick(criteria, ctx, instructions),
                            "probabilities": {}, "confidence": self.default_choice_conf})
        probs = rec.get("probabilities") or {k: 1.0 / max(len(criteria), 1) for k in criteria}
        return ChoiceAnswer(rec["choice"], probs, rec.get("confidence", self.default_choice_conf))

    def noul(self, state, question: str) -> float:
        return float(self._lookup(state, "noul", question, self.default_noul))
