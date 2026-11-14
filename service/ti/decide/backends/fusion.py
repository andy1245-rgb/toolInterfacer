"""fusion — the 3B and a cheap text member, averaged, with two ops the 3B owns.

This is the shipping form of the arm that passed a pre-registered test on 2026-11-10
(`docs/fusion-2026-11-10.md`, option 3): the equal-weight mean of the two backends' distributions
over the offered options, except that when the base's own top choice is `web_search` or
`web_fetch` the member is not consulted and the base answers alone. Measured on the 415 routing
rows: 349 correct against the base's 325, +24 rows, exact McNemar p = 3.0e-06, every label netting
zero or better, positive in 5 of 5 fold seeds.

**It is off, and it says so.** `decide.fusion.enabled` must be true *and*
`decide.backends_in_use.choice` must be `"fusion"`. If the block is absent, or present with
`enabled: false`, asking for this backend raises with both lines it needs — it does not quietly
serve the base's answer instead. A disabled component that silently behaves like an enabled one is
the defect class this repository spent a night cataloguing (`docs/rot-audit-2026-11-10.md`).

Cost, stated where the code is: this backend calls **both** members on every request. The member is
cheap (0.13 ms measured), so the p50 is the base's own latency plus 0.13 ms — it buys accuracy, not
time. The front-door variant that buys time is a different configuration
(`docs/classical-revival-2026-11-10.md`), and the two should not be confused.

Config block, off by default (absent means off):

    "fusion": {
      "enabled": false,
      "member": "classical",
      "base": "d1_systemone",
      "weights": 0.5,
      "protect": ["web_search", "web_fetch"]
    }

`weights` is the weight on the **base**; 0.5 is the null and the only value ever measured. `protect`
is a list of labels the base owns outright. Both are copied from the arm's spec and neither was
tuned.
"""
from __future__ import annotations

from typing import Any

from ..primitives import ChoiceAnswer, DecideBackend, make_backend, register

DEFAULT_PROTECT = ("web_search", "web_fetch")
ENABLE_HINT = ('set decide.fusion.enabled = true and decide.backends_in_use.choice = "fusion" '
               '(preset in docs/fusion-2026-11-10.md)')


def _argmax(known: list[str], values: dict[str, float]) -> str:
    return min(known, key=lambda option: (-values[option], known.index(option), option))


@register("fusion")
class FusionBackend(DecideBackend):
    name = "fusion"

    def __init__(self, config: dict[str, Any] | None = None,
                 member: DecideBackend | None = None,
                 base: DecideBackend | None = None) -> None:
        if config is None:
            from ...conf import load
            config = dict((load().get("decide") or {}).get("fusion") or {})
        self.cfg = dict(config)
        if not self.cfg.get("enabled"):
            raise NotImplementedError(
                f"fusion: the backend is configured but disabled — {ENABLE_HINT}. It refuses "
                f"rather than falling back to the base, because a silent fallback here would "
                f"report the fusion's name on the base's answer")
        self.member = member or make_backend(str(self.cfg.get("member") or "classical"))
        self.base = base or make_backend(str(self.cfg.get("base") or "d1_systemone"))
        self.weights = float(self.cfg.get("weights", 0.5))
        if not 0.0 <= self.weights <= 1.0:
            raise ValueError(f"fusion: weights={self.weights} must be in [0, 1]")
        protect = self.cfg.get("protect")
        self.protect = tuple(protect) if protect is not None else DEFAULT_PROTECT

    # -- the decision ---------------------------------------------------

    def choice(self, state: Any, instructions: str, criteria: dict[str, str]) -> ChoiceAnswer:
        offered = [c for c in criteria]
        if not offered:
            return ChoiceAnswer(None, {}, 0.0)
        base = self.base.choice(state, instructions, criteria)
        base_probs = self._dist(base, offered)
        if base.choice is not None and base.choice in self.protect:
            # the ops the base owns: the member is not consulted at all, not even for a tie-break
            return ChoiceAnswer(base.choice, base_probs, base.confidence)

        member = self.member.choice(state, instructions, criteria)
        member_probs = self._dist(member, offered)
        if member_probs is None:
            return ChoiceAnswer(base.choice, base_probs, base.confidence)
        blended = {o: self.weights * base_probs.get(o, 0.0)
                      + (1.0 - self.weights) * member_probs.get(o, 0.0) for o in offered}
        total = sum(blended.values())
        if not total > 0.0:
            return ChoiceAnswer(base.choice, base_probs, base.confidence)
        distribution = {o: v / total for o, v in blended.items()}
        pick = _argmax(offered, distribution)
        return ChoiceAnswer(pick, distribution, distribution[pick])

    @staticmethod
    def _dist(answer: ChoiceAnswer | None, offered: list[str]) -> dict[str, float] | None:
        """A distribution over the offered options, or None when the answer carries none."""
        probs = dict(getattr(answer, "probabilities", None) or {})
        if not probs:
            return None
        kept = {o: float(probs[o]) for o in offered if o in probs}
        total = sum(kept.values())
        if not total > 0.0:
            return None
        return {o: v / total for o, v in kept.items()}

    def noul(self, state: Any, question: str) -> float | None:
        """Relevance is the base's judgement; the member has no opinion on it."""
        return self.base.noul(state, question)
