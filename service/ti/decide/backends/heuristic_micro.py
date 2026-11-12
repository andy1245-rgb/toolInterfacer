"""heuristic_micro — pattern gate that keeps the decision model off trivial hops.

Deliberately tiny: a URL present → fetch; the reddit scheme present → reddit. Not a router:
resolve.FastPathBackend runs it as the pre-gate in front of the decision model, and it abstains
(None) unless one of the criteria is a key it recognised, which leaves most questions to the
model.

**The third detector is gone, and this is why.** It used to emit `hits["filesys"]` for a
path-like token. `filesys` is not one of the eight op names — the router offers `filesys_read`,
`filesys_glob` and `filesys_grep` — so the filter `if c in criteria` dropped it and the branch
answered nothing, ever. Measured over the 415 routing rows: `_PATH` matched **0 of 415**. It was
dead in two independent ways at once.

It could not be repaired by loosening the pattern, which is what a reader would try first. A
loosened path regex matches **149 of 415** rows and only **60** of them are truly `filesys_*`
(40% precise): it would answer 38 rows that are `web_fetch` and 23 that are `reddit_digest`, both
of which the surviving detectors get right today, and 28 rows across four other ops at 0%. One key
cannot name three operations, and the pattern that would need to tell `filesys_read` from
`filesys_grep` is not a regex over the surface text — that is the model's job.

The two surviving detectors are what the pre-gate is actually for, and they are exact:

    web_fetch      38 of 415 rows, 38 of them correct
    reddit_digest  36 of 415 rows, 36 of them correct

Coverage, not accuracy, is the number to watch here: a detector that stops matching has a
perfect accuracy on the rows it no longer answers. `bench/backend_coverage.py` prints answered,
abstained and precision for every offline backend against the eval split.
"""
from __future__ import annotations

import re
from typing import Any

from ..primitives import ChoiceAnswer, DecideBackend, register

_URL = re.compile(r"https?://\S+")
_REDDIT = re.compile(r"\br/\w+|\breddit\b", re.I)


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
