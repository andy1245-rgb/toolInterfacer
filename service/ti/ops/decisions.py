"""Decision steps surfaced as ops so templates speak one step vocabulary.

decide_extract — fill a list of literal values (queries, candidate keys) from
the intent + context. Uses the backend's extract primitive (fixture replayable
via recorded map; d1 extract lands in P2/P4).

The mined payload is `source_artifact` when a step names one, else
`intent_context` — the intent text is the context a first-step extractor has,
and a step that mines nothing would answer ([], 0.1) on every research run.
Instructions steer the *ask* (queries vs urls vs paths); the payload supplies
the words.

decide_pick — choose top_k items from a source artifact. Uses `choice` with
criteria built from the candidates (id + description lines preferred).

Both ops report the confidence their backend measured, and None when it measured
none: `run_extract` defaults to no number (the shipped rule_extract matches
patterns, it does not score them) rather than to a plausible-looking constant,
and the runner, the artifacts and the run-level `confidence` all carry that
absence through instead of substituting one.
"""
from __future__ import annotations

import json
import re
from typing import Any

from ..context.packer import guard_state
from ..decide.resolve import resolve as resolve_backend
from .op_api import op

# The number of candidates that reach the decision model. Its prefill dominates
# latency on a CPU-only box (~19 ms/token measured), so this is a latency knob;
# _rank keeps the most relevant ones so the cap costs no accuracy.
DEFAULT_MAX_CANDIDATES = 6

# The number of characters of each candidate description that reach the decision
# model. `guard_state` bounds the state but not the criteria, so without this a
# single candidate carrying a multi-megabyte snippet is a wedged service, not a
# slow decision (docs/review-2026-11-03.md B5 measured 12 MB -> ~3M tokens).
DEFAULT_MAX_DESC_CHARS = 200


def _max_desc_chars() -> int:
    from ..conf import load
    try:
        return max(1, int(load()["decide"].get("max_criteria_chars",
                                               DEFAULT_MAX_DESC_CHARS)))
    except (KeyError, TypeError, ValueError):
        return DEFAULT_MAX_DESC_CHARS


def _max_candidates() -> int:
    from ..conf import load
    try:
        return max(1, int(load()["decide"].get("max_candidates", DEFAULT_MAX_CANDIDATES)))
    except (KeyError, TypeError, ValueError):
        return DEFAULT_MAX_CANDIDATES


def _rank(items: list[Any], intent: str | None) -> list[Any]:
    """Lexical prefilter: rank candidates by overlap with the intent, keep the head.

    Deliberately cheap and deterministic — it exists so that `max_candidates` can
    be small without hiding the right answer from the model.
    """
    words = {w for w in re.findall(r"[a-z0-9_]+", (intent or "").lower()) if len(w) > 2}
    if not words or len(items) <= 1:
        return items

    def score(pair: tuple[int, Any]) -> tuple[int, int]:
        idx, item = pair
        text = (item if isinstance(item, str) else json.dumps(item, default=str)).lower()
        return (-sum(1 for w in words if w in text), idx)

    return [item for _, item in sorted(enumerate(items), key=score)]


def _describe(candidates: list[Any]) -> dict[str, str]:
    """maps candidate -> {id: desc}. Handles path strings and {title, snippet} dicts.

    Each description is clipped to `decide.max_criteria_chars`. The option set is
    the one part of a decision payload that does not pass through
    `guard_state`, so the clip is what keeps a candidate's snippet from
    deciding the latency (or the survival) of the call.

    Ids are deliberately *not* clipped: an id is the label the model answers
    with, and `run_pick` maps that answer back to the source item, so truncating
    a key would silently drop the pick. They are path/title/url-shaped and
    bounded by whatever produced the artifact.
    """
    cap = _max_desc_chars()
    out: dict[str, str] = {}
    for c in candidates:
        if isinstance(c, dict):
            cid = str(c.get("id") or c.get("url") or c.get("title") or len(out))
            desc = str(c.get("title") or "") + " :: " + str(c.get("snippet") or c.get("desc") or "")
        else:
            cid, desc = str(c), ""
        out[cid] = desc.strip(" :")[:cap]
    return out


@op("decide_extract", kind="value_list", required=["instructions"],
    optional=["source_artifact", "count", "intent_context"])
def run_extract(params: dict[str, Any], store: Any) -> tuple[str, Any, float | None]:
    src = params.get("source_artifact")
    source = store.try_get(src) if src else None
    ctx = source.data if source is not None else {}
    # The extractor mines the payload for literals, and the intent is payload: a
    # step with no source artifact (research's `queries`) otherwise handed it
    # nothing but its own instruction text ("Extract 3-6 search queries ..."),
    # which has no topic in it. `intent_context` is the runner's per-step intent
    # (see core/runner.run_plan), declared optional so it lands as a real arg.
    intent = params.get("intent_context")
    payload: Any = {"intent": intent, "source": ctx} if intent else ctx
    backend = resolve_backend("extract")
    state = guard_state({"context": payload, "instructions": params["instructions"]})
    values = backend.extract(state,
                             {"type": "value_list", "count": int(params.get("count", 4))})
    conf = _measured_conf(values)
    data = values.get("values", values) if isinstance(values, dict) else values
    return "value_list", data, conf


def _measured_conf(values: Any) -> float | None:
    """The conf the extract backend measured, or None when it measured none.

    Absent is the normal case: a rule (or any non-probabilistic) extractor has no
    distribution to report, and the 0.7 this used to fall back to was a number
    nobody computed standing in for a measurement that did not happen.
    """
    if not isinstance(values, dict):
        return None
    raw = values.get("conf")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return float(raw)


@op("decide_pick", kind="picked_items",
    required=["source_artifact", "instructions"], optional=["top_k", "intent_context"])
def run_pick(params: dict[str, Any], store: Any) -> tuple[str, Any, float]:
    art = store.try_get(params["source_artifact"])
    if art is None:
        raise ValueError("decide_pick: source artifact missing")
    candidates = art.data
    raw = candidates if isinstance(candidates, list) else [candidates]
    if not raw:
        # No candidate is not the same answer as "this one": picking would hand
        # the next step a fabricated id ("__none__") that no op can run, and the
        # failure would surface one step later as a nonsense URL or path.
        raise ValueError(f"decide_pick: nothing to pick from — artifact "
                         f"{params['source_artifact']!r} is empty")
    chosen = _rank(raw, params.get("intent_context"))[:_max_candidates()]
    criteria = _describe(chosen)
    if not criteria:
        # Nothing to choose from is not a decision the backend can make: asking it
        # anyway made d1 abstain and the pair surface "pair: no stage produced a
        # choice" for an empty directory (observed live). Name the real cause.
        raise ValueError(f"decide_pick: {params['source_artifact']!r} has no candidates "
                         "to choose from — the previous step found nothing")
    backend = resolve_backend("choice")
    # The candidate list travels once, in `criteria` (the backends render it as the
    # option set / the questions.criteria block). Repeating it in the state doubled
    # the payload for no information: measured 218 -> 125 tokens and 2.6 s -> 1.5 s
    # per decision on the CPU box. Every choice backend — cascade included, it
    # forwards the state it is given — sees a state bounded by
    # conf.context.state_max_tokens.
    state = guard_state({"intent_context": params.get("intent_context"),
                         "source_kind": art.kind})
    answer = backend.choice(state, params["instructions"], criteria)
    k = int(params.get("top_k", 1))
    keys = answer.probabilities and sorted(answer.probabilities, key=answer.probabilities.get, reverse=True)
    picks = (keys[:k] if k > 1 else [answer.choice]) or [answer.choice]
    return "picked_items", picks, answer.confidence
