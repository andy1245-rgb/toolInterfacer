"""Decision steps surfaced as ops so templates speak one step vocabulary.

decide_extract — fill a list of literal values (queries, candidate keys) from
the intent + context. Uses the backend's extract primitive (fixture replayable
via recorded map; d1 extract lands in P2/P4).

decide_pick — choose top_k items from a source artifact. Uses `choice` with
criteria built from the candidates (id + description lines preferred).
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
    """maps candidate -> {id: desc}. Handles path strings and {title, snippet} dicts."""
    out: dict[str, str] = {}
    for c in candidates:
        if isinstance(c, dict):
            cid = str(c.get("id") or c.get("url") or c.get("title") or len(out))
            desc = str(c.get("title") or "") + " :: " + str(c.get("snippet") or c.get("desc") or "")
        else:
            cid, desc = str(c), ""
        out[cid] = desc.strip(" :")
    return out


@op("decide_extract", kind="value_list", required=["instructions"], optional=["source_artifact", "count"])
def run_extract(params: dict[str, Any], store: Any) -> tuple[str, Any, float]:
    src = params.get("source_artifact")
    ctx = store.try_get(src).data if (src and store.try_get(src)) else {}
    backend = resolve_backend("extract")
    state = guard_state({"context": ctx, "instructions": params["instructions"]})
    values = backend.extract(state,
                             {"type": "value_list", "count": int(params.get("count", 4))})
    conf = float(values.get("conf", 0.7)) if isinstance(values, dict) else 0.7
    data = values.get("values", values) if isinstance(values, dict) else values
    return "value_list", data, conf


@op("decide_pick", kind="picked_items",
    required=["source_artifact", "instructions"], optional=["top_k", "intent_context"])
def run_pick(params: dict[str, Any], store: Any) -> tuple[str, Any, float]:
    art = store.try_get(params["source_artifact"])
    if art is None:
        raise ValueError("decide_pick: source artifact missing")
    candidates = art.data
    raw = candidates if isinstance(candidates, list) else [candidates]
    chosen = _rank(raw, params.get("intent_context"))[:_max_candidates()]
    criteria = _describe(chosen)
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
