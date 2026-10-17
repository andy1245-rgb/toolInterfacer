"""Decision steps surfaced as ops so templates speak one step vocabulary.

decide_extract — fill a list of literal values (queries, candidate keys) from
the intent + context. Uses the backend's extract primitive (fixture replayable
via recorded map; d1 extract lands in P2/P4).

decide_pick — choose top_k items from a source artifact. Uses `choice` with
criteria built from the candidates (id + description lines preferred).
"""
from __future__ import annotations

from typing import Any

from ..decide.resolve import resolve as resolve_backend
from .op_api import op


def _describe(candidates: list[Any]) -> dict[str, str]:
    """maps candidate -> {id: desc}. Handles path strings and {title, snippet} dicts."""
    out: dict[str, str] = {}
    for c in candidates[:64]:
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
    values = backend.extract({"context": ctx, "instructions": params["instructions"]},
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
    criteria = _describe(candidates if isinstance(candidates, list) else [candidates])
    backend = resolve_backend("choice")
    answer = backend.choice(
        {"intent_context": params.get("intent_context"), "candidates": criteria,
         "source_kind": art.kind},
        params["instructions"], criteria)
    k = int(params.get("top_k", 1))
    keys = answer.probabilities and sorted(answer.probabilities, key=answer.probabilities.get, reverse=True)
    picks = (keys[:k] if k > 1 else [answer.choice]) or [answer.choice]
    return "picked_items", picks, answer.confidence
