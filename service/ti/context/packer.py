"""Packer — assemble the `state` for a decide call, under the token budget.

State payload = intent + params-as-context + session digest slice + candidate
lists, recent-weighted, capped via conf.context.state_max_tokens. Input tokens
are the decision model's latency lever, so the packer trims to budget rather
than shipping everything.
"""
from __future__ import annotations

from typing import Any

# guard_state trim degrees. Every one of them is a clip, never a removal: the
# state keeps its outer keys and their types, so a backend sees a smaller
# payload of the same shape rather than a different one.
DESC_CHARS = 200      # a single candidate description line
CANDIDATE_CAP = 16    # candidate entries that survive the drop pass
STRING_CHARS = 1000   # any other string value left in the state
STUB_DIGEST: dict[str, Any] = {"summary": "trim: over budget"}
_CHAR_PER_TOKEN = 4   # whole-payload approximation; P3 swaps in a real counter


def pack(intent: str, params: dict[str, Any], digest: dict[str, Any],
         candidates: dict[str, Any], max_tokens: int | None = None) -> dict[str, Any]:
    from ..conf import load
    cap = max_tokens or int(load()["context"]["state_max_tokens"])
    state = {
        "intent": intent,
        "params_context": {k: v for k, v in params.items()},
        "digest": digest,
        "candidates": candidates,
    }
    return trim(state, cap)


def trim(state: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    char_cap = max_tokens * _CHAR_PER_TOKEN
    current = len(str(state))
    if current <= char_cap:
        return state
    state = dict(state)
    # 1) collapse the session digest to a stub
    state["digest"] = {"summary": "trim: over budget"}
    current = len(str(state))
    if current <= char_cap:
        return state
    # 2) drop largest params_context values until we fit (context hints are the
    #    only elastic part; intent + candidates must survive)
    ctx = dict(state.get("params_context") or {})
    for k in sorted(ctx, key=lambda key: -len(str(ctx.get(key, "")))):
        if current <= char_cap:
            break
        current -= len(str(ctx.pop(k))) - len(k)  # approx chars freed
        ctx[k] = "(trimmed)"
    state["params_context"] = ctx
    return state


def guard_state(state: Any, max_tokens: int | None = None) -> Any:
    """Bound a decide `state` to conf.context.state_max_tokens, shape-preserving.

    A no-op for anything already inside the budget — the same object comes back
    with the same contents, so the prompt the decision model was validated on is
    the prompt it receives. Over budget it trims in a fixed order, and every
    step is a clip or a drop of surplus *entries*: no outer key disappears and
    no key changes type.

      a) long candidate descriptions   -> clipped to DESC_CHARS
      b) surplus candidate entries     -> first CANDIDATE_CAP kept
      c) the session digest            -> STUB_DIGEST
      d) every remaining string value  -> clipped to STRING_CHARS, then smaller

    The budget is characters, not tokens (len(str(state)) <= max_tokens * 4),
    the same approximation trim() uses. The last pass re-clips at tighter limits
    until the state fits, so a single oversized value cannot offload its budget
    onto the rest of the payload. The key skeleton is the floor: a cap smaller
    than the keys themselves cannot be met without dropping keys, which this
    never does.
    """
    from ..conf import load
    cap = int(max_tokens or load()["context"]["state_max_tokens"])
    char_cap = cap * _CHAR_PER_TOKEN
    if len(str(state)) <= char_cap:
        return state
    if not isinstance(state, dict):
        return _clip(state, char_cap)  # str state (backends accept one) -> bounded as-is
    out = dict(state)
    for step in (_clip_descriptions, _drop_extra_candidates, _stub_digest):
        out = step(out)
        if len(str(out)) <= char_cap:
            return out
    for limit in (STRING_CHARS, DESC_CHARS, 40):
        out = _clip_strings(out, limit)
        if len(str(out)) <= char_cap:
            break
    return out


def _clip(value: Any, limit: int) -> Any:
    if not isinstance(value, str) or len(value) <= limit:
        return value
    return value[:limit] + "..."


def _clip_strings(value: Any, limit: int) -> Any:
    """Recursive string clip: containers keep their type, keys and order."""
    if isinstance(value, str):
        return _clip(value, limit)
    if isinstance(value, dict):
        return {k: _clip_strings(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [_clip_strings(v, limit) for v in value]
    return value


def _clip_descriptions(state: dict[str, Any]) -> dict[str, Any]:
    cands = state.get("candidates")
    if cands is None:
        return state
    return {**state, "candidates": _clip_strings(cands, DESC_CHARS)}


def _drop_extra_candidates(state: dict[str, Any]) -> dict[str, Any]:
    cands = state.get("candidates")
    if isinstance(cands, dict) and len(cands) > CANDIDATE_CAP:
        return {**state,
                "candidates": dict(list(cands.items())[:CANDIDATE_CAP])}
    if isinstance(cands, list) and len(cands) > CANDIDATE_CAP:
        return {**state, "candidates": cands[:CANDIDATE_CAP]}
    return state


def _stub_digest(state: dict[str, Any]) -> dict[str, Any]:
    if "digest" not in state:
        return state
    return {**state, "digest": dict(STUB_DIGEST)}
