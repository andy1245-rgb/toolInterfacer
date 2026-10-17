"""Packer — assemble the `state` for a decide call, under the token budget.

State payload = intent + params-as-context + session digest slice + candidate
lists, recent-weighted, capped via conf.context.state_max_tokens. Input tokens
are the decision model's latency lever, so the packer trims to budget rather
than shipping everything.
"""
from __future__ import annotations

from typing import Any


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
    # P0: whole-payload approximation (~4 chars/token); P3 swaps in a real counter.
    char_cap = max_tokens * 4
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
