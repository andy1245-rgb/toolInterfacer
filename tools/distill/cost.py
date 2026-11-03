"""cost — what a labelling run cost, measured when the reply says so, else estimated.

Token cost is the reason the gate in ``label.py`` exists, so the number has to be
honest about its own provenance. The hosted backend asks OpenRouter to include the
charge in the reply (``usage.cost``, via the request's ``usage: {include: true}``),
and when it arrives it is the billed amount. When it does not, the fallback is a
token estimate at a declared price, and below that a character estimate — and the
per-row ``label_method`` records which one a row used, so a projection can never be
reported as if it were measured.

The fallback prices are an ASSUMPTION about a cheap flash-tier model, not a quote.
They are here so an estimate exists at all; if the measured cost is available it is
always preferred, and every report says which of the two it printed.
"""
from __future__ import annotations

from typing import Any

# Assumed list price, USD per 1k tokens. Deliberately on the high side: an optimistic
# fallback would under-project a run and defeat the gate it feeds.
FALLBACK_IN_PER_1K = 0.000_30
FALLBACK_OUT_PER_1K = 0.001_20
CHARS_PER_TOKEN = 4.0

MEASURED = "measured"
EST_TOKENS = "estimated_tokens"
EST_CHARS = "estimated_chars"
FIXTURE = "fixture"


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None  # NaN is not a number here


def row_cost(usage: dict[str, Any] | None, prompt_chars: int,
             reply_chars: int) -> tuple[float, str]:
    """(usd, method) for one call."""
    usage = usage or {}
    direct = _num(usage.get("cost"))
    if direct is not None:
        return direct, MEASURED
    prompt_tokens = _num(usage.get("prompt_tokens"))
    reply_tokens = _num(usage.get("completion_tokens"))
    if prompt_tokens is not None and reply_tokens is not None:
        usd = (prompt_tokens / 1000.0 * FALLBACK_IN_PER_1K
               + reply_tokens / 1000.0 * FALLBACK_OUT_PER_1K)
        return usd, EST_TOKENS
    world_tokens = (prompt_chars + reply_chars) / CHARS_PER_TOKEN
    reply_guess = max(reply_chars / CHARS_PER_TOKEN, 1.0)
    usd = ((world_tokens - reply_guess) / 1000.0 * FALLBACK_IN_PER_1K
           + reply_guess / 1000.0 * FALLBACK_OUT_PER_1K)
    return usd, EST_CHARS


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Totals over labelled rows, split by how each number was obtained."""
    total = 0.0
    methods: dict[str, int] = {}
    usd_by_method: dict[str, float] = {}
    for row in rows:
        cost = _num(row.get("cost_usd"))
        if cost is None:
            continue
        method = str(row.get("label_method") or "unknown")
        methods[method] = methods.get(method, 0) + 1
        usd_by_method[method] = usd_by_method.get(method, 0.0) + cost
        total += cost
    return {"rows": len(rows), "usd": round(total, 6),
            "method": (next(iter(methods)) if len(methods) == 1
                       else ("mixed" if methods else "none")),
            "by_method": {k: {"rows": methods[k], "usd": round(usd_by_method[k], 6)}
                          for k in sorted(methods)},
            "per_row": round(total / max(len(rows), 1), 8)}


def project(labelled: int, spent: float, remaining: int) -> dict[str, Any]:
    """Extrapolate one full run from the rows measured so far."""
    per_row = spent / labelled if labelled else 0.0
    return {"labelled": labelled, "spent_usd": round(spent, 6),
            "per_row_usd": round(per_row, 8), "remaining": remaining,
            "projected_total_usd": round(spent + per_row * remaining, 6),
            "projected_rest_usd": round(per_row * remaining, 6)}


def price_note() -> str:
    return (f"measured from usage.cost when present; otherwise {FALLBACK_IN_PER_1K}/1k in, "
            f"{FALLBACK_OUT_PER_1K}/1k out (assumption), otherwise "
            f"{CHARS_PER_TOKEN:.0f} chars/token")
