"""Session digest — consumes the pi extension's rolling transcript digest.

Shape the extension sends per useTool call (~spec §6):

  {turns: [{"role": "user"|"assistant", "summary": str, "mentions": [str]}],
   antecedents: [{"phrase": "the file we discussed", "resolved": str|None}],
   cwd?: str}

Recent-weighted, budgeted by conf.context.digest_max_tokens. sanitize() is the
service-side bound: nothing downstream of ingress sees a wider digest than the
8 x 240 window, whatever the caller sent.
"""
from __future__ import annotations

from typing import Any

EMPTY: dict[str, Any] = {"turns": [], "antecedents": []}

# Recent-weighted window + per-turn cap. The extension applies both at its end
# (session_push.ts: last 8 turns, 240 chars each); the service re-applies them
# because ingress cannot assume the caller is the extension.
TURN_CAP = 8
TURN_CHARS = 240


def sanitize(digest: Any) -> dict[str, Any]:
    """Cap a caller-supplied digest to the shape the service budgets for.

    For an extension-shaped digest this is the identity: <= 8 turns x 240 chars,
    <= 8 antecedents, `cwd` carried through (the L1 disk-slot-fill bridge reads
    it, and every path comes out of it). Anything else survives only as far as
    those bounds allow — extra keys are dropped rather than forwarded.
    """
    if not isinstance(digest, dict):
        return dict(EMPTY)
    out = {"turns": _window(digest.get("turns")),
           "antecedents": _window(digest.get("antecedents"))}
    if "cwd" in digest:
        out["cwd"] = digest["cwd"]
    return out


def _window(items: Any) -> list[Any]:
    """Last TURN_CAP entries, each clipped to TURN_CHARS of summary text."""
    if not isinstance(items, list):
        return []
    return [_clip_turn(item) for item in items[-TURN_CAP:]]


def _clip_turn(turn: Any) -> Any:
    if not isinstance(turn, dict):
        return turn
    out = dict(turn)
    summary = out.get("summary")
    if isinstance(summary, str) and len(summary) > TURN_CHARS:
        out["summary"] = summary[:TURN_CHARS]
    return out


def mentioned(keywords: list[str], digest: dict[str, Any]) -> list[str]:
    """Artifact keys/items the digest mentions containing any keyword."""
    hits: list[str] = []
    for ant in digest.get("antecedents", []):
        for kw in keywords:
            if kw and kw.lower() in str(ant.get("phrase", "")).lower():
                hits.append(str(ant))
    return hits
