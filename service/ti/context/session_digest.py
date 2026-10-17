"""Session digest — consumes the pi extension's rolling transcript digest.

Shape the extension sends per useTool call (~spec §6):

  {turns: [{"role": "user"|"assistant", "summary": str, "mentions": [str]}],
   antecedents: [{"phrase": "the file we discussed", "resolved": str|None}]}

Recent-weighted, budgeted by conf.context.digest_max_tokens.
"""
from __future__ import annotations

from typing import Any

EMPTY: dict[str, Any] = {"turns": [], "antecedents": []}


def sanitize(digest: Any) -> dict[str, Any]:
    if not isinstance(digest, dict):
        return EMPTY
    return {
        "turns": digest.get("turns", [])[-8:],
        "antecedents": digest.get("antecedents", [])[-8:],
    }


def mentioned(keywords: list[str], digest: dict[str, Any]) -> list[str]:
    """Artifact keys/items the digest mentions containing any keyword."""
    hits: list[str] = []
    for ant in digest.get("antecedents", []):
        for kw in keywords:
            if kw and kw.lower() in str(ant.get("phrase", "")).lower():
                hits.append(str(ant))
    return hits
