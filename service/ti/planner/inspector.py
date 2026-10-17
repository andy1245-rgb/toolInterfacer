"""Inspector — classify the intent, resolve level + template.

v1 classification (meta decision, one `choice`): passthrough (L0) /
one_hop slot-fill (L1) / research (L2) / reddit_digest (L2) / disk_lookup (L2).
P1 implements rule-first classification + a `decide` fallback call site;
P2 replaces the rule layer with a routed d1 choice.
"""
from __future__ import annotations

from typing import Any


def inspect(intent: str, params: dict[str, Any], digest: dict[str, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {"intent": intent, "level": None, "template": None}
    if "research" in intent.lower() or "compare" in intent.lower():
        meta["level"], meta["template"] = "L2", "research"
    elif "reddit" in intent.lower() or "r/" in intent:
        meta["level"], meta["template"] = "L2", "reddit_digest"
    elif params.get("tool"):
        meta["level"], meta["template"] = "L0", None
    else:
        meta["level"], meta["template"] = "L1", "disk_lookup"
    return meta
