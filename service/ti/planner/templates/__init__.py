"""templates — authored L2/L1 pipelines + the apply dispatch.

Templates publish the *shape* of a pipeline; every *degree* (k, caps, gates)
resolves from conf at run time. apply() never hardcodes tuning specifics.
"""
from __future__ import annotations

from typing import Any

from ...core.runner import Step

FINAL_KEYS: dict[str, str] = {
    "research": "pages",
    "reddit_digest": "threads",
    "disk_lookup": "content",
}


def apply_template(name: str, intent: str, digest: dict[str, Any]) -> tuple[list[Step], str]:
    final_key = FINAL_KEYS.get(name)
    if final_key is None:
        raise KeyError(f"unknown template {name!r}")
    if name == "research":
        from .research import apply
    elif name == "reddit_digest":
        from .reddit_digest import apply
    elif name == "disk_lookup":
        from .disk_lookup import apply
    else:  # pragma: no cover — FINAL_KEYS gate keeps this unreachable
        raise KeyError(name)
    _ = final_key  # per-template final step id; kept for the run_plan contract
    return apply(intent, digest), final_key
