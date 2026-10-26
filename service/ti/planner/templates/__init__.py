"""templates — authored L2/L1 pipelines + the apply dispatch.

Templates publish the *shape* of a pipeline; every *degree* (k, caps, gates)
resolves from conf at run time. apply() never hardcodes tuning specifics.
"""
from __future__ import annotations

from typing import Any

from ...core.runner import Step

# Fixed terminals only. A template whose terminal depends on the intent shape
# (disk_lookup: listing vs read) reports it from its own build(), so the key can
# never disagree with the steps that were built.
FINAL_KEYS: dict[str, str] = {
    "research": "pages",
    "reddit_digest": "threads",
}


def apply_template(name: str, intent: str, digest: dict[str, Any],
                   params: dict[str, Any] | None = None,
                   ) -> tuple[list[Step], str, list[str]]:
    """(steps, final artifact key, assumptions the plan itself establishes)."""
    if name == "disk_lookup":
        from .disk_lookup import build
        return build(intent, digest, params)
    final_key = FINAL_KEYS.get(name)
    if final_key is None:
        raise KeyError(f"unknown template {name!r}")
    if name == "research":
        from .research import apply
    elif name == "reddit_digest":
        from .reddit_digest import apply
    else:  # pragma: no cover — FINAL_KEYS gate keeps this unreachable
        raise KeyError(name)
    return apply(intent, digest), final_key, []
