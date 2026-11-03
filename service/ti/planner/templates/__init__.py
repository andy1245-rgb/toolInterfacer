"""templates — authored L2/L1 pipelines + the apply dispatch.

Templates publish the *shape* of a pipeline; every *degree* (k, caps, gates)
resolves from conf at run time. apply() never hardcodes tuning specifics.

Each template exposes `build(intent, digest, params) -> (steps, final_key,
assumptions)`. The final artifact key is returned by the builder that produced
the steps, never looked up from a table: a plan whose terminal key disagrees with
the steps it built reports success on an artifact the run never wrote.
"""
from __future__ import annotations

from typing import Any, Callable

from ...core.runner import Step

Builder = Callable[[str, dict[str, Any], dict[str, Any] | None],
                   tuple[list[Step], str, list[str]]]


def _disk_lookup(intent: str, digest: dict[str, Any],
                 params: dict[str, Any] | None = None) -> tuple[list[Step], str, list[str]]:
    from .disk_lookup import build
    return build(intent, digest, params)


def _research(intent: str, digest: dict[str, Any],
              params: dict[str, Any] | None = None) -> tuple[list[Step], str, list[str]]:
    from .research import build
    return build(intent, digest, params)


def _reddit_digest(intent: str, digest: dict[str, Any],
                   params: dict[str, Any] | None = None) -> tuple[list[Step], str, list[str]]:
    from .reddit_digest import build
    return build(intent, digest, params)


#: Every authored plan shape, by the name `planner.inspector` resolves to.
TEMPLATES: dict[str, Builder] = {
    "research": _research,
    "reddit_digest": _reddit_digest,
    "disk_lookup": _disk_lookup,
}


def apply_template(name: str, intent: str, digest: dict[str, Any],
                   params: dict[str, Any] | None = None,
                   ) -> tuple[list[Step], str, list[str]]:
    """(steps, final artifact key, assumptions the plan itself establishes)."""
    builder = TEMPLATES.get(name)
    if builder is None:
        raise KeyError(f"unknown template {name!r}; known: {sorted(TEMPLATES)}")
    return builder(intent, digest, params)
