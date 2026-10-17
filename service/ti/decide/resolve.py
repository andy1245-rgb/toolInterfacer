from __future__ import annotations

from typing import Any, Callable

from .primitives import DecideBackend, make_backend


def resolve(kind: str) -> DecideBackend:
    """Pick the backend serving a decision kind, per conf.decide.backends_in_use."""
    mapping = load_backends_in_use()
    name = mapping.get(kind) or mapping.get("default")
    if not name:
        raise KeyError(f"no decide backend configured for kind {kind!r}")
    return make_backend(name)


def load_backends_in_use() -> dict[str, str]:
    from ..conf import load
    return dict(load()["decide"]["backends_in_use"])


def decide_factory() -> Callable[[str], DecideBackend]:
    """The runner-facing factory: kind name → backend."""
    return resolve


def as_state(x: Any) -> str:
    return x if isinstance(x, str) else json_str(x)


def json_str(x: Any) -> str:
    import json
    return json.dumps(x, default=str)
