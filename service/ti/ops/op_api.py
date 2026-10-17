"""Op registration + step-param schema validation.

An op is a function `run(params, store) -> (kind, data, conf)`.
Templates reference ops by name; the runner resolves names through
`registry.get()`. Every op declares a param schema the runner validates before
calling — LLM-injected params that match the schema land as real args, the
rest ride along as soft context.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

RunFn = Callable[..., tuple[str, Any, float]]


@dataclass
class OpDef:
    name: str
    kind: str                      # artifact kind it emits
    required: list[str]            # required params
    optional: list[str]
    run: RunFn


_REGISTRY: dict[str, OpDef] = {}


def op(name: str, kind: str, required: list[str] | None = None,
       optional: list[str] | None = None) -> Callable[[RunFn], RunFn]:
    def _wrap(fn: RunFn) -> RunFn:
        _REGISTRY[name] = OpDef(name=name, kind=kind,
                                required=required or [], optional=optional or [], run=fn)
        return fn

    return _wrap


def get(name: str) -> OpDef:
    if name not in _REGISTRY:
        raise KeyError(f"unknown op {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def all_ops() -> list[str]:
    return sorted(_REGISTRY)


def validate_params(op_name: str, params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split LLM params into (matching args, soft context)."""
    d = get(op_name)
    args = {k: params[k] for k in d.required if k in params}
    missing = [k for k in d.required if k not in args]
    if missing:
        raise ValueError(f"missing required params for {op_name!r}: {missing}")
    soft = {k: v for k, v in params.items() if k not in d.required + d.optional}
    for k in d.optional:
        if k in params:
            args[k] = params[k]
    return args, soft


def import_all() -> None:
    """Import every op module so registrations fire (called by http.api boot)."""
    from . import decisions  # noqa: F401  (registration)
    from .web import fetch as _wf, search as _ws  # noqa: F401
    from .reddit import digest as _rd  # noqa: F401
    from .filesys import impl as _fi  # noqa: F401
