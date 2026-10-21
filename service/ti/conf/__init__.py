"""conf — every tuning knob in one place.

P0 ships a skeleton defaults.json; every specific (top-k, gates, caps,
backend map, budgets) resolves through load() at run time so template/op
files stay free of magic numbers.

Two environment hooks, both optional:

``TI_CONF``          a whole alternate config file (what a dev bench or a test
                    points at defaults.dev.json-style files)
``TI_CONF_OVERLAY``  a partial config deep-merged over whatever ``TI_CONF``
                    selected — one section is enough to repoint ``ops.*`` at a
                    replay wrapper or a fake CLI without copying the file

Lists replace rather than concatenate, so an overlay cannot silently grow a
list it meant to pin.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_DEFAULTS = Path(__file__).resolve().parent / "defaults.json"
_cache: dict[str, Any] | None = None


def load() -> dict[str, Any]:
    global _cache
    if _cache is None:
        path = os.environ.get("TI_CONF", str(_DEFAULTS))
        with open(path, "rb") as fh:
            conf = json.load(fh)
        overlay = os.environ.get("TI_CONF_OVERLAY")
        if overlay:
            with open(overlay, "rb") as fh:
                conf = _merge(conf, json.load(fh))
        _cache = conf
    return _cache  # type: ignore[return-value]


def reload() -> dict[str, Any]:
    global _cache
    _cache = None
    return load()


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge `over` onto `base` (dicts recurse, everything else replaces)."""
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out
