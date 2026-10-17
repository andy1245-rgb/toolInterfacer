"""conf — every tuning knob in one place.

P0 ships a skeleton defaults.json; every specific (top-k, gates, caps,
backend map, budgets) resolves through load() at run time so template/op
files stay free of magic numbers.
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
            _cache = json.load(fh)
    return _cache  # type: ignore[return-value]


def reload() -> dict[str, Any]:
    global _cache
    _cache = None
    return load()
