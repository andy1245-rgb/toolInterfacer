"""predictors — the registered implementations.

Importing this package registers every built-in. The optional ones (sklearn, torch) import
their dependency *inside the factory*, so this package imports cleanly on a box with neither,
and a missing dependency is an actionable error at build time instead of an ImportError at
import time.

If a module is missing outright, its kind is still registered — as a factory that raises with
the import error as its message. A kind that silently vanishes from the registry is how a
"predictor does not exist" mystery starts.
"""
from __future__ import annotations

from ..protocol import PredictorError, register
from . import constant, fake, fusion, llama, logit_read, recorded  # noqa: F401

OPTIONAL = ("sklearn_text", "torch_text", "ensemble")
UNAVAILABLE: dict[str, str] = {}


def _broken(kind: str, reason: str):
    def factory(**_kw):
        raise PredictorError(f"{kind}: this predictor module is unavailable here ({reason})")
    return factory


for _name in OPTIONAL:
    try:
        __import__(f"{__name__}.{_name}")
    except Exception as _exc:  # noqa: BLE001 - recorded and re-raised at build time
        UNAVAILABLE[_name] = f"{type(_exc).__name__}: {_exc}"
        register(_name)(_broken(_name, UNAVAILABLE[_name]))
