"""Decision primitives — the interface every backend implements.

Backends never leak their wire format past this module. All state payloads are
plain JSON-safe values (str / dict / image-data-URL).
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Callable

# decision step limits ("state" size guidance lives in context.packer)


@dataclass
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float


class DecideBackend(abc.ABC):
    """One backend implements all primitives for one model/family.

    Registered by name via ti.decide.registry (module-level dict).
    """

    name: str = "?"

    @abc.abstractmethod
    def choice(self, state: Any, instructions: str, criteria: dict[str, str]) -> ChoiceAnswer: ...

    @abc.abstractmethod
    def noul(self, state: Any, question: str) -> float: ...

    def score(self, state: Any, question: str, levels: list[str]) -> float:
        raise NotImplementedError(f"{self.name}: score")

    def extract(self, state: Any, schema: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(f"{self.name}: extract")

    def health(self) -> bool:
        return True


_REGISTRY: dict[str, Callable[[], DecideBackend]] = {}


def _ensure_backends_loaded() -> None:
    if _REGISTRY:
        return
    from .backends import d1_systemone, fixture, heuristic_micro, logit_read  # noqa: F401  (registration)


def register(name: str) -> Callable[[type[DecideBackend]], type[DecideBackend]]:
    def _wrap(cls: type[DecideBackend]) -> type[DecideBackend]:
        _REGISTRY[name] = cls  # type: ignore[assignment]
        return cls

    return _wrap


def make_backend(name: str) -> DecideBackend:
    _ensure_backends_loaded()
    if name not in _REGISTRY:
        raise KeyError(f"unknown decide backend {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()  # type: ignore[return-value]
