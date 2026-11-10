"""protocol — the predictor seam.

A predictor sees exactly what a router sees at run time: the input text and the closed
option set, in a fixed order. It returns a choice, optional probabilities, and the latency
it actually took. Nothing else crosses the seam, which is why a scikit-learn pipeline, a
fine-tuned encoder, and a llama-server are the same kind of arm.

Spec strings keep the CLI free of imports: ``kind|key=value|key=value``. Example::

    recorded|rows=bench/results/p3-transfer/rows.jsonl|arm=floor_a
    sklearn_text|vectorizer=tfidf_word_char|clf=linsvc|calibrate=sigmoid

Get the order right: ``options`` is passed through untouched and the harness records it,
plus a sha256 of it, in every artifact.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .rows import Row, split_spec

__all__ = ["Prediction", "Predictor", "register", "build", "available", "option_order_sha"]


class PredictorError(RuntimeError):
    """A predictor that cannot answer, with a message the caller can act on."""


@dataclass
class Prediction:
    choice: str
    probabilities: dict[str, float] | None
    latency_ms: float


@runtime_checkable
class Predictor(Protocol):
    name: str

    def predict(self, text: str, options: tuple[str, ...]) -> Prediction:
        ...


@dataclass
class FitPredictorMixin:
    """Optional capability: a predictor that can be fitted in-process.

    Grouped cross-validation needs to build a *fresh* unfitted one per fold, which is what
    the ``factory`` half of the registry entry provides.
    """

    def fit(self, rows: list[Row]) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


_REGISTRY: dict[str, object] = {}


def register(kind: str):
    """Register a predictor factory: ``fn(**kwargs) -> Predictor``."""
    def deco(fn):
        _REGISTRY[kind] = fn
        return fn
    return deco


def available() -> list[str]:
    return sorted(_REGISTRY)


def build(spec: str) -> Predictor:
    kind, kw = split_spec(spec)
    if not _REGISTRY:
        # importing the package is what registers the built-ins; do it on first use so a
        # caller that only needs the metrics does not pay for it
        from . import predictors  # noqa: F401,PLC0415
    if kind not in _REGISTRY:
        raise PredictorError(f"unknown predictor {kind!r}; known: {', '.join(available())}")
    predictor = _REGISTRY[kind](**kw)
    if not getattr(predictor, "name", ""):
        predictor.name = kind
    return predictor


def option_order_sha(options: tuple[str, ...]) -> str:
    """A stable fingerprint of the option set *and its order*.

    This is the field that makes an ordering experiment auditable: the eight descriptions
    reversed hash differently from the eight in their frozen order, with no text change.
    """
    return hashlib.sha256("\x1f".join(options).encode("utf-8")).hexdigest()[:16]


@dataclass
class Record:
    """One scored row. Everything a metric needs, and the audit trail around it."""
    arm: str
    id: str
    group: str
    label: str
    text: str
    choice: str
    correct: bool
    latency_ms: float
    probabilities: dict[str, float] | None = None
    order_sha: str = ""
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"arm": self.arm, "id": self.id, "group": self.group, "label": self.label,
                "text": self.text, "choice": self.choice, "correct": self.correct,
                "latency_ms": round(self.latency_ms, 3),
                "probabilities": self.probabilities, "order_sha": self.order_sha,
                "meta": self.meta}


def score(arm: str, rows: list[Row], predictor: Predictor,
          options_for=None) -> list[Record]:
    """Ask ``predictor`` every row once and return records.

    ``options_for(row)`` lets a control permute the order per run (or per row, which is
    never what you want) without touching the predictor. A predictor that raises is a
    recorded failure, not a dropped row: a dropped row reads as a miss and inflates every
    count below it.
    """
    out: list[Record] = []
    for row in rows:
        options = options_for(row) if options_for else row.options
        t0 = time.perf_counter()
        try:
            # replay needs row identity, a live predictor needs only the text
            if hasattr(predictor, "predict_for"):
                pred = predictor.predict_for(row)
            else:
                pred = predictor.predict(row.text, options)
            latency = pred.latency_ms if pred.latency_ms else (time.perf_counter() - t0) * 1000
            choice, probs = pred.choice, pred.probabilities
            meta = dict(row.meta)
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            latency = (time.perf_counter() - t0) * 1000
            choice, probs = "", None
            meta = dict(row.meta, error=f"{type(exc).__name__}: {exc}")
        if choice not in options:
            meta.setdefault("invalid_choice", choice)
        out.append(Record(arm=arm, id=row.id, group=row.group, label=row.label,
                          text=row.text, choice=choice, correct=(choice == row.label),
                          latency_ms=latency, probabilities=probs,
                          order_sha=option_order_sha(tuple(options)), meta=meta))
    return out
