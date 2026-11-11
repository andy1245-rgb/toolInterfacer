"""classical — the shipped local tier, measured as the front door it is supposed to be.

This predictor does not fit anything of its own: it builds `service/ti/decide/backends/classical.py`
from **the shipped config** and asks it, so a run measures the deployed tier and not a copy of it.
That is the point. The tier was dead for its whole life — pointed at a corpus whose labels shared
no names with the eight options it is offered — and no benchmark could see it, because the only
number anyone kept was accuracy on rows the expensive model had already answered.

Two consequences a reader should know:

* **An abstention is recorded as an abstention, not as a wrong answer.** `choice=""` with
  `meta["abstained"]`, and the run reports how many rows it declined. A tier that answers nothing
  reads as `answered 0/207` in the summary, which is loud; it used to read as a normal accuracy.
* **Probabilities are the backend's own, and its temperature is the shipped one.** So a tier can
  answer correctly and still be useless as a front door if its confidences sit below the gate —
  which is visible in `coverage@precision=0.90`, and was not visible in anything before.

Spec: ``classical`` (shipped config), or ``classical|corpus_path=...|temperature=0.17`` to score
a variant without touching the config.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from ..protocol import Prediction, PredictorError, register
from ..rows import Row

ROOT = Path(__file__).resolve().parents[3]
SERVICE = ROOT / "service"
CONFIG_KEYS = ("corpus_path", "temperature", "loss", "c", "epochs", "eta", "ngram_max", "min_df",
               "sublinear_tf", "seed", "abstain_label")


def _imports():
    """The service package, lazily: the harness has to import on a box without it."""
    if str(SERVICE) not in sys.path:
        sys.path.insert(0, str(SERVICE))
    try:
        from ti.conf import load  # noqa: PLC0415
        from ti.decide.backends.classical import ClassicalBackend  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 - the message is the value here
        raise PredictorError(
            f"classical: the service package did not import from {SERVICE} "
            f"({type(exc).__name__}: {exc})") from None
    return load, ClassicalBackend


class ClassicalPredictor:
    """The shipped classical backend, read through its own config."""

    def __init__(self, **overrides: str) -> None:
        load, ClassicalBackend = _imports()
        self.overrides = {k: v for k, v in overrides.items() if v != ""}
        unknown = sorted(set(self.overrides) - set(CONFIG_KEYS))
        if unknown:
            raise PredictorError(
                f"classical: unknown key(s) {unknown}; known: {', '.join(CONFIG_KEYS)}")
        cfg = dict(load()["decide"]["classical"])
        for key, raw in self.overrides.items():
            cfg[key] = raw if isinstance(cfg.get(key), str) else _coerce(cfg.get(key), raw)
        self.cfg = cfg
        self.backend = ClassicalBackend(config=cfg)
        self.name = f"classical:{Path(str(cfg['corpus_path'])).name}"

    def fitted_labels(self) -> tuple[str, ...]:
        """The label set the model can actually name — empty means it can name nothing."""
        return tuple(sorted(self.backend.model().model.classes))

    def predict(self, text: str, options: tuple[str, ...]) -> Prediction:
        t0 = time.perf_counter()
        answer = self.backend.choice(text, "", {o: o for o in options})
        latency = (time.perf_counter() - t0) * 1000.0
        choice = getattr(answer, "choice", None) or ""
        probabilities = dict(getattr(answer, "probabilities", None) or {}) or None
        return Prediction(choice=choice, probabilities=probabilities, latency_ms=latency,
                          meta={"abstained": not choice})


def _coerce(current, raw: str):
    if isinstance(current, bool):
        return str(raw).lower() in ("1", "true", "yes")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


@register("classical")
def build(**kw) -> ClassicalPredictor:
    return ClassicalPredictor(**{k: str(v) for k, v in kw.items()})


def fitted_label_overlap(predictor: ClassicalPredictor,
                         options: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(fitted labels that are also offered, fitted labels nobody offers) — the rot detector."""
    fitted = predictor.fitted_labels()
    return (tuple(lab for lab in fitted if lab in options),
            tuple(lab for lab in fitted if lab not in options))
