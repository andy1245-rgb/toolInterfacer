"""classical — TF-IDF + linear SVM behind the `choice` primitive.

The baseline the sibling router benchmark recorded at 93.19% accuracy / 0.60
none-detection / 0.61 ms p50 over 382 utterances and 18 classes
(`experiments/exp_classic.py`, model `tfidf-word+linearsvc`).
scikit-learn is not installed in this repo and it declares no third-party
dependencies, so the vectoriser and the solver are the stdlib
re-implementation in `..linear_text` rather than the original objects.

**Probabilities.** `LinearSVC` has no `predict_proba`; its decision values are
uncalibrated margins. The distribution returned here is the softmax of those
margins over the alternatives actually on offer — the caller's candidates plus
the fitted abstain class — with `decide.classical.temperature` (default 1.0, i.e.
the raw softmax, no calibration set was used to fit it). `confidence` is the
peak probability `max(probabilities)`, because that is the axis
`policy.gate_route` is expressed in (`policy/confidence.py`): a backend that
reports an entropy-derived number instead silently moves the threshold.

**The abstain class.** The model is fitted with `__none__` as a real class, the
way the sibling benchmark fitted it, so abstention is a learned decision and not
a threshold bolted on afterwards. The abstain class competes in every call even
though callers never list it in `criteria`; when it wins, `choice` is None and
`probabilities` carries the `__none__` mass, which is the same "no answer" the
cascade and `resolve.FastPathBackend` already understand (`answer.choice is
None`). None-detection therefore means: of the utterances that really are
off-scaffold, how many did the model route to `__none__`.

`extract`, `score` and `noul` are not served — a linear text classifier answers
one question (which label fits this text), and `rule_extract` is the backend
that owns `noul` in the shipped config.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...conf import load
from ..linear_text import TextClassifier, fit, softmax
from ..primitives import ChoiceAnswer, DecideBackend, register

ROOT = Path(__file__).resolve().parents[4]
_CACHE: dict[tuple, TextClassifier] = {}


@register("classical")
class ClassicalBackend(DecideBackend):
    name = "classical"

    def __init__(self, config: dict[str, Any] | None = None,
                 corpus: list[tuple[str, str]] | None = None) -> None:
        """`config`/`corpus` exist for tests and bench runs that pin a corpus;
        make_backend() constructs with neither and reads decide.classical."""
        self.cfg = dict(config if config is not None else load()["decide"]["classical"])
        self._corpus = corpus

    # -- corpus -----------------------------------------------------------

    def training_corpus(self) -> list[tuple[str, str]]:
        """(text, label) rows, resolved from config — never baked-in weights."""
        if self._corpus is not None:
            return list(self._corpus)
        path = resolve_path(str(self.cfg["corpus_path"]))
        return load_corpus(path)

    def model(self) -> TextClassifier:
        corpus = self.training_corpus()
        key = (tuple(sorted(self.cfg.items())), tuple(corpus))
        cached = _CACHE.get(key)
        if cached is None:
            texts = [t for t, _ in corpus]
            labels = [label for _, label in corpus]
            cached = _CACHE[key] = fit(
                texts, labels,
                loss=str(self.cfg.get("loss", "squared_hinge")),
                c=float(self.cfg.get("c", 1.0)),
                epochs=int(self.cfg.get("epochs", 100)),
                eta=float(self.cfg.get("eta", 0.3)),
                seed=int(self.cfg.get("seed", 42)),
                ngram_max=int(self.cfg.get("ngram_max", 2)),
                min_df=int(self.cfg.get("min_df", 1)),
                sublinear_tf=bool(self.cfg.get("sublinear_tf", True)),
                fingerprint_extra={"corpus_rows": len(corpus),
                                   "corpus_labels": len(set(labels))},
            )
        return cached

    # -- primitive --------------------------------------------------------

    def choice(self, state: Any, instructions: str,
               criteria: dict[str, str]) -> ChoiceAnswer:
        text = state if isinstance(state, str) else json.dumps(state, default=str)
        candidates = list(criteria)
        if not candidates:
            # nothing on offer: an empty distribution is the honest answer
            return ChoiceAnswer(None, {}, 0.0)

        model = self.model()
        abstain = str(self.cfg.get("abstain_label", "__none__"))
        temperature = float(self.cfg.get("temperature", 1.0))
        margins = model.scores(text)

        fitted = [label for label in candidates if label in margins]
        if not fitted:
            # the model was never shown any of these labels; picking one would
            # be a coin flip dressed up as a distribution
            uniform = 1.0 / len(candidates)
            return ChoiceAnswer(None, {c: uniform for c in candidates}, uniform)

        # One floor below every fitted margin for a candidate the model was not
        # fitted on (it keeps the distribution normalized over the real offer
        # set without pretending a score exists), and the abstain class is
        # always on the ballot even though the caller never names it.
        floor = min(margins.values()) - 5.0
        ballot = set(candidates)
        if abstain in margins:
            ballot.add(abstain)
        ranked = sorted(ballot)
        probs = softmax([margins.get(label, floor) for label in ranked], temperature)
        distribution = dict(zip(ranked, probs))

        top = max(range(len(ranked)), key=lambda i: probs[i])
        confidence = probs[top]
        if ranked[top] == abstain:
            return ChoiceAnswer(None, distribution, confidence)
        return ChoiceAnswer(ranked[top], distribution, confidence)

    def noul(self, state: Any, question: str) -> float:
        raise NotImplementedError(
            "classical: choice-only backend; it serves no noul decisions. "
            "Point conf.decide.backends_in_use.noul at a noul backend.")

    def score(self, state: Any, question: str, levels: list[str]) -> float:
        raise NotImplementedError("classical: choice-only backend, not a scorer")


def resolve_path(raw: str) -> Path:
    """corpus_path: absolute as given, `~`-relative, or repo-relative."""
    path = Path(raw).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_corpus(path: Path) -> list[tuple[str, str]]:
    """Read training rows from either corpus shape.

    * a tool registry — `tools[].id` with `references`, plus `none_route`
      (that is `bench/registry_382.json`, the corpus the 382-row benchmark
      trained on);
    * a flat row list of `{"text"|"utterance", "label"|"tool"}`.
    """
    rows = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        corpus = [(ref, tool["id"]) for tool in rows.get("tools", [])
                  for ref in tool.get("references", [])]
        none_route = rows.get("none_route") or {}
        corpus += [(ref, none_route.get("id", "__none__"))
                   for ref in none_route.get("references", [])]
    else:
        corpus = []
        for row in rows:
            text = row.get("text", row.get("utterance"))
            label = row.get("label", row.get("tool"))
            if text is None or label is None:
                raise ValueError(f"{path}: row missing text/label: {sorted(row)}")
            corpus.append((text, label))
    if not corpus:
        raise ValueError(f"{path}: no training rows")
    return corpus
