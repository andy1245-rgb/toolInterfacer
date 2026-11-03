"""linear_text — TF-IDF + linear classifier, stdlib only.

A re-implementation of the two scikit-learn objects the sibling benchmark used
for its classical baseline (``TfidfVectorizer(ngram_range=(1, 2), min_df=1,
sublinear_tf=True)`` and a linear SVM), because this repo declares no
third-party dependencies and scikit-learn is not installed here. The vectoriser
defaults are reproduced exactly — lowercase, ``\\b\\w\\w+\\b`` tokens, smoothed
idf ``ln((1+n)/(1+df)) + 1``, sublinear tf ``1 + ln(count)``, L2 row norm — so a
score from this module is comparable with the recorded one rather than merely
similar to it.

The classifier is one-vs-rest linear with an L2 penalty, trained by stochastic
subgradient descent with Polyak weight averaging and a ``eta0 / (1 + eta0*l2*t)``
step schedule, so ``epochs`` is a convergence budget rather than a tuned
hyper-parameter: past the point where the objective stops moving, more epochs
change nothing. Two losses:

``squared_hinge``
    ``sum_i max(0, 1 - y_i (w.x_i))^2`` per class — what ``LinearSVC``
    minimises. Decision values are uncalibrated margins.

``log``
    multinomial logistic — the softmax of the scores is a probability
    distribution natively, which is what ``LogisticRegression`` gives.

Determinism: the document order for every epoch comes from ``random.Random(seed)``
and the averaged weights are computed in a fixed order, so the same corpus,
config and seed produce identical weights. ``TextClassifier.fingerprint`` hashes
the fitted weights together with the whole recipe (hyper-parameters, seed,
corpus size and any extra the caller supplies), so a recorded accuracy can be
tied to the model that produced it.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, field

_TOKEN = re.compile(r"\b\w\w+\b")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens of 2+ word characters — sklearn's default pattern."""
    return _TOKEN.findall(text.lower())


@dataclass
class TfidfVectorizer:
    """Sparse TF-IDF over word n-grams, matching sklearn's defaults."""

    ngram_max: int = 2
    min_df: int = 1
    sublinear_tf: bool = True
    _terms: dict[str, int] = field(default_factory=dict)
    _idf: list[float] = field(default_factory=list)

    def fit(self, docs: list[str]) -> "TfidfVectorizer":
        df: dict[str, int] = {}
        for doc in docs:
            for term in set(self._ngrams(doc)):
                df[term] = df.get(term, 0) + 1
        kept = sorted(t for t, c in df.items() if c >= self.min_df)
        self._terms = {t: i for i, t in enumerate(kept)}
        n = len(docs)
        self._idf = [math.log((1.0 + n) / (1.0 + df[t])) + 1.0 for t in kept]
        return self

    def transform(self, docs: list[str]) -> list[tuple[list[int], list[float]]]:
        return [self.vectorize(doc) for doc in docs]

    def fit_transform(self, docs: list[str]) -> list[tuple[list[int], list[float]]]:
        return self.fit(docs).transform(docs)

    def _ngrams(self, doc: str) -> list[str]:
        toks = tokenize(doc)
        out = list(toks)
        for n in range(2, self.ngram_max + 1):
            out.extend(" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1))
        return out

    def vectorize(self, doc: str) -> tuple[list[int], list[float]]:
        counts: dict[int, int] = {}
        for term in self._ngrams(doc):
            idx = self._terms.get(term)
            if idx is not None:
                counts[idx] = counts.get(idx, 0) + 1
        if not counts:
            return [], []
        vals = []
        for idx, count in counts.items():
            tf = 1.0 + math.log(count) if self.sublinear_tf else float(count)
            vals.append(tf * self._idf[idx])
        norm = math.sqrt(sum(v * v for v in vals))
        idxs = list(counts)
        if norm > 0.0:
            vals = [v / norm for v in vals]
        return idxs, vals

    @property
    def n_features(self) -> int:
        return len(self._terms)


@dataclass
class LinearModel:
    """One weight row per class plus a bias — enough to score any text."""

    classes: list[str]
    weights: list[list[float]]
    bias: list[float]
    loss: str

    def scores(self, idxs: list[int], vals: list[float]) -> list[float]:
        """Raw decision values, one per class, in ``classes`` order."""
        out = []
        for k, row in enumerate(self.weights):
            total = self.bias[k]
            for i, v in zip(idxs, vals):
                total += row[i] * v
            out.append(total)
        return out

    def fingerprint(self) -> str:
        """Stable digest of the fitted weights (this is the model's identity)."""
        h = hashlib.sha256()
        h.update(",".join(self.classes).encode())
        h.update(self.loss.encode())
        for row, b in zip(self.weights, self.bias):
            h.update((",".join(f"{w:.6g}" for w in row) + f"|{b:.6g}").encode())
        return h.hexdigest()


def softmax(values: list[float], temperature: float = 1.0) -> list[float]:
    """Softmax over scores — the probability reading of a set of margins."""
    if not values:
        return []
    t = temperature if temperature > 0 else 1.0
    top = max(values)
    exps = [math.exp((v - top) / t) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


@dataclass
class TextClassifier:
    """Vectoriser + linear model, fitted together; the unit the backend caches."""

    vectorizer: TfidfVectorizer
    model: LinearModel
    seed: int
    fingerprint: str

    def scores(self, text: str) -> dict[str, float]:
        idxs, vals = self.vectorizer.vectorize(text)
        return dict(zip(self.model.classes, self.model.scores(idxs, vals)))


def fit(texts: list[str], labels: list[str], *, loss: str = "squared_hinge",
        c: float = 1.0, epochs: int = 100, eta: float = 0.1, seed: int = 42,
        ngram_max: int = 2, min_df: int = 1, sublinear_tf: bool = True,
        fingerprint_extra: dict | None = None) -> TextClassifier:
    """Fit TF-IDF + an L2-regularised one-vs-rest linear model.

    ``c`` is the per-sample loss weight (sklearn's ``C``); the L2 strength is
    ``1 / (c * n_samples)``, which is sklearn's own scaling, so a ``c`` of 1.0
    here means what ``LinearSVC(C=1.0)`` means there.
    """
    if len(texts) != len(labels):
        raise ValueError("texts and labels must be the same length")
    if not texts:
        raise ValueError("cannot fit on an empty corpus")
    if loss not in ("squared_hinge", "log"):
        raise ValueError(f"unknown loss {loss!r}")

    vec = TfidfVectorizer(ngram_max=ngram_max, min_df=min_df, sublinear_tf=sublinear_tf)
    docs = vec.fit_transform(texts)
    classes = sorted(set(labels))
    k, n = len(classes), len(docs)
    pos = {c: i for i, c in enumerate(classes)}
    y = [pos[label] for label in labels]

    dim = vec.n_features
    weights = [[0.0] * dim for _ in range(k)]
    bias = [0.0] * k
    avg_w = [[0.0] * dim for _ in range(k)]
    avg_b = [0.0] * k
    l2 = 1.0 / (c * n)
    rng = random.Random(seed)
    order = list(range(n))
    passes = max(epochs, 1)

    # The L2 term multiplies every weight by (1 - eta*l2) on every step. Doing
    # that densely costs dim*k per document, which is the whole training budget
    # at this corpus size; instead one factor is kept for the lot and weights are
    # materialised only when the factor is folded back in (at every epoch end and
    # whenever it has drifted far enough to lose precision).
    scale = 1.0
    step = 0
    for _ in range(passes):
        rng.shuffle(order)
        for d in order:
            idxs, vals = docs[d]
            if not idxs:
                continue
            step += 1
            eta_t = eta / (1.0 + eta * l2 * step)
            scale *= 1.0 - eta_t * l2
            for j in range(k):
                row = weights[j]
                s = bias[j]
                for i, v in zip(idxs, vals):
                    s += row[i] * v
                s *= scale
                target = 1.0 if j == y[d] else -1.0
                if loss == "squared_hinge":
                    margin = 1.0 - target * s
                    ngrad = 0.0 if margin <= 0.0 else 2.0 * c * margin * target
                else:
                    # -d/ds of log(1 + exp(-target*s)), the ascent direction for
                    # the target class; clipped so exp() cannot overflow.
                    z = -target * s
                    ngrad = 0.0 if z > 30.0 else target / (1.0 + math.exp(-z))
                if ngrad != 0.0:
                    g = eta_t * ngrad / scale
                    for i, v in zip(idxs, vals):
                        row[i] += g * v
                    bias[j] += g
            if scale < 0.5:
                for j in range(k):
                    row = weights[j]
                    for i in range(dim):
                        row[i] *= scale
                    bias[j] *= scale
                scale = 1.0
        for j in range(k):
            row, aw = weights[j], avg_w[j]
            for i in range(dim):
                aw[i] += row[i] * scale  # scale is per-epoch: average the
            avg_b[j] += bias[j] * scale  # weights that actually scored

    weights = [[w / passes for w in row] for row in avg_w]
    bias = [b / passes for b in avg_b]
    model = LinearModel(classes=classes, weights=weights, bias=bias, loss=loss)
    fp = hashlib.sha256()
    fp.update(model.fingerprint().encode())
    fp.update(json.dumps({"loss": loss, "c": c, "epochs": epochs, "eta": eta,
                          "seed": seed, "ngram_max": ngram_max, "min_df": min_df,
                          "sublinear_tf": sublinear_tf, "n_docs": n,
                          "n_features": dim, "classes": classes,
                          "corpus": (fingerprint_extra or {})},
                         sort_keys=True).encode())
    return TextClassifier(vectorizer=vec, model=model, seed=seed,
                          fingerprint=fp.hexdigest())
