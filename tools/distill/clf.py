"""clf — the op classifier the distilled corpus trains, and why it is shaped this way.

The corpus (`data/distill/`) supervises exactly one decision: which of the eight
ops should handle a short utterance. This module is the smallest model that can
learn it on 329 rows without a dependency — `pyproject.toml` declares none, and
the core is stdlib-only by policy, so there is no numpy and no scikit-learn here.
Everything is dict-of-index sparse with plain Python floats.

Two choices carry the result, and each was measured rather than asserted
(`tools/distill/train.py --cv` and `bench/distill_transfer.py --sweep` reproduce
the numbers):

**Features are the utterance twice — raw, and with its objects abstracted to their
kind.** Word unigrams let the model memorise the corpus's own slot nouns ("tau_lo",
"bench/lanes.json"), which is exactly the vocabulary that does *not* survive the
transfer to `utterances_v2`. `abstract()` rewrites those nouns to `<url>`, `<sub>`,
`<path>`, `<file>`, `<ident>`, `<lit>`, `<num>` from the surface form alone, so an
object the corpus never saw is typed the same way at inference time, and the `a:`
features describe the operation's frame instead of its noun. A shape block (`s:url`,
`s:sub`, `s:ext`, word count, first token, …) sits on top. Under family-grouped CV on
`train.jsonl` the raw-only model scores 0.740 and adding the abstracted view moves it
to 0.763; adding boundary-anchored char 3-5 grams buys nothing for three times the
fit, so it is opt-in (`char_grams`).

**The learner is closed-form dual ridge, not a fitted ensemble.** Unbounded Adagrad
on these features reaches 100% in-sample with weights at |w| = 4.4e3 and eight seeds
that disagree on 109 of 150 out-of-corpus rows: it scores 0.774 under grouped CV, the
highest of any candidate tried, and transfers at 60.0%. Ridge — one hyperparameter, no
seed — scores 0.763 in-corpus and transfers at 68.7%. `SoftmaxAdagrad` is kept because
the comparison is the point: in-corpus CV ranks it *first* and it is the worse model,
which is `docs/distill-classifier-2026-11-04.md` §6.

Neither choice beats the teacher: this is a linear model on a corpus whose labels are
authored, and it loses to a keyword table. The report is
`docs/distill-classifier-2026-11-04.md`.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import random
import re
from typing import Iterable

ROOT = pathlib.Path(__file__).resolve().parents[2]
CORPUS_DIR = ROOT / "data" / "distill"

FORMAT = "ti-distill-clf/1"

# The eight ops, in the order the corpus ballot uses. Index order is part of the
# artefact: a reordered tuple is a different model.
LABELS = (
    "decide_extract",
    "decide_pick",
    "filesys_glob",
    "filesys_grep",
    "filesys_read",
    "reddit_digest",
    "web_fetch",
    "web_search",
)
LABEL_INDEX = {lab: i for i, lab in enumerate(LABELS)}

# The shipped configuration: candidate C6 in `bench/distill_transfer.py` — dual
# ridge on the raw + object-abstracted feature views, lambda 1.0. Chosen by
# family-grouped CV on `train.jsonl` among the candidates whose fit is bounded
# (a closed-form ridge solution, not eight unbounded Adagrad seeds);
# `docs/distill-classifier-2026-11-04.md` §5 shows the CV ranks the unbounded
# ensembles higher and that this preference does not transfer.
DEFAULT = {
    "min_df": 1,
    "epochs": 10,
    "lr": 0.5,
    "n_seeds": 8,
    "char_grams": False,
    "mode": "both",
    "learner": "ridge",
    "decay": 1.0,
    "lam": 1.0,
}

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_URL = re.compile(r"https?://|www\.")
_SUBREDDIT = re.compile(r"\br/")
_EXTENSION = re.compile(r"\.[a-z0-9]{1,5}\b")
_DIGIT = re.compile(r"[0-9]")


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #

def tokens(text: str) -> list[str]:
    """Lowercase, punctuation as separator. `bench/lanes.json` ->
    ['bench', 'lanes', 'json'], so the extension survives as its own token."""
    return [t for t in _NON_ALNUM.split(text.lower()) if t]


_ABSTRACT_PATTERNS = (
    (re.compile(r"https?://\S+"), "<url>"),
    (re.compile(r"\br/\w+", re.I), "<sub>"),
    (re.compile(r"[\w.-]*/[\w./-]*\w"), "<path>"),
    (re.compile(r"\b[\w-]+\.(?:py|md|json|toml|ya?ml|txt|sh|cfg|ini|proto|xml|html|lock|rs|go|c|h|cpp)\b",
                re.I), "<file>"),
    (re.compile(r"\b\w+_\w+\b"), "<ident>"),
    (re.compile(r"'[^']{1,60}'"), "<lit>"),
    (re.compile(r'"[^"]{1,60}"'), "<lit>"),
    (re.compile(r"\b\d+(?:\.\d+)*\b"), "<num>"),
)


def abstract(text: str) -> str:
    """Replace the utterance's *objects* with their kind, keeping its frame.

    `print the whole pyproject.toml for me` and `open service/ti/registry.py`
    both become `<verb> the whole <file> for me` / `open <path>`: the operation
    survives, the noun does not. This is the feature form of the thesis
    `docs/p1-diagnose` reached from the 3B's failures — it routes on the object
    noun and ignores the operation — and it is a normalisation of the utterance's
    own surface form, not a lookup of the training corpus's slot list: an object
    the corpus never saw (`tests/data/utterances.json`, `wallet.json`) is typed
    by the same patterns at inference time.

    Order matters: a URL contains a `/` and a `<host>` word, so it is replaced
    before paths are, and a path is replaced before the identifier rule can fire
    on its stem.
    """
    out = text
    for pattern, tag in _ABSTRACT_PATTERNS:
        out = pattern.sub(f" {tag} ", out)
    return re.sub(r"\s+", " ", out).strip()


def shape_features(text: str) -> list[str]:
    """What *kind* of thing the utterance names, not which one.

    Every feature here is a property of the surface form alone, so it transfers
    across slot values: `fetch https://a.example` and `fetch https://b.example`
    share `s:url` and share nothing lexical that matters.
    """
    low = text.lower()
    toks = tokens(text)
    out = [
        "s:url" if _URL.search(low) else "s:no-url",
        "s:sub" if _SUBREDDIT.search(low) else "s:no-sub",
        "s:ext" if _EXTENSION.search(low) else "s:no-ext",
        f"s:nw:{min(len(toks), 12)}",
    ]
    if toks:
        out.append(f"s:head:{toks[0]}")
    if _DIGIT.search(low):
        out.append("s:digit")
    if any("_" in t for t in toks):
        out.append("s:ident")
    if any("/" in t or "\\" in t for t in text.split()):
        out.append("s:path")
    if "?" in text:
        out.append("s:question")
    if "'" in text or '"' in text:
        out.append("s:quoted")
    return out


def features(text: str, char_grams: bool = False, mode: str = "both") -> list[str]:
    """Word unigrams, word bigrams, the shape block, and (optionally) the
    boundary-anchored char 3-5 grams `bench/local_clf.py` uses.

    `mode` selects which text the lexical features are taken from: `raw` (the
    utterance), `abstract` (objects replaced by their kind, `abstract()`), or
    `both` — the raw tokens prefixed `w:` and the abstracted ones prefixed `a:`,
    so the two views cannot collide and a weight learned on one is not silently
    reused by the other.
    """
    out: list[str] = []
    views = []
    if mode in ("raw", "both"):
        views.append(("w", tokens(text)))
    if mode in ("abstract", "both"):
        views.append(("a", tokens(abstract(text))))
    for prefix, toks in views:
        out += [f"{prefix}:{t}" for t in toks]
        out += [f"{prefix}b:{a}_{b}" for a, b in zip(toks, toks[1:])]
        if char_grams:
            for t in toks:
                padded = f"^{t}$"
                for n in (3, 4, 5):
                    for i in range(max(0, len(padded) - n + 1)):
                        out.append(f"{prefix}c{n}:{padded[i:i + n]}")
    out += shape_features(text)
    return out


class Featurizer:
    """Sublinear tf * idf on a training-fitted vocabulary; rows l2-normalised.

    Rows are dicts of `{feature_index: weight}` rather than dense vectors: the
    vocabulary is a few thousand entries and each utterance touches a few dozen,
    and pure-Python sparse arithmetic is what keeps an eight-seed ensemble under
    a second per fit.
    """

    def __init__(self, min_df: int = 1, char_grams: bool = False,
                 mode: str = "both") -> None:
        self.min_df = min_df
        self.char_grams = char_grams
        self.mode = mode
        self.vocab: dict[str, int] = {}
        self.idf: list[float] = []

    def _feats(self, text: str) -> list[str]:
        return features(text, self.char_grams, self.mode)

    def fit(self, texts: list[str]) -> "Featurizer":
        df: dict[str, int] = {}
        for t in texts:
            for f in set(self._feats(t)):
                df[f] = df.get(f, 0) + 1
        kept = sorted(f for f, c in df.items() if c >= self.min_df)
        self.vocab = {f: i for i, f in enumerate(kept)}
        n = len(texts)
        self.idf = [math.log((1.0 + n) / (1.0 + df[f])) + 1.0 for f in kept]
        return self

    def transform(self, texts: list[str]) -> list[dict[int, float]]:
        if not self.vocab:
            raise RuntimeError("Featurizer.transform before fit")
        rows: list[dict[int, float]] = []
        for t in texts:
            counts: dict[int, int] = {}
            for f in self._feats(t):
                j = self.vocab.get(f)
                if j is not None:
                    counts[j] = counts.get(j, 0) + 1
            row = {j: (1.0 + math.log(c)) * self.idf[j] for j, c in counts.items()}
            norm = math.sqrt(sum(v * v for v in row.values()))
            if norm > 0.0:
                row = {j: v / norm for j, v in row.items()}
            rows.append(row)
        return rows


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #

def _softmax(z: list[float]) -> list[float]:
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


class SoftmaxAdagrad:
    """Multinomial softmax trained by sparse Adagrad SGD.

    Deterministic: the row order is a `random.Random(seed).shuffle` and every
    float accumulates in list order, so two runs with the same seed produce
    byte-identical weights (asserted by `tests/test_distill_clf.py`).

    Adagrad rather than plain gradient descent because it needs a handful of
    epochs: the momentum variant in `bench/local_clf.py` wants 400 passes over
    the same rows to reach the same place, which is 30x the wall clock for an
    ensemble of eight.

    **Not the shipped learner.** It is the reference for the comparison in
    `docs/distill-classifier-2026-11-04.md` §6: it wins in-corpus CV and loses on
    transfer, because nothing here bounds its weights. `decay` (a per-epoch,
    decoupled shrink) makes it merely worse instead of unbounded, which is why the
    decayed candidate is also in that table.
    """

    def __init__(self, epochs: int = 10, lr: float = 0.5, seed: int = 0,
                 decay: float = 1.0) -> None:
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.decay = decay
        self.W: list[list[float]] = []
        self.dim = 0

    def fit(self, X: list[dict[int, float]], y: list[int]) -> "SoftmaxAdagrad":
        dim = max((max(r) for r in X if r), default=-1) + 1
        k = len(LABELS)
        W = [[0.0] * k for _ in range(dim + 1)]
        A = [[1e-8] * k for _ in range(dim + 1)]
        rows = [list(r.items()) for r in X]
        order = list(range(len(X)))
        rng = random.Random(self.seed)

        for _ in range(self.epochs):
            rng.shuffle(order)
            for i in order:
                row, target = rows[i], y[i]
                bias = W[dim]
                z = list(bias)
                for j, v in row:
                    Wj = W[j]
                    for c in range(k):
                        z[c] += v * Wj[c]
                p = _softmax(z)
                for c in range(k):
                    g = p[c] - (1.0 if c == target else 0.0)
                    bias[c] -= self.lr * g / math.sqrt(A[dim][c])
                    A[dim][c] += g * g
                    if g == 0.0:
                        continue
                    for j, v in row:
                        gj = v * g
                        W[j][c] -= self.lr * gj / math.sqrt(A[j][c])
                        A[j][c] += gj * gj
            if self.decay != 1.0:
                # Decoupled decay, once per epoch. Inside the per-row update it
                # compounds over every feature x class the row touches, and at
                # Adagrad's step size that is a collapse to chance, not a prior.
                for Wj in W:
                    for c in range(k):
                        Wj[c] *= self.decay
        self.W = W
        self.dim = dim
        return self

    def proba(self, x: dict[int, float]) -> list[float]:
        bias = self.W[self.dim]
        z = list(bias)
        for j, v in x.items():
            Wj = self.W[j]
            for c in range(len(LABELS)):
                z[c] += v * Wj[c]
        return _softmax(z)


def _dot(a: dict[int, float], b: dict[int, float]) -> float:
    """Sparse dot product, iterating the shorter row."""
    if len(a) > len(b):
        a, b = b, a
    s = 0.0
    for j, v in a.items():
        w = b.get(j)
        if w:
            s += v * w
    return s


class Ridge:
    """Closed-form dual ridge regression on one-hot targets.

    No seed, no learning rate, no epoch count: `alpha = (K + lam*I)^-1 Y` with
    `K = X X^T`, and a row's scores are `k @ alpha` for its kernel row `k`. With
    329 rows the dual is 329x329, which a pure-Python Cholesky factorises in a
    couple of seconds — cheaper than the ensemble and, unlike it, bounded by
    construction. The single knob is `lam`.

    Included because the Adagrad ensemble can only be made well-behaved by
    bounding it, and a closed-form solution with one honest hyperparameter is a
    stronger claim on 329 rows than eight random draws from an unbounded one.
    """

    def __init__(self, lam: float = 1.0) -> None:
        self.lam = lam
        self.alpha: list[list[float]] = []
        self.X: list[dict[int, float]] = []

    def fit(self, X: list[dict[int, float]], y: list[int]) -> "Ridge":
        n, k = len(X), len(LABELS)
        K = [[0.0] * n for _ in range(n)]
        for i in range(n):
            xi = X[i]
            for j in range(i, n):
                K[i][j] = K[j][i] = _dot(xi, X[j])
        for i in range(n):
            K[i][i] += self.lam
        Y = [[0.0] * k for _ in range(n)]
        for i, target in enumerate(y):
            Y[i][target] = 1.0

        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            Li, Ki = L[i], K[i]
            for j in range(i + 1):
                Lj = L[j]
                s = Ki[j] - sum(Li[t] * Lj[t] for t in range(j))
                Li[j] = math.sqrt(s) if i == j else s / Lj[j]

        alpha = [[0.0] * k for _ in range(n)]
        for c in range(k):
            z = [0.0] * n
            for i in range(n):
                Li = L[i]
                z[i] = (Y[i][c] - sum(Li[t] * z[t] for t in range(i))) / Li[i]
            for i in range(n - 1, -1, -1):
                Li = L[i]
                alpha[i][c] = (z[i] - sum(Li[t] * alpha[t][c] for t in range(i + 1, n))) / Li[i]
        self.alpha = alpha
        self.X = X
        return self

    def proba(self, x: dict[int, float]) -> list[float]:
        scores = [0.0] * len(LABELS)
        for j, xj in enumerate(self.X):
            kern = _dot(x, xj)
            if kern == 0.0:
                continue
            a = self.alpha[j]
            for c in range(len(LABELS)):
                scores[c] += kern * a[c]
        return _softmax([s * 4.0 for s in scores])


def _model_matrix(model) -> list[list[float]]:
    """The fitted parameter block, whichever learner produced it."""
    return model.alpha if isinstance(model, Ridge) else model.W


class DistillClassifier:
    """A featurizer plus its fitted learner — one object, one artefact."""

    def __init__(self, min_df: int = 1, epochs: int = 10, lr: float = 0.5,
                 n_seeds: int = 8, char_grams: bool = False, mode: str = "both",
                 learner: str = "ensemble", decay: float = 1.0,
                 lam: float = 1.0) -> None:
        self.min_df = min_df
        self.epochs = epochs
        self.lr = lr
        self.n_seeds = n_seeds
        self.char_grams = char_grams
        self.mode = mode
        self.learner = learner
        self.decay = decay
        self.lam = lam
        self.feat = Featurizer(min_df=min_df, char_grams=char_grams, mode=mode)
        self.models: list = []
        self.n_train = 0

    @classmethod
    def from_config(cls, config: dict) -> "DistillClassifier":
        cfg = dict(DEFAULT, **config)
        return cls(min_df=cfg["min_df"], epochs=cfg["epochs"], lr=cfg["lr"],
                   n_seeds=cfg["n_seeds"], char_grams=cfg["char_grams"],
                   mode=cfg["mode"], learner=cfg["learner"], decay=cfg["decay"],
                   lam=cfg["lam"])

    def config(self) -> dict:
        return {"min_df": self.min_df, "epochs": self.epochs, "lr": self.lr,
                "n_seeds": self.n_seeds, "char_grams": self.char_grams,
                "mode": self.mode, "learner": self.learner, "decay": self.decay,
                "lam": self.lam}

    def fit(self, rows: Iterable[dict]) -> "DistillClassifier":
        """`rows` are corpus rows; the label is `label` (the authored `intended`
        op), never the teacher's answer."""
        rows = list(rows)
        texts = [r["utterance"] for r in rows]
        y = [LABEL_INDEX[r["label"]] for r in rows]
        self.n_train = len(rows)
        self.feat = Featurizer(min_df=self.min_df, char_grams=self.char_grams,
                               mode=self.mode).fit(texts)
        X = self.feat.transform(texts)
        if self.learner == "ridge":
            self.models = [Ridge(lam=self.lam).fit(X, y)]
        else:
            self.models = [
                SoftmaxAdagrad(epochs=self.epochs, lr=self.lr, seed=s,
                               decay=self.decay).fit(X, y)
                for s in range(self.n_seeds)
            ]
        return self

    def predict_proba(self, text: str) -> list[float]:
        if not self.models:
            raise RuntimeError("predict_proba before fit")
        x = self.feat.transform([text])[0]
        acc = [0.0] * len(LABELS)
        for m in self.models:
            for c, p in enumerate(m.proba(x)):
                acc[c] += p
        return [v / len(self.models) for v in acc]

    def predict(self, text: str) -> tuple[str, float]:
        """(label, confidence). There is no abstain: the corpus has no abstain
        class, so an empty utterance becomes whatever the bias favours."""
        p = self.predict_proba(text)
        best = max(range(len(LABELS)), key=lambda c: (p[c], -c))
        return LABELS[best], p[best]

    # -- artefact ----------------------------------------------------------- #

    def to_artefact(self) -> dict:
        return {
            "format": FORMAT,
            "labels": list(LABELS),
            "config": self.config(),
            "n_train": self.n_train,
            "vocab": sorted(self.feat.vocab, key=self.feat.vocab.get),
            "idf": [round(v, 12) for v in self.feat.idf],
            "weights": [[[round(v, 12) for v in row] for row in _model_matrix(m)]
                        for m in self.models],
        }

    def artefact_sha(self) -> str:
        """sha256 over the canonical serialisation — the recipe's proof."""
        blob = json.dumps(self.to_artefact(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# corpus access
# --------------------------------------------------------------------------- #

def read_jsonl(path: pathlib.Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_split(split: str, corpus_dir: pathlib.Path | str = CORPUS_DIR) -> list[dict]:
    return read_jsonl(pathlib.Path(corpus_dir) / f"{split}.jsonl")


def train_default(corpus_dir: pathlib.Path | str = CORPUS_DIR) -> DistillClassifier:
    """The shipped model: `DEFAULT` fitted on `train.jsonl`.

    Deliberately reads one file and one field. `utterances_v2` is not reachable
    from here, which is the structural half of "do not train on the eval set" —
    `tests/test_distill_clf.py` asserts the other half (the model's training
    utterances are disjoint from all 150 v2 rows).
    """
    return DistillClassifier.from_config(DEFAULT).fit(load_split("train", corpus_dir))
