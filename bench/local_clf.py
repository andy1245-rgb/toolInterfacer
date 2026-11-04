#!/usr/bin/env python3
"""Offline 8-way op classifier + cascade economics for the local/hosted frontier.

Question this answers: how much of the routing decision can a zero-cost local
classifier take off `z-ai/glm-5.3-flash` before blended accuracy drops below the
hosted model's own, and what does that buy per 1k decisions?

No network, no model calls, no service imports: the dataset is 150 short
utterances and everything here is arithmetic. numpy is the only dependency.

Usage:
    python bench/local_clf.py [--data tests/data/utterances_v2.json]
        [--folds 5] [--inner-folds 3] [--heldout tests/data/heldout_phase3.json]
        [--llm-results bench/results/dell/v2-d1-3B-Q4.json]
        [--json bench/results/local_clf_v2.json] [--no-latency]

Protocol notes (read before quoting any number):
  * Folds are GROUPED. t3 rows come in boundary families (label/tier_target
    pair) and t1/t2 rows come in miss-paraphrase clusters, so a random split
    puts near-duplicates on both sides of the split and inflates accuracy. A
    group never straddles folds; both splits are reported so the inflation is
    visible rather than asserted.
  * Hyperparameters are chosen by an inner grouped CV on the training portion of
    each outer fold. The test fold is never consulted for model selection.
  * The 3B column is read from a recorded probe file (already out-of-sample: it
    is a frozen model, no fitting). It is NOT cross-validated, and v2 was built
    from that model's misses, so its column is pessimistic here by construction.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
import statistics
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def repo_rel(path: str | Path) -> str:
    """Path as recorded in the artifact: repo-relative when it is inside the
    tree, bare filename when it is not.

    The recording is committed, so it must not carry the machine it was made on
    (`tests/test_no_personal_paths.py`). Absolutising `--data` into the JSON made
    every re-run re-leak whatever worktree it ran in.
    """
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return p.name


LABELS = (
    "filesys_read",
    "filesys_glob",
    "filesys_grep",
    "decide_pick",
    "decide_extract",
    "web_search",
    "web_fetch",
    "reddit_digest",
)
LABEL_INDEX = {lab: i for i, lab in enumerate(LABELS)}
BANDS = ("t1", "t2", "t3")

# Reference points supplied with the task, not re-derived here.
HOSTED_NAME = "z-ai/glm-5.3-flash"
HOSTED_ACC = 0.953          # 143/150 on this same v2 set
HOSTED_COST_1K = 0.0221     # USD per 1k decisions
HOSTED_COST_1K_CORRECT = 0.0232
HOSTED_P50_MS = 588.0
LOCAL_COST_PER_1K = 0.0     # on-box CPU, sub-millisecond; treated as free below

Z95 = 1.959963984540054


# --------------------------------------------------------------------------- #
# text features
# --------------------------------------------------------------------------- #

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def tokens(text: str) -> list[str]:
    """Lowercase, punctuation-as-separator. `service/ti/registry.py` ->
    ['service','ti','registry','py'] so the extension stays a token."""
    return [t for t in _NON_ALNUM.split(text.lower()) if t]


def features(text: str) -> list[str]:
    """Word 1-2 grams + boundary-anchored char 3-5 grams."""
    toks = tokens(text)
    out = [f"w:{t}" for t in toks]
    out += [f"b:{a}_{b}" for a, b in zip(toks, toks[1:])]
    for t in toks:
        padded = f"^{t}$"
        for n in (3, 4, 5):
            if len(padded) < n:
                continue
            for i in range(len(padded) - n + 1):
                out.append(f"c{n}:{padded[i:i + n]}")
    return out


class Featurizer:
    """Sublinear tf * idf on a training-fitted vocabulary, rows l2-normalised.

    Built with a plain dict scan: 150 rows and a few thousand features, so a
    dense float64 matrix is cheaper in wall time than a sparse type would be to
    write here.
    """

    def __init__(self, min_df: int = 2, char_only: bool = False) -> None:
        self.min_df = min_df
        self.char_only = char_only
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None

    def _feats(self, text: str) -> list[str]:
        fs = features(text)
        if self.char_only:
            fs = [f for f in fs if f.startswith("c")]
        return fs

    def fit(self, texts: list[str]) -> "Featurizer":
        df: dict[str, int] = {}
        for t in texts:
            for f in set(self._feats(t)):
                df[f] = df.get(f, 0) + 1
        kept = sorted(f for f, c in df.items() if c >= self.min_df)
        self.vocab = {f: i for i, f in enumerate(kept)}
        n = len(texts)
        self.idf = np.array(
            [math.log((1.0 + n) / (1.0 + df[f])) + 1.0 for f in kept], dtype=np.float64
        )
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        if self.idf is None:
            raise RuntimeError("Featurizer.transform before fit")
        d = len(self.vocab)
        X = np.zeros((len(texts), d), dtype=np.float64)
        for r, t in enumerate(texts):
            counts: dict[int, int] = {}
            for f in self._feats(t):
                j = self.vocab.get(f)
                if j is not None:
                    counts[j] = counts.get(j, 0) + 1
            for j, c in counts.items():
                X[r, j] = (1.0 + math.log(c)) * self.idf[j]
            norm = float(np.sqrt((X[r] ** 2).sum()))
            if norm > 0.0:
                X[r] /= norm
        return X


# --------------------------------------------------------------------------- #
# multinomial logistic regression (plain full-batch gradient descent)
# --------------------------------------------------------------------------- #

class SoftmaxRegression:
    def __init__(self, l2: float = 3e-3, lr: float = 0.5, epochs: int = 400,
                 momentum: float = 0.9) -> None:
        self.l2 = l2
        self.lr = lr
        self.epochs = epochs
        self.momentum = momentum
        self.W: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: list[int]) -> "SoftmaxRegression":
        n, d = X.shape
        k = len(LABELS)
        Xb = np.hstack([X, np.ones((n, 1))])
        Y = np.zeros((n, k))
        Y[np.arange(n), np.asarray(y, dtype=int)] = 1.0
        W = np.zeros((d + 1, k))
        V = np.zeros_like(W)
        for _ in range(self.epochs):
            P = _softmax(Xb @ W)
            G = (P - Y) / n
            grad = Xb.T @ G
            grad[:-1] += self.l2 * W[:-1]          # bias is not regularised
            V = self.momentum * V + self.lr * grad
            W = W - V
        self.W = W
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("predict_proba before fit")
        Xb = np.hstack([X, np.ones((X.shape[0], 1))])
        return _softmax(Xb @ self.W)


def _softmax(Z: np.ndarray) -> np.ndarray:
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #

class NearestNeighbour:
    """Char-ngram 1-NN, cosine on the same tf-idf char features."""

    def __init__(self, min_df: int = 2) -> None:
        self.feat = Featurizer(min_df=min_df, char_only=True)
        self.X: np.ndarray | None = None
        self.y: list[str] = []

    def fit(self, texts: list[str], labels: list[str]) -> "NearestNeighbour":
        self.feat.fit(texts)
        self.X = self.feat.transform(texts)
        self.y = list(labels)
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if self.X is None:
            raise RuntimeError("predict_proba before fit")
        Q = self.feat.transform(texts)
        sim = Q @ self.X.T                                   # rows are unit norm
        out = np.zeros((len(texts), len(LABELS)))
        for r in range(len(texts)):
            best = int(np.argmax(sim[r]))
            out[r, LABEL_INDEX[self.y[best]]] = 1.0
        return out


# Keyword cues written from the op contract (docs/ops.md §7 semantics), not from
# the labels in this file. A row with no cue abstains rather than guessing; the
# abstention rate is reported because a rules gate that always answers is a
# different (worse) baseline than one that declines.
RULES: tuple[tuple[str, float, str], ...] = (
    # web_fetch — a resolvable target
    ("web_fetch", 3.0, r"https?://"),
    ("web_fetch", 2.0, r"\b(?:that|the|this|linked)\s+(?:page|url|link|post|address|advisory|blog|article|card)\b"),
    ("web_fetch", 1.5, r"\b(?:fetch|download|load|carry on reading|click through)\b"),
    ("web_fetch", 1.5, r"\b(?:body|content|text|gist|full text)\s+of\s+(?:that|the|this)\b"),
    ("web_fetch", 1.5, r"\b(?:at|under)\s+(?:that|this|the)\s+(?:address|url|link)\b"),
    ("web_fetch", 1.0, r"\b(?:page|url|link)\b"),
    # reddit_digest — reddit-native or community-shaped discourse
    ("reddit_digest", 3.0, r"\br/\w+"),
    ("reddit_digest", 2.0, r"\b(?:reddit|subreddit|the sub|megathread)\b"),
    ("reddit_digest", 1.5, r"\b(?:community|boards?|forum|elsewhere)\b"),
    ("reddit_digest", 1.0, r"\b(?:comments|threads|reactions|respond|replies|asking|saying|chatter)\b"),
    # web_search — discovery of information that is not on disk
    ("web_search", 2.5, r"\b(?:look up|search the web|web search|find out|look into)\b"),
    ("web_search", 1.5, r"\b(?:anything out there|anything published|any coverage|coverage|news|latest|chatter|out there|published|writeups?|benchmarks|release notes|docs on|official docs)\b"),
    ("web_search", 1.5, r"\b(?:what changed in|who ships|who else|how do other projects|these days)\b"),
    ("web_search", 1.0, r"\b(?:alternate|other|compar|alternatives)\b"),
    # filesys_grep — find by content
    ("filesys_grep", 2.0, r"\b(?:call ?site|call sites?|defined|definition|references?|referenced|imports?|uses? of|use of|mention|mentions|usages?|occurs?|catch|calls?)\b"),
    ("filesys_grep", 2.0, r"\b(?:which|where|find|locate|search|look for|track down|grep|anything|any)\b.*\b(?:defined|referenced|mention|import|use|uses|calls?|catch|hardcoded|left|touch|do)\b"),
    ("filesys_grep", 1.5, r"\bgrep\b"),
    ("filesys_grep", 1.0, r"\b(?:hardcoded|in the tree|in src|in tests|in bin|in the fixtures?)\b"),
    # filesys_glob — find by name / shape of the tree
    ("filesys_glob", 2.0, r"\b(?:list|names? of|how many|which files?|find all|find every|any files?|are there any|do we have|what files?)\b"),
    ("filesys_glob", 2.0, r"\.(?:py|md|toml|ya?ml|txt|json|sh|proto|bak|har|patch|cfg|ini|lock|rst)\b"),
    ("filesys_glob", 1.5, r"\b(?:ending in|ends in|start(?:s)? with|match the pattern|the pattern|dotfiles?|top level|anywhere in)\b"),
    ("filesys_glob", 1.0, r"\b(?:under|in the|in|here)\s+\S+/?\s*$"),
    ("filesys_glob", 1.0, r"\b(?:fixtures?|docs?|dirs?|director(?:y|ies)|subdirs?)\b"),
    # filesys_read — the contents of one named artifact
    ("filesys_read", 2.0, r"\b(?:print|cat|dump|show|display|read|open|pull up|walk me through|look at|check the top of)\b"),
    ("filesys_read", 2.0, r"\b(?:contents? of|what'?s inside|what'?s written in|full text of|body of|schema block|the whole)\b"),
    ("filesys_read", 1.5, r"\b[\w./-]+\.(?:py|md|toml|ya?ml|txt|json|cfg|ini|lock)\b"),
    ("filesys_read", 1.0, r"\b(?:implemented in|how does|how is)\b"),
    # decide_pick — triage of an existing candidate set
    ("decide_pick", 2.5, r"\b(?:which one|which of|which is|which do we|of these|of those|among|out of the|of the (?:dozen|two|three|four|five|ten)|narrow .* down|keep the (?:best|two|three)|sort the|rank)\b"),
    ("decide_pick", 2.0, r"\b(?:candidates?|hits?|matches?|results?|threads?|pages?|files found|shortlist|drafts?|queue|pile)\b"),
    ("decide_pick", 1.5, r"\b(?:most (?:useful|relevant|on-topic|strongest)|best|strongest|more on-topic|right one|right entry|the right one|should we (?:ship|actually visit|open)|point me at)\b"),
    ("decide_pick", 1.0, r"\b(?:choose|pick|select)\b"),
    # decide_extract — derive queries/names/paths from the ask itself
    ("decide_extract", 2.5, r"\b(?:phrasings?|wordings?|keyword|keywords|search terms?|search strings?|queries|query|variants?|angles|alternative ways|alternate ways|distinct ways)\b"),
    ("decide_extract", 2.0, r"\b(?:turn .* into|come up with|give me (?:a few|some|three|five|several|a fresh)|write out|work out|extract|pull out|pull the|name the|list the exact|titles?|identifiers?)\b"),
    ("decide_extract", 1.5, r"\b(?:paths?|dirs?|filenames?|file names?|the names|wording)\b"),
    ("decide_extract", 1.0, r"\b(?:the ask|the request|the brief|the instruction|the goal|the user'?s)\b"),
)


def rules_predict(text: str) -> tuple[str | None, dict[str, float]]:
    score = {lab: 0.0 for lab in LABELS}
    for lab, weight, pattern in RULES:
        if re.search(pattern, text.lower()):
            score[lab] += weight
    best = max(score.items(), key=lambda kv: (kv[1], -LABELS.index(kv[0])))
    if best[1] <= 0.0:
        return None, score
    return best[0], score


# --------------------------------------------------------------------------- #
# grouping and folds
# --------------------------------------------------------------------------- #

def _char_shingles(text: str, n: int = 3) -> set[str]:
    s = " " + _NON_ALNUM.sub(" ", text.lower()).strip() + " "
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def boundary_family(row: dict) -> str:
    pair = tuple(sorted((row["label"], row["tier_target"])))
    return f"t3|{pair[0]}~{pair[1]}"


def build_groups(rows: list[dict], dup_threshold: float = 0.25) -> list[str]:
    """Group id per row.

    t3: one group per boundary family (label, tier_target) — the pair the row
        was constructed inside, from both directions.
    t1/t2: single-link clusters of near-duplicate paraphrase inside one label
        (char-3gram Jaccard >= dup_threshold). Labelling these by label alone
        would put an entire class in one fold, which measures leave-classes-out
        generalisation and is not what production looks like.

    At 0.25 the t1/t2 side yields 91 clusters over 95 rows (4 pairs merged,
    largest cluster 2); 0.34 merges 1 pair and 0.20 chains to 7. The threshold
    is stated rather than tuned: it only has to stop verbatim siblings from
    straddling a fold, and the t3 families — where the paraphrase pressure
    actually is — are grouped by construction, not by this number.
    """
    groups = [""] * len(rows)
    by_label_12: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        if r["band"] == "t3":
            groups[i] = boundary_family(r)
        else:
            by_label_12.setdefault(r["label"], []).append(i)

    for label, idxs in by_label_12.items():
        parent = {i: i for i in idxs}

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        shingles = {i: _char_shingles(rows[i]["utterance"]) for i in idxs}
        for a, b in itertools.combinations(idxs, 2):
            if _jaccard(shingles[a], shingles[b]) >= dup_threshold:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
        for i in idxs:
            groups[i] = f"t12|{label}|{find(i)}"
    return groups


def make_folds(group_ids: list[str], labels: list[str], k: int) -> list[int]:
    """Assign whole groups to k folds, stratified by label.

    Groups are placed largest-first so the big boundary families cannot end up
    all in one fold. Each placement minimises, lexicographically: (a) how much
    of this group's labels the fold already holds, (b) how full the fold is.
    (a) first is what keeps every fold containing every label; (b) breaks ties
    so the folds stay the same size. Deterministic.
    """
    members: dict[str, list[int]] = {}
    for i, g in enumerate(group_ids):
        members.setdefault(g, []).append(i)
    order = sorted(members, key=lambda g: (-len(members[g]), g))
    k = max(1, min(k, len(order)))

    fold_label: list[dict[str, int]] = [dict() for _ in range(k)]
    fold_size = [0] * k
    assign = [0] * len(group_ids)
    for g in order:
        idxs = members[g]
        lab_counts: dict[str, int] = {}
        for i in idxs:
            lab_counts[labels[i]] = lab_counts.get(labels[i], 0) + 1
        best_f, best_key = 0, None
        for f in range(k):
            held = sum(fold_label[f].get(lab, 0) for lab in lab_counts)
            key = (held, fold_size[f], f)
            if best_key is None or key < best_key:
                best_f, best_key = f, key
        for i in idxs:
            assign[i] = best_f
        fold_size[best_f] += len(idxs)
        for lab, c in lab_counts.items():
            fold_label[best_f][lab] = fold_label[best_f].get(lab, 0) + c
    return assign


def random_folds(labels: list[str], k: int, seed: int = 0) -> list[int]:
    rng = random.Random(seed)
    assign = [0] * len(labels)
    per_label: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        per_label.setdefault(lab, []).append(i)
    for lab, idxs in per_label.items():
        shuffled = idxs[:]
        rng.shuffle(shuffled)
        for j, i in enumerate(shuffled):
            assign[i] = j % k
    return assign


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def norm_ppf(q: float) -> float:
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def rows_to_certify(p: float, p0: float = HOSTED_ACC, alpha: float = 0.05,
                    power: float = 0.8) -> int | None:
    """Committed rows needed for a one-sided binomial test to show that local
    precision p beats the hosted accuracy p0 (normal approximation)."""
    if p <= p0:
        return None
    za, zb = norm_ppf(1.0 - alpha), norm_ppf(power)
    num = za * math.sqrt(p0 * (1.0 - p0)) + zb * math.sqrt(p * (1.0 - p))
    return int(math.ceil(num * num / (p - p0) ** 2))


def accuracy(preds: list[str | None], labels: list[str]) -> float:
    if not labels:
        return 0.0
    hits = sum(1 for p, y in zip(preds, labels) if p == y)
    return hits / len(labels)


# --------------------------------------------------------------------------- #
# the model config grid + CV driver
# --------------------------------------------------------------------------- #

# The gradient is a mean over rows (order 1e-2 per weight here), so l2 only
# starts biting around 1e-3: below ~1e-3 nothing is regularised, above ~1e-2 the
# solution collapses toward the majority class. The range is the honest one.
GRID = [
    {"min_df": md, "l2": l2}
    for md in (1, 2)
    for l2 in (3e-4, 1e-3, 3e-3, 1e-2)
]


def _fit_predict(train_texts: list[str], train_labels: list[str],
                 test_texts: list[str], cfg: dict) -> np.ndarray:
    feat = Featurizer(min_df=cfg["min_df"]).fit(train_texts)
    Xtr = feat.transform(train_texts)
    Xte = feat.transform(test_texts)
    model = SoftmaxRegression(l2=cfg["l2"]).fit(
        Xtr, [LABEL_INDEX[y] for y in train_labels]
    )
    return model.predict_proba(Xte)


def grouped_cv(rows: list[dict], fold_of: list[int], k: int,
               inner_k: int = 3) -> dict:
    """Outer grouped CV with an inner grouped CV for hyperparameter choice.

    Returns out-of-fold probabilities (no row is ever predicted by a model that
    saw it), the chosen config per outer fold, and inner-OOF predictions on each
    training portion (used later so per-class expert selection is not made on the
    rows it is scored on).
    """
    texts = [r["utterance"] for r in rows]
    labels = [r["label"] for r in rows]
    groups = [r["group"] for r in rows]
    n = len(rows)
    oof_probs = np.zeros((n, len(LABELS)))
    chosen: list[dict] = []
    inner_oof_pred: list[str | None] = [None] * n   # from inner CV, train portions only

    for f_out in range(k):
        train_idx = [i for i in range(n) if fold_of[i] != f_out]
        test_idx = [i for i in range(n) if fold_of[i] == f_out]
        inner_folds = make_folds([groups[i] for i in train_idx],
                                 [labels[i] for i in train_idx], inner_k)
        best_cfg, best_acc, best_pred = None, -1.0, None
        for cfg in GRID:
            preds: dict[int, np.ndarray] = {}
            for f_in in range(inner_k):
                tr = [train_idx[j] for j in range(len(train_idx))
                      if inner_folds[j] != f_in]
                te = [train_idx[j] for j in range(len(train_idx))
                      if inner_folds[j] == f_in]
                if not tr or not te:
                    continue
                P = _fit_predict([texts[i] for i in tr], [labels[i] for i in tr],
                                 [texts[i] for i in te], cfg)
                for row_pos, i in enumerate(te):
                    preds[i] = P[row_pos]
            if not preds:
                continue
            acc = accuracy(
                [LABELS[int(np.argmax(preds[i]))] if i in preds else None
                 for i in train_idx],
                [labels[i] for i in train_idx],
            )
            if acc > best_acc + 1e-12:
                best_cfg, best_acc, best_pred = cfg, acc, dict(preds)
        if best_cfg is None:                     # degenerate fold
            best_cfg = GRID[0]
        chosen.append({**best_cfg, "inner_acc": round(best_acc, 4)})
        for i, P in (best_pred or {}).items():
            inner_oof_pred[i] = LABELS[int(np.argmax(P))]
        P_te = _fit_predict([texts[i] for i in train_idx],
                            [labels[i] for i in train_idx],
                            [texts[i] for i in test_idx], best_cfg)
        for row_pos, i in enumerate(test_idx):
            oof_probs[i] = P_te[row_pos]

    return {"oof_probs": oof_probs, "chosen": chosen, "inner_oof_pred": inner_oof_pred}


def oof_predictions(rows: list[dict], fold_of: list[int], k: int) -> np.ndarray:
    """Out-of-fold probability matrix for the 1-NN baseline."""
    texts = [r["utterance"] for r in rows]
    labels = [r["label"] for r in rows]
    n = len(rows)
    P = np.zeros((n, len(LABELS)))
    for f in range(k):
        tr = [i for i in range(n) if fold_of[i] != f]
        te = [i for i in range(n) if fold_of[i] == f]
        if not tr or not te:
            continue
        m = NearestNeighbour().fit([texts[i] for i in tr], [labels[i] for i in tr])
        Pte = m.predict_proba([texts[i] for i in te])
        for row_pos, i in enumerate(te):
            P[i] = Pte[row_pos]
    return P


def max_prob(P: np.ndarray) -> np.ndarray:
    return P.max(axis=1)


def margin(P: np.ndarray) -> np.ndarray:
    S = np.sort(P, axis=1)
    return S[:, -1] - S[:, -2]


# --------------------------------------------------------------------------- #
# cascade economics
# --------------------------------------------------------------------------- #

def cascade_row(coverage: float, precision: float) -> dict:
    blended = precision * coverage + HOSTED_ACC * (1.0 - coverage)
    cost_1k = HOSTED_COST_1K * (1.0 - coverage) + LOCAL_COST_PER_1K * coverage
    cost_1k_correct = cost_1k / blended if blended > 0 else float("inf")
    return {
        "coverage": coverage,
        "precision": precision,
        "blended_acc": blended,
        "cost_1k": cost_1k,
        "cost_1k_correct": cost_1k_correct,
        "wrong_commits_1k": (1.0 - precision) * coverage * 1000.0,
        "vs_baseline_acc": blended - HOSTED_ACC,
        "vs_baseline_cost_pct": (cost_1k / HOSTED_COST_1K - 1.0) * 100.0,
    }


def gate_curve(preds: list[str | None], labels: list[str], conf: np.ndarray,
               taus: list[float]) -> list[dict]:
    rows = []
    for tau in taus:
        sel = [i for i in range(len(labels)) if conf[i] >= tau and preds[i] is not None]
        n_sel = len(sel)
        if n_sel == 0:
            rows.append({"tau": tau, "coverage": 0.0, "n": 0, "precision": None,
                         "precision_lo": None, "precision_hi": None})
            continue
        hits = sum(1 for i in sel if preds[i] == labels[i])
        lo, hi = wilson(hits, n_sel)
        rows.append({
            "tau": tau,
            "coverage": n_sel / len(labels),
            "n": n_sel,
            "precision": hits / n_sel,
            "precision_lo": lo,
            "precision_hi": hi,
        })
    return rows


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def ci_str(lo: float, hi: float, digits: int = 0) -> str:
    return f"[{lo * 100:.{digits}f}, {hi * 100:.{digits}f}]"


def fmt_pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "   —  "
    return f"{x * 100:5.1f}%"


def leak_audit(rows: list[dict], fold_of: list[int], group_key: str = "group") -> dict:
    """How much of a row's own family leaks into its training portion.

    `straddle` is the share of rows whose group also appears in another fold —
    the direct measure of the paraphrase leak. `max_jaccard_any` is the closest
    string in the training portion regardless of label, which is what a
    boundary-pair row sees when its sibling is on the other side.
    """
    shingles = [_char_shingles(r["utterance"]) for r in rows]
    straddle = 0
    same_group_frac = 0
    max_any: list[float] = []
    for i, r in enumerate(rows):
        groups_elsewhere = {rows[j][group_key] for j in range(len(rows))
                            if fold_of[j] != fold_of[i]}
        if r[group_key] in groups_elsewhere:
            straddle += 1
        best = 0.0
        best_same = 0.0
        for j in range(len(rows)):
            if fold_of[j] == fold_of[i]:
                continue
            jac = _jaccard(shingles[i], shingles[j])
            best = max(best, jac)
            if rows[j]["label"] == r["label"]:
                best_same = max(best_same, jac)
        max_any.append(best)
        same_group_frac += 1 if best_same >= 0.25 else 0
    return {
        "group_straddle_frac": straddle / len(rows),
        "mean_max_jaccard_any": statistics.fmean(max_any),
        "median_max_jaccard_any": statistics.median(max_any),
        "frac_same_label_dup_ge_0.25": same_group_frac / len(rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "tests/data/utterances_v2.json"))
    ap.add_argument("--heldout", default=str(ROOT / "tests/data/heldout_phase3.json"))
    ap.add_argument("--llm-results",
                    default=str(ROOT / "bench/results/dell/v2-d1-3B-Q4.json"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--inner-folds", type=int, default=3)
    ap.add_argument("--json", default=str(ROOT / "bench/results/local_clf_v2.json"))
    ap.add_argument("--no-latency", action="store_true")
    args = ap.parse_args()

    rows = json.loads(Path(args.data).read_text())
    labels = [r["label"] for r in rows]
    n = len(rows)
    k = args.folds
    group_ids = build_groups(rows)
    for i, g in enumerate(group_ids):
        rows[i]["group"] = g
    fold_of = make_folds(group_ids, labels, k)

    report: dict = {
        "data": {"path": repo_rel(args.data), "n": n,
                 "bands": {b: sum(1 for r in rows if r["band"] == b) for b in BANDS}},
        "reference": {"hosted": HOSTED_NAME, "hosted_acc": HOSTED_ACC,
                      "hosted_cost_1k": HOSTED_COST_1K,
                      "hosted_cost_1k_correct": HOSTED_COST_1K_CORRECT,
                      "local_cost_1k": LOCAL_COST_PER_1K},
    }

    print("# local 8-way op classifier — grouped CV + cascade economics")
    band_counts = {b: sum(1 for r in rows if r["band"] == b) for b in BANDS}
    band_str = ", ".join(f"{b}={c}" for b, c in band_counts.items())
    print(f"data: {args.data} — {n} rows ({band_str})")
    print(f"groups: {len(set(group_ids))} unique "
          f"({len({g for g in group_ids if g.startswith('t3|')})} boundary families, "
          f"{len({g for g in group_ids if g.startswith('t12|')})} paraphrase clusters), "
          f"{k} folds")
    print()

    # ---- fold balance ----------------------------------------------------- #
    print("## fold balance (rows per label)")
    header = f"{'label':<16}" + "".join(f"{'f' + str(f):>6}" for f in range(k)) + "  total"
    print(header)
    for lab in LABELS:
        counts = [sum(1 for i, r in enumerate(rows)
                      if r["label"] == lab and fold_of[i] == f) for f in range(k)]
        print(f"{lab:<16}" + "".join(f"{c:>6d}" for c in counts) + f"  {sum(counts):>5d}")
    print(f"{'ALL':<16}"
          + "".join(f"{sum(1 for i in range(n) if fold_of[i] == f):>6d}" for f in range(k))
          + f"  {n:>5d}")
    report["fold_sizes"] = [sum(1 for i in range(n) if fold_of[i] == f) for f in range(k)]
    print()

    # ---- fit + predictions ------------------------------------------------ #
    t0 = time.perf_counter()
    cv = grouped_cv(rows, fold_of, k, args.inner_folds)
    fit_seconds = time.perf_counter() - t0
    P_lr = cv["oof_probs"]
    pred_lr = [LABELS[int(np.argmax(P_lr[i]))] for i in range(n)]
    P_nn = oof_predictions(rows, fold_of, k)
    pred_nn = [LABELS[int(np.argmax(P_nn[i]))] for i in range(n)]
    pred_rules: list[str | None] = []
    for r in rows:
        p, _ = rules_predict(r["utterance"])
        pred_rules.append(p)

    # random-split control, same pipeline, to size the leakage inflation
    fold_rand = random_folds(labels, k, seed=0)
    pred_lr_rand = [LABELS[int(np.argmax(row))]
                    for row in grouped_cv(rows, fold_rand, k, args.inner_folds)["oof_probs"]]

    print(f"## CV results — {k}-fold grouped, inner {args.inner_folds}-fold for "
          f"hyperparameters ({fit_seconds:.1f}s to fit all folds)")
    methods = [
        ("lr (softmax, nested)", pred_lr),
        ("1nn (char 3-5)", pred_nn),
        ("rules (keyword)", pred_rules),
    ]
    print(f"{'method':<24}{'pooled':>8}{'95% CI (Wilson)':>20}"
          + "".join(f"{b:>8}" for b in BANDS) + f"{'fold min..max':>16}")
    for name, preds in methods:
        hits = sum(1 for p, y in zip(preds, labels) if p == y)
        lo, hi = wilson(hits, n)
        folds = []
        for f in range(k):
            sel = [i for i in range(n) if fold_of[i] == f]
            folds.append(accuracy([preds[i] for i in sel], [labels[i] for i in sel]))
        band_cells = "".join(
            fmt_pct(accuracy([p for p, r in zip(preds, rows) if r["band"] == b],
                             [r["label"] for r in rows if r["band"] == b])).rjust(8)
            for b in BANDS
        )
        print(f"{name:<24}{fmt_pct(hits / n):>8}"
              f"{ci_str(lo, hi, 1):>20}{band_cells}"
              f"{f'{min(folds) * 100:.1f}..{max(folds) * 100:.1f}':>16}")
        report.setdefault("cv", {})[name] = {
            "pooled": hits / n, "wilson95": [lo, hi],
            "per_fold": folds,
            "per_band": {b: accuracy([p for p, r in zip(preds, rows) if r["band"] == b],
                                     [r["label"] for r in rows if r["band"] == b])
                         for b in BANDS},
        }
    lr_folds = report["cv"]["lr (softmax, nested)"]["per_fold"]
    print("   lr per fold: "
          + ", ".join(f"f{f} {v * 100:.1f}%" for f, v in enumerate(lr_folds))
          + f"   (fold spread {min(lr_folds) * 100:.0f}-{max(lr_folds) * 100:.0f}% "
          f"on 30-row folds)")
    abstain_rules = sum(1 for p in pred_rules if p is None)
    print(f"{'  (rules abstentions)':<24}{abstain_rules:>8d} of {n}"
          f"   — counted as misses above")
    print(f"{'random-split lr':<24}{fmt_pct(accuracy(pred_lr_rand, labels)):>8}"
          f"   <- same model, near-duplicate leak across folds")
    report["cv"]["random_split_lr_pooled"] = accuracy(pred_lr_rand, labels)
    print(f"{'hosted glm-5.3-flash':<24}{fmt_pct(HOSTED_ACC):>8}   <- reference, "
          f"one pass over the same {n} rows")
    print()

    # ---- per label -------------------------------------------------------- #
    print("## per-label accuracy")
    print(f"{'label':<16}{'n':>5}{'lr':>9}{'1nn':>9}{'rules':>9}{'3b':>9}{'delta lr-3b':>13}")
    llm_rows = None
    llm_path = Path(args.llm_results)
    m3b_pred: list[str | None] | None = None
    m3b_conf: np.ndarray | None = None
    if llm_path.exists():
        llm_rows = json.loads(llm_path.read_text())["per_row"]
        m3b_pred = [r["choice"] for r in llm_rows]
        m3b_conf = np.array([max(r["probabilities"].values()) if r.get("probabilities")
                             else float(r.get("conf") or 0.0) for r in llm_rows])
    for lab in LABELS:
        sel = [i for i in range(n) if labels[i] == lab]
        a_lr = accuracy([pred_lr[i] for i in sel], [labels[i] for i in sel])
        a_nn = accuracy([pred_nn[i] for i in sel], [labels[i] for i in sel])
        a_ru = accuracy([pred_rules[i] for i in sel], [labels[i] for i in sel])
        a_3b = (accuracy([m3b_pred[i] for i in sel], [labels[i] for i in sel])
                if m3b_pred else None)
        delta = f"{(a_lr - a_3b) * 100:+.0f} pts" if m3b_pred else "—"
        print(f"{lab:<16}{len(sel):>5}{fmt_pct(a_lr):>9}{fmt_pct(a_nn):>9}"
              f"{fmt_pct(a_ru):>9}{fmt_pct(a_3b):>9}{delta:>13}")
    print("lr chosen configs per fold: "
          + ", ".join(f"md{c['min_df']}/l2={c['l2']:g}(inner {c['inner_acc']:.2f})"
                      for c in cv["chosen"]))
    report["chosen_configs"] = cv["chosen"]
    report["per_label"] = {
        lab: {
            "n": sum(1 for x in labels if x == lab),
            "lr": accuracy([pred_lr[i] for i in range(n) if labels[i] == lab],
                           [lab] * sum(1 for x in labels if x == lab)),
            "nn1": accuracy([pred_nn[i] for i in range(n) if labels[i] == lab],
                            [lab] * sum(1 for x in labels if x == lab)),
            "rules": accuracy([pred_rules[i] for i in range(n) if labels[i] == lab],
                              [lab] * sum(1 for x in labels if x == lab)),
            "3b": (accuracy([m3b_pred[i] for i in range(n) if labels[i] == lab],
                            [lab] * sum(1 for x in labels if x == lab))
                   if m3b_pred else None),
        }
        for lab in LABELS
    }
    print()

    # ---- leakage audit ---------------------------------------------------- #
    leak_g = leak_audit(rows, fold_of)
    leak_r = leak_audit(rows, fold_rand)
    print("## leakage audit — does a row's family show up in its own training portion?")
    print(f"{'split':<10}{'group straddle':>16}{'mean max jaccard':>18}"
          f"{'median':>9}{'same-label dup >=0.25':>23}")
    for name, lk in (("grouped", leak_g), ("random", leak_r)):
        print(f"{name:<10}{lk['group_straddle_frac'] * 100:>15.1f}%"
              f"{lk['mean_max_jaccard_any']:>18.2f}"
              f"{lk['median_max_jaccard_any']:>9.2f}"
              f"{lk['frac_same_label_dup_ge_0.25'] * 100:>22.0f}%")
    print("   (straddle is 0% by construction on the grouped split; the drop from")
    print("   random to grouped accuracy above is what that 100%-vs-0% difference buys)")
    report["leak_audit"] = {"grouped": leak_g, "random": leak_r}
    print()

    # ---- coverage / precision -------------------------------------------- #
    conf_lr = max_prob(P_lr)
    marg_lr = margin(P_lr)
    taus = [0.0, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    curve = gate_curve(pred_lr, labels, conf_lr, taus)
    print("## coverage / precision — gate on max probability (lr, out-of-fold)")
    print(f"{'tau':>6}{'n':>5}{'coverage':>10}{'precision':>11}"
          f"{'wilson95':>18}{'blended acc':>13}{'cost/1k':>10}{'cost/1k ok':>12}")
    print(f"{'':>6}{'':>5}{'':>10}{'':>11}{'':>18}{'(cascade)':>13}{'$':>10}{'$':>12}")
    for row in curve:
        if row["n"] == 0:
            print(f"{row['tau']:>6.2f}{0:>5d}{'0.0%':>10}{'—':>11}{'—':>18}")
            continue
        casc = cascade_row(row["coverage"], row["precision"])
        print(f"{row['tau']:>6.2f}{row['n']:>5d}{fmt_pct(row['coverage']):>10}"
              f"{fmt_pct(row['precision']):>11}"
              f"{ci_str(row['precision_lo'], row['precision_hi']):>18}"
              f"{fmt_pct(casc['blended_acc']):>13}"
              f"{casc['cost_1k']:>10.4f}{casc['cost_1k_correct']:>12.4f}")
        report.setdefault("gate_curve_maxprob", []).append({**row, **casc})
    # ---- the frontier as coverage, not as tau ---------------------------- #
    order = list(np.argsort(-conf_lr, kind="stable"))
    frontier: list[dict] = []
    print()
    print("## the frontier, indexed by local coverage (rows taken highest-confidence first)")
    print("   this is the table that answers 'how much can we get away with'; it does")
    print("   not depend on the confidence being calibrated, only ordered.")
    print(f"{'cover':>7}{'n':>5}{'precision':>11}{'wilson95':>18}{'blended':>9}"
          f"{'Δacc':>8}{'$/1k dec':>10}{'$/1k ok':>10}{'saved':>8}{'wrong/1k':>10}")
    for q in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70,
              0.80, 0.90, 1.00):
        m = max(1, int(round(q * n)))
        sel = order[:m]
        hits = sum(1 for i in sel if pred_lr[i] == labels[i])
        lo, hi = wilson(hits, m)
        casc = cascade_row(m / n, hits / m)
        frontier.append({"target_coverage": q, "n": m, "coverage": m / n,
                         "precision": hits / m, "precision_lo": lo,
                         "precision_hi": hi, **casc})
        print(f"{fmt_pct(m / n):>7}{m:>5}{fmt_pct(hits / m):>11}"
              f"{ci_str(lo, hi):>18}{fmt_pct(casc['blended_acc']):>9}"
              f"{casc['vs_baseline_acc'] * 100:>+8.1f}{casc['cost_1k']:>10.4f}"
              f"{casc['cost_1k_correct']:>10.4f}"
              f"{(1 - casc['cost_1k'] / HOSTED_COST_1K) * 100:>7.0f}%"
              f"{casc['wrong_commits_1k']:>10.1f}")
    report["frontier_by_coverage"] = frontier

    print()
    print("## accuracy budget ladder — how much local coverage a stated loss buys")
    print("   Δacc = blended - hosted. Coverage is the largest frontier cell inside")
    print("   the budget; precision is what that cell measured on committed rows.")
    print(f"{'budget (pts)':>13}{'max cover':>11}{'n':>5}{'precision':>11}"
          f"{'blended':>9}{'Δacc':>7}{'cost saved':>11}{'$/1k ok':>10}")
    ladder = []
    for budget in (0.5, 1.0, 2.0, 4.0, 8.0):
        best = frontier[0]
        for row in frontier:
            if row["vs_baseline_acc"] >= -budget / 100.0:
                best = row
        saved = (1 - best["cost_1k"] / HOSTED_COST_1K) * 100
        ladder.append({"budget_pts": budget, **best})
        print(f"{budget:>13.1f}{fmt_pct(best['coverage']):>11}{best['n']:>5}"
              f"{fmt_pct(best['precision']):>11}{fmt_pct(best['blended_acc']):>9}"
              f"{best['vs_baseline_acc'] * 100:>+7.1f}{saved:>10.0f}%"
              f"{best['cost_1k_correct']:>10.4f}")
    report["budget_ladder"] = ladder
    op = ladder[2]                       # the 2-point-budget row
    print()
    print(f"   per band at the 2-point budget ({op['coverage'] * 100:.1f}% coverage, "
          f"{op['n']} rows committed overall), because a pooled number hides this:")
    print(f"   {'band':<6}{'rows':>6}{'committed':>11}{'cov':>8}{'precision':>11}")
    top = set(order[:op["n"]])
    for b in BANDS:
        band_idx = [i for i in range(n) if rows[i]["band"] == b]
        committed = [i for i in band_idx if i in top]
        hits = sum(1 for i in committed if pred_lr[i] == labels[i])
        print(f"   {b:<6}{len(band_idx):>6}{len(committed):>11}"
              f"{fmt_pct(len(committed) / len(band_idx)):>8}"
              + (f"{fmt_pct(hits / len(committed)):>11}" if committed else f"{'—':>11}"))
        report.setdefault("per_band_at_op", {})[b] = {
            "rows": len(band_idx), "committed": len(committed),
            "coverage": len(committed) / len(band_idx),
            "precision": (hits / len(committed)) if committed else None,
        }
    report["operating_point"] = op
    print("   (committed-row accuracy IS the precision column: a committed row is")
    print("   never seen by the hosted model, so a wrong commit cannot be corrected)")
    print()

    report["gate_curve_margin"] = gate_curve(pred_lr, labels, marg_lr, taus)
    print()
    print("  margin gate (top1-top2) for comparison:")
    print(f"  {'tau':>5}{'n':>5}{'coverage':>10}{'precision':>11}")
    for row in report["gate_curve_margin"]:
        if row["n"]:
            print(f"  {row['tau']:>5.2f}{row['n']:>5d}{fmt_pct(row['coverage']):>10}"
                  f"{fmt_pct(row['precision']):>11}")
    print()

    # ---- the 3B as the gate, measured on the same rows -------------------- #
    if m3b_pred is not None and m3b_conf is not None:
        curve3b = gate_curve(m3b_pred, labels, m3b_conf, taus)
        print("## coverage / precision — gate on max probability (d1-3B Q4_K_M, "
              "measured one pass)")
        print("   v2's t3 rows were built from this model's misses, so this "
              "column is pessimistic by construction.")
        print(f"{'tau':>6}{'n':>5}{'coverage':>10}{'precision':>11}{'wilson95':>18}"
              f"{'blended acc':>13}{'$/1k dec':>10}{'saved':>8}")
        for row in curve3b:
            if row["n"] == 0:
                continue
            casc = cascade_row(row["coverage"], row["precision"])
            print(f"{row['tau']:>6.2f}{row['n']:>5d}{fmt_pct(row['coverage']):>10}"
                  f"{fmt_pct(row['precision']):>11}"
                  f"{ci_str(row['precision_lo'], row['precision_hi']):>18}"
                  f"{fmt_pct(casc['blended_acc']):>13}"
                  f"{casc['cost_1k']:>10.4f}"
                  f"{(1 - casc['cost_1k'] / HOSTED_COST_1K) * 100:>7.0f}%")
            report.setdefault("gate_curve_3b", []).append({**row, **casc})
        print("   confidence quality at MATCHED coverage (same rows sorted by each")
        print("   model's own confidence; a gate is only useful if this lifts)")
        print(f"   {'coverage':>9}{'n':>5}{'lr precision':>14}{'3b precision':>14}"
              f"{'rules precision':>16}")
        matched = []
        for q in (0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00):
            m = max(1, int(round(q * n)))
            cell = {"coverage": m / n, "n": m}
            for tag, preds, cf in (
                ("lr", pred_lr, conf_lr),
                ("3b", m3b_pred, m3b_conf),
                ("rules", pred_rules, None),
            ):
                if tag == "rules":
                    # rules carry no confidence: use its score margin as the ranking
                    sc = [max(rules_predict(r["utterance"])[1].values()) for r in rows]
                    ordr = list(np.argsort(-np.array(sc), kind="stable"))
                else:
                    ordr = list(np.argsort(-cf, kind="stable"))
                sel = ordr[:m]
                # same label multiset, so compare on the same number of rows only
                hits = sum(1 for i in sel if preds[i] == labels[i])
                cell[tag] = hits / m
            matched.append(cell)
            print(f"   {fmt_pct(cell['coverage']):>9}{m:>5}"
                  f"{fmt_pct(cell['lr']):>14}{fmt_pct(cell['3b']):>14}"
                  f"{fmt_pct(cell['rules']):>16}")
        report["matched_coverage_precision"] = matched
        print()

    # ---- cascade economics ---------------------------------------------- #
    print("## cascade economics — local commits at tau, else hosted")
    print(f"   hosted {HOSTED_NAME}: acc {HOSTED_ACC * 100:.1f}%, "
          f"${HOSTED_COST_1K:.4f}/1k decisions, ${HOSTED_COST_1K_CORRECT:.4f}/1k correct, "
          f"p50 {HOSTED_P50_MS:.0f} ms")
    print("   local inference cost treated as $0/1k (on-box CPU, measured below)")
    print(f"{'tau':>6}{'cover':>8}{'prec':>8}{'blended':>9}{'Δacc':>8}"
          f"{'$/1k dec':>10}{'$/1k ok':>10}{'saved':>8}{'wrong/1k':>10}{'net vs hosted':>15}")
    print(f"{'none':>6}{'0.0%':>8}{'—':>8}{fmt_pct(HOSTED_ACC):>9}{'—':>8}"
          f"{HOSTED_COST_1K:>10.4f}{HOSTED_COST_1K_CORRECT:>10.4f}{'0%':>8}"
          f"{'0.0':>10}{'baseline':>15}")
    inner_rows = [r for r in report.get("gate_curve_maxprob", []) if r["n"] > 0]
    for row in inner_rows:
        casc = cascade_row(row["coverage"], row["precision"])
        saved = (1 - casc["cost_1k"] / HOSTED_COST_1K) * 100
        verdict = f"{casc['vs_baseline_acc'] * 100:+.1f} pts"
        print(f"{row['tau']:>6.2f}{fmt_pct(row['coverage']):>8}"
              f"{fmt_pct(row['precision']):>8}{fmt_pct(casc['blended_acc']):>9}"
              f"{casc['vs_baseline_acc'] * 100:>+8.1f}"
              f"{casc['cost_1k']:>10.4f}{casc['cost_1k_correct']:>10.4f}"
              f"{saved:>7.0f}%{casc['wrong_commits_1k']:>10.1f}{verdict:>15}")
    print()
    print("break-even, algebraically: blended_acc - hosted_acc = coverage * (prec - "
          f"{HOSTED_ACC:.3f})")
    print(f"  -> accuracy break-even is local precision = {HOSTED_ACC * 100:.1f}% "
          "(the hosted accuracy). Above it every commit adds accuracy; below it "
          "every commit removes accuracy.")
    print("  -> with local inference free, cost is strictly monotone improving in "
          "coverage: any commit saves money. The cost side never sets the limit; "
          "precision does.")
    best_cov = 0.0
    for row in inner_rows:
        if row["precision"] >= HOSTED_ACC and row["coverage"] > best_cov:
            best_cov = row["coverage"]
    print(f"  -> largest measured coverage whose committed precision still >= "
          f"{HOSTED_ACC * 100:.1f}%: {best_cov * 100:.1f}% "
          f"(cell-level; the CI on precision is wide at these n)")
    report["breakeven"] = {
        "precision_breakeven": HOSTED_ACC,
        "max_coverage_precision_ge_hosted": best_cov,
    }
    print()
    print("   how many committed rows it takes to CERTIFY a local gate beats the")
    print(f"   hosted {HOSTED_ACC * 100:.1f}% (one-sided binomial, alpha 0.05, power 0.8):")
    print(f"   {'local precision':>16}{'committed rows needed':>24}")
    for p_t in (0.96, 0.97, 0.98, 0.99, 0.995):
        need = rows_to_certify(p_t)
        print(f"   {p_t * 100:>15.1f}%{(str(need) if need else '—'):>24}")
    report["rows_to_certify"] = {
        f"{p_t:.3f}": rows_to_certify(p_t) for p_t in (0.96, 0.97, 0.98, 0.99, 0.995)
    }
    _live_n = 400
    print(f"   at the measured operating point ({op['coverage'] * 100:.1f}% coverage) "
          f"{_live_n} unselected live rows would commit "
          f"~{round(op['coverage'] * _live_n)} of them — see whether that clears the "
          f"row above before trusting the gate.")
    print()

    # ---- per-class hybrid ------------------------------------------------ #
    print("## per-class hybrid expert routing")
    print("   experts: lr (OOF), 3b (measured). Selection is per class, which is a")
    print("   weaker requirement than a per-row trust gate.")
    static_map = {
        "filesys_read": "lr",
        "filesys_glob": "3b",
        "filesys_grep": "lr",
        "decide_pick": "3b",
        "decide_extract": "lr",
        "web_search": "3b",
        "web_fetch": "3b",
        "reddit_digest": "3b",
    }
    # A per-class map answers "which expert should serve this row". At inference
    # the class is the unknown being predicted, so the map must be indexed by the
    # class the cheap model PREDICTS (`pred_lr`, an out-of-fold prediction) — never
    # by `labels[i]`. Indexing it by the row's own label is an oracle: it requires
    # the answer to look up the answer, and no deployed system can run it. Both are
    # kept and both are reported, because the difference between them is the
    # quantity that matters. Measured at n=150 (5-fold × 5 seeds, cluster
    # bootstrap) in `bench/router_policy.py`: 84.5% oracle against 79.3% deployed.
    def _dispatch(choice_map: dict[str, str], key) -> list[str | None]:
        return [pred_lr[i] if choice_map[key[i]] == "lr" else m3b_pred[i]
                for i in range(n)]

    acc_static = acc_static_routed = 0.0
    acc_nested = acc_oracle = 0.0
    if m3b_pred is not None:
        pred_oracle: list[str | None] = [None] * n
        pred_routed: list[str | None] = [None] * n
        for f in range(k):
            train_idx = [i for i in range(n) if fold_of[i] != f]
            test_idx = [i for i in range(n) if fold_of[i] == f]
            choice_map: dict[str, str] = {}
            for lab in LABELS:
                sel = [i for i in train_idx if labels[i] == lab]
                if not sel:
                    choice_map[lab] = "3b"
                    continue
                lr_correct = sum(1 for i in sel
                                 if cv["inner_oof_pred"][i] == labels[i])
                b3_correct = sum(1 for i in sel if m3b_pred[i] == labels[i])
                choice_map[lab] = "lr" if lr_correct > b3_correct else "3b"
            for i in test_idx:
                routed = _dispatch(choice_map, pred_lr)
                oracle = _dispatch(choice_map, labels)
                pred_routed[i] = routed[i]
                pred_oracle[i] = oracle[i]
        acc_nested = accuracy(pred_routed, labels)
        acc_oracle = accuracy(pred_oracle, labels)
        acc_static_routed = accuracy(_dispatch(static_map, pred_lr), labels)
        acc_static = accuracy(_dispatch(static_map, labels), labels)

    # optimistic variant: select on the pooled OOF numbers themselves
    acc_opt = 0.0
    if m3b_pred is not None:
        per_lab = report["per_label"]
        opt_map = {lab: ("lr" if per_lab[lab]["lr"] >= (per_lab[lab]["3b"] or 0)
                         else "3b") for lab in LABELS}
        acc_opt = accuracy(_dispatch(opt_map, labels), labels)
        print(f"   static table, dispatched on PREDICTED class (deployable): "
              f"{acc_static_routed * 100:.1f}%")
        print(f"   static table, dispatched on TRUE label (oracle):          "
              f"{acc_static * 100:.1f}%")
        print(f"   per-class map from training folds, PREDICTED class "
              f"(deployable): {acc_nested * 100:.1f}%")
        print(f"   per-class map from training folds, TRUE label (oracle, "
              f"not deployable): {acc_oracle * 100:.1f}%")
        print(f"   per-class map from pooled OOF, TRUE label (optimistic ub):  "
              f"{acc_opt * 100:.1f}%")
        print(f"   lr alone {accuracy(pred_lr, labels) * 100:.1f}%   "
              f"3b alone {accuracy(m3b_pred, labels) * 100:.1f}%   "
              f"hosted {HOSTED_ACC * 100:.1f}%")
        union_lr_3b = sum(1 for i in range(n)
                          if pred_lr[i] == labels[i] or m3b_pred[i] == labels[i]) / n
        union_all = sum(1 for i in range(n)
                        if pred_lr[i] == labels[i] or m3b_pred[i] == labels[i]
                        or pred_rules[i] == labels[i]) / n
        print(f"   oracle union lr|3b {union_lr_3b * 100:.1f}%  "
              f"lr|3b|rules {union_all * 100:.1f}%  "
              f"<- ceiling for any per-class choice; the deployable map keeps "
              f"{(acc_nested - 0.740) / (union_lr_3b - 0.740) * 100:.0f}% "
              f"of the lr|3b gap, the oracle {(acc_oracle - 0.740) / (union_lr_3b - 0.740) * 100:.0f}%")
        local_classes = [lab for lab in LABELS if opt_map[lab] == "lr"]
        local_rows = [i for i in range(n) if labels[i] in local_classes]
        cov = len(local_rows) / n
        prec = accuracy([pred_lr[i] for i in local_rows],
                        [labels[i] for i in local_rows])
        casc = cascade_row(cov, prec)
        print(f"   as a gate, class-shaped: coverage {cov * 100:.1f}% "
              f"({len(local_rows)} rows), precision {prec * 100:.1f}%, "
              f"blended {casc['blended_acc'] * 100:.1f}%, "
              f"${casc['cost_1k']:.4f}/1k decisions")
        report["hybrid"] = {
            "static_map": static_map,
            "routed_selection_acc": acc_nested,
            "oracle_selection_acc": acc_oracle,
            "static_routed_acc": acc_static_routed,
            "static_oracle_acc": acc_static,
            "optimistic_selection_acc": acc_opt,
            "routed_note": "dispatch key is the predicted class (pred_lr); deployable",
            "oracle_note": "dispatch key is the row's true label; NOT deployable",
            "lr_alone": accuracy(pred_lr, labels),
            "3b_alone": accuracy(m3b_pred, labels),
            "optimistic_map": opt_map,
            "union_lr_3b": union_lr_3b, "union_all_three": union_all,
            "class_shaped_coverage": cov, "class_shaped_precision": prec,
            "class_shaped_cascade": casc,
        }
    print()

    # ---- latency --------------------------------------------------------- #
    if not args.no_latency:
        feat = Featurizer(min_df=cv["chosen"][0]["min_df"]).fit(
            [r["utterance"] for r in rows])
        X = feat.transform([r["utterance"] for r in rows])
        model = SoftmaxRegression(l2=cv["chosen"][0]["l2"]).fit(
            X, [LABEL_INDEX[y] for y in labels])
        texts = [r["utterance"] for r in rows]
        for t in texts[:20]:
            m1 = SoftmaxRegression(l2=cv["chosen"][0]["l2"])
            m1.W = model.W
            m1.predict_proba(feat.transform([t]))
        single = []
        for _ in range(400):
            t = texts[random.randrange(len(texts))]
            start = time.perf_counter()
            model.predict_proba(feat.transform([t]))
            single.append((time.perf_counter() - start) * 1000.0)
        batch = []
        for _ in range(30):
            start = time.perf_counter()
            model.predict_proba(feat.transform(texts))
            batch.append((time.perf_counter() - start) * 1000.0)
        single.sort()
        p50 = single[len(single) // 2]
        p95 = single[int(len(single) * 0.95)]
        per_row = statistics.fmean(batch) / len(texts)
        print("## latency (indicative — 12-thread laptop with other load on it)")
        print(f"   features: {X.shape[1]} dims, {int((X > 0).sum())} non-zeros "
              f"in the {len(texts)}-row corpus")
        print(f"   single utterance (tokenise + tf-idf + predict): "
              f"p50 {p50:.2f} ms, p95 {p95:.2f} ms over 400 draws")
        print(f"   batch of {len(texts)} (amortised per row): {per_row:.3f} ms")
        print(f"   hosted p50 for the same decision: {HOSTED_P50_MS:.0f} ms "
              f"({HOSTED_P50_MS / p50:.0f}x slower than one local row)")
        report["latency_ms"] = {
            "single_p50": p50, "single_p95": p95, "batch_per_row": per_row,
            "n_rows": len(texts), "n_features": int(X.shape[1]),
            "note": "indicative: shared 12-thread laptop, other load present",
        }
        print()

    # ---- held-out -------------------------------------------------------- #
    hp = Path(args.heldout)
    hold = None
    if hp.exists():
        candidate = json.loads(hp.read_text())
        # A held-out set is only scoreable if it carries this file's 8-way op
        # labels. `heldout_phase3.json` is a *shape* set (76 rows keyed
        # expect/family), so the resolved name is a different task's data and
        # scoring it would print a number about shape classification.
        if candidate and all(r.get("label") in LABEL_INDEX for r in candidate):
            hold = candidate
    if hold is not None:
        feat = Featurizer(min_df=cv["chosen"][0]["min_df"]).fit(
            [r["utterance"] for r in rows])
        model = SoftmaxRegression(l2=cv["chosen"][0]["l2"]).fit(
            feat.transform([r["utterance"] for r in rows]),
            [LABEL_INDEX[y] for y in labels])
        hp_probs = model.predict_proba(feat.transform([r["utterance"] for r in hold]))
        hp_pred = [LABELS[int(np.argmax(p))] for p in hp_probs]
        hold_labels = [r["label"] for r in hold]
        acc_hold = accuracy(hp_pred, hold_labels)
        print(f"## held-out — {hp.name} ({len(hold)} rows), evaluated, never trained on")
        print(f"   accuracy {acc_hold * 100:.1f}% "
              f"(trained on all {n} v2 rows; not tuned against this set)")
        report["heldout"] = {"path": repo_rel(hp), "n": len(hold),
                             "accuracy": acc_hold}
    else:
        reason = ("file absent" if not hp.exists()
                  else f"{hp.name} carries no 8-way op labels")
        print(f"## held-out — {reason}; nothing was used for evaluation beyond "
              f"the v2 rows above")
        report["heldout"] = {"path": repo_rel(hp), "used": False, "reason": reason}
    print()

    # ---- honesty --------------------------------------------------------- #
    print("## what these numbers do not show")
    print("   Caveats, worst first:")
    print("   1. v2 is 150 rows built adversarially — t3 is boundary pairs and every")
    print("      3B miss got >=3 paraphrases. Grouped CV removes the near-duplicate")
    print("      leak (random 5-fold scores 69.3% vs 61.3% grouped for the same")
    print("      model: 8 points of pure leak) but it cannot model a phrasing nobody")
    print("      anticipated. Treat 61.3% as a ceiling on live performance.")
    print("   2. The rules row (78.7%) is CONTAMINATED and is not evidence: the cue")
    print("      table was written after reading these 150 utterances. Its edge over")
    print("      the fitted model is partly the author having seen the test data.")
    print("      It is in the table as a reference point for what a hand table can")
    print("      reach on the set it was written against, and for nothing else.")
    print("   3. Asymmetry in the comparison: the hosted 95.3% was measured on these")
    print("      same rows, but v2 was built against the 3B's misses rather than the")
    print("      hosted model's, so the hosted number is the least flattered figure")
    print("      here while our local figures are the most flattered (feature design,")
    print("      hyperparameter grid, and cue table were all chosen with v2 in view).")
    print("   4. Per-class hybrid runs on 15-24 rows per class: a class-level")
    print("      difference under ~15 points is noise, and 3 of 8 classes flip with a")
    print("      single row. The static prior table and the fitted selection agreeing")
    print("      on 8/8 classes is the only reason to trust the shape of that result.")
    print("   5. Local inference is priced at $0 because it is on-box and sub-ms. It")
    print("      is not free of failure: a committed wrong op runs a wrong tool call")
    print("      on the user's machine, and no column above prices that in.")
    print("   6. Nothing here is live traffic. Every number is a 150-row artifact.")
    print()
    print("   What survives the caveat, and what does not:")
    print("   SURVIVES — the break-even algebra: a local commit only improves blended")
    print("     accuracy when committed-row precision beats the hosted accuracy, and")
    print("     since local inference is free the cost constraint never binds first.")
    print("     This is arithmetic, not measurement; it holds for any dataset.")
    print("   SURVIVES — the shape of the error profile: lexical features carry")
    print("     decide_extract (+54 pts over the 3B) and filesys_grep (+14), and are")
    print("     far behind on web_search (-61), web_fetch (-47), reddit (-40), pick")
    print("     (-22). That direction reproduces the prior measurement and is what")
    print("     makes per-class routing the right device rather than a per-row gate.")
    print("   SURVIVES — confidence quality ordering: a model's max-probability")
    print("     discriminates its own correctness (3B: 74% pooled -> 97.6% at tau")
    print("     0.90) while the n-gram model's barely does (61.3% -> 86.7% at 10%")
    print("     coverage). Gating on the model, not on the lexical classifier, is")
    print("     the only version of this that has a margin.")
    print("   DOES NOT SURVIVE — any specific coverage number. 14.7% at a 2-point")
    print("     accuracy budget is a property of 150 rows written by one author to")
    print("     break a specific 3B; live traffic will move both the coverage and")
    print("     the precision, and the move is unmodellable from here.")
    print("   DOES NOT SURVIVE — the 1-NN and rules columns as 'what local costs'.")
    print("     They are three different local methods; the fitted model at 61.3% is")
    print("     the one that generalises, the hand table at 78.7% is the one that")
    print("     memorised this set.")
    print()

    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
