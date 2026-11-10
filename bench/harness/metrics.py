"""metrics — every number the harness will print, as pure functions over records.

Two conventions are stated here once so a reader never has to guess:

* **conf** of a row is ``max(probabilities.values())``. A row without probabilities is
  excluded from every calibration number and the exclusion is counted (``n_scored``),
  never silently dropped.
* **precision** on the reject curve means accuracy on the *accepted* set, which is the
  quantity a confidence gate actually buys. ``coverage@precision=0.9`` is the largest
  fraction of rows that can be answered while the answered set stays at or above 0.9.

McNemar is the exact two-sided binomial on the discordant pairs, the same test the repo has
used since `bench/results/p1-diagnose`; `tests` cross-check this implementation against
`bench/ti_bench/stats.mcnemar_exact` so the two cannot drift.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from math import comb
from typing import Callable, Iterable, Sequence

from .protocol import Record, score
from .rows import Row

Z95 = 1.959963984540054
EPS = 1e-12


# ------------------------------------------------------------------ basics

def accuracy(recs: Sequence[Record]) -> dict:
    n = len(recs)
    correct = sum(1 for r in recs if r.correct)
    acc = correct / n if n else 0.0
    return {"n": n, "correct": correct, "accuracy": acc, "wilson95": wilson_ci(correct, n)}


def wilson_ci(correct: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar: binomial on the discordant pairs, doubled one tail."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def paired(reference: Sequence[Record], treatment: Sequence[Record]) -> dict:
    """Fixed / broke / kept over the ids both arms scored, plus exact McNemar."""
    a = {r.id: r for r in reference}
    b = {r.id: r for r in treatment}
    ids = sorted(set(a) & set(b))
    fixed = [i for i in ids if not a[i].correct and b[i].correct]
    broke = [i for i in ids if a[i].correct and not b[i].correct]
    return {"n": len(ids), "fixed": len(fixed), "broke": len(broke),
            "p": mcnemar_exact(len(broke), len(fixed)),
            "reference_correct": sum(1 for i in ids if a[i].correct),
            "treatment_correct": sum(1 for i in ids if b[i].correct),
            "fixed_ids": fixed, "broke_ids": broke,
            "only_in_reference": sorted(set(a) - set(b)),
            "only_in_treatment": sorted(set(b) - set(a))}


def a_a_flips(first: Sequence[Record], second: Sequence[Record]) -> dict:
    """The floor: the reference arm run twice. Choice flips and correctness flips."""
    a = {r.id: r for r in first}
    b = {r.id: r for r in second}
    ids = sorted(set(a) & set(b))
    flips = [i for i in ids if a[i].choice != b[i].choice]
    flips_correct = [i for i in ids if a[i].correct != b[i].correct]
    return {"n": len(ids), "choice_flips": len(flips),
            "correctness_flips": len(flips_correct),
            "rate": len(flips) / len(ids) if ids else 0.0,
            "flip_ids": flips}


# ------------------------------------------------------------------ the ledger

def ledger(reference: Sequence[Record], treatment: Sequence[Record]) -> dict:
    """The per-label ledger, the unit that is actually edited.

    Booked by the *reference* label, so a row that moves from `web_search` to
    `filesys_grep` is charged to `web_search` — the option that lost it — which is how the
    grep arm's damage was attributed.
    """
    a = {r.id: r for r in reference}
    b = {r.id: r for r in treatment}
    ids = sorted(set(a) & set(b))
    out: dict[str, dict] = {}
    for label in sorted({a[i].label for i in ids}):
        sel = [i for i in ids if a[i].label == label]
        base = sum(1 for i in sel if a[i].correct)
        treat = sum(1 for i in sel if b[i].correct)
        fixed = sum(1 for i in sel if not a[i].correct and b[i].correct)
        broke = sum(1 for i in sel if a[i].correct and not b[i].correct)
        out[label] = {"n": len(sel), "base": base, "treat": treat,
                      "fixed": fixed, "broke": broke, "net": treat - base}
    return {"by_label": out,
            "net": sum(v["net"] for v in out.values()),
            "fixed": sum(v["fixed"] for v in out.values()),
            "broke": sum(v["broke"] for v in out.values())}


def ledger_net(led: dict, labels: Iterable[str]) -> int:
    """Net rows over a named subset of labels — 'untouched' is this with the edited ones out."""
    keep = set(labels)
    return sum(v["net"] for k, v in led["by_label"].items() if k in keep)


# ------------------------------------------------------------------ calibration

def _conf(rec: Record) -> float | None:
    if not rec.probabilities:
        return None
    return max(rec.probabilities.values())


def calibration(recs: Sequence[Record], bins: int = 10) -> dict | None:
    """ECE (equal-width, population weighted), multi-class Brier, and NLL."""
    scored = [(r, _conf(r)) for r in recs]
    scored = [(r, c) for r, c in scored if c is not None]
    if not scored:
        return None
    n = len(scored)
    bucket: dict[int, list[tuple[float, bool]]] = {}
    for r, c in scored:
        i = min(bins - 1, int(c * bins))
        bucket.setdefault(i, []).append((c, r.correct))
    ece = 0.0
    for i in range(bins):
        rows = bucket.get(i, [])
        if not rows:
            continue
        mean_conf = sum(c for c, _ in rows) / len(rows)
        mean_acc = sum(1 for _, ok in rows if ok) / len(rows)
        ece += (len(rows) / n) * abs(mean_acc - mean_conf)
    brier = 0.0
    nll = 0.0
    for r, _c in scored:
        probs = r.probabilities or {}
        y = r.label
        brier += sum((p - (1.0 if k == y else 0.0)) ** 2 for k, p in probs.items())
        nll -= math.log(max(probs.get(y, 0.0), EPS))
    return {"n_scored": n, "n_skipped": len(recs) - n,
            "ece": ece,
            "brier": brier / n, "nll": nll / n}


# ------------------------------------------------------------------ reject curve

def reject_curve(recs: Sequence[Record], step: float = 0.05) -> list[dict]:
    """Coverage and accuracy as the confidence gate sweeps, highest threshold first."""
    scored = [(r, _conf(r)) for r in recs]
    scored = [(r, c) for r, c in scored if c is not None]
    if not scored:
        return []
    out = []
    t = 0.0
    while t <= 1.0 + 1e-9:
        kept = [r for r, c in scored if c >= t]
        n_kept = len(kept)
        out.append({"threshold": round(t, 3), "coverage": n_kept / len(scored),
                    "precision": (sum(1 for r in kept if r.correct) / n_kept) if n_kept else None,
                    "n": n_kept})
        t += step
    return out


def coverage_at_precision(recs: Sequence[Record], target: float = 0.9) -> dict:
    """Largest accepted fraction whose accepted set stays at or above ``target`` accuracy.

    The set is defined *by the reported threshold*: every row with confidence at or above it
    is accepted, ties included, and the coverage and precision returned are that set's. A
    threshold read off the k-th confidence while the set is the first k rows would describe
    two different operating points, which is the defect this replaced.
    """
    scored = [(r, _conf(r)) for r in recs]
    scored = [(r, c) for r, c in scored if c is not None]
    if not scored:
        return {"coverage": None, "precision": None, "threshold": None, "n_accepted": 0}
    total = len(scored)
    best = {"coverage": 0.0, "precision": None, "threshold": None, "n_accepted": 0}
    for threshold in sorted({c for _, c in scored}, reverse=True):
        accepted = [r for r, c in scored if c >= threshold]
        precision = sum(1 for r in accepted if r.correct) / len(accepted)
        if precision >= target and len(accepted) > best["n_accepted"]:
            best = {"coverage": len(accepted) / total, "precision": precision,
                    "threshold": threshold, "n_accepted": len(accepted)}
    return best


# ------------------------------------------------------------------ latency

def latency(recs: Sequence[Record]) -> dict:
    ms = sorted(r.latency_ms for r in recs)
    if not ms:
        return {"n": 0, "p50": None, "p95": None, "mean": None}
    def pct(p: float) -> float:
        if len(ms) == 1:
            return ms[0]
        idx = min(len(ms) - 1, max(0, int(round(p * (len(ms) - 1)))))
        return ms[idx]
    return {"n": len(ms), "p50": pct(0.50), "p95": pct(0.95),
            "mean": sum(ms) / len(ms), "max": ms[-1]}


# ------------------------------------------------------------------ grouped CV

def assign_folds(groups: Iterable[str], folds: int = 5, seed: int = 0,
                 labels: Iterable[tuple[str, str]] | None = None,
                 stratify: bool = True) -> dict[str, int]:
    """Stable group -> fold map. Hashed, not random: the same corpus always folds the same way.

    With ``labels`` (``(group, label)`` pairs) and ``stratify`` the assignment balances each fold
    by label: groups are placed largest-first into whichever fold currently holds the fewest rows
    of their labels. That is the difference between a measurement and a wall — an unstratified
    map can put every row of a label in one fold, so the training side never sees that label and
    the classifier is asked to predict something it was never shown.

    ``stratify=False`` keeps the hashed map, and it is a supported experiment rather than a
    legacy path: the two rules differ by more than 20 rows on this corpus alone, so a delta
    quoted under one of them is conditional on it. The rule is recorded in every artifact
    (``grouped_cv.stratified``); quote it beside the number.
    """
    uniq = sorted(set(groups))
    if folds < 2:
        raise ValueError("grouped CV needs at least 2 folds")
    if labels is None or not stratify:
        return {g: int(hashlib.sha256(f"{seed}|{g}".encode()).hexdigest(), 16) % folds
                for g in uniq}
    per_group: dict[str, Counter] = {}
    for group, label in labels:
        per_group.setdefault(group, Counter())[label] += 1
    load = [Counter() for _ in range(folds)]
    size = [0] * folds
    out: dict[str, int] = {}
    order = sorted(uniq, key=lambda g: (-sum(per_group.get(g, {}).values()),
                                        hashlib.sha256(f"{seed}|{g}".encode()).hexdigest()))
    for group in order:
        counts = per_group.get(group, Counter())
        chosen = min(range(folds), key=lambda k: (sum(load[k][label] for label in counts),
                                                  size[k], k))
        out[group] = chosen
        size[chosen] += sum(counts.values())
        for label, n in counts.items():
            load[chosen][label] += n
    return out


def grouped_cv(rows: Sequence[Row], make: Callable[[], object], folds: int = 5,
               seed: int = 0, options_for=None, stratify: bool = True) -> dict:
    """Fit on the other folds, score the held-out fold, and record the assignment.

    ``make()`` must return a *fresh, unfitted* predictor exposing ``fit(rows)``. The point
    of the fold map being written into the artifact is that a 97% in-sample number and an
    88% shipped number differ by which rows shared a fold.
    """
    fold_of = assign_folds((r.group for r in rows), folds, seed,
                           labels=((r.group, r.label) for r in rows), stratify=stratify)
    per_fold = []
    oof: list[Record] = []
    for k in range(folds):
        test = [r for r in rows if fold_of[r.group] == k]
        train = [r for r in rows if fold_of[r.group] != k]
        if not test:
            per_fold.append({"fold": k, "n_test": 0, "n_train": len(train), "accuracy": None,
                             "train_missing_labels": []})
            continue
        pred = make()
        if not hasattr(pred, "fit"):
            raise TypeError(f"{getattr(pred, 'name', pred)} has no fit(); grouped CV needs one")
        pred.fit(list(train))
        recs = score(f"cv{k}", test, pred, options_for=options_for)
        oof.extend(recs)
        acc = accuracy(recs)
        trained = {r.label for r in train}
        per_fold.append({"fold": k, "n_test": len(test), "n_train": len(train),
                         "accuracy": acc["accuracy"], "correct": acc["correct"],
                         "train_missing_labels": sorted({r.label for r in test} - trained)})
    return {"folds": folds, "seed": seed, "fold_of": fold_of, "per_fold": per_fold,
            "stratified": bool(stratify),
            "n_groups": len(set(fold_of)), "n_labels": len({r.label for r in rows}),
            "oof": oof, "oof_accuracy": accuracy(oof)}


# ------------------------------------------------------------------ reporting helpers

def group_report(recs: Sequence[Record]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for g in sorted({r.group for r in recs}):
        sub = [r for r in recs if r.group == g]
        acc = accuracy(sub)
        out[g] = {"n": acc["n"], "correct": acc["correct"], "accuracy": acc["accuracy"]}
    return out


def confusion(recs: Sequence[Record]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r in recs:
        out.setdefault(r.label, Counter())[r.choice] += 1
    return {k: dict(v) for k, v in out.items()}


def order_summary(recs: Sequence[Record]) -> dict:
    return {"order_sha": sorted({r.order_sha for r in recs}),
            "n_distinct_orders": len({r.order_sha for r in recs})}
