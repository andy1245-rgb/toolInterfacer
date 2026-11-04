#!/usr/bin/env python3
"""router_policy — what a *deployable* per-class router buys, and what the oracle hid.

The defect this module replaces
-------------------------------
`bench/local_clf.py` published `nested_selection_acc = 84.7%` as per-class
routing. The map is learned honestly, but it is applied as

    pred[i] = lr[i] if choice_map[LABELS[i]] == "lr" else 3b[i]

where `LABELS[i]` is the *ground-truth* class of the row being routed. At
inference the class is the unknown being predicted, so no router can index the
map that way: the number is an oracle, and the honest analogue is to index the
same map with a *predicted* class.

This module measures four policies through the same folds, seeds and
cluster-bootstrap interval:

| policy | dispatch key | map learned on | deployable |
|---|---|---|---|
| `oracle[per-class]` | true label of the scored row | train folds | **no** |
| `router[per-class, A]` | cheap model's predicted class | train folds | yes |
| `router[per-class, B]` | cheap model's predicted class | train folds, keyed by *predicted* class | yes |
| `router[static]` | cheap model's predicted class | nothing (prior table) | yes |

`A` is the one-line honest analogue of the oracle: same learned config, honest
dispatch. `B` aligns the estimator with the deployed decision — it scores each
expert on the training rows the dispatcher would actually send to it.

Then the cost-aware cascade: commit the cheap signal when it is confident,
escalate to the frozen 3B when it is not. Fixed-τ points are policies with no
free parameter, so their accuracy is measured out-of-fold on every row; the
*selected* τ is additionally chosen on the training fold only, and reported as a
separate row so a threshold is never tuned on the rows it is scored on.

The 3B is never re-driven: its per-row choices and probability vectors come from
`bench/results/p1-diagnose/baseline_live.json` (150/150 rows, byte-identical to
`bench/results/dell/v2-d1-3B-Q4.json`, 0/150 argmax disagreement across two
builds). This module makes no network calls.

Dataset caveat, stated once and repeated in the artifact
--------------------------------------------------------
`utterances_v2.json` has no phrasing-family field: 95 of its 150 rows (63.3%)
are atomic, so the group split holds out the 11 declared t3 boundary families and
nothing else. It therefore **cannot measure generalisation** — the same split on
the 382-row set, where 110 real families exist, costs the same model class 23.5
points (`docs/eval-harness-2026-11-04.md`). Every number here is optimistic by an
unknown amount of that order.

Usage
-----
    python3 bench/router_policy.py                # every table, ~5 seeds
    python3 bench/router_policy.py --seeds 2      # faster smoke
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "service"), str(ROOT / "bench")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import eval_harness as H  # noqa: E402
import local_clf as lc  # noqa: E402
from eval_signals import mcnemar_exact, union_ceiling  # noqa: E402

D1_PATH = ROOT / "bench/results/p1-diagnose/baseline_live.json"
OUT_JSON = ROOT / "bench/results/router-policy-2026-11-04.json"

# The cheap expert, fixed a priori at the mid-range of `local_clf.GRID` — the
# same configuration `bench/eval_signals.py` publishes as `lr`. Nothing here is
# tuned; a different config would be a different number, not a better policy.
CHEAP_MIN_DF = 1
CHEAP_L2 = 1e-3
KNN_MIN_DF = 2

# prior-knowledge table from `bench/local_clf.py` (no fitting, indexed by class)
STATIC_MAP = {
    "filesys_read": "lr",
    "filesys_glob": "3b",
    "filesys_grep": "lr",
    "decide_pick": "3b",
    "decide_extract": "lr",
    "web_search": "3b",
    "web_fetch": "3b",
    "reddit_digest": "3b",
}

# Fixed-τ cascade grid. Every point is a policy with no free parameter, so its
# accuracy is measured on rows no threshold ever looked at. The grid runs down to
# 0.10 because the cheap model's probability scale is not the 3B's: `lr` is
# confident on few rows and its useful gate sits low, which a 0.5-anchored grid
# would have hidden.
TAUS = (0.0, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80,
        0.90, 1.01)
NESTED_TAUS = tuple(round(0.10 + 0.05 * i, 2) for i in range(18))  # 0.10 … 0.95
NOMINAL_3B_MS = 1900.0   # the unloaded per-row cost quoted in the task brief


# --------------------------------------------------------------------------- #
# the one frozen signal: the recorded 3B
# --------------------------------------------------------------------------- #

class ThreeB:
    """d1-3B-Q4_K_M's recorded choice, probability vector and latency per row.

    Frozen and never refitted, so its predictions are out-of-fold by
    construction and identical under every split and seed. `align` refuses a
    recording that is not the scored dataset: pairing rows by position with the
    wrong file would produce a plausible accuracy for a different model.
    """

    def __init__(self, path: Path = D1_PATH) -> None:
        rec = json.loads(Path(path).read_text())
        self.path = Path(path)
        self.meta = {k: rec[k] for k in ("timestamp", "model", "n_rows", "accuracy")
                     if k in rec}
        self.rows = rec["per_row"]
        self.by_text: dict[str, dict] = {}

    def align(self, ds: H.Dataset) -> "ThreeB":
        texts = [e.text for e in ds.examples]
        if len(self.rows) != len(texts):
            raise SystemExit(f"{self.path.name}: {len(self.rows)} recorded rows vs "
                             f"dataset {len(texts)}")
        for i, (row, text) in enumerate(zip(self.rows, texts)):
            if row["utterance"] != text:
                raise SystemExit(f"{self.path.name}: row {i} is {row['utterance']!r}, "
                                 f"dataset has {text!r} — recording is not aligned")
            if row["label"] != ds.examples[i].label:
                raise SystemExit(f"{self.path.name}: row {i} label {row['label']!r} vs "
                                 f"{ds.examples[i].label!r}")
        self.by_text = {r["utterance"]: r for r in self.rows}
        return self

    def choice(self, text: str) -> str | None:
        return self.by_text[text]["choice"]

    def probs(self, text: str) -> np.ndarray:
        rec = self.by_text[text]["probabilities"]
        return np.array([rec[lab] for lab in lc.LABELS], dtype=np.float64)

    def ms(self, text: str) -> float:
        return float(self.by_text[text]["ms"])


# --------------------------------------------------------------------------- #
# the cheap signal
# --------------------------------------------------------------------------- #

def _conf(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two confidence readings from one probability matrix.

    `maxp` is the gate the repo already uses. `margin` is top-1 minus top-2 —
    the reading that separates a model that has one answer from a model that is
    torn between two, which maxp cannot distinguish.
    """
    S = np.sort(P, axis=1)
    return S[:, -1], S[:, -1] - S[:, -2]


class Cheap:
    """tf-idf word+char features, softmax regression — `lr`, fit on whatever it
    is given. The class it predicts is the router's dispatch key."""

    def __init__(self) -> None:
        self.feat = lc.Featurizer(min_df=CHEAP_MIN_DF)
        self.model = lc.SoftmaxRegression(l2=CHEAP_L2)

    def fit(self, texts: list[str], labels: list[str]) -> "Cheap":
        self.feat = lc.Featurizer(min_df=CHEAP_MIN_DF).fit(texts)
        self.model = lc.SoftmaxRegression(l2=CHEAP_L2).fit(
            self.feat.transform(texts), [lc.LABEL_INDEX[y] for y in labels])
        return self

    def proba(self, texts: list[str]) -> np.ndarray:
        return self.model.predict_proba(self.feat.transform(texts))


class Neighbour:
    """char-ngram cosine 1-NN — a second, structurally different dispatch key,
    used only to ask whether the router is limited by its dispatcher."""

    def __init__(self) -> None:
        self.model = lc.NearestNeighbour(min_df=KNN_MIN_DF)

    def fit(self, texts: list[str], labels: list[str]) -> "Neighbour":
        self.model = lc.NearestNeighbour(min_df=KNN_MIN_DF).fit(texts, labels)
        return self

    def proba(self, texts: list[str]) -> np.ndarray:
        return self.model.predict_proba(texts)


def _argmax_labels(P: np.ndarray) -> list[str]:
    return [lc.LABELS[int(i)] for i in P.argmax(axis=1)]


# --------------------------------------------------------------------------- #
# per-class expert selection
# --------------------------------------------------------------------------- #

def _expert_choice(cheap_hits: int, cheap_n: int, b3_hits: int, b3_n: int,
                   min_n: int) -> str:
    """Which expert to install for one class.

    Compared on equal footing (both scored on the same training rows) and only
    where the training fold holds at least `min_n` such rows; ties go to the 3B,
    which is the expert with the higher pooled accuracy, so the tie-break does
    not smuggle in a preference for the cheaper wrong answer.
    """
    if cheap_n < min_n or b3_n < min_n:
        return "3b"
    return "lr" if cheap_hits * b3_n > b3_hits * cheap_n else "3b"


def _class_map(train_rows: list[dict], key: str, labels: tuple[str, ...],
               min_n: int) -> dict[str, str]:
    """Learn `class -> expert` on training rows only.

    `key="true"` files a row under its ground-truth label — legitimate at
    training time, and the estimator the published 84.7% uses.
    `key="pred"` files it under the class the dispatcher predicted, which is the
    population the choice is actually applied to.
    """
    out: dict[str, str] = {}
    for lab in labels:
        sel = [r for r in train_rows if r[key] == lab]
        cheap_hits = sum(1 for r in sel if r["cheap"] == r["true"])
        b3_hits = sum(1 for r in sel if r["b3"] == r["true"])
        out[lab] = _expert_choice(cheap_hits, len(sel), b3_hits, len(sel), min_n)
    return out


# --------------------------------------------------------------------------- #
# one (seed, fold) pass
# --------------------------------------------------------------------------- #

def _fold_pass(ds: H.Dataset, fold_of: list[int], f: int, seed: int, three_b: ThreeB,
               k_inner: int, min_class_n: int) -> dict:
    """Everything one honest outer fold can produce, computed once.

    Returns per-test-row decisions for every policy under test plus the learned
    state, so no policy is ever scored with anything but a model fitted on
    training rows.
    """
    n = ds.n
    texts = [e.text for e in ds.examples]
    labels = [e.label for e in ds.examples]
    train_idx = [i for i in range(n) if fold_of[i] != f]
    test_idx = [i for i in range(n) if fold_of[i] == f]

    # --- inner grouped CV: out-of-fold cheap predictions on the TRAINING rows.
    # Needed because the per-class choice and the cascade threshold are both
    # learned on training rows, and the cheap model must not be scored in-sample
    # while learning them.
    tr_groups = [ds.examples[i].group for i in train_idx]
    tr_labels = [labels[i] for i in train_idx]
    inner_folds = H.group_folds(tr_groups, tr_labels, k_inner, seed)
    oof_pred = [None] * len(train_idx)
    oof_maxp = [0.0] * len(train_idx)
    oof_margin = [0.0] * len(train_idx)
    oof_nn = [None] * len(train_idx)
    for f_in in range(k_inner):
        pos_tr = [j for j in range(len(train_idx)) if inner_folds[j] != f_in]
        pos_te = [j for j in range(len(train_idx)) if inner_folds[j] == f_in]
        if not pos_tr or not pos_te:
            continue
        tr_t = [texts[train_idx[j]] for j in pos_tr]
        tr_y = [labels[train_idx[j]] for j in pos_tr]
        te_t = [texts[train_idx[j]] for j in pos_te]
        P = Cheap().fit(tr_t, tr_y).proba(te_t)
        maxp, marg = _conf(P)
        pred = _argmax_labels(P)
        nn = _argmax_labels(Neighbour().fit(tr_t, tr_y).proba(te_t))
        for row, j in enumerate(pos_te):
            oof_pred[j] = pred[row]
            oof_maxp[j] = float(maxp[row])
            oof_margin[j] = float(marg[row])
            oof_nn[j] = nn[row]

    train_rows = [
        {"i": train_idx[j],
         "cheap": oof_pred[j], "true": labels[train_idx[j]],
         "pred": oof_pred[j], "nn": oof_nn[j],
         "maxp": oof_maxp[j], "margin": oof_margin[j],
         "b3": three_b.choice(texts[train_idx[j]])}
        for j in range(len(train_idx)) if oof_pred[j] is not None
    ]

    map_true = _class_map(train_rows, "true", lc.LABELS, min_class_n)
    map_pred = _class_map(train_rows, "pred", lc.LABELS, min_class_n)

    # --- the deployed cheap model, fitted on the whole training portion
    fit_t = [texts[i] for i in train_idx]
    fit_y = [labels[i] for i in train_idx]
    P_te = Cheap().fit(fit_t, fit_y).proba([texts[i] for i in test_idx])
    NN_te = Neighbour().fit(fit_t, fit_y).proba([texts[i] for i in test_idx])
    maxp_te, marg_te = _conf(P_te)
    cheap_te = _argmax_labels(P_te)
    nn_te = _argmax_labels(NN_te)

    # --- cascade thresholds, chosen on the training portion only
    tau_sel = _select_tau(train_rows, three_b, texts)
    tau_sel_margin = _select_tau(train_rows, three_b, texts, signal="margin")
    tau_override = _select_override_tau(train_rows)

    decisions: dict[str, list] = {}

    def record(name: str, preds: list) -> None:
        decisions[name] = preds

    b3_te = [three_b.choice(texts[i]) for i in test_idx]
    lab_te = [labels[i] for i in test_idx]

    record("lr", cheap_te)
    record("3b", b3_te)
    record("rules", [lc.rules_predict(texts[i])[0] for i in test_idx])
    record("1nn", nn_te)
    record("oracle[per-class]", [
        cheap_te[r] if map_true[lab_te[r]] == "lr" else b3_te[r]
        for r in range(len(test_idx))])
    record("router[per-class, A]", [
        cheap_te[r] if map_true[cheap_te[r]] == "lr" else b3_te[r]
        for r in range(len(test_idx))])
    record("router[per-class, B]", [
        cheap_te[r] if map_pred[cheap_te[r]] == "lr" else b3_te[r]
        for r in range(len(test_idx))])
    record("router[static]", [
        cheap_te[r] if STATIC_MAP[cheap_te[r]] == "lr" else b3_te[r]
        for r in range(len(test_idx))])
    record("router[A, 1nn dispatch]", [
        cheap_te[r] if map_true[nn_te[r]] == "lr" else b3_te[r]
        for r in range(len(test_idx))])
    for tau in TAUS:
        record(f"cascade[maxp>={tau}]", [
            cheap_te[r] if maxp_te[r] >= tau else b3_te[r]
            for r in range(len(test_idx))])
        record(f"cascade[margin>={tau}]", [
            cheap_te[r] if marg_te[r] >= tau else b3_te[r]
            for r in range(len(test_idx))])
    record("cascade[nested tau]", [
        cheap_te[r] if maxp_te[r] >= tau_sel else b3_te[r]
        for r in range(len(test_idx))])
    record("cascade[nested tau, margin]", [
        cheap_te[r] if marg_te[r] >= tau_sel_margin else b3_te[r]
        for r in range(len(test_idx))])
    # arbitration: the 3B answers every row and the cheap model may override it
    # where the two disagree. Cost is the 3B's either way, so this is an accuracy
    # lever, and the map above is not involved.
    for tau in TAUS:
        record(f"override[maxp>={tau}]", [
            cheap_te[r] if (cheap_te[r] != b3_te[r] and maxp_te[r] >= tau) else b3_te[r]
            for r in range(len(test_idx))])
    record("override[nested tau]", [
        cheap_te[r] if (cheap_te[r] != b3_te[r] and maxp_te[r] >= tau_override)
        else b3_te[r] for r in range(len(test_idx))])
    for tau in TAUS:
        record(f"overrode[maxp>={tau}]", [
            1.0 if (cheap_te[r] != b3_te[r] and maxp_te[r] >= tau) else 0.0
            for r in range(len(test_idx))])
    record("overrode[nested tau]", [
        1.0 if (cheap_te[r] != b3_te[r] and maxp_te[r] >= tau_override) else 0.0
        for r in range(len(test_idx))])

    # how much of the traffic each router actually sends to the cheap expert —
    # the cost side of a per-class policy, which no accuracy number shows
    for tag, mapper in (("A", map_true), ("B", map_pred), ("static", STATIC_MAP)):
        record(f"routed_to_cheap[{tag}]",
               [1.0 if mapper[c] == "lr" else 0.0 for c in cheap_te])

    # escalation is a property of the gate, not of the answer
    for tau in TAUS:
        record(f"escalate[maxp>={tau}]", [0.0 if maxp_te[r] >= tau else 1.0
                                          for r in range(len(test_idx))])
        record(f"escalate[margin>={tau}]", [0.0 if marg_te[r] >= tau else 1.0
                                           for r in range(len(test_idx))])
    record("escalate[nested tau]", [0.0 if maxp_te[r] >= tau_sel else 1.0
                                    for r in range(len(test_idx))])
    record("escalate[nested tau, margin]",
           [0.0 if marg_te[r] >= tau_sel_margin else 1.0
            for r in range(len(test_idx))])

    return {"test_idx": test_idx, "decisions": decisions,
            "map_true": map_true, "map_pred": map_pred,
            "tau": tau_sel, "tau_margin": tau_sel_margin,
            "tau_override": tau_override,
            "cheap_maxp": [float(x) for x in maxp_te],
            "cheap_margin": [float(x) for x in marg_te],
            "cheap_pred": cheap_te, "b3": b3_te, "labels": lab_te}


def _routed_accuracy(rows: list[dict], three_b: ThreeB, texts: list[str],
                     tau: float, signal: str = "maxp") -> float:
    hits = 0
    for r in rows:
        b3 = three_b.choice(texts[r["i"]])
        pred = r["cheap"] if r[signal] >= tau else b3
        hits += 1 if pred == r["true"] else 0
    return hits / len(rows) if rows else 0.0


def _select_tau(train_rows: list[dict], three_b: ThreeB, texts: list[str],
                signal: str = "maxp") -> float:
    """The cheapest threshold that ties the best training-fold accuracy.

    Selection happens here, on the training portion of one outer fold, and
    nowhere else. Among thresholds the training fold cannot separate, the rule
    takes the *smaller* τ — the one that commits more rows and therefore
    escalates fewer — because an unmeasurable difference is not a reason to
    spend 1.9 s. A τ is only ever raised by a training-fold accuracy gain the
    fold can actually see.
    """
    best_acc, best_tau = -1.0, NESTED_TAUS[0]
    for tau in NESTED_TAUS:
        acc = _routed_accuracy(train_rows, three_b, texts, tau, signal)
        if acc > best_acc + 1e-12 or (abs(acc - best_acc) <= 1e-12 and tau < best_tau):
            best_acc, best_tau = acc, tau
    return best_tau


def _select_override_tau(train_rows: list[dict]) -> float:
    """The highest cheap-confidence bar at which overriding still pays on the
    training fold.

    Ties break toward the *larger* τ — fewer overrides — because overriding the
    3B is the action that can lose rows, and a difference the training fold
    cannot see is not a reason to take it. Selected on training rows only.
    """
    best_acc, best_tau = -1.0, NESTED_TAUS[-1]
    for tau in NESTED_TAUS:
        hits = 0
        for r in train_rows:
            b3 = r["b3"]
            pred = r["cheap"] if (r["cheap"] != b3 and r["maxp"] >= tau) else b3
            hits += 1 if pred == r["true"] else 0
        acc = hits / len(train_rows) if train_rows else 0.0
        if acc > best_acc + 1e-12 or (abs(acc - best_acc) <= 1e-12 and tau > best_tau):
            best_acc, best_tau = acc, tau
    return best_tau


# --------------------------------------------------------------------------- #
# the sweep
# --------------------------------------------------------------------------- #

def run(ds: H.Dataset, three_b: ThreeB, *, split: str = "group", k: int = H.DEFAULT_K,
        seeds: tuple[int, ...] = H.DEFAULT_SEEDS, k_inner: int = 3,
        min_class_n: int = 3, n_boot: int = H.DEFAULT_BOOT) -> dict:
    n = ds.n
    labels = [e.label for e in ds.examples]
    texts = [e.text for e in ds.examples]
    variants: list[str] = []
    outcomes: dict[str, list[float]] = {}
    per_seed_preds: dict[str, list[list]] = {}
    per_seed_acc: dict[str, list[float]] = {}
    escalations: dict[str, list[float]] = {}
    taus: list[float] = []
    maps: list[dict] = []
    dispatch_acc: dict[str, list[float]] = {}

    for seed in seeds:
        fold_of = H.folds_for(ds, split, k, seed)
        seed_preds: dict[str, list] = {}
        seed_escal: dict[str, list[float]] = {}
        for f in range(k):
            out = _fold_pass(ds, fold_of, f, seed, three_b, k_inner, min_class_n)
            taus.append(out["tau"])
            maps.append({"seed": seed, "fold": f, "tau": out["tau"],
                         "tau_margin": out["tau_margin"],
                         "tau_override": out["tau_override"],
                         "map_true": out["map_true"], "map_pred": out["map_pred"]})
            for name, preds in out["decisions"].items():
                if (name.startswith("escalate") or name.startswith("overrode")
                        or name.startswith("routed_to_cheap")):
                    bucket = seed_escal
                else:
                    bucket = seed_preds
                if name not in bucket:
                    bucket[name] = [None] * n
                for r, i in enumerate(out["test_idx"]):
                    bucket[name][i] = preds[r]
            for name in ("dispatch[lr]", "dispatch[3b]"):
                dispatch_acc.setdefault(name, [0.0] * n)
            for r, i in enumerate(out["test_idx"]):
                dispatch_acc["dispatch[lr]"][i] += (
                    1.0 if out["cheap_pred"][r] == out["labels"][r] else 0.0)
                dispatch_acc["dispatch[3b]"][i] += (
                    1.0 if out["b3"][r] == out["labels"][r] else 0.0)
        for name in seed_preds:
            if name not in variants:
                variants.append(name)
                outcomes[name] = [0.0] * n
                per_seed_preds[name] = []
                per_seed_acc[name] = []
            hits = 0
            for i in range(n):
                ok = 1.0 if seed_preds[name][i] == labels[i] else 0.0
                outcomes[name][i] += ok
                hits += ok
            per_seed_acc[name].append(round(hits / n, 4))
            per_seed_preds[name].append(seed_preds[name])
        for name in seed_escal:
            escalations.setdefault(name, []).extend(
                v for v in seed_escal[name] if v is not None)

    results: dict[str, dict] = {}
    rep = ds.group_report()
    for name in variants:
        mean_out = [o / len(seeds) for o in outcomes[name]]
        acc = sum(mean_out) / n
        lo, hi = H.cluster_bootstrap_ci(mean_out, ds.groups, n_boot=n_boot)
        entry = {
            "predictor": name, "split": split, "n": n, "groups": rep["groups"],
            "k": k, "seeds": len(seeds), "acc": round(acc, 4),
            "ci95": [lo, hi], "per_seed": per_seed_acc[name],
            "acc_min": round(min(per_seed_acc[name]), 4),
            "acc_max": round(max(per_seed_acc[name]), 4),
            "atomic_row_fraction": rep["singleton_row_fraction"],
            "outcomes": mean_out,
        }
        esc_key = name.replace("cascade", "escalate")
        if esc_key in escalations:
            entry["escalation_rate"] = round(
                sum(escalations[esc_key]) / len(escalations[esc_key]), 4)
        own_key = name.replace("override", "overrode")
        if own_key in escalations:
            entry["override_rate"] = round(
                sum(escalations[own_key]) / len(escalations[own_key]), 4)
        cheap_key = name.replace("router[per-class, A]", "routed_to_cheap[A]")
        cheap_key = cheap_key.replace("router[per-class, B]", "routed_to_cheap[B]")
        cheap_key = cheap_key.replace("router[static]", "routed_to_cheap[static]")
        if cheap_key in escalations:
            entry["cheap_coverage"] = round(
                sum(escalations[cheap_key]) / len(escalations[cheap_key]), 4)
        results[name] = entry

    for name, values in escalations.items():
        rate = round(sum(values) / len(values), 4)
        if name.startswith("overrode"):
            key = "override_rate"
        elif name.startswith("routed_to_cheap"):
            key = "cheap_coverage"
        else:
            key = "escalation_rate"
        results[name] = {
            "predictor": name, "split": split, "n": n, "groups": rep["groups"],
            "k": k, "seeds": len(seeds), key: rate, "rate_only": True,
        }

    for name, out in dispatch_acc.items():
        out = [o / len(seeds) for o in out]
        acc = sum(out) / n
        lo, hi = H.cluster_bootstrap_ci(out, ds.groups, n_boot=n_boot)
        results[name] = {"predictor": name, "split": split, "n": n,
                         "groups": rep["groups"], "k": k, "seeds": len(seeds),
                         "acc": round(acc, 4), "ci95": [lo, hi],
                         "outcomes": out, "dispatch_only": True}

    return {"results": results, "tau_per_fold": taus, "maps": maps,
            "dataset": ds.name, "group_report": rep, "split": split,
            "seeds": list(seeds), "k": k, "three_b": three_b.meta}


def _as_result(entry: dict) -> H.Result:
    return H.Result(
        predictor=entry["predictor"], split=entry["split"], n=entry["n"],
        n_groups=entry["groups"], k=entry["k"], seeds=entry["seeds"],
        acc_mean=entry["acc"], acc_sd=0.0, acc_min=entry.get("acc_min", entry["acc"]),
        acc_max=entry.get("acc_max", entry["acc"]), ci_lo=entry["ci95"][0],
        ci_hi=entry["ci95"][1], per_seed=entry.get("per_seed", []),
        dataset=entry.get("dataset", ""),
        atomic_row_fraction=entry.get("atomic_row_fraction", 0.0),
        outcomes=entry["outcomes"])


# --------------------------------------------------------------------------- #
# the uncertainty question
# --------------------------------------------------------------------------- #

def _auc(scores: list[float], positives: list[bool]) -> float:
    """Rank AUC: P(score of a positive > score of a negative), ties as 0.5.

    Implemented with mid-ranks so a plateau of identical scores cannot be
    counted as discrimination.
    """
    pos = sum(1 for p in positives if p)
    neg = len(positives) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        mid = (i + j) / 2.0 + 1.0
        for k2 in range(i, j + 1):
            ranks[order[k2]] = mid
        i = j + 1
    rank_sum = sum(r for r, p in zip(ranks, positives) if p)
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def uncertainty_report(ds: H.Dataset, three_b: ThreeB, sweep: dict,
                       k_inner: int = 3) -> dict:
    """Does a cheaper or better uncertainty signal catch the 3B's confident misses?

    Three readings, all computed from artifacts already on disk plus one OOF pass
    of the cheap model:

    - `maxp_3b` / `margin_3b`: the 3B's own top probability vs its top-1/top-2
      margin, as a gate on its own answers;
    - `maxp_lr` / `margin_lr`: the cheap model's confidence;
    - `disagree`: 1 when the cheap model's answer differs from the 3B's — a
      signal that costs one cheap inference *in addition to* the 3B, so it is an
      arbitration device, not a cheap front-end.
    """
    n = ds.n
    texts = [e.text for e in ds.examples]
    labels = [e.label for e in ds.examples]

    # one OOF pass of the cheap model over all rows, so the cheap reading is
    # never in-sample (same folds as the sweep's first seed)
    fold_of = H.folds_for(ds, "group", H.DEFAULT_K, 0)
    cheap_pred = [None] * n
    cheap_maxp = [0.0] * n
    cheap_margin = [0.0] * n
    for f in range(H.DEFAULT_K):
        tr = [i for i in range(n) if fold_of[i] != f]
        te = [i for i in range(n) if fold_of[i] == f]
        P = Cheap().fit([texts[i] for i in tr], [labels[i] for i in tr]).proba(
            [texts[i] for i in te])
        maxp, marg = _conf(P)
        pred = _argmax_labels(P)
        for row, i in enumerate(te):
            cheap_pred[i] = pred[row]
            cheap_maxp[i] = float(maxp[row])
            cheap_margin[i] = float(marg[row])

    b3_pred = [three_b.choice(t) for t in texts]
    b3_maxp = [float(three_b.probs(t).max()) for t in texts]
    b3_sorted = [np.sort(three_b.probs(t)) for t in texts]
    b3_margin = [float(s[-1] - s[-2]) for s in b3_sorted]

    b3_wrong = [b3_pred[i] != labels[i] for i in range(n)]
    cheap_wrong = [cheap_pred[i] != labels[i] for i in range(n)]
    disagree = [cheap_pred[i] != b3_pred[i] for i in range(n)]

    # a "miss detector" should score HIGH on a row the 3B got wrong
    auc = {
        "maxp_3b (low = suspicious)": _auc([-v for v in b3_maxp], b3_wrong),
        "margin_3b (low = suspicious)": _auc([-v for v in b3_margin], b3_wrong),
        "maxp_lr (low = suspicious)": _auc([-v for v in cheap_maxp], b3_wrong),
        "margin_lr (low = suspicious)": _auc([-v for v in cheap_margin], b3_wrong),
        "disagreement (1 = suspicious)": _auc([1.0 if d else 0.0 for d in disagree],
                                              b3_wrong),
        "cheap wrong (1 = suspicious)": _auc([1.0 if w else 0.0 for w in cheap_wrong],
                                             b3_wrong),
    }

    # the confident-miss set the task names
    hi_conf_miss = [i for i in range(n) if b3_wrong[i] and b3_maxp[i] >= 0.5]
    all_miss = [i for i in range(n) if b3_wrong[i]]

    def capture(idx: list[int], flag) -> int:
        return sum(1 for i in idx if flag(i))

    detections = {
        "n_3b_misses": len(all_miss),
        "n_3b_miss_conf_ge_0.5": len(hi_conf_miss),
        "maxp_3b<0.5 catches": capture(all_miss, lambda i: b3_maxp[i] < 0.5),
        "margin_3b<0.3 catches": capture(all_miss, lambda i: b3_margin[i] < 0.3),
        "margin_3b<0.5 catches": capture(all_miss, lambda i: b3_margin[i] < 0.5),
        "maxp_lr<0.5 catches": capture(all_miss, lambda i: cheap_maxp[i] < 0.5),
        "disagreement catches": capture(all_miss, lambda i: disagree[i]),
        "on_confident_misses: maxp_lr<0.5": capture(hi_conf_miss,
                                                    lambda i: cheap_maxp[i] < 0.5),
        "on_confident_misses: margin_lr<0.3": capture(hi_conf_miss,
                                                      lambda i: cheap_margin[i] < 0.3),
        "on_confident_misses: disagreement": capture(hi_conf_miss,
                                                     lambda i: disagree[i]),
    }

    # escalation-precision: of the rows each gate escalates, how many are rows
    # the 3B would have got right (wasted spend) vs wrong (a save)
    gate_quality: list[dict] = []
    for name, conf, tau in (
        ("3b maxp", b3_maxp, 0.5), ("3b margin", b3_margin, 0.3),
        ("3b margin", b3_margin, 0.5), ("lr maxp", cheap_maxp, 0.5),
        ("lr maxp", cheap_maxp, 0.7), ("lr margin", cheap_margin, 0.3),
    ):
        esc = [i for i in range(n) if conf[i] < tau]
        gate_quality.append({
            "signal": name, "tau": tau, "escalated": len(esc),
            "rate": round(len(esc) / n, 4),
            "of_which_3b_wrong": capture(esc, lambda i, b=b3_wrong: b[i]),
            "of_which_3b_right": len(esc) - capture(esc, lambda i, b=b3_wrong: b[i]),
        })

    # matched-coverage comparison: at the same escalation budget, which reading
    # catches more of the 3B's misses? A gate is only better if it catches more
    # at the SAME spend, so the comparison is made at a shared rate, never at a
    # shared threshold.
    matched: list[dict] = []
    for target in (0.10, 0.20, 0.30, 0.40, 0.50):
        cut = max(1, round(target * n))
        entry: dict = {"rate": round(cut / n, 4)}
        for label, score in (("3b maxp", b3_maxp), ("3b margin", b3_margin),
                             ("lr margin", cheap_margin), ("lr maxp", cheap_maxp)):
            order = sorted(range(n), key=lambda i, s=score: s[i])
            entry[label] = capture(order[:cut], lambda i, b=b3_wrong: b[i])
        entry["disagreement"] = {
            "rate": round(sum(1 for d in disagree if d) / n, 4),
            "catches": capture(range(n), lambda i: disagree[i] and b3_wrong[i])}
        matched.append(entry)

    # arbitration: the 3B answers every row; override it with the cheap model
    # when they disagree and the cheap model is confident. Cost is unchanged
    # (the 3B is always asked) so this is an accuracy lever only.
    arb: list[dict] = []
    for tau in (0.0, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 1.01):
        hits_3b = sum(1 for i in range(n) if not b3_wrong[i])
        over = {i for i in range(n) if disagree[i] and cheap_maxp[i] >= tau}
        arb_pred = [cheap_pred[i] if i in over else b3_pred[i] for i in range(n)]
        hits_arb = sum(1 for i in range(n) if arb_pred[i] == labels[i])
        lo, hi = lc.wilson(hits_arb, n)
        m = mcnemar_exact(
            SimpleNamespace(outcomes=[1.0 if arb_pred[i] == labels[i] else 0.0
                                      for i in range(n)]),
            SimpleNamespace(outcomes=[1.0 if not b3_wrong[i] else 0.0
                                      for i in range(n)]))
        arb.append({"tau": tau, "overrides": len(over),
                    "overrides_cheap_right": capture(sorted(over),
                                                     lambda i: not cheap_wrong[i]),
                    "acc": round(hits_arb / n, 4), "acc_ci": [round(lo, 4), round(hi, 4)],
                    "acc_3b": round(hits_3b / n, 4),
                    "delta_rows": hits_arb - hits_3b, "p_vs_3b": m["p"]})

    # pairwise agreement between the three readings on the message that matters
    both_wrong = sum(1 for i in range(n) if b3_wrong[i] and cheap_wrong[i])
    return {"auc": {k: round(v, 4) for k, v in auc.items()},
            "detections": detections, "gate_quality": gate_quality,
            "matched_coverage": matched,
            "arbitration": arb, "both_wrong": both_wrong,
            "disagreement_rate": round(sum(1 for d in disagree if d) / n, 4),
            "hi_conf_miss_rows": [
                {"utterance": texts[i], "label": labels[i], "3b": b3_pred[i],
                 "3b_conf": round(b3_maxp[i], 4),
                 "3b_margin": round(b3_margin[i], 4),
                 "lr": cheap_pred[i], "lr_conf": round(float(cheap_maxp[i]), 4),
                 "lr_margin": round(float(cheap_margin[i]), 4)}
                for i in hi_conf_miss]}


def confident_miss_report(ds: H.Dataset, three_b: ThreeB, sweep: dict) -> dict:
    """Accuracy on the subset where the 3B is confident and wrong.

    This is the set a per-row gate cannot fix by construction (the 3B's own
    probability says commit), so it is the right place to ask whether a class-
    level policy does better than a confidence-level one.
    """
    miss = []
    for i, e in enumerate(ds.examples):
        p = three_b.probs(e.text)
        if three_b.choice(e.text) != e.label and float(p.max()) >= 0.5:
            miss.append(i)
    out: dict = {"n_confident_misses": len(miss),
                 "rows": [ds.examples[i].text for i in miss],
                 "by_label": {}}
    for i in miss:
        lab = ds.examples[i].label
        out["by_label"][lab] = out["by_label"].get(lab, 0) + 1
    out["policies"] = {}
    for name in ("3b", "lr", "router[per-class, A]", "router[static]",
                 "cascade[nested tau]", "override[nested tau]",
                 "oracle[per-class]"):
        e = sweep["results"].get(name)
        if not e or not miss:
            continue
        hits = sum(e["outcomes"][i] for i in miss)
        out["policies"][name] = {"correct": round(hits, 2), "of": len(miss),
                                 "rate": round(hits / len(miss), 4)}
    return out


# --------------------------------------------------------------------------- #
# latency
# --------------------------------------------------------------------------- #

def latency_report(ds: H.Dataset, three_b: ThreeB, sweep: dict) -> dict:
    """Induced latency per policy, from measured quantities only.

    The cheap model is timed live in this process (single-row, after a warm-up).
    The 3B per-row cost comes from the recording, which was taken against the
    *shared* server, so it carries other tenants' queueing; the nominal 1.9 s
    figure is reported alongside it as the unloaded cost the brief quotes.
    """
    texts = [e.text for e in ds.examples]
    labels = [e.label for e in ds.examples]
    model = Cheap().fit(texts, labels)
    for t in texts[:20]:
        model.proba([t])
    samples = []
    for t in texts:
        start = time.perf_counter()
        model.proba([t])
        samples.append((time.perf_counter() - start) * 1000.0)
    cheap_ms = statistics.median(samples)
    b3_ms_recorded = statistics.median([three_b.ms(t) for t in texts])
    b3_ms_mean = statistics.mean([three_b.ms(t) for t in texts])
    out: dict = {"cheap_ms_p50": round(cheap_ms, 3),
                 "cheap_ms_mean": round(statistics.mean(samples), 3),
                 "3b_ms_p50_recorded": round(b3_ms_recorded, 1),
                 "3b_ms_mean_recorded": round(b3_ms_mean, 1),
                 "3b_ms_nominal": NOMINAL_3B_MS,
                 "policies": {}}
    for name, entry in sweep["results"].items():
        if entry.get("dispatch_only") or entry.get("rate_only"):
            continue
        if "escalation_rate" not in entry:
            continue
        rate = entry["escalation_rate"]
        out["policies"][name] = {
            "escalation_rate": rate,
            "latency_ms_nominal": round(cheap_ms + rate * NOMINAL_3B_MS, 1),
            "latency_ms_recorded": round(cheap_ms + rate * b3_ms_recorded, 1),
        }
    for name in ("lr", "1nn", "rules", "3b"):
        if name in sweep["results"]:
            out["policies"].setdefault(name, {"escalation_rate":
                                              1.0 if name == "3b" else 0.0,
                                              "latency_ms_nominal":
                                              round(cheap_ms + (NOMINAL_3B_MS if name == "3b" else 0.0), 1),
                                              "latency_ms_recorded":
                                              round(cheap_ms + (b3_ms_recorded if name == "3b" else 0.0), 1)})
    return out


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

MAIN_ROWS = ("lr", "1nn", "rules", "3b", "router[static]", "router[per-class, A]",
             "router[per-class, B]", "router[A, 1nn dispatch]",
             "oracle[per-class]")


def table_main(sweep: dict, split_note: str) -> str:
    r = sweep["results"]
    lines = ["| policy | dispatch key | n | groups | acc | 95% CI (cluster) | seeds "
             "| sent to cheap |",
             "|---|---|---:|---:|---:|---|---|---|"]
    keys = {"lr": "—", "1nn": "—", "rules": "—", "3b": "—",
            "router[static]": "cheap argmax",
            "router[per-class, A]": "cheap argmax",
            "router[per-class, B]": "cheap argmax",
            "router[A, 1nn dispatch]": "1nn argmax",
            "oracle[per-class]": "**ground truth**"}
    for name in MAIN_ROWS:
        e = r.get(name)
        if not e:
            continue
        ci = f"{e['ci95'][0] * 100:.1f}–{e['ci95'][1] * 100:.1f}%"
        spread = (f"{e['acc_min'] * 100:.1f}–{e['acc_max'] * 100:.1f}%"
                  if len(e.get("per_seed", [])) > 1 else "—")
        cov = (f" | {e['cheap_coverage'] * 100:.0f}%"
               if "cheap_coverage" in e else " | —")
        lines.append(f"| `{name}` | {keys.get(name, '—')} | {e['n']} | "
                     f"{e['groups']} | **{e['acc'] * 100:.1f}%** | {ci} | "
                     f"{spread}{cov} |")
    lines.append("")
    lines.append(split_note)
    return "\n".join(lines)


def table_cascade(sweep: dict, latency: dict) -> str:
    r = sweep["results"]
    lines = ["| τ | gate | committed | escalation | acc | 95% CI | latency (ms, "
             "3B@1.9 s) |", "|---:|---|---:|---:|---:|---|---:|"]
    for gate in ("maxp", "margin"):
        for tau in TAUS:
            name = f"cascade[{gate}>={tau}]"
            e = r.get(name)
            if not e:
                continue
            esc = r[f"escalate[{gate}>={tau}]"]["escalation_rate"]
            lat = latency["policies"].get(name, {}).get("latency_ms_nominal", 0.0)
            shown = "≥1" if tau > 1.0 else f"{tau:.2f}"
            lines.append(f"| {shown} | {gate} | {1 - esc:.0%} | {esc:.0%} | "
                         f"**{e['acc'] * 100:.1f}%** | "
                         f"{e['ci95'][0] * 100:.1f}–{e['ci95'][1] * 100:.1f}% | "
                         f"{lat:.0f} |")
    for name in ("cascade[nested tau]", "cascade[nested tau, margin]"):
        e = r.get(name)
        if not e:
            continue
        esc = r[name.replace("cascade", "escalate")]["escalation_rate"]
        lat = latency["policies"].get(name, {}).get("latency_ms_nominal", 0.0)
        taus = sweep["tau_per_fold"]
        lines.append(f"| {statistics.median(taus):.2f} (median) | nested | "
                     f"{1 - esc:.0%} | {esc:.0%} | **{e['acc'] * 100:.1f}%** | "
                     f"{e['ci95'][0] * 100:.1f}–{e['ci95'][1] * 100:.1f}% | {lat:.0f} |")
    for name, label in (("lr", "lr only (no gate)"), ("3b", "3B only")):
        e = r.get(name)
        lat = latency["policies"].get(name, {}).get("latency_ms_nominal", 0.0)
        esc = 0.0 if name == "lr" else 1.0
        lines.append(f"| — | {label} | {1 - esc:.0%} | {esc:.0%} | "
                     f"**{e['acc'] * 100:.1f}%** | "
                     f"{e['ci95'][0] * 100:.1f}–{e['ci95'][1] * 100:.1f}% | {lat:.0f} |")
    return "\n".join(lines)


def table_override(sweep: dict) -> str:
    """The 3B answers every row; the cheap model may override it on disagreement."""
    r = sweep["results"]
    lines = ["| τ (override bar) | overrides | acc | 95% CI | vs 3B | exact p |",
             "|---|---:|---|---:|---:|---:|"]
    for name in [f"override[maxp>={t}]" for t in TAUS] + ["override[nested tau]"]:
        e = r.get(name)
        if not e:
            continue
        rate = e.get("override_rate", 0.0)
        shown = ("nested" if "nested" in name
                 else "≥1" if name.endswith("=1.01]") else f"{name.split('>=')[1][:-1]}")
        m = mcnemar_exact(_as_result(e), _as_result(r["3b"]))
        lines.append(f"| {shown} | {rate * 100:.0f}% ({round(rate * e['n'])} rows) | "
                     f"**{e['acc'] * 100:.1f}%** | "
                     f"{e['ci95'][0] * 100:.1f}–{e['ci95'][1] * 100:.1f}% | "
                     f"{(e['acc'] - r['3b']['acc']) * 100:+.1f} pts | {m['p']:.4f} |")
    return "\n".join(lines)


def table_pairs(sweep: dict, ds: H.Dataset) -> tuple[str, dict]:
    r = sweep["results"]

    def res(name: str) -> H.Result:
        return _as_result(r[name])

    lines = ["| comparison | A only | B only | Δ acc | 95% CI on Δ | exact p |",
             "|---|---:|---:|---:|---|---:|"]
    pairs = {}
    for a, b in (("router[per-class, A]", "3b"),
                 ("router[per-class, A]", "lr"),
                 ("router[per-class, B]", "router[per-class, A]"),
                 ("oracle[per-class]", "router[per-class, A]"),
                 ("router[per-class, A]", "router[static]"),
                 ("cascade[nested tau]", "3b"),
                 ("cascade[maxp>=0.3]", "3b"),
                 ("cascade[maxp>=0.35]", "3b"),
                 ("cascade[maxp>=0.3]", "router[per-class, A]"),
                 ("cascade[maxp>=0.5]", "3b"),
                 ("override[nested tau]", "3b")):
        if a not in r or b not in r:
            continue
        m = mcnemar_exact(res(a), res(b))
        diff = [x - y for x, y in zip(r[a]["outcomes"], r[b]["outcomes"])]
        lo, hi = H.cluster_bootstrap_ci(diff, ds.groups)
        delta = sum(diff) / len(diff)
        pairs[f"{a} vs {b}"] = {"acc_a": r[a]["acc"], "acc_b": r[b]["acc"],
                                "delta": round(delta, 4),
                                "delta_ci95": [lo, hi], **m}
        lines.append(f"| `{a}` vs `{b}` | {m['a_only']} | {m['b_only']} | "
                     f"{delta * 100:+.1f} pts | {lo * 100:+.1f}–{hi * 100:+.1f} | "
                     f"{m['p']:.4f} |")
    lines.append("")
    lines.append("Δ is A minus B. The interval is a cluster bootstrap over groups on "
                 "the paired per-row difference, so it is the instrument; the "
                 "exact p beside it is unadjusted for the eleven comparisons above.")
    return "\n".join(lines), pairs


def table_union(sweep: dict, ds: H.Dataset) -> tuple[str, dict]:
    r = sweep["results"]
    out = {}
    lines = ["| pair | union (oracle) | 95% CI | ceiling over which policy |",
             "|---|---|---|---|"]
    for a, b in (("lr", "3b"), ("rules", "3b"), ("lr", "rules")):
        if a not in r or b not in r:
            continue
        u = union_ceiling(_as_result(r[a]), _as_result(r[b]), ds)
        name = f"union({a}|{b})"
        out[name] = {"acc": u.acc_mean, "ci95": [u.ci_lo, u.ci_hi]}
        lines.append(f"| `{a}` \\| `{b}` | **{u.acc_mean * 100:.1f}%** | "
                     f"{u.ci_lo * 100:.1f}–{u.ci_hi * 100:.1f}% | any per-row or "
                     f"per-class selector between them |")
    return "\n".join(lines), out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="group", choices=["group", "row"])
    ap.add_argument("--seeds", type=int, default=len(H.DEFAULT_SEEDS))
    ap.add_argument("--k", type=int, default=H.DEFAULT_K)
    ap.add_argument("--k-inner", type=int, default=3)
    ap.add_argument("--boot", type=int, default=H.DEFAULT_BOOT)
    ap.add_argument("--grouping", default="boundary",
                    choices=["boundary", "surface", "row"])
    ap.add_argument("--json", default=str(OUT_JSON))
    ap.add_argument("--no-latency", action="store_true")
    args = ap.parse_args()

    ds = H.load_v2(grouping=args.grouping)
    three_b = ThreeB().align(ds)
    rep = ds.group_report()
    print("# a deployable per-class router — measured, not asserted")
    print()
    print(f"data: {ds.name} — {ds.n} rows; groups {rep['groups']} "
          f"({rep['singleton_groups']} singleton, "
          f"{rep['singleton_row_fraction'] * 100:.1f}% of rows atomic)")
    print(f"split: {args.split}; {args.k}-fold × {args.seeds} seeds; "
          f"cluster bootstrap {args.boot} resamples over groups")
    print(f"3B: {three_b.meta.get('model', {}).get('path', 'd1-3B-Q4_K_M.gguf')} "
          f"recorded, {three_b.meta.get('n_rows')} rows, frozen (no server call)")
    print()

    sweep = run(ds, three_b, split=args.split, k=args.k,
                seeds=tuple(range(args.seeds)), k_inner=args.k_inner,
                n_boot=args.boot)

    split_note = (f"n={ds.n}, {rep['groups']} groups, {rep['singleton_row_fraction'] * 100:.0f}% "
                  f"of rows atomic — this set cannot measure generalisation.")
    print("## Every policy on the same folds\n")
    print(table_main(sweep, split_note))
    print()

    print("## Dispatch accuracy — the router cannot beat its own class predictor\n")
    for name in ("dispatch[lr]", "dispatch[3b]"):
        e = sweep["results"].get(name)
        if e:
            print(f"- `{name}` {e['acc'] * 100:.1f}% "
                  f"[{e['ci95'][0] * 100:.1f}–{e['ci95'][1] * 100:.1f}%] "
                  f"(n={e['n']})")
    print()

    print("## Paired (McNemar exact) — does anything actually differ?\n")
    pairs_tbl, pairs = table_pairs(sweep, ds)
    print(pairs_tbl)
    print()

    print("## Oracle ceilings\n")
    union_tbl, unions = table_union(sweep, ds)
    print(union_tbl)
    print()

    latency = {"policies": {}} if args.no_latency else latency_report(ds, three_b, sweep)
    print("## The cost-aware cascade\n")
    print(table_cascade(sweep, latency))
    print()

    print("## Arbitration: the 3B always answers, the cheap model may override it\n")
    print(table_override(sweep))
    print()
    print(f"nested override τ per fold: median {statistics.median([m['tau_override'] for m in sweep['maps']]):.2f}, "
          f"range {min(m['tau_override'] for m in sweep['maps']):.2f}–"
          f"{max(m['tau_override'] for m in sweep['maps']):.2f}")
    print()

    print("## Does a better uncertainty signal exist?\n")
    unc = uncertainty_report(ds, three_b, sweep)
    cmiss = confident_miss_report(ds, three_b, sweep)
    print(f"The 3B's {cmiss['n_confident_misses']} confident misses (conf ≥ 0.5) by "
          f"true class: {cmiss['by_label']}\n")
    print("| policy | right on those rows |")
    print("|---|---:|")
    for name, row in cmiss["policies"].items():
        print(f"| `{name}` | {row['correct']:g}/{row['of']} "
              f"({row['rate'] * 100:.0f}%) |")
    print()
    print(f"3B misses: {unc['detections']['n_3b_misses']}; of those at conf ≥ 0.5: "
          f"{unc['detections']['n_3b_miss_conf_ge_0.5']}. "
          f"cheap/3B disagreement rate {unc['disagreement_rate'] * 100:.1f}%.\n")
    print("| detector of a 3B miss | AUC |")
    print("|---|---:|")
    for k, v in unc["auc"].items():
        print(f"| {k} | {v:.3f} |")
    print()
    print("At a matched escalation budget, misses caught:")
    print()
    print("| budget | 3b maxp | 3b margin | lr margin | lr maxp |")
    print("|---:|---:|---:|---:|---:|")
    for row in unc["matched_coverage"]:
        print(f"| {row['rate'] * 100:.0f}% | {row['3b maxp']} | {row['3b margin']} | "
              f"{row['lr margin']} | {row['lr maxp']} |")
    dis = unc["matched_coverage"][0]["disagreement"]
    print(f"\n(misses caught out of {unc['detections']['n_3b_misses']}. The "
          f"cheap/3B disagreement flag fires on {dis['rate'] * 100:.0f}% of rows and "
          f"catches {dis['catches']} of them — a flag, not a ranking.)")
    print()
    print(json.dumps(unc["detections"], indent=1))
    print()
    print("| arbitration: 3B always, override on disagreement | overrides | of which "
          "cheap right | acc (Wilson 95%) | vs 3B | exact p |")
    print("|---|---:|---:|---|---:|---:|")
    for row in unc["arbitration"]:
        print(f"| cheap maxp ≥ {row['tau']:.2f} | {row['overrides']} | "
              f"{row['overrides_cheap_right']} | {row['acc'] * 100:.1f}% "
              f"[{row['acc_ci'][0] * 100:.0f}–{row['acc_ci'][1] * 100:.0f}] | "
              f"{row['delta_rows']:+d} rows | {row['p_vs_3b']:.4f} |")
    print()

    if args.json:
        payload = {
            "dataset": {"name": ds.name, "n": ds.n, "group_report": rep,
                        "caveat": "no family field; 63.3% atomic rows; cannot "
                                  "measure generalisation — every number is "
                                  "optimistic by an unknown amount"},
            "protocol": {"split": args.split, "k": args.k, "seeds": list(range(args.seeds)),
                         "k_inner": args.k_inner, "bootstrap": args.boot,
                         "ci": "cluster bootstrap over groups, 95%"},
            "three_b": three_b.meta,
            "results": {k: {kk: vv for kk, vv in v.items() if kk != "outcomes"}
                        for k, v in sweep["results"].items()},
            "tau_per_fold": sweep["tau_per_fold"],
            "tau_override_per_fold": [m["tau_override"] for m in sweep["maps"]],
            "maps": sweep["maps"],
            "pairs": pairs, "unions": unions,
            "latency": latency, "uncertainty": unc,
            "confident_misses": cmiss,
        }
        Path(args.json).write_text(json.dumps(payload, indent=1) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
