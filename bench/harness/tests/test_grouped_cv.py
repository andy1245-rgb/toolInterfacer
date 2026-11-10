"""Grouped cross-validation, with a predictor small enough to fit by hand.

The spec's reason for `--cv` is blunt: a random split on utterances from the same family is how a
classifier reports 97% in sample and ships 88%. This test uses a nearest-prototype classifier on
twelve synthetic rows, so the answer is checkable on paper, and asserts the two things that make
cross-validation honest: a group never straddles two folds, and the fold map is recorded.
"""
from __future__ import annotations

import math

import pytest

from bench.harness import metrics
from bench.harness.protocol import Prediction
from bench.harness.rows import Row


class NearestPrototype:
    """Fits one prototype per label; a predictor with a real `fit`, in twenty lines."""

    name = "nearest_prototype"

    def __init__(self):
        self.prototypes: dict[str, list[float]] = {}

    @staticmethod
    def _vec(text: str) -> list[float]:
        v = [0.0, 0.0]
        for ch in text:
            v[0] += 1.0 if ch in "abc" else 0.0
            v[1] += 1.0 if ch in "xyz" else 0.0
        n = math.sqrt(sum(c * c for c in v)) or 1.0
        return [c / n for c in v]

    def fit(self, rows: list[Row]) -> None:
        acc: dict[str, list[float]] = {}
        for r in rows:
            v = self._vec(r.text)
            cur = acc.setdefault(r.label, [0.0, 0.0])
            cur[0] += v[0]
            cur[1] += v[1]
        self.prototypes = {k: v for k, v in acc.items()}

    def predict(self, text: str, options: tuple[str, ...]) -> Prediction:
        v = self._vec(text)
        best, best_d = options[0], -1.0
        for o in options:
            if o not in self.prototypes:
                continue
            p = self.prototypes[o]
            d = sum(a * b for a, b in zip(v, p))
            if d > best_d:
                best, best_d = o, d
        return Prediction(choice=best, probabilities=None, latency_ms=0.0)


def _rows() -> list[Row]:
    rows = []
    for i in range(12):
        label = "alpha" if i % 2 == 0 else "omega"
        text = "aaa" if label == "alpha" else "xxx"
        # every group carries both labels: a fold that trained on one label only would make the
        # out-of-fold number a statement about the fixture, not about the predictor
        rows.append(Row(id=f"r{i:02d}", text=text, label=label, group=f"g{i // 3}",
                        options=("alpha", "omega")))
    return rows


def test_the_fold_map_never_splits_a_group():
    rows = _rows()
    fold_of = metrics.assign_folds((r.group for r in rows), folds=2, seed=5)
    for g in {r.group for r in rows}:
        assert len({fold_of[g]}) == 1
    assert len(fold_of) == 4
    for k in (0, 1):
        train = [r for r in rows if fold_of[r.group] != k]
        assert {r.label for r in train} == {"alpha", "omega"}, "a fold must train on both labels"


def test_grouped_cv_records_the_assignment_and_scores_out_of_fold():
    rows = _rows()
    out = metrics.grouped_cv(rows, NearestPrototype, folds=2, seed=5)
    assert out["fold_of"] == metrics.assign_folds(
        (r.group for r in rows), folds=2, seed=5,
        labels=((r.group, r.label) for r in rows))
    assert len(out["oof"]) == len(rows)
    assert out["oof_accuracy"]["correct"] == len(rows)   # separable by construction
    assert sum(f["n_test"] for f in out["per_fold"]) == len(rows)
    assert all(f["n_train"] > 0 for f in out["per_fold"])


def test_grouped_cv_shows_the_gap_when_a_prototype_leaks():
    """A predictor that memorises the row must do worse out of fold than in sample."""
    class Memoriser(NearestPrototype):
        def fit(self, rows):
            super().fit(rows)
            self.seen = {r.text for r in rows}

        def predict(self, text, options):
            if text not in self.seen:
                return Prediction(choice=options[0], probabilities=None, latency_ms=0.0)
            return Prediction(choice=self.prototypes and max(self.prototypes, key=lambda l:
                                                            sum(self.prototypes[l])),
                              probabilities=None, latency_ms=0.0)

    rows = [Row(id=f"r{i}", text=f"utterance {i}",
                label=("alpha" if i % 2 else "omega"), group=f"g{i // 3}",
                options=("alpha", "omega")) for i in range(12)]
    leaky = metrics.grouped_cv(rows, Memoriser, folds=2, seed=1)
    in_sample = sum(1 for r in rows)  # every row is seen during some fit
    assert len(leaky["oof"]) == in_sample
    assert leaky["oof_accuracy"]["correct"] < len(rows)


def test_cv_needs_at_least_two_folds():
    with pytest.raises(ValueError):
        metrics.assign_folds(["g1", "g2"], folds=1)
