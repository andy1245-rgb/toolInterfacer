"""The seams a result measured elsewhere needs: an explicit fold map, an explicit fit pool.

Without these two options the harness can only score a *new* experiment. Re-scoring one that
already exists means its folds and its training set, not folds and a training set that merely
look similar — on this corpus the difference between training on the reference utterances and
training on the other folds of the generated rows is several points.
"""

from __future__ import annotations

import json

import pytest

from harness import metrics
from harness import rows as R
from harness.protocol import build


def tiny_rows(groups_and_labels, options=("a", "b", "c")):
    """Rows from ``[(group, label), ...]``, one row each, text carrying the label."""
    return [
        R.Row(id=f"r{i}", text=f"{label} {i}", label=label, group=group, options=options)
        for i, (group, label) in enumerate(groups_and_labels)
    ]


class Counter:
    """A predictor that records which rows each fold was fitted on."""

    def __init__(self) -> None:
        self.fits: list[list[str]] = []
        self.rows: list = []

    def fit(self, rows):
        self.fits.append([row.id for row in rows])
        self.rows = rows

    def predict(self, text, options):
        from harness.protocol import Prediction

        return Prediction(choice=options[0], probabilities=None, latency_ms=0.0)


# ------------------------------------------------------------------ explicit folds

def test_an_explicit_fold_map_is_used_instead_of_assigning_one():
    rows = tiny_rows([("g1", "a"), ("g2", "b"), ("g3", "c"), ("g4", "a")])
    mapping = {"g1": 0, "g2": 0, "g3": 1, "g4": 1}
    cv = metrics.grouped_cv(rows, Counter, fold_of=mapping, stratify=True)
    assert cv["fold_map"] == "explicit"
    assert cv["fold_of"] == mapping
    assert cv["folds"] == 2
    assert [fold["n_test"] for fold in cv["per_fold"]] == [2, 2]


def test_a_fold_map_that_misses_a_group_is_an_error_not_a_silent_drop():
    rows = tiny_rows([("g1", "a"), ("g2", "b")])
    with pytest.raises(ValueError, match="does not cover"):
        metrics.grouped_cv(rows, Counter, fold_of={"g1": 0})


# ------------------------------------------------------------------ explicit fit pool

def test_fit_pool_supplies_the_training_rows_instead_of_the_other_folds():
    rows = tiny_rows([("g1", "a"), ("g2", "b")])
    pool = [
        R.Row(id="p1", text="see a", label="a", group="refs", options=("a", "b", "c")),
        R.Row(id="p2", text="see b", label="b", group="refs", options=("a", "b", "c")),
    ]
    make = lambda: Counter()  # noqa: E731 - a fresh recorder per fold
    cv = metrics.grouped_cv(rows, make, fold_of={"g1": 0, "g2": 1}, fit_pool=pool)
    assert cv["fit_pool"] == 2
    assert [fold["n_train"] for fold in cv["per_fold"]] == [2, 2]
    # the pool, not the other fold's rows, is what the model saw
    assert all(fold["fit_pool_dropped"] == 0 for fold in cv["per_fold"])


def test_a_pool_row_that_leaks_into_the_held_out_group_is_dropped():
    rows = tiny_rows([("g1", "a"), ("g2", "b")])
    pool = [
        R.Row(id="clean", text="x", label="a", group="refs", options=("a", "b", "c"),
              meta={"leaks_into": []}),
        R.Row(id="leaky", text="y", label="b", group="refs", options=("a", "b", "c"),
              meta={"leaks_into": ["g1"]}),
    ]
    cv = metrics.grouped_cv(rows, Counter, fold_of={"g1": 0, "g2": 1}, fit_pool=pool)
    # fold 0 holds g1 out, so "leaky" is dropped there and kept in fold 1
    assert cv["per_fold"][0]["n_train"] == 1
    assert cv["per_fold"][1]["n_train"] == 2
    assert cv["per_fold"][0]["fit_pool_dropped"] == 1


def test_a_pool_row_whose_own_group_is_held_out_is_dropped():
    rows = tiny_rows([("g1", "a"), ("g2", "b")])
    pool = [
        R.Row(id="p1", text="x", label="a", group="g1", options=("a", "b", "c")),
        R.Row(id="p2", text="y", label="b", group="refs", options=("a", "b", "c")),
    ]
    cv = metrics.grouped_cv(rows, Counter, fold_of={"g1": 0, "g2": 1}, fit_pool=pool)
    assert cv["per_fold"][0]["n_train"] == 1
    assert cv["per_fold"][1]["n_train"] == 2


# ------------------------------------------------------------------ the localalexa datasets

def test_localalexa_can_group_by_phrasing_family():
    by_family = R.load("localalexa_intents")
    by_phrasing = R.load("localalexa_intents|group_by=phrasing")
    assert len(by_family) == len(by_phrasing) == 382
    assert len(by_family.groups()) == 7
    assert len(by_phrasing.groups()) == 110
    # the domain column survives on the row either way
    assert by_phrasing.rows[0].meta["family"] == by_family.rows[0].group


def test_an_unknown_group_by_is_refused():
    with pytest.raises(R.DatasetError, match="group_by"):
        R.load("localalexa_intents|group_by=topic")


def test_localalexa_refs_is_the_training_pool_with_its_leaks_named():
    refs = R.load("localalexa_refs")
    assert len(refs) == 200
    assert len(refs.labels()) == 18
    leaking = [row for row in refs.rows if row.meta["leaks_into"]]
    # 29 of the 200 references are literal slot fills of a template; those are the rows a
    # fold has to drop along with the phrasing family it is holding out
    assert len(leaking) == 29
    assert all(isinstance(row.meta["leaks_into"], list) for row in refs.rows)
    # every reference is an option, so a fold can always fit on it
    assert all(row.label in row.options for row in refs.rows)


def test_the_reference_pool_and_the_scored_set_agree_on_the_option_order():
    refs = R.load("localalexa_refs")
    scored = R.load("localalexa_intents")
    assert refs.options == scored.options


# ------------------------------------------------------------------ ensemble standardisation

def test_standardise_is_only_for_the_mean_mode():
    spec = "ensemble|members=a:sklearn_text|mode=vote|standardise=true"
    with pytest.raises(Exception, match="standardise applies to mode=mean"):
        build(spec)


def test_standardise_shows_on_the_arm_name():
    spec = ("ensemble|members=a:sklearn_text~clf=logreg;;b:sklearn_text~clf=linsvc"
            "|mode=mean|standardise=true")
    predictor = build(spec)
    assert predictor.standardise is True
    assert predictor.name.endswith(":standardised")


def test_standardising_rescales_without_reordering_a_member():
    from harness.predictors.ensemble import _standardise

    shares = {"a": 0.8, "b": 0.15, "c": 0.05}
    standardised = _standardise(shares)
    # z-scores, not a distribution: they centre on zero and keep the ordering
    assert abs(sum(standardised.values())) < 1e-9
    assert sorted(standardised, key=lambda o: -standardised[o]) == ["a", "b", "c"]


def test_a_sharp_member_and_a_flat_member_vote_differently_after_standardising():
    from harness.predictors.ensemble import _standardise

    sharp = {"a": 0.97, "b": 0.02, "c": 0.01}
    flat = {"a": 0.40, "b": 0.35, "c": 0.25}
    sharp_z, flat_z = _standardise(sharp), _standardise(flat)
    # in raw probabilities the flat member's opinion on "b" outweighs the sharp member's by
    # about 18x, so the sharp one is effectively silent; standardised, the two are within a
    # factor of two of each other and the disagreement actually arrives
    raw_ratio = flat["b"] / sharp["b"]
    z_ratio = abs(flat_z["b"] / sharp_z["b"])
    assert raw_ratio > 10
    assert z_ratio < 2
    assert sharp_z["b"] < 0.0 < flat_z["b"]


# ------------------------------------------------------------------ torch without a file yet

def test_a_torch_arm_with_a_trainer_may_start_without_a_checkpoint(tmp_path):
    from harness.predictors.torch_text import build as build_torch

    target = tmp_path / "not-written-yet.pt"
    predictor = build_torch(checkpoint=str(target),
                            trainer="localalexa.harness_adapters:train_setfit")
    assert str(predictor.checkpoint) == str(target)


def test_a_torch_arm_without_a_trainer_still_needs_the_file(tmp_path):
    from harness.predictors.torch_text import build as build_torch

    with pytest.raises(Exception, match="does not exist"):
        build_torch(checkpoint=str(tmp_path / "absent.pt"))
