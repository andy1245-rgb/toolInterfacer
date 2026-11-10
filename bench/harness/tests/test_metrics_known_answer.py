"""Every metric against a hand-computed case, and the two statistics against the repo's own.

The tests are deliberately arithmetic a reader can redo on paper: no golden files, no
snapshots, nothing that can be "updated" to match a bug.
"""
from __future__ import annotations

import math

import pytest

from bench.harness import metrics
from bench.harness.protocol import Record

# ------------------------------------------------------------------ fixtures


def rec(id, label, choice, conf=None, label_prob=None, latency=1.0, group="g1"):
    probs = None
    if conf is not None:
        probs = {label: conf}
        # a second option so a multi-class Brier is exercised
        probs["other"] = 1.0 - conf
        if label_prob is not None:
            probs[label] = label_prob
            probs["other"] = 1.0 - label_prob
    return Record(arm="a", id=id, group=group, label=label, text=id, choice=choice,
                  correct=(choice == label), latency_ms=latency, probabilities=probs)


# ------------------------------------------------------------------ McNemar

@pytest.mark.parametrize("b,c,expected", [
    (0, 0, 1.0),
    (0, 1, 1.0),
    (1, 1, 1.0),
    (0, 5, 0.0625),          # 2 * (1/2^5)
    (3, 12, 0.03515625),     # 2 * (1 + 12 + 66 + 220) / 2^15
    (10, 7, 0.629058837890625),
])
def test_mcnemar_known_answers(b, c, expected):
    assert metrics.mcnemar_exact(b, c) == pytest.approx(expected, abs=1e-12)


def test_mcnemar_agrees_with_the_repo_implementation():
    """The harness must not drift from the statistic the repo's reports were computed with."""
    from bench.ti_bench.stats import mcnemar_exact as repo_mcnemar
    for b in range(0, 21):
        for c in range(0, 21):
            assert metrics.mcnemar_exact(b, c) == pytest.approx(repo_mcnemar(b, c), abs=1e-12)


def test_mcnemar_is_symmetric():
    for b, c in ((2, 9), (0, 0), (7, 7)):
        assert metrics.mcnemar_exact(b, c) == metrics.mcnemar_exact(c, b)


# ------------------------------------------------------------------ Wilson

def test_wilson_matches_the_published_interval():
    """325/415 is the floor_a record; the doc quotes [74.1, 82.0]."""
    lo, hi = metrics.wilson_ci(325, 415)
    assert (round(100 * lo, 1), round(100 * hi, 1)) == (74.1, 82.0)


def test_wilson_edges():
    assert metrics.wilson_ci(0, 0) == (0.0, 0.0)
    lo, hi = metrics.wilson_ci(0, 10)
    assert lo == 0.0 and 0.0 < hi < 0.35


# ------------------------------------------------------------------ accuracy, ledger, pairs

def test_accuracy_counts():
    recs = [rec("1", "a", "a"), rec("2", "a", "b"), rec("3", "b", "b"), rec("4", "b", "b")]
    acc = metrics.accuracy(recs)
    assert acc["n"] == 4 and acc["correct"] == 3
    assert acc["accuracy"] == pytest.approx(0.75)


def test_ledger_books_by_reference_label():
    base = [rec("1", "web_search", "web_search"), rec("2", "web_search", "web_search"),
            rec("3", "filesys_grep", "filesys_glob")]
    treat = [rec("1", "web_search", "filesys_grep"),      # broken by the treatment
             rec("2", "web_search", "web_search"),        # kept
             rec("3", "filesys_grep", "filesys_grep")]    # fixed
    led = metrics.ledger(base, treat)
    assert led["by_label"]["web_search"] == {"n": 2, "base": 2, "treat": 1,
                                             "fixed": 0, "broke": 1, "net": -1}
    assert led["by_label"]["filesys_grep"] == {"n": 1, "base": 0, "treat": 1,
                                               "fixed": 1, "broke": 0, "net": 1}
    assert led["net"] == 0 and led["fixed"] == 1 and led["broke"] == 1
    assert metrics.ledger_net(led, ["web_search"]) == -1
    assert metrics.ledger_net(led, ["filesys_grep"]) == 1


def test_paired_reports_both_directions_and_the_p():
    base = [rec(str(i), "a", "a") for i in range(4)]
    treat = [rec("0", "a", "b"), rec("1", "a", "b"), rec("2", "a", "a"), rec("3", "a", "a")]
    p = metrics.paired(base, treat)
    assert (p["n"], p["fixed"], p["broke"]) == (4, 0, 2)
    assert p["p"] == pytest.approx(0.5)          # 2*0.25
    assert p["treatment_correct"] - p["reference_correct"] == -2


def test_a_a_floor_counts_choice_flips_not_only_correctness():
    first = [rec("1", "a", "a"), rec("2", "a", "a"), rec("3", "a", "b")]
    second = [rec("1", "a", "a"), rec("2", "a", "b"), rec("3", "a", "b")]
    floor = metrics.a_a_flips(first, second)
    assert floor["choice_flips"] == 1
    assert floor["correctness_flips"] == 1
    assert floor["rate"] == pytest.approx(1 / 3)


# ------------------------------------------------------------------ calibration

def test_ece_hand_computed():
    # two confident-and-half-right rows, two half-confident-and-half-right rows
    recs = [rec("1", "a", "a", conf=1.0), rec("2", "a", "b", conf=1.0),
            rec("3", "a", "a", conf=0.5), rec("4", "a", "b", conf=0.5)]
    cal = metrics.calibration(recs)
    # bin 9: conf 1.0, n=2, acc 0.5 -> |0.5-1.0| * 2/4 = 0.25 ; bin 5: acc 0.5, conf 0.5 -> 0
    assert cal["ece"] == pytest.approx(0.25)
    assert cal["n_scored"] == 4 and cal["n_skipped"] == 0


def test_brier_and_nll_hand_computed():
    r = rec("1", "a", "a", conf=0.8)
    cal = metrics.calibration([r])
    # Brier over {a:0.8, other:0.2} with label a: (0.8-1)^2 + (0.2-0)^2 = 0.08
    assert cal["brier"] == pytest.approx(0.08)
    assert cal["nll"] == pytest.approx(-math.log(0.8))


def test_rows_without_probabilities_are_excluded_and_counted():
    recs = [rec("1", "a", "a", conf=0.9), Record(arm="a", id="2", group="g1", label="a",
                                                 text="x", choice="a", correct=True,
                                                 latency_ms=1.0, probabilities=None)]
    cal = metrics.calibration(recs)
    assert cal["n_scored"] == 1 and cal["n_skipped"] == 1
    assert metrics.calibration([recs[1]]) is None


# ------------------------------------------------------------------ reject curve

def test_reject_curve_shape_and_precision():
    recs = [rec(str(i), "a", "a", conf=0.8) for i in range(5)]
    curve = metrics.reject_curve(recs, step=0.1)
    assert curve[0]["threshold"] == 0.0 and curve[-1]["threshold"] == pytest.approx(1.0)
    at80 = [p for p in curve if p["threshold"] == pytest.approx(0.8)][0]
    assert at80["coverage"] == 1.0 and at80["precision"] == 1.0
    above = [p for p in curve if p["threshold"] == pytest.approx(0.9)][0]
    assert above["coverage"] == 0.0 and above["precision"] is None


def test_coverage_at_precision_is_the_largest_clean_prefix():
    recs = ([rec(f"hi{i}", "a", "a", conf=0.95) for i in range(5)] +
            [rec(f"lo{i}", "a", "b", conf=0.10) for i in range(5)])
    cap = metrics.coverage_at_precision(recs, target=0.9)
    assert cap["coverage"] == pytest.approx(0.5) and cap["n_accepted"] == 5
    assert cap["precision"] == 1.0 and cap["threshold"] == pytest.approx(0.95)


def test_coverage_at_precision_cannot_be_bought_by_answering_nothing():
    recs = [rec("1", "a", "b", conf=0.2)]
    cap = metrics.coverage_at_precision(recs, target=0.9)
    assert cap["coverage"] == 0.0 and cap["n_accepted"] == 0


# ------------------------------------------------------------------ latency

def test_latency_percentiles():
    recs = [Record(arm="a", id=str(i), group="g", label="a", text="x", choice="a",
                   correct=True, latency_ms=float(i), probabilities=None)
            for i in range(1, 101)]
    lat = metrics.latency(recs)
    assert lat["n"] == 100
    assert lat["p50"] == pytest.approx(51.0, abs=1.0)
    assert lat["p95"] == pytest.approx(96.0, abs=1.0)
    assert lat["max"] == 100.0


# ------------------------------------------------------------------ folds

def test_fold_assignment_is_stable_and_group_safe():
    groups = [f"fam{i % 7}" for i in range(50)]
    a = metrics.assign_folds(groups, folds=5, seed=11)
    b = metrics.assign_folds(groups, folds=5, seed=11)
    assert a == b
    assert set(a.values()) <= set(range(5))
    assert metrics.assign_folds(groups, folds=5, seed=12) != a or len(a) == 1


def test_grouped_cv_needs_fit():
    from bench.harness.rows import Row
    rows = [Row(id=f"r{i}", text=f"t{i}", label="a", group=f"g{i % 5}",
                options=("a", "b")) for i in range(20)]

    class NoFit:
        name = "nofit"

        def predict(self, text, options):
            raise AssertionError("should not be called")

    with pytest.raises(TypeError):
        metrics.grouped_cv(rows, NoFit, folds=2)
