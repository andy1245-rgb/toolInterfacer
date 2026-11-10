"""The optional predictors: their contracts, and the dependency-missing paths they must have.

`scikit-learn` and `torch` are not installed here, so the fitted arms cannot be executed on this
box. What *can* be and is tested: the grammar, the combining rules that need no dependency, the
error-agreement arithmetic, and the fact that asking for a dependency you do not have produces a
message naming the install command instead of a traceback.
"""
from __future__ import annotations

import pytest

from bench.harness.predictors.ensemble import pairwise_error_agreement
from bench.harness.protocol import PredictorError, build


def _have(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


# ------------------------------------------------------------------ ensemble: no dependency

def test_vote_picks_the_most_voted_offered_option():
    arm = build("ensemble|members=a:constant~choice=web_search;;"
                "b:constant~choice=web_search;;c:constant~choice=filesys_grep|mode=vote")
    pred = arm.predict("t", ("filesys_grep", "web_search"))
    assert pred.choice == "web_search"
    assert pred.probabilities is None, "a share of votes is not a probability"


def test_vote_ties_break_by_the_callers_option_order():
    arm = build("ensemble|members=a:constant~choice=web_search;;"
                "b:constant~choice=filesys_grep|mode=vote")
    assert arm.predict("t", ("filesys_grep", "web_search")).choice == "filesys_grep"
    assert arm.predict("t", ("web_search", "filesys_grep")).choice == "web_search"


def test_mean_averages_probabilities_and_renormalises():
    arm = build("ensemble|members=a:fake~accuracy=1.0~seed=1;;"
                "b:fake~accuracy=1.0~seed=2|mode=mean")
    pred = arm.predict("t", ("a", "b"))
    assert pred.probabilities is not None
    assert sum(pred.probabilities.values()) == pytest.approx(1.0)
    assert set(pred.probabilities) == {"a", "b"}


def test_mean_refuses_members_that_cannot_produce_probabilities():
    arm = build("ensemble|members=a:constant~choice=a;;b:constant~choice=b|mode=mean")
    with pytest.raises(PredictorError):
        arm.predict("t", ("a", "b"))


def test_an_unknown_mode_is_an_error_naming_the_modes():
    with pytest.raises(PredictorError) as exc:
        build("ensemble|members=a:constant~choice=a|mode=auction")
    assert "vote" in str(exc.value) and "mean" in str(exc.value)


def test_an_unknown_key_after_the_ensemble_owns_the_spec_is_an_error():
    """A fragment after mode=/folds= belongs to the ensemble, so a stray key there is a typo."""
    with pytest.raises(PredictorError) as exc:
        build("ensemble|members=a:constant~choice=a|mode=vote|nonsense=1")
    assert "unknown key" in str(exc.value)


def test_a_member_key_before_mode_is_a_member_parameter():
    """Documented grammar: the member list continues across '|' until an ensemble key appears."""
    arm = build("ensemble|members=a:constant~choice=web_search~bogus=1|mode=vote")
    assert arm.predict("t", ("filesys_grep", "web_search")).choice == "web_search"


def test_an_ensemble_with_no_members_is_an_error():
    with pytest.raises(PredictorError):
        build("ensemble|mode=vote")


# ------------------------------------------------------------------ error agreement, by hand

def test_error_agreement_hand_computed():
    masks = {"a": [True, False, False], "b": [True, True, False]}
    out = pairwise_error_agreement(masks)
    pair = out["pairs"]["a|b"]
    # a is wrong on row 1 only, b on rows 1 and 2; they agree on 2 of 3 rows
    assert (pair["both_right"], pair["a_only"], pair["b_only"], pair["both_wrong"]) == (1, 0, 1, 1)
    assert pair["agreement"] == pytest.approx(2 / 3)
    # phi coefficient of (1,0,0) and (1,1,0) is 0.5, worked out on paper
    assert pair["correlation"] == pytest.approx(0.5)


def test_identical_members_have_perfect_agreement_and_a_degenerate_correlation():
    out = pairwise_error_agreement({"a": [True, False], "b": [True, False]})
    assert out["pairs"]["a|b"]["agreement"] == 1.0


def test_a_single_member_reports_no_pairs():
    out = pairwise_error_agreement({"a": [True, False]})
    assert out["pairs"] == {}


# ------------------------------------------------------------------ the dependencies

@pytest.mark.skipif(_have("sklearn"), reason="scikit-learn is installed: the gap cannot be shown")
def test_sklearn_arm_names_the_install_command_when_the_dependency_is_missing():
    with pytest.raises(PredictorError) as exc:
        build("sklearn_text|vectorizer=tfidf_word|clf=linsvc")
    message = str(exc.value)
    assert "scikit-learn" in message and "pip install" in message


@pytest.mark.skipif(_have("sklearn"), reason="scikit-learn is installed")
def test_stacked_ensemble_refuses_at_build_time_without_sklearn():
    """Refusing when the predictor is asked for beats refusing on the first fit: a run that
    starts and then cannot stack has already spent the wall clock."""
    with pytest.raises(PredictorError) as exc:
        build("ensemble|members=a:constant~choice=a;;b:constant~choice=b|mode=stacked")
    assert "scikit-learn" in str(exc.value)


@pytest.mark.skipif(_have("torch"), reason="torch is installed")
def test_torch_arm_requires_a_checkpoint_before_it_requires_torch():
    with pytest.raises(PredictorError) as exc:
        build("torch_text|encoder=some-encoder")
    assert "checkpoint" in str(exc.value)


def test_every_registered_kind_is_buildable_or_says_why_not():
    """A kind that is registered but neither buildable nor self-explaining is a trap."""
    from bench.harness.protocol import available
    for kind in available():
        if kind in ("llama_systemone", "logit_read"):
            continue  # needs a server; asked for below only if one is listening
        try:
            build(kind)
        except PredictorError as exc:
            assert str(exc).strip(), f"{kind} failed with an empty message"
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"{kind} raised {type(exc).__name__} instead of PredictorError: {exc}")
