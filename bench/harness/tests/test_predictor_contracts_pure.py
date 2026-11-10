"""The pure parts of the fitted predictors, pinned without the dependency.

`sklearn` and `torch` are not installed here, so the two fitted arms cannot be executed. Their
*contracts* can be: both reduce the decision to `_argmax(known, values)` and the torch arm
validates its checkpoint before it ever imports torch. Those are the pieces that decide whether a
17-intent model can be scored on an 8-way subset, and whether a malformed checkpoint fails with a
sentence instead of a traceback, so they are tested directly rather than trusted.
"""
from __future__ import annotations

import pytest

from bench.harness.predictors import sklearn_text, torch_text
from bench.harness.protocol import PredictorError

# ------------------------------------------------------------------ restriction to the offer


def test_argmax_honours_the_callers_option_order_on_a_tie():
    values = {"a": 1.0, "b": 1.0}
    assert sklearn_text._argmax(["a", "b"], values) == "a"
    assert sklearn_text._argmax(["b", "a"], values) == "b"
    assert torch_text._argmax(["a", "b"], values) == "a"


def test_argmax_breaks_the_tie_by_name_before_dict_order():
    # same option name twice is impossible in a tuple, but the third key must still be stable
    values = {"b": 2.0, "a": 2.0, "c": 1.0}
    assert sklearn_text._argmax(["a", "b", "c"], values) == "a"


def test_argmax_picks_the_highest_and_ignores_absent_options():
    values = {"grep": 0.2, "read": 0.7}
    assert sklearn_text._argmax(["grep", "read"], values) == "read"
    assert torch_text._argmax(["grep", "read"], values) == "read"


# ------------------------------------------------------------------ the checkpoint contract

@pytest.fixture
def torch_arm(tmp_path):
    """A predictor with no torch: construction only stats the checkpoint path."""
    p = tmp_path / "head.pt"
    p.write_bytes(b"not really a checkpoint")
    return torch_text.TorchTextPredictor(checkpoint=str(p))


GOOD = {"state_dict": {"head.weight": object(), "head.bias": object()},
        "labels": ["filesys_read", "filesys_grep"]}


def test_a_well_formed_checkpoint_validates(torch_arm):
    state_dict, labels, pooling, encoder, max_length = torch_arm._validate(dict(GOOD))
    assert set(state_dict) == {"head.weight", "head.bias"}
    assert labels == ("filesys_read", "filesys_grep")
    assert pooling == "mean" and isinstance(encoder, str) and max_length >= 1


@pytest.mark.parametrize("blob,expected", [
    ("a string", "does not hold a dict"),
    ({}, "no non-empty state_dict"),
    ({"state_dict": {}, "labels": ["a"]}, "no non-empty state_dict"),
    ({"state_dict": {"head.weight": 1}, "labels": []}, "no labels list"),
    ({"state_dict": {"head.weight": 1}, "labels": ["a", "a"]}, "labels repeat"),
    ({"state_dict": {"head.weight": 1}, "labels": ["a"], "pooling": "max"}, "pooling"),
    ({"state_dict": {"head.bias": 1}, "labels": ["a"]}, None),
    ({"state_dict": {"other.weight": 1, "head.bias": 1}, "labels": ["a"]}, "outside head.*"),
    ({"state_dict": {"head.weight": 1}, "labels": [""]}, "non-empty strings"),
])
def test_a_malformed_checkpoint_names_the_defect(torch_arm, blob, expected):
    if expected is None:
        # a head with a bias and no weight is structurally odd but not rejected here: the width
        # check belongs to load, which needs torch. Assert the labels so the case is not a no-op.
        assert torch_arm._validate(blob)[1] == ("a",)
        return
    with pytest.raises(PredictorError) as exc:
        torch_arm._validate(blob)
    assert expected in str(exc.value)


def test_a_directory_checkpoint_finds_the_head_file(tmp_path):
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "head.pt").write_bytes(b"x")
    assert torch_text._checkpoint_file(str(d)).name == "head.pt"


def test_a_directory_without_a_head_file_is_refused(tmp_path):
    d = tmp_path / "ckpt"
    d.mkdir()
    with pytest.raises(PredictorError) as exc:
        torch_text._checkpoint_file(str(d))
    assert "head.pt" in str(exc.value)


def test_a_missing_checkpoint_path_is_refused(tmp_path):
    with pytest.raises(PredictorError):
        torch_text._checkpoint_file(str(tmp_path / "nope.pt"))
