"""The pipeline with a predictor that needs no model, no server and no network.

FakePredictor is right with probability `accuracy` per row, decided by a hash of the seed and
the text, so every count below is exactly computable. The tests recompute the hashes
independently rather than trusting the predictor, because a test that asks the code how many
rows it got right proves nothing.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from bench.harness import controls, metrics
from bench.harness.predictors.fake import FakePredictor, truth_from_rows
from bench.harness.predictors.constant import ConstantPredictor
from bench.harness.protocol import build, option_order_sha, score
from bench.harness.rows import load as load_dataset


@pytest.fixture(scope="module")
def ds():
    return load_dataset("tool_routing")


@pytest.fixture(scope="module")
def small(ds):
    from bench.harness.rows import Dataset
    return Dataset(ds.name, ds.rows[:60], ds.options, ds.source)


def expected_hits(rows, accuracy, seed="0"):
    """Independent reimplementation of the fixture's coin."""
    n = 0
    for r in rows:
        h = hashlib.sha256("\x1f".join(["fake", str(seed), r.text]).encode()).hexdigest()
        if int(h[:16], 16) / float(16 ** 16) < accuracy:
            n += 1
    return n


# ------------------------------------------------------------------ the fixture itself

def test_fake_with_a_cheat_sheet_and_full_accuracy_is_perfect(small):
    p = FakePredictor(truth_from_rows(list(small)), accuracy=1.0)
    recs = score("a", list(small), p)
    assert metrics.accuracy(recs)["correct"] == len(small)


def test_fake_accuracy_zero_never_answers_the_label(small):
    p = FakePredictor(truth_from_rows(list(small)), accuracy=0.0)
    recs = score("a", list(small), p)
    assert metrics.accuracy(recs)["correct"] == 0


@pytest.mark.parametrize("rate", [0.25, 0.5, 0.75])
def test_fake_hit_rate_is_exactly_the_hash_count(small, rate):
    p = FakePredictor(truth_from_rows(list(small)), accuracy=rate, seed="7")
    recs = score("a", list(small), p)
    assert metrics.accuracy(recs)["correct"] == expected_hits(list(small), rate, seed="7")


def test_fake_without_a_cheat_sheet_is_deterministic(small):
    one = score("a", list(small), FakePredictor(accuracy=0.0, seed="x"))
    two = score("a", list(small), FakePredictor(accuracy=0.0, seed="x"))
    assert [r.choice for r in one] == [r.choice for r in two]


def test_fake_probabilities_are_a_distribution(small):
    p = FakePredictor(truth_from_rows(list(small)), accuracy=1.0)
    recs = score("a", list(small), p)
    for r in recs:
        assert r.probabilities is not None
        assert sum(r.probabilities.values()) == pytest.approx(1.0)
        assert max(r.probabilities.values()) == pytest.approx(0.6)


def test_constant_predictor_is_the_baseline_and_needs_no_probabilities(small):
    recs = score("a", list(small), ConstantPredictor(choice="web_search"))
    acc = metrics.accuracy(recs)
    assert all(r.choice == "web_search" for r in recs)
    assert acc["correct"] == sum(1 for r in small if r.label == "web_search")
    assert metrics.calibration(recs) is None


# ------------------------------------------------------------------ the seams together

def test_spec_strings_build_predictors():
    p = build("constant|choice=decide_pick")
    assert p.predict("anything", ("a", "decide_pick")).choice == "decide_pick"
    with pytest.raises(Exception):
        build("nosuchpredictor")


def test_every_row_is_scored_even_when_the_predictor_raises():
    from bench.harness.rows import Dataset, Row

    class Explodes:
        name = "boom"

        def predict(self, text, options):
            raise RuntimeError("nope")

    rows = [Row(id="r1", text="t", label="a", group="g", options=("a", "b"))]
    recs = score("a", rows, Explodes())
    assert len(recs) == 1 and recs[0].choice == "" and recs[0].correct is False
    assert "RuntimeError" in recs[0].meta["error"]
    assert metrics.accuracy(recs)["n"] == 1


def test_an_invalid_choice_is_recorded_not_reinterpreted(small):
    class OffMenu:
        name = "offmenu"

        def predict(self, text, options):
            from bench.harness.protocol import Prediction
            return Prediction(choice="not_an_option", probabilities=None, latency_ms=0.1)

    recs = score("a", list(small)[:3], OffMenu())
    assert all(r.correct is False for r in recs)
    assert recs[0].meta["invalid_choice"] == "not_an_option"


# ------------------------------------------------------------------ order control

def test_order_rev_reverses_and_records_a_new_hash(ds):
    frozen = tuple(ds.options)
    rev = controls.permute(frozen, "rev")
    assert rev == tuple(reversed(frozen))
    assert option_order_sha(rev) != option_order_sha(frozen)
    assert set(rev) == set(frozen)


def test_order_shuffle_is_deterministic_and_a_permutation(ds):
    a = controls.permute(tuple(ds.options), "shuffle:42")
    b = controls.permute(tuple(ds.options), "shuffle:42")
    c = controls.permute(tuple(ds.options), "shuffle:43")
    assert a == b and set(a) == set(ds.options)
    assert a != c


def test_order_control_reaches_the_predictor(small):
    seen = {}

    class Watcher:
        name = "watch"

        def predict(self, text, options):
            from bench.harness.protocol import Prediction
            seen.setdefault("order", tuple(options))
            return Prediction(choice=options[0], probabilities=None, latency_ms=0.0)

    score("a", list(small)[:1], Watcher(), options_for=controls.options_for("rev"))
    assert seen["order"] == tuple(reversed(small.options))


def test_unknown_order_mode_is_an_error(ds):
    with pytest.raises(ValueError):
        controls.permute(tuple(ds.options), "sideways")


# ------------------------------------------------------------------ paraphrase control

def test_paraphrase_rewrites_descriptions_and_never_labels(ds):
    mapping = {"filesys_grep": "Hunt through local files for a pattern."}
    out = controls.apply_paraphrase(ds, mapping)
    assert out.options == ds.options
    before = ds.rows[0]
    after = out.by_id()[before.id]
    assert after.label == before.label and after.options == before.options
    assert after.meta["descriptions"]["filesys_grep"] == "Hunt through local files for a pattern."
    assert out.by_id()[before.id].meta["paraphrased"] == ["filesys_grep"]


def test_paraphrase_refuses_an_option_the_dataset_does_not_have(ds):
    with pytest.raises(Exception) as exc:
        controls.apply_paraphrase(ds, {"not_an_op": "words"})
    assert "not in this dataset" in str(exc.value)


# ------------------------------------------------------------------ pre-registration

def test_condition_grammar_evaluates_both_ways():
    ctx = {"mcnemar_p": 0.004, "delta": 9, "ledger": {"web_search": {"broke": 3}}}
    assert controls.evaluate_condition("mcnemar_p < 0.05", ctx)["passed"] is True
    assert controls.evaluate_condition("mcnemar_p > 0.05", ctx)["passed"] is False
    assert controls.evaluate_condition("delta >= 9", ctx)["passed"] is True
    assert controls.evaluate_condition("ledger.web_search.broke <= 3", ctx)["passed"] is True
    assert controls.evaluate_condition("ledger.web_search.broke < 3", ctx)["passed"] is False


def test_condition_grammar_refuses_a_path_it_cannot_resolve():
    with pytest.raises(KeyError):
        controls.evaluate_condition("nope.here < 1", {"mcnemar_p": 0.1})


def test_condition_grammar_refuses_an_expression_without_an_operator():
    with pytest.raises(Exception):
        controls.evaluate_condition("mcnemar_p", {})


def test_check_reports_every_condition_and_the_conjunction():
    ctx = {"mcnemar_p": 0.004, "ledger": {"web_search": {"broke": 3}}}
    prereg = {"conditions": [{"name": "p", "expr": "mcnemar_p < 0.05"},
                            {"name": "ws", "expr": "ledger.web_search.broke <= 3"}]}
    out = controls.check(prereg, ctx)
    assert out["passed"] is True and len(out["conditions"]) == 2
    prereg["conditions"][1]["expr"] = "ledger.web_search.broke <= 2"
    assert controls.check(prereg, ctx)["passed"] is False


def test_the_floor_gate_is_strictly_more_than_three_times():
    assert controls.gate_verdict(3, 1)["clears"] is False
    assert controls.gate_verdict(4, 1)["clears"] is True
    assert controls.gate_verdict(-4, 1)["clears"] is True      # a loss counts too
    assert controls.gate_verdict(0, 0)["clears"] is False      # a zero floor is not a licence


# ------------------------------------------------------------------ the CLI, end to end

def _truth_file(small, tmp_path, n=25):
    """The fake's cheat sheet, written where a spec string can point at it."""
    truth = {r.text: r.label for r in list(small)[:n]}
    p = tmp_path / "truth.json"
    p.write_text(json.dumps(truth))
    return p


def test_cli_writes_an_artifact_and_prints_no_verdict_without_a_preregistration(
        small, tmp_path, capsys):
    from bench.harness.run import main
    out = tmp_path / "run"
    truth = _truth_file(small, tmp_path)
    rc = main(["--dataset", "tool_routing", "--limit", "25",
               "--arm", f"ref=fake|accuracy=1.0|seed=1|truth={truth}",
               "--arm", f"treat=fake|accuracy=0.0|seed=1|truth={truth}",
               "--reference", "ref", "--aa", "--write-rows", "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 0
    assert "no --preregister file" in text and "VERDICT" not in text
    summary = json.loads((out / "summary.json").read_text())
    assert summary["dataset"]["order_sha"] == option_order_sha(summary["dataset"]["options"])
    assert summary["arms"]["ref"]["accuracy"]["correct"] == 25
    assert summary["arms"]["treat"]["accuracy"]["correct"] == 0
    assert summary["arms"]["treat"]["ledger_vs_reference"]["net"] == -25
    assert (out / "rows.jsonl").read_text().count("\n") == 50


def test_cli_with_a_preregistration_prints_a_verdict(small, tmp_path, capsys):
    from bench.harness.run import main
    pre = tmp_path / "pre.json"
    pre.write_text(json.dumps({
        "arm": "treat", "edited_labels": ["filesys_grep"],
        "conditions": [{"name": "material", "expr": "delta <= -20"},
                       {"name": "not_noise", "expr": "aa_flips >= 0"}]}))
    truth = _truth_file(small, tmp_path)
    rc = main(["--dataset", "tool_routing", "--limit", "25",
               "--arm", f"ref=fake|accuracy=1.0|seed=1|truth={truth}",
               "--arm", f"treat=fake|accuracy=0.0|seed=1|truth={truth}",
               "--reference", "ref", "--aa", "--preregister", str(pre),
               "--out", str(tmp_path / "run")])
    text = capsys.readouterr().out
    assert rc == 0
    assert "VERDICT: PASS" in text
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["prereg"]["passed"] is True


def test_cli_order_control_changes_the_recorded_hash(tmp_path, capsys):
    from bench.harness.run import main
    main(["--dataset", "tool_routing", "--limit", "10", "--order", "rev",
          "--predictor", "constant|choice=filesys_grep", "--out", str(tmp_path / "a")])
    frozen = json.loads((tmp_path / "a" / "summary.json").read_text())["dataset"]
    main(["--dataset", "tool_routing", "--limit", "10",
          "--predictor", "constant|choice=filesys_grep", "--out", str(tmp_path / "b")])
    plain = json.loads((tmp_path / "b" / "summary.json").read_text())["dataset"]
    assert frozen["order_sha"] != plain["order_sha"]
    assert frozen["order_mode"] == "rev" and plain["order_mode"] == "frozen"
    assert sorted(frozen["options"]) == sorted(plain["options"])
