"""The two seams tonight's plan depends on: a fold map that cannot hide a label, and a
fusion arm that can outvote the 3B.

The fusion numbers these tests guard are 351/415 = 84.6% (equal-weight mean, exact McNemar
p = 0.00022) against the 3B's 325/415 = 78.3%, and 330/415 for the confidence gate. Those are
run-level results; what is pinned here is the arithmetic and the refusal paths, so a later
edit cannot quietly turn a blend into a coin flip.
"""
from __future__ import annotations

import json

import pytest

from bench.harness import controls, metrics, run
from bench.harness.protocol import Prediction, Record, build
from bench.harness.rows import Row


# ------------------------------------------------------------------ helpers

class StubMember:
    """A member with a hand-set distribution, so the blend is checkable on paper."""

    name = "stub_member"

    def __init__(self, choice: str, probabilities: dict[str, float] | None):
        self.choice = choice
        self.probabilities = probabilities
        self.seen_options: tuple[str, ...] = ()

    def predict(self, text: str, options: tuple[str, ...]) -> Prediction:
        self.seen_options = tuple(options)
        return Prediction(choice=self.choice, probabilities=self.probabilities, latency_ms=0.0)


def _write_base(tmp_path, rows: list[tuple[str, str, dict[str, float]]]) -> str:
    path = tmp_path / "base.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for id_, choice, probs in rows:
            fh.write(json.dumps({"arm": "base", "id": id_, "choice": choice,
                                 "probabilities": probs}) + "\n")
    return str(path)


def _row(id_: str, label: str, group: str = "g", options=("a", "b")) -> Row:
    return Row(id=id_, text=f"text {id_}", label=label, group=group, options=tuple(options))


# ------------------------------------------------------------------ fusion

def test_mean_blends_the_two_distributions_including_the_weight(tmp_path):
    base_path = _write_base(tmp_path, [("r1", "a", {"a": 0.6, "b": 0.4})])
    fusion = build(f"fusion|member=fake|base_arm=base|base_path={base_path}|weights=0.5")
    fusion.member = StubMember("a", {"a": 0.2, "b": 0.8})
    pred = fusion.predict_for(_row("r1", "a"))
    assert pred.probabilities == pytest.approx({"a": 0.4, "b": 0.6})
    assert pred.choice == "b"            # the member outvotes a base that weakly prefers a
    assert pred.meta["fusion"] == "mean"
    assert sum(pred.probabilities.values()) == pytest.approx(1.0)


def test_the_weight_is_the_base_weight_and_zero_copies_the_member(tmp_path):
    base_path = _write_base(tmp_path, [("r1", "a", {"a": 0.9, "b": 0.1})])
    fusion = build(f"fusion|member=fake|base_arm=base|base_path={base_path}|weights=0.0")
    fusion.member = StubMember("a", {"a": 0.2, "b": 0.8})
    assert fusion.predict_for(_row("r1", "a")).choice == "b"


def test_gate_takes_the_member_below_the_threshold_and_the_base_above_it(tmp_path):
    # "unsure" is a flat distribution, not a confident pick of the other option: the gate
    # reads the base's *max* probability, so a 0.70 spread is a confident answer at 0.70
    base_path = _write_base(tmp_path, [("low", "a", {"a": 0.33, "b": 0.31}),
                                       ("high", "a", {"a": 0.80, "b": 0.20})])
    fusion = build(f"fusion|member=fake|base_arm=base|base_path={base_path}"
                   f"|mode=gate|t=0.38")
    fusion.member = StubMember("b", None)          # a choice and no probabilities: gate allows it
    low = fusion.predict_for(_row("low", "a"))
    high = fusion.predict_for(_row("high", "a"))
    assert (low.choice, low.meta["fusion"]) == ("b", "member")
    assert (high.choice, high.meta["fusion"]) == ("a", "base")


def test_mean_refuses_a_member_without_probabilities(tmp_path):
    base_path = _write_base(tmp_path, [("r1", "a", {"a": 0.6, "b": 0.4})])
    fusion = build(f"fusion|member=fake|base_arm=base|base_path={base_path}|weights=0.5")
    fusion.member = StubMember("a", None)
    with pytest.raises(Exception, match="emits no probabilities"):
        fusion.predict_for(_row("r1", "a"))


def test_a_row_the_base_never_saw_is_an_error_not_a_guess(tmp_path):
    base_path = _write_base(tmp_path, [("r1", "a", {"a": 0.6, "b": 0.4})])
    fusion = build(f"fusion|member=fake|base_arm=base|base_path={base_path}")
    fusion.member = StubMember("a", {"a": 0.5, "b": 0.5})
    with pytest.raises(Exception, match="no row 'other'"):
        fusion.predict_for(_row("other", "a"))


def test_the_member_spec_escapes_its_own_pipes():
    pytest.importorskip("sklearn")
    fusion = build("fusion|member=sklearn_text~vectorizer=tfidf_word~clf=logreg|weights=0.5")
    assert fusion.member_spec == "sklearn_text|vectorizer=tfidf_word|clf=logreg"
    assert "logreg" in fusion.member.name


def test_the_member_sees_the_permuted_option_order(tmp_path):
    base_path = _write_base(tmp_path, [("r1", "a", {"a": 0.9, "b": 0.1})])
    fusion = build(f"fusion|member=fake|base_arm=base|base_path={base_path}|mode=gate|t=0.1")
    stub = StubMember("b", None)
    fusion.member = stub
    fusion.predict_for(_row("r1", "a"), ("b", "a"))
    assert stub.seen_options == ("b", "a")


# ------------------------------------------------------------------ folds

class MajorityLabel:
    """The smallest possible fitted predictor: it needs the training labels and nothing else."""

    name = "majority_label"

    def fit(self, rows):
        counts: dict[str, int] = {}
        for r in rows:
            counts[r.label] = counts.get(r.label, 0) + 1
        self.label = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

    def predict(self, text, options):
        return Prediction(choice=self.label if self.label in options else options[0],
                          probabilities=None, latency_ms=0.0)


def test_every_training_fold_keeps_every_label_on_the_real_corpus():
    from bench.harness.rows import load

    rows = list(load("tool_routing"))
    cv = metrics.grouped_cv(rows, MajorityLabel, folds=5, seed=0)
    assert (cv["n_groups"], cv["n_labels"]) == (133, 8)
    for fold in cv["per_fold"]:
        assert fold["n_test"] > 0
        assert fold["train_missing_labels"] == [], fold
    assert len(cv["oof"]) == len(rows)


def test_a_corpus_with_one_group_per_label_is_flagged_not_silently_averaged():
    labels = [f"op{i}" for i in range(7)]
    rows = [Row(id=f"r{i}", text=f"say {lab}", label=lab, group=lab, options=tuple(labels))
            for i, lab in enumerate(labels)]
    cv = metrics.grouped_cv(rows, MajorityLabel, folds=5, seed=0)
    missing = [f for f in cv["per_fold"] if f.get("train_missing_labels")]
    assert missing, "a held-out label must be reported, not averaged in as a zero"
    assert {lab for f in cv["per_fold"] for lab in f["train_missing_labels"]} == set(labels)


def test_the_unstratified_map_is_still_available_and_differs():
    groups = [f"g{i}" for i in range(30)]
    plain = metrics.assign_folds(groups, folds=5, seed=0)
    strat = metrics.assign_folds(groups, folds=5, seed=0,
                                 labels=[(g, g) for g in groups])
    assert set(plain) == set(strat) == set(groups)
    assert len(set(plain.values())) == 5


# ------------------------------------------------------------------ metrics that lied

def _recs(pairs):
    return [Record(arm="a", id=f"r{i}", group="g", label=p[2], text="t", choice=p[1],
                   correct=p[0] == p[1], latency_ms=1.0, probabilities=p[3])
            for i, p in enumerate(pairs)]


def test_the_reported_threshold_defines_the_reported_set():
    recs = _recs([("a", "a", "a", {"a": 0.95, "b": 0.05}),
                  ("a", "a", "a", {"a": 0.95, "b": 0.05}),
                  ("b", "a", "b", {"a": 0.50, "b": 0.50}),
                  ("a", "a", "a", {"a": 0.30, "b": 0.70})])
    cap = metrics.coverage_at_precision(recs, 0.9)
    accepted = [r for r in recs if max(r.probabilities.values()) >= cap["threshold"]]
    assert len(accepted) / len(recs) == pytest.approx(cap["coverage"])
    assert sum(1 for r in accepted if r.correct) / len(accepted) == pytest.approx(cap["precision"])


def test_a_one_row_delta_does_not_clear_a_zero_floor():
    assert controls.gate_verdict(1, 0)["clears"] is False
    assert controls.gate_verdict(4, 0)["clears"] is True
    assert controls.gate_verdict(4, 2)["clears"] is False      # 3x the floor is 6
    assert controls.gate_verdict(7, 2)["clears"] is True


# ------------------------------------------------------------------ the CLI

def test_cv_artifacts_carry_the_out_of_fold_rows_not_an_unfitted_pass(tmp_path):
    pytest.importorskip("sklearn")
    out = tmp_path / "artifact"
    rc = run.main(["--dataset", "tool_routing", "--limit", "80",
                   "--arm", "base=recorded|arm=floor_a",
                   "--arm", "member=sklearn_text|vectorizer=tfidf_word|clf=logreg",
                   "--reference", "base", "--cv", "2", "--out", str(out)])
    assert rc == 0
    summary = json.loads((out / "summary.json").read_text())
    member = summary["arms"]["member"]
    assert member["accuracy"]["n"] == 80
    assert member["accuracy"]["correct"] > 0, "an unfitted pass would read 0/80"
    assert member["grouped_cv"]["n_labels"] == 8
    rows = [json.loads(line) for line in (out / "rows.jsonl").read_text().splitlines()]
    member_rows = [r for r in rows if r["arm"] == "member"]
    assert len(member_rows) == 80
    assert sum(1 for r in member_rows if r["correct"]) == member["accuracy"]["correct"]


def test_order_is_refused_for_an_arm_that_cannot_see_it():
    with pytest.raises(SystemExit, match="cannot be honoured"):
        run.main(["--dataset", "tool_routing", "--limit", "20",
                  "--arm", "base=recorded|arm=floor_a", "--order", "rev"])


# ------------------------------------------------------------------ fixes from the review

def test_stratification_repairs_a_map_that_loses_a_label():
    """The unstratified map is not just different, it is sometimes unanswerable.

    Two groups per label and two folds: a label whose two groups both hash into the test fold
    never appears in training. The search over seeds is deterministic, so this is a proof and
    not a flake.
    """
    options = tuple(f"op{i}" for i in range(8))
    rows = [Row(id=f"r{i}-{g}", text=f"say {group}", label=label, group=group, options=options)
            for i, label in enumerate(options) for g in range(2)
            for group in [f"{label}-{g}"]]
    broken = None
    for seed in range(8):
        cv = metrics.grouped_cv(rows, MajorityLabel, folds=2, seed=seed, stratify=False)
        if any(f["train_missing_labels"] for f in cv["per_fold"]):
            broken = seed
            break
    assert broken is not None, "the hashed map is supposed to be able to lose a label"
    fixed = metrics.grouped_cv(rows, MajorityLabel, folds=2, seed=broken, stratify=True)
    assert all(f["train_missing_labels"] == [] for f in fixed["per_fold"])


def test_the_two_fold_rules_disagree_on_the_real_corpus():
    from bench.harness.rows import load

    rows = list(load("tool_routing"))
    labels = [(r.group, r.label) for r in rows]
    hashed = metrics.assign_folds((r.group for r in rows), 5, 0, labels=labels, stratify=False)
    strat = metrics.assign_folds((r.group for r in rows), 5, 0, labels=labels, stratify=True)
    assert set(hashed) == set(strat)
    assert hashed != strat, "if the rules produced the same map, one of them is dead code"


def test_fit_and_score_on_the_same_rows_is_refused():
    with pytest.raises(SystemExit, match="must not contain the scoring rows"):
        run.main(["--dataset", "tool_routing", "--split", "test", "--fit-on", "test",
                  "--arm", "b=recorded|arm=floor_a",
                  "--arm", "m=sklearn_text|vectorizer=tfidf_word|clf=logreg",
                  "--reference", "b"])


def test_the_untouched_fit_protocol_works():
    pytest.importorskip("sklearn")
    rc = run.main(["--dataset", "tool_routing", "--split", "test", "--fit-on", "train",
                   "--arm", "b=recorded|arm=floor_a",
                   "--arm", "m=sklearn_text|vectorizer=tfidf_word|clf=logreg",
                   "--reference", "b"])
    assert rc == 0


def test_a_replay_capability_not_a_signature_decides_the_order_refusal(tmp_path):
    base_path = _write_base(tmp_path, [("r1", "a", {"a": 0.6, "b": 0.4})])
    fusion = build(f"fusion|member=fake|base_arm=base|base_path={base_path}")
    assert fusion.replays_a_file is True
    assert build("recorded|arm=floor_a").replays_a_file is True
    with pytest.raises(SystemExit, match="cannot be honoured"):
        run.main(["--dataset", "tool_routing", "--limit", "10", "--order", "rev",
                  "--arm", f"f=fusion|member=fake|base_arm=base|base_path={base_path}"])


def test_the_base_spec_nests_and_selects_the_arm(tmp_path):
    base_path = _write_base(tmp_path, [("r1", "a", {"a": 0.9, "b": 0.1})])
    fusion = build(f"fusion|member=fake|base=recorded~arm=floor_b|base_path={base_path}",
                   )
    assert fusion.base_spec == "recorded|arm=floor_b"
    assert fusion.base_arm == "floor_b"


def test_an_unknown_key_is_refused_rather_than_swallowed(tmp_path):
    base_path = _write_base(tmp_path, [("r1", "a", {"a": 0.6, "b": 0.4})])
    with pytest.raises(Exception, match="unknown key"):
        build(f"fusion|member=fake|swaped=1|base_arm=base|base_path={base_path}")


def test_a_duplicate_row_id_in_the_base_is_an_error(tmp_path):
    path = tmp_path / "dup.jsonl"
    lines = [{"arm": "base", "id": "r1", "choice": "a", "probabilities": {"a": 0.6, "b": 0.4}},
             {"arm": "base", "id": "r1", "choice": "b", "probabilities": {"a": 0.1, "b": 0.9}}]
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    fusion = build(f"fusion|member=fake|base_arm=base|base_path={path}")
    with pytest.raises(Exception, match="duplicate row id"):
        fusion.base_rows()


def test_errors_are_counted_in_the_summary(tmp_path):
    pytest.importorskip("sklearn")
    out = tmp_path / "art"
    base_dir = tmp_path / "base.jsonl"
    base_dir.write_text(json.dumps({"arm": "floor_a", "id": "v3-0001", "choice": "web_search",
                                    "probabilities": {"web_search": 0.9}}) + "\n")
    rc = run.main(["--dataset", "tool_routing", "--limit", "6",
                   "--arm", "b=recorded|arm=floor_a",
                   "--arm", f"f=fusion|member=fake|base_arm=floor_a|base_path={base_dir}",
                   "--reference", "b", "--out", str(out)])
    assert rc == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["arms"]["f"]["n_errors"] >= 1, "a row that never ran must be visible"


def test_the_artifact_source_has_no_absolute_path(tmp_path):
    out = tmp_path / "art"
    run.main(["--dataset", "tool_routing", "--limit", "5",
              "--arm", "b=recorded|arm=floor_a", "--out", str(out)])
    summary = json.loads((out / "summary.json").read_text())
    assert not summary["dataset"]["source"].startswith("/")
    assert "spec" in summary["arms"]["b"]


def test_the_cue_feature_scale_is_real_and_changes_the_member():
    pytest.importorskip("sklearn")
    from bench.harness.rows import load

    rows = list(load("tool_routing"))[:120]
    base = build("sklearn_text|vectorizer=tfidf_word|clf=logreg|features=rules")
    scaled = build("sklearn_text|vectorizer=tfidf_word|clf=logreg|features=rules"
                   "|feature_scale=maxabs")
    base.fit(rows)
    scaled.fit(rows)
    p1 = base.predict(rows[0].text, rows[0].options)
    p2 = scaled.predict(rows[0].text, rows[0].options)
    assert p1.probabilities is not None and p2.probabilities is not None
    assert p1.probabilities != p2.probabilities
