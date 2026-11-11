"""The test that would have caught a tier that was dead for its whole life.

`decide.classical.corpus_path` pointed at a 17-tool voice-assistant registry while the router
offered eight code/file operations. Zero label overlap means the fitted model can name none of the
candidates, so it returned "no opinion" on every request ever made, the expensive model answered,
and every accuracy number stayed where it was. Nothing logged coverage, so nothing noticed.

Four tests, in the order the failure escalates:

* the vocabulary can name the options it is offered — the rot itself, checked directly;
* the shipped tier answers something on the eval split — the one-line assertion that was missing;
* the corpus is built from the training split and is not stale — so the revival cannot rot quietly;
* the rot reproduced deliberately is **loud** in the harness summary — so the next one is visible.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))

from bench.harness import run  # noqa: E402
from bench.harness.protocol import build as build_predictor  # noqa: E402
from bench.harness.rows import load as load_dataset  # noqa: E402

OPS = ("filesys_read", "filesys_glob", "filesys_grep", "decide_pick", "decide_extract",
       "web_search", "web_fetch", "reddit_digest")


@pytest.fixture(scope="module")
def tier():
    return build_predictor("classical")


def test_the_fitted_vocabulary_can_name_the_offered_options(tier):
    from bench.harness.predictors.classical import fitted_label_overlap

    overlap, strangers = fitted_label_overlap(tier, OPS)
    assert not strangers, (
        f"the classical corpus is fitted on labels nobody offers: {strangers}. That is the exact "
        f"rot this file exists for — it abstains forever and looks like a cautious tier.")
    assert set(overlap) == set(OPS)


def test_the_shipped_tier_answers_something_on_the_eval_split(tier):
    """One line, permanently: a front door that answers nothing is not a front door."""
    rows = [r for r in load_dataset("tool_routing").rows if r.meta.get("split") == "test"][:40]
    answers = [tier.predict(r.text, r.options) for r in rows]
    answered = [a for a in answers if a.choice]
    assert answered, (
        f"the classical tier answered 0 of {len(rows)} rows — it is configured against a "
        f"vocabulary that cannot name the offered options, and it has been abstaining instead of "
        f"answering. Check decide.classical.corpus_path and run "
        f"`python3 bench/build_classical_corpus.py`.")
    assert all(a.choice in r.options for a, r in zip(answers, rows) if a.choice)


def test_the_corpus_is_the_training_split_and_is_current():
    proc = subprocess.run([sys.executable, str(ROOT / "bench/build_classical_corpus.py"),
                           "--check"], capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    corpus = json.loads((ROOT / "service/ti/data/classical_v3_train.json").read_text())
    v3 = json.loads((ROOT / "bench/data/heldout_v3.json").read_text())
    trained_on = {r["utterance"] for r in corpus}
    eval_rows = {r["utterance"] for r in v3 if r.get("split") == "test"}
    assert not (trained_on & eval_rows), "the corpus contains eval rows: every number is leakage"
    assert len(corpus) == len({r["utterance"] for r in v3 if r.get("split") == "train"})


def test_the_rot_reproduced_on_purpose_is_loud_in_the_summary(tmp_path):
    """Point the tier back at the old registry and assert the harness reports a dead tier.

    Before the coverage column existed this run read as a normal accuracy: the tier abstained on
    every row, so `correct` was 0/40 and there was nothing to distinguish "wrong" from "silent".
    """
    out = tmp_path / "dead"
    rc = run.main(["--dataset", "tool_routing", "--split", "test", "--limit", "40",
                   "--arm", "dead=classical|corpus_path=bench/registry_382.json",
                   "--out", str(out)])
    assert rc == 0
    arm = json.loads((out / "summary.json").read_text())["arms"]["dead"]
    assert arm["n_abstained"] == arm["accuracy"]["n"], (
        "the old configuration is supposed to abstain on every row; if it no longer does, the "
        "vocabulary overlap changed and this test is measuring something else")
    assert arm["n_answered"] == 0
    assert arm["coverage_at_precision90"]["coverage"] in (0.0, None)
