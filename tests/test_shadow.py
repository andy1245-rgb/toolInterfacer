"""Shadow mode: compute what the fused policy would have said, change nothing.

The guarantee under test is narrow and absolute — `choice()` returns the served answer, whatever
the shadow thinks — and the value under test is that both sides land in the log beside the outcome
that actually happened.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))

from ti.decide import shadow  # noqa: E402
from ti.decide.primitives import ChoiceAnswer  # noqa: E402

CRITERIA = {"filesys_read": "read", "filesys_glob": "glob", "filesys_grep": "grep",
            "decide_pick": "pick", "decide_extract": "extract", "web_search": "search",
            "web_fetch": "fetch", "reddit_digest": "digest"}


class Stub:
    def __init__(self, choice, probabilities=None, last_source="primary"):
        self.name = "stub"
        self._choice, self._probs = choice, probabilities
        self.last_source = last_source
        self.calls = 0

    def choice(self, state, instructions, criteria):
        self.calls += 1
        if isinstance(self._choice, Exception):
            raise self._choice
        probs = dict(self._probs or {})
        return ChoiceAnswer(self._choice, probs, probs.get(self._choice, 0.0) if probs else 0.0)

    def noul(self, state, question):
        return 0.4

    def health(self):
        return True


def _enabled(tmp_path, member, **cfg):
    config = {"enabled": True, "member": "classical", "weights": 0.5,
              "protect": ["web_search", "web_fetch"], "log_path": str(tmp_path / "shadow.jsonl")}
    config.update(cfg)
    return shadow.ShadowBackend(Stub("filesys_glob", {"filesys_glob": 0.6, "filesys_read": 0.4}),
                                config=config, member=member)


def _lines(tmp_path):
    path = tmp_path / "shadow.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ------------------------------------------------------------------ the guarantee

def test_the_served_answer_is_returned_untouched(tmp_path):
    """Even when the shadow disagrees loudly, the caller gets exactly the inner answer."""
    member = Stub("filesys_read", {"filesys_read": 0.99, "filesys_glob": 0.01})
    backend = _enabled(tmp_path, member)
    answer = backend.choice("anything", "", CRITERIA)
    assert answer.choice == "filesys_glob"
    assert answer.confidence == pytest.approx(0.6)


def test_the_inner_backend_is_asked_exactly_once(tmp_path):
    inner = Stub("filesys_glob", {"filesys_glob": 0.6, "filesys_read": 0.4})
    backend = shadow.ShadowBackend(
        inner, config={"enabled": True, "member": "classical", "log_path": str(tmp_path / "shadow.jsonl")},
        member=Stub("filesys_read", {"filesys_read": 0.9, "filesys_glob": 0.1}))
    backend.choice("anything", "", CRITERIA)
    assert inner.calls == 1, "the shadow must never re-ask the model"


def test_a_broken_member_cannot_break_the_request(tmp_path):
    backend = _enabled(tmp_path, Stub(RuntimeError("member is down")))
    answer = backend.choice("anything", "", CRITERIA)
    assert answer.choice == "filesys_glob"
    line = _lines(tmp_path)[0]
    assert "member is down" in line["shadow_error"]


def test_an_inner_failure_still_propagates(tmp_path):
    backend = shadow.ShadowBackend(Stub(RuntimeError("the model is down")),
                                   config={"enabled": True, "member": "classical",
                                           "log_path": str(tmp_path / "shadow.jsonl")},
                                   member=Stub("filesys_read", {"filesys_read": 1.0}))
    with pytest.raises(RuntimeError, match="the model is down"):
        backend.choice("anything", "", CRITERIA)


# ------------------------------------------------------------------ the comparison

def test_both_sides_are_logged(tmp_path):
    member = Stub("filesys_read", {"filesys_read": 0.9, "filesys_glob": 0.1})
    _enabled(tmp_path, member).choice("anything", "", CRITERIA)
    line = _lines(tmp_path)[0]
    assert line["kind"] == "decision"
    assert line["served"] == "filesys_glob"
    assert line["member"] == "filesys_read"
    assert line["fused"] == "filesys_read", "the member's 0.9 must win the mean"
    assert line["agrees"] is False and line["member_agrees_with_served"] is False
    assert line["source"] == "primary"


def test_a_protected_op_does_not_consult_the_member(tmp_path):
    inner = Stub("web_search", {"web_search": 0.7, "filesys_read": 0.3})
    backend = shadow.ShadowBackend(
        inner, config={"enabled": True, "member": "classical",
                       "log_path": str(tmp_path / "shadow.jsonl")},
        member=Stub("filesys_read", {"filesys_read": 0.99, "web_search": 0.01}))
    assert backend.choice("anything", "", CRITERIA).choice == "web_search"
    line = _lines(tmp_path)[0]
    assert line["fused"] == "web_search" and line["why"] == "protected"
    assert line["member"] is None


def test_one_side_without_a_distribution_logs_why(tmp_path):
    inner = Stub("filesys_glob", None)
    backend = shadow.ShadowBackend(
        inner, config={"enabled": True, "member": "classical",
                       "log_path": str(tmp_path / "shadow.jsonl")},
        member=Stub("filesys_read", {"filesys_read": 0.9, "filesys_glob": 0.1}))
    backend.choice("anything", "", CRITERIA)
    line = _lines(tmp_path)[0]
    assert line["fused"] == "filesys_glob" and "no distribution" in line["why"]


# ------------------------------------------------------------------ the outcome

def test_the_outcome_joins_the_decision_and_counts_attempts(tmp_path):
    backend = _enabled(tmp_path, Stub("filesys_read", {"filesys_read": 0.9, "filesys_glob": 0.1}))
    backend.choice("anything", "", CRITERIA)
    decision = _lines(tmp_path)[0]["id"]
    cfg = {"enabled": True, "log_path": str(tmp_path / "shadow.jsonl")}
    shadow.note_outcome(op="filesys_grep", ok=True, ms=12.0, step="s1", cfg=cfg)
    shadow.note_outcome(op="filesys_grep", ok=True, ms=11.0, step="s1", cfg=cfg)
    outcomes = [line for line in _lines(tmp_path) if line["kind"] == "outcome"]
    assert [o["id"] for o in outcomes] == [decision, decision]
    assert [o["attempt"] for o in outcomes] == [1, 2], "a retried step must be visible as attempt 2"
    assert outcomes[0]["op"] == "filesys_grep"


def test_an_outcome_without_a_decision_is_dropped(tmp_path):
    shadow.clear()
    shadow.note_outcome(op="filesys_read", ok=True,
                        cfg={"enabled": True, "log_path": str(tmp_path / "shadow.jsonl")})
    assert _lines(tmp_path) == []


# ------------------------------------------------------------------ the switch

def test_wrapping_is_the_identity_when_disabled():
    inner = Stub("filesys_glob", {"filesys_glob": 1.0})
    assert shadow.wrap(inner, cfg={"enabled": False}) is inner


def test_the_shipped_config_has_shadow_on():
    """The operator approved shadowing for live traffic on 2026-11-10; this pins the decision."""
    assert shadow.enabled() is True
    assert shadow.enabled({"enabled": False}) is False


def test_the_log_never_defaults_inside_the_repository():
    path = shadow.log_path({})
    assert ROOT not in path.parents, "the log carries traffic and must live outside the repo"


def test_the_log_path_can_be_redirected(monkeypatch, tmp_path):
    monkeypatch.setenv(shadow.ENV_PATH, str(tmp_path / "from-env.jsonl"))
    assert shadow.log_path({}) == tmp_path / "from-env.jsonl"
    assert shadow.log_path({"log_path": str(tmp_path / "explicit.jsonl")}) == \
        tmp_path / "explicit.jsonl"


# ------------------------------------------------------------------ the diagnostic that matters on live traffic

def test_an_abstaining_member_is_logged_as_abstaining_not_blended(tmp_path):
    """The only live choice question is `decide_pick`, whose options are candidate ids.

    `classical` is an op router, so it abstains there by construction. Averaging a uniform
    distribution into the served answer would have rewritten it towards the middle and reported
    agreement that came from arithmetic rather than from competence.
    """
    backend = _enabled(tmp_path, Stub(None, {"filesys_read": 0.166, "filesys_glob": 0.166}))
    answer = backend.choice("anything", "", {"cand-1": "a", "cand-2": "b"})
    assert answer.choice == "filesys_glob"
    line = _lines(tmp_path)[0]
    assert line["why"] == "member abstained"
    assert line["member_answered"] is False
    assert line["fused"] == line["served"]


def test_the_vocabulary_overlap_is_recorded_per_request(tmp_path):
    class VocabStub(Stub):
        def fitted_labels(self):
            return ("filesys_read", "web_search")

    backend = _enabled(tmp_path, VocabStub("filesys_read", {"filesys_read": 0.9,
                                                            "web_search": 0.1}))
    backend.choice("anything", "", {"filesys_read": "r", "web_search": "w", "other": "o"})
    line = _lines(tmp_path)[0]
    assert line["vocab_match"] == pytest.approx(0.667), "two of three offered options are known"


def test_a_classical_tier_can_report_its_vocabulary():
    """The check that would have caught the dead tier, available from the backend itself."""
    from ti.decide.backends.classical import ClassicalBackend

    labels = ClassicalBackend().fitted_labels()
    assert set(labels) == {"filesys_read", "filesys_glob", "filesys_grep", "decide_pick",
                           "decide_extract", "web_search", "web_fetch", "reddit_digest"}
