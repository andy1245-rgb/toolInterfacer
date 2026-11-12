"""The pre-gate, measured by coverage — the number that exposes a detector that stopped matching.

`heuristic_micro` had a third detector that emitted `hits["filesys"]`, a key that is not one of the
eight op names, so the `if c in criteria` filter dropped it; and its path regex required a leading
`/` and matched 0 of 415 rows. Two independent reasons to be dead, neither visible in any accuracy
number, because a detector that answers nothing is never wrong.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))

from ti.decide.backends.heuristic_micro import HeuristicMicroBackend  # noqa: E402

OPS = {"filesys_read": "read", "filesys_glob": "glob", "filesys_grep": "grep",
       "decide_pick": "pick", "decide_extract": "extract", "web_search": "search",
       "web_fetch": "fetch", "reddit_digest": "digest"}


def _rows():
    from bench.harness.rows import load

    return [r for r in load("tool_routing").rows if r.meta.get("split") == "test"]


def test_the_pre_gate_answers_something():
    """The rot was a branch that answered nothing for the life of the repo."""
    rows = _rows()
    backend = HeuristicMicroBackend()
    answers = [backend.choice(r.text, "", OPS) for r in rows]
    answered = [a for a in answers if a is not None]
    assert answered, "the pre-gate answered none of the eval rows — every detector is dead"
    assert len(answered) >= 20, f"only {len(answered)} rows answered; the detectors went quiet"


def test_every_answer_the_pre_gate_gives_is_right():
    rows = _rows()
    backend = HeuristicMicroBackend()
    wrong = []
    for row in rows:
        answer = backend.choice(row.text, "", OPS)
        if answer is not None and answer.choice != row.label:
            wrong.append((row.id, row.label, answer.choice))
    assert not wrong, f"the pre-gate is guessing: {wrong[:5]}"


def test_an_answer_is_always_one_of_the_offered_options():
    rows = _rows()
    backend = HeuristicMicroBackend()
    for row in rows:
        answer = backend.choice(row.text, "", OPS)
        if answer is not None:
            assert answer.choice in OPS
            assert set(answer.probabilities) <= set(OPS), (
                "the pre-gate named a label the router does not offer — that is the exact shape "
                "of the retired filesys branch")


def test_the_retired_path_branch_is_retired():
    """A path-only utterance is the model's job now; the pre-gate must abstain on it."""
    backend = HeuristicMicroBackend()
    for text in ("open bench/data/heldout_v3.json", "read ../../etc/passwd",
                 "what's in src/localalexa/cli.py"):
        assert backend.choice(text, "", OPS) is None, (
            "a path-only utterance was answered; one key cannot name three filesys ops, and a "
            "loosened pattern is 40% precise over 149 rows")


def test_the_coverage_table_can_see_a_dead_backend():
    """`bench/backend_coverage.py` is the command the install message points at."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("backend_coverage",
                                                  ROOT / "bench/backend_coverage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = _rows()
    measured = module.measure("classical", rows)
    assert measured["answered"] == len(rows)
    assert measured["right"] / measured["answered"] > 0.6
    gate = module.measure("heuristic_micro", rows)
    assert gate["answered"] > 0 and gate["right"] == gate["answered"]
