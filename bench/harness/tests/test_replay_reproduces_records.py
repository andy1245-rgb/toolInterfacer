"""The harness must reproduce the records it was built from, or it is not finished.

Every arm in `bench/results/p3-transfer/rows.jsonl` is a published number with a document
attached. Replaying them through the four seams is the regression test: if `rows.py`, the
predictor seam, the ledger or the McNemar ever drift, these fail.

The numbers asserted here are the ones in `docs/grep-arm-2026-11-10.md` and
`docs/rewrite-transfer-2026-11-04.md`.
"""
from __future__ import annotations

import pytest

from bench.harness import metrics
from bench.harness.protocol import build, option_order_sha, score
from bench.harness.rows import load as load_dataset

ROWS = "bench/results/p3-transfer/rows.jsonl"


@pytest.fixture(scope="module")
def ds():
    return load_dataset("tool_routing")


@pytest.fixture(scope="module")
def floor_a(ds):
    return score("floor_a", list(ds), build(f"recorded|path={ROWS}|arm=floor_a"))


def test_floor_a_reproduces_325_of_415(floor_a):
    acc = metrics.accuracy(floor_a)
    assert acc["n"] == 415
    assert acc["correct"] == 325, "floor_a is the record this harness was built from"
    assert acc["accuracy"] == pytest.approx(0.7831, abs=5e-5)
    lo, hi = acc["wilson95"]
    assert (round(100 * lo, 1), round(100 * hi, 1)) == (74.1, 82.0)


def test_the_grep_arm_ledger_reproduces(floor_a):
    grep = score("grep_only", list(load_dataset("tool_routing")),
                 build(f"recorded|path={ROWS}|arm=grep_only"))
    led = metrics.ledger(floor_a, grep)
    assert led["by_label"]["filesys_grep"] == {"n": 52, "base": 36, "treat": 46,
                                               "fixed": 10, "broke": 0, "net": 10}
    assert led["by_label"]["web_search"]["broke"] == 3
    assert led["by_label"]["web_search"]["net"] == -3
    assert led["by_label"]["decide_extract"]["net"] == -2
    assert led["net"] == 3
    assert led["fixed"] == 10 and led["broke"] == 7


def test_the_grep_arm_mcnemar_reproduces(floor_a):
    grep = score("grep_only", list(load_dataset("tool_routing")),
                 build(f"recorded|path={ROWS}|arm=grep_only"))
    p = metrics.paired(floor_a, grep)
    assert p["p"] == pytest.approx(0.6291, abs=5e-5)


def test_the_order_arm_reproduces(floor_a):
    rev = score("order_rev", list(load_dataset("tool_routing")),
                build(f"recorded|path={ROWS}|arm=order_rev"))
    acc = metrics.accuracy(rev)
    assert acc["correct"] == 316
    p = metrics.paired(floor_a, rev)
    assert (p["fixed"], p["broke"]) == (6, 15)
    assert p["p"] == pytest.approx(0.0784, abs=5e-5)


def test_the_a_a_floor_reproduces(floor_a):
    again = score("floor_b", list(load_dataset("tool_routing")),
                  build(f"recorded|path={ROWS}|arm=floor_b"))
    floor = metrics.a_a_flips(floor_a, again)
    assert floor["n"] == 415
    assert floor["choice_flips"] == 1, "the published floor is one flip in 415"


def test_the_artifacts_are_gate_free(floor_a):
    """P2 from the plan: a gate answer has a one-hot 1.0 vector and a sub-second latency."""
    for r in floor_a:
        assert r.latency_ms > 1000, f"{r.id} answered in {r.latency_ms:.1f} ms — not the model"
        if r.probabilities:
            assert sorted(r.probabilities.values())[-1] < 0.999999


def test_replay_records_the_order_it_was_scored_under(ds, floor_a):
    assert {r.order_sha for r in floor_a} == {option_order_sha(ds.options)}


def test_recorded_refuses_an_unknown_arm():
    with pytest.raises(Exception) as exc:
        build(f"recorded|path={ROWS}|arm=no_such_arm")
    assert "not in" in str(exc.value)


def test_recorded_refuses_to_replay_under_a_permuted_order(ds):
    frozen = option_order_sha(ds.options)
    p = build(f"recorded|path={ROWS}|arm=floor_a|order_sha={frozen}")
    row = ds.rows[0]
    with pytest.raises(Exception) as exc:
        p.predict_for(row.__class__(id=row.id, text=row.text, label=row.label,
                                    group=row.group, options=tuple(reversed(row.options)),
                                    meta=row.meta))
    assert "recorded under order" in str(exc.value)


def test_replay_of_every_arm_is_deterministic(ds):
    a = score("a", list(ds)[:50], build(f"recorded|path={ROWS}|arm=floor_a"))
    b = score("b", list(ds)[:50], build(f"recorded|path={ROWS}|arm=floor_a"))
    assert [r.choice for r in a] == [r.choice for r in b]
