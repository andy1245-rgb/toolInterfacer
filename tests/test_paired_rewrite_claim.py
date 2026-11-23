"""The paired rewrite, pinned as the FAIL it was — so nobody re-runs it hoping for another answer.

`docs/closed-experiments-2026-11-10.md` §4 closes it: `filesys_grep` gains the ten rows the
one-of-eight rewrite gained, `web_search` pays nothing (the compensating line worked), and the cost
lands on `reddit_digest`. The overall delta is +2 at p = 0.81.

This recomputes every condition from the committed artifact with the same metric functions the run
used, so the claim stays checkable without re-running 415 rows against a live model.
"""
from __future__ import annotations

import json
from pathlib import Path

ARTIFACT = Path(__file__).resolve().parents[1] / "bench/results/paired-rewrite-2026-11-10"


def _records():
    from bench.harness.protocol import Record

    rows = [json.loads(line) for line in (ARTIFACT / "rows.jsonl").read_text().splitlines()
            if line.strip()]
    by_arm: dict[str, list] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(Record(
            arm=row["arm"], id=row["id"], group=row["group"], label=row["label"],
            text=row["text"], choice=row["choice"], correct=row["correct"],
            latency_ms=row["latency_ms"], probabilities=row["probabilities"],
            order_sha=row.get("order_sha", ""), meta=row.get("meta") or {}))
    return by_arm


def test_the_artifact_records_the_instrument_it_used():
    dataset = json.loads((ARTIFACT / "summary.json").read_text())["dataset"]
    assert dataset["order_mode"] == "frozen", "the order must be frozen: wording is the variable"
    assert dataset["order_sha"] == dataset["frozen_order_sha"]
    assert sorted(dataset["paraphrase"]) == ["filesys_grep", "web_search"], (
        "a paired rewrite edits exactly the two labels the claim is about")
    assert dataset["paraphrase_sha"]


def test_the_paired_rewrite_gains_on_grep_and_pays_nothing_on_web_search():
    from bench.harness import metrics

    by_arm = _records()
    led = metrics.ledger(by_arm["floor_a"], by_arm["order_rewrite"])
    assert led["by_label"]["filesys_grep"]["net"] == 10, (
        "the grep gain is the reason the experiment exists; if it changed, the claim changed")
    assert led["by_label"]["web_search"]["net"] == 0, (
        "the compensating line is the whole point: web_search must not pay")


def test_the_cost_lands_on_reddit_digest_and_the_result_is_still_a_fail():
    from bench.harness import metrics

    by_arm = _records()
    led = metrics.ledger(by_arm["floor_a"], by_arm["order_rewrite"])
    paired = metrics.paired(by_arm["floor_a"], by_arm["order_rewrite"])
    delta = paired["treatment_correct"] - paired["reference_correct"]
    assert led["by_label"]["reddit_digest"]["net"] == -4
    assert delta == 2 and paired["p"] > 0.5, (
        "the overall result is a wash; a future run that changes this needs a new pre-registration")


def test_the_declared_verdict_matches_the_recomputation():
    summary = json.loads((ARTIFACT / "summary.json").read_text())
    conditions = {c["expr"]: c["passed"] for c in summary["prereg"]["conditions"]}
    assert summary["prereg"]["passed"] is False
    assert conditions["delta >= 4"] is False
    assert conditions["mcnemar_p < 0.05"] is False
    assert conditions["min_label_net >= -3"] is False
    assert conditions["ledger.filesys_grep.net >= 0"] is True
    assert conditions["ledger.web_search.net >= 0"] is True
