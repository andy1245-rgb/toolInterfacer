"""The one claim in this repo that passed a pre-registered test, checkable from the repo alone.

`docs/fusion-2026-11-10.md` option 3 passed all four conditions. A claim that can only be checked
by re-running 415 rows against a recorded base artifact is a claim that will quietly stop being
true, so this test recomputes every condition from
`bench/results/fusion-option3-2026-11-10/` with the same metric functions the run used:

* C1 delta >= +20 rows, C2 exact McNemar p < 0.01 and C3 no label worse than -3, all recomputed
  from `rows.jsonl`;
* C4 positive in >= 4 of 5 fold seeds, recomputed from the summary's per-seed counts against the
  reference arm's own count;
* the summary's declared verdict is checked against the recomputation, so editing one without the
  other fails the suite.

No model, no server, no network: two files.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "bench/results/fusion-option3-2026-11-10"

C1_MIN_ROWS = 20
C2_MAX_P = 0.01
C3_WORST = -3
C4_MIN_SEEDS = 4


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


def _summary():
    return json.loads((ARTIFACT / "summary.json").read_text())


def test_the_artifact_is_the_one_this_claim_is_about():
    summary = _summary()
    assert summary["dataset"]["name"] == "tool_routing"
    assert summary["reference"] == "floor_a"
    assert "protect=web_search,web_fetch" in summary["arms"]["r_fusion"]["spec"]
    assert summary["prereg"]["passed"] is True


def test_c1_and_c2_recompute_from_the_rows():
    from bench.harness import metrics

    by_arm = _records()
    paired = metrics.paired(by_arm["floor_a"], by_arm["r_fusion"])
    delta = paired["treatment_correct"] - paired["reference_correct"]
    assert by_arm["floor_a"].__len__() == 415
    assert paired["reference_correct"] == 325, "the reference arm's count moved"
    assert delta >= C1_MIN_ROWS, f"C1: {delta} rows, the bar is {C1_MIN_ROWS}"
    assert paired["p"] < C2_MAX_P, f"C2: p = {paired['p']:.3g}, the bar is {C2_MAX_P}"
    assert paired["treatment_correct"] == 349


def test_c3_recomputes_from_the_ledger():
    from bench.harness import metrics

    by_arm = _records()
    led = metrics.ledger(by_arm["floor_a"], by_arm["r_fusion"])
    worst_label, worst = min(led["by_label"].items(), key=lambda kv: kv[1]["net"])
    assert worst["net"] >= C3_WORST, (
        f"C3: {worst_label} nets {worst['net']}, the bar is {C3_WORST}")
    owned = {label: led["by_label"][label]["net"]
             for label in ("web_search", "web_fetch") if label in led["by_label"]}
    assert all(net >= 0 for net in owned.values()), (
        f"the two ops the base owns must not lose ground: {owned}")


def test_c4_recomputes_from_the_per_seed_counts():
    summary = _summary()
    base_correct = summary["arms"]["floor_a"]["accuracy"]["correct"]
    per_seed = summary["arms"]["r_fusion"]["grouped_cv"]["per_seed"]
    assert len(per_seed) == 5, f"C4 needs five fold seeds, found {len(per_seed)}"
    deltas = {seed: rec["oof_accuracy"]["correct"] - base_correct
              for seed, rec in per_seed.items()}
    positive = sum(1 for d in deltas.values() if d > 0)
    assert positive >= C4_MIN_SEEDS, f"C4: {deltas} — {positive} of {len(deltas)} positive"
    declared = summary["prereg"]["delta_by_seed"]
    assert {k: int(v) for k, v in declared.items()} == deltas, (
        "the declared per-seed deltas and the per-seed counts disagree")


def test_the_verdict_and_the_numbers_are_in_the_same_file():
    """A PASS with a failing number in it is the failure mode this test exists for."""
    summary = _summary()
    conditions = {c.get("name") or c["expr"]: c["passed"]
                  for c in summary["prereg"]["conditions"]}
    assert len(conditions) == 4
    assert all(conditions.values()), f"a condition failed in a PASS artifact: {conditions}"


def test_the_latency_cost_is_what_the_doc_says():
    """The fusion calls both members, so its p50 is the base's plus the member's."""
    summary = _summary()
    fusion_ms = summary["arms"]["r_fusion"]["latency"]["p50"]
    base_ms = summary["arms"]["floor_a"]["latency"]["p50"]
    assert fusion_ms < base_ms, (
        "the recorded base is a replay, so this arm's latency is the member's alone; the doc says "
        "so explicitly and must not claim the fusion saves the model call")


def test_the_eval_split_gain_is_much_smaller_than_the_pooled_gain():
    """The claim was pre-registered over all 415 rows; the held-out split tells a smaller story.

    Pinned so nobody quotes +24 as if the corpus were held out end to end. The 207 eval rows are
    the harder half — the base scores 79.7% there against 78.3% pooled — and the fusion gains
    three rows on them.
    """
    from bench.harness import metrics

    by_arm = _records()
    base, fusion = by_arm["floor_a"], by_arm["r_fusion"]
    test_ids = {r.id for r in base if r.meta.get("split") == "test"}
    base_test = [r for r in base if r.id in test_ids]
    fusion_test = [r for r in fusion if r.id in test_ids]
    assert len(base_test) == 207
    paired = metrics.paired(base_test, fusion_test)
    delta = paired["treatment_correct"] - paired["reference_correct"]
    assert paired["reference_correct"] == 165
    assert delta == 3, f"the eval-split delta moved to {delta}; the doc says +3"
    assert paired["p"] > 0.05, (
        f"the eval-split delta is now significant (p = {paired['p']:.3f}); that would be a "
        f"different claim and needs its own pre-registration")
