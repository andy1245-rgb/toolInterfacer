"""The harness, through the repo's own suite.

`bench/harness/tests/` holds the detailed known-answer tests; this file exists so that
`python3 -m pytest tests/ -q` — the command CI runs — covers the harness too, and so that the one
assertion that matters most cannot rot: the harness reproduces the record it was built from.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.harness import metrics  # noqa: E402
from bench.harness.protocol import build, option_order_sha, score  # noqa: E402
from bench.harness.rows import load as load_dataset  # noqa: E402

ROWS = "bench/results/p3-transfer/rows.jsonl"


def test_harness_reproduces_floor_a():
    ds = load_dataset("tool_routing")
    recs = score("floor_a", list(ds), build(f"recorded|path={ROWS}|arm=floor_a"))
    acc = metrics.accuracy(recs)
    assert acc["n"] == 415
    assert acc["correct"] == 325
    assert {r.order_sha for r in recs} == {option_order_sha(ds.options)}


def test_harness_reproduces_the_grep_ledger():
    ds = load_dataset("tool_routing")
    base = score("floor_a", list(ds), build(f"recorded|path={ROWS}|arm=floor_a"))
    grep = score("grep_only", list(ds), build(f"recorded|path={ROWS}|arm=grep_only"))
    led = metrics.ledger(base, grep)
    assert led["by_label"]["filesys_grep"]["net"] == 10
    assert led["by_label"]["filesys_grep"]["broke"] == 0
    assert led["by_label"]["web_search"]["net"] == -3
    assert metrics.paired(base, grep)["p"] == pytest.approx(0.6291, abs=5e-5)


def test_mcnemar_does_not_drift_from_the_repo_statistic():
    from bench.ti_bench.stats import mcnemar_exact as repo_mcnemar
    for b in range(0, 13):
        for c in range(0, 13):
            assert metrics.mcnemar_exact(b, c) == pytest.approx(repo_mcnemar(b, c), abs=1e-12)


def test_the_cli_runs_end_to_end_without_a_model(tmp_path):
    from bench.harness.run import main
    truth = tmp_path / "truth.json"
    ds = load_dataset("tool_routing")
    truth.write_text(json.dumps({r.text: r.label for r in ds.rows[:12]}))
    out = tmp_path / "run"
    assert main(["--dataset", "tool_routing", "--limit", "12",
                 "--predictor", f"fake|accuracy=1.0|seed=1|truth={truth}",
                 "--aa", "--write-rows", "--out", str(out)]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["arms"]["arm"]["accuracy"]["correct"] == 12
    assert summary["dataset"]["order_sha"] == option_order_sha(summary["dataset"]["options"])
    assert (out / "rows.jsonl").read_text().count("\n") == 12


def test_a_predictor_that_needs_the_network_is_not_needed_by_the_suite():
    """Importing the harness must never reach for a model, a socket, or a dependency."""
    import bench.harness.predictors as predictors
    assert "fake" in __import__("bench.harness.protocol", fromlist=["x"]).available()
    # the optional kinds are registered whether or not their dependency is installed
    assert set(predictors.OPTIONAL) <= set(
        __import__("bench.harness.protocol", fromlist=["x"]).available())
