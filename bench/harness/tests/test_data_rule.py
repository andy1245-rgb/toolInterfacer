"""The data rule: a dataset adapter is a door into a public repository.

Rows are read from a path and identified by a sha256 in the artifact, never committed. Only
aggregates are written. Row-level output for a dataset that was not authored here goes beside the
dataset, not into `bench/results/`. The test when unsure is a projector in a room with an
employer in it.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from bench.harness import rows, run


def _csv(tmp_path, name="mined.csv", n=6):
    path = tmp_path / name
    lines = ["text,label"] + [f"private sentence number {i},op{i % 3}" for i in range(n)]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_the_artifact_identifies_a_dataset_by_hash_and_never_by_its_rows(tmp_path):
    out = tmp_path / "art"
    rc = run.main(["--dataset", "tool_routing", "--limit", "12",
                   "--arm", "b=recorded|arm=floor_a", "--out", str(out)])
    assert rc == 0
    summary = json.loads((out / "summary.json").read_text())
    meta = summary["dataset"]
    expected = hashlib.sha256((rows.ROOT / "bench/data/heldout_v3.json").read_bytes()).hexdigest()
    assert meta["sha256"] == expected
    assert len(meta["sha256"]) == 64
    assert meta["authored"] is True
    assert meta["source_kind"] == "in-repo"
    assert meta["source"] == "bench/data/heldout_v3.json"
    assert meta["rows"] == 12          # the count, not the rows


def test_only_aggregates_are_written_unless_asked(tmp_path):
    out = tmp_path / "art"
    run.main(["--dataset", "tool_routing", "--limit", "8",
              "--arm", "b=recorded|arm=floor_a", "--out", str(out)])
    assert (out / "summary.json").exists()
    assert not (out / "rows.jsonl").exists()
    blob = (out / "summary.json").read_text()
    assert "utterance" not in blob and "is ChoiceAnswer" not in blob


def test_no_utterance_appears_in_an_aggregate(tmp_path):
    from bench.harness.rows import load

    out = tmp_path / "art"
    run.main(["--dataset", "tool_routing", "--limit", "40",
              "--arm", "b=recorded|arm=floor_a", "--out", str(out)])
    blob = (out / "summary.json").read_text()
    texts = sorted(load("tool_routing").rows, key=lambda r: -len(r.text))[:5]
    for row in texts:
        assert row.text not in blob, f"{row.id}: an utterance reached an aggregate"


def test_a_non_authored_dataset_is_flagged_and_cannot_write_rows_into_results(tmp_path):
    path = _csv(tmp_path)
    dataset = rows.load(f"generic_csv|path={path}")
    assert dataset.authored is False
    assert dataset.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()

    out = tmp_path / "art"
    with pytest.raises(SystemExit, match="not authored here"):
        run.main(["--dataset", f"generic_csv|path={path}",
                  "--arm", "c=constant|index=0",
                  "--write-rows", "--rows-to", str(rows.ROOT / "bench/results/never")])


def test_an_external_dataset_is_named_by_basename_only(tmp_path):
    path = _csv(tmp_path)
    out = tmp_path / "art"
    rc = run.main(["--dataset", f"generic_csv|path={path}",
                   "--arm", "c=constant|index=0", "--out", str(out)])
    assert rc == 0
    meta = json.loads((out / "summary.json").read_text())["dataset"]
    assert meta["source"] == path.name
    assert meta["source_kind"] == "external"
    assert "/tmp/" not in meta["source"]


def test_row_level_output_for_an_external_dataset_defaults_beside_it(tmp_path):
    path = _csv(tmp_path)
    out = tmp_path / "art"
    rc = run.main(["--dataset", f"generic_csv|path={path}",
                   "--arm", "c=constant|index=0",
                   "--write-rows", "--out", str(out)])
    assert rc == 0
    beside = tmp_path / "generic_csv-rows.jsonl"
    assert beside.exists()
    assert not (out / "rows.jsonl").exists()
    assert beside.read_text().count("\n") == 6


def test_the_authored_dataset_may_write_rows_into_its_own_artifact(tmp_path):
    out = tmp_path / "art"
    rc = run.main(["--dataset", "tool_routing", "--limit", "5",
                   "--arm", "b=recorded|arm=floor_a",
                   "--write-rows", "--out", str(out)])
    assert rc == 0
    assert (out / "rows.jsonl").read_text().count("\n") == 5


def test_the_hash_survives_a_limit_and_a_split(tmp_path):
    whole = rows.load("tool_routing")
    limited = run.main(["--dataset", "tool_routing", "--limit", "9",
                        "--arm", "b=recorded|arm=floor_a", "--out", str(tmp_path / "a")])
    assert limited == 0
    meta = json.loads((tmp_path / "a" / "summary.json").read_text())["dataset"]
    assert meta["sha256"] == whole.sha256, "the identity is the file, not the slice"
    assert meta["rows"] == 9
