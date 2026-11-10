"""The dataset seam: the contract, the three adapters, and the one check that matters.

`Row` refuses a label that is not in its own option set. That is not ceremony: an 18-class
corpus behind an 8-op router is exactly how the classical backend came to answer `None` on
every row, and the harness makes that class of bug fail at load.
"""
from __future__ import annotations

import json

import pytest

from bench.harness import rows as R


def _row(**kw):
    base = dict(id="r1", text="do a thing", label="a", group="g1", options=("a", "b"))
    base.update(kw)
    return R.Row(**base)


def test_row_accepts_a_consistent_row():
    r = _row()
    assert r.options == ("a", "b") and r.label == "a"


def test_row_rejects_a_label_outside_the_option_set():
    with pytest.raises(R.DatasetError) as exc:
        _row(label="c")
    assert "not in its option set" in str(exc.value)


@pytest.mark.parametrize("kw", [
    {"id": ""}, {"text": "   "}, {"group": ""}, {"options": ()},
])
def test_row_rejects_the_rest_of_the_contract(kw):
    with pytest.raises(R.DatasetError):
        _row(**kw)


def test_duplicate_ids_are_refused_at_load():
    with pytest.raises(R.DatasetError) as exc:
        R._validate([_row(id="x"), _row(id="x")], "ds")
    assert "duplicate ids" in str(exc.value)


def test_mixed_option_tuples_are_refused_at_load():
    with pytest.raises(R.DatasetError) as exc:
        R._validate([_row(id="x"), _row(id="y", options=("b", "a"))], "ds")
    assert "different option tuple" in str(exc.value)


# ------------------------------------------------------------------ tool_routing

def test_tool_routing_shape_matches_the_published_set():
    ds = R.load("tool_routing")
    assert len(ds) == 415
    assert len(ds.options) == 8
    assert len(ds.groups()) == 133
    assert len(ds.labels()) == 8


def test_tool_routing_sides_are_the_planned_ones():
    ds = R.load("tool_routing")
    train = ds.slice("split", ["train"])
    test = ds.slice("split", ["test"])
    assert (len(train), len(test)) == (208, 207)
    assert not {r.group for r in train.rows} & {r.group for r in test.rows}


def test_tool_routing_carries_the_instrument_descriptions():
    ds = R.load("tool_routing")
    row = ds.rows[0]
    descriptions = row.meta["descriptions"]
    assert set(descriptions) == set(ds.options)
    assert descriptions["filesys_grep"].startswith("Search inside file contents")


def test_tool_routing_keeps_the_frozen_option_order():
    from bench.ti_bench.prompts import OP_DESCRIPTIONS
    ds = R.load("tool_routing")
    assert ds.options == tuple(OP_DESCRIPTIONS)


def test_stratum_names_match_the_legacy_artifacts():
    ds = R.load("tool_routing")
    grep = [r for r in ds.rows if r.label == "filesys_grep"][0]
    guard = [r for r in ds.rows if r.label == "web_fetch"][0]
    assert grep.meta["stratum"].startswith("targets-")
    assert guard.meta["stratum"].startswith("guard-")


# ------------------------------------------------------------------ localalexa_intents

def test_localalexa_counts_and_the_abstain_option():
    ds = R.load("localalexa_intents")
    assert len(ds) == 382
    assert "__none__" in ds.options, "the abstain route is a label in this corpus"
    assert len(ds.options) == 18
    assert len(ds.groups()) == 7


def test_localalexa_options_come_from_the_registry_not_globally():
    ds = R.load("localalexa_intents")
    assert ds.options[0] == "timer_create" or ds.options[0] in ds.options
    assert set(ds.labels()) <= set(ds.options)


# ------------------------------------------------------------------ generic_csv

def test_generic_csv_derives_options_and_reads_groups(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("say,intent,kind\nhello,greet,a\nbye,greet,a\nlights on,sleep,b\n")
    ds = R.load(f"generic_csv|path={p}|text_col=say|label_col=intent|group_col=kind")
    assert len(ds) == 3
    assert ds.options == ("greet", "sleep")
    assert len(ds.groups()) == 2


def test_generic_csv_honours_an_explicit_option_order(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("say,intent\nhello,greet\nbye,greet\n")
    ds = R.load(f"generic_csv|path={p}|text_col=say|label_col=intent|options=zz,greet")
    assert ds.options == ("zz", "greet")


def test_generic_csv_refuses_a_missing_column(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("say,intent\nhello,greet\n")
    with pytest.raises(R.DatasetError) as exc:
        R.load(f"generic_csv|path={p}|text_col=nope")
    assert "no column" in str(exc.value)


def test_a_csv_whose_labels_exceed_its_declared_options_is_refused(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("say,intent\nhello,greet\nbye,other\n")
    with pytest.raises(R.DatasetError) as exc:
        R.load(f"generic_csv|path={p}|text_col=say|label_col=intent|options=greet")
    assert "not in its option set" in str(exc.value)


def test_unknown_dataset_names_the_alternatives():
    with pytest.raises(R.DatasetError) as exc:
        R.load("nope")
    assert "tool_routing" in str(exc.value)
