"""The record schema and its validator.

One record is one trainable decision. Three kinds:

``choice``   state + criteria (option id -> description) + label, and the label
             must be one of the criteria's option ids — that constraint is what
             makes the row answerable at all.
``noul``     state + yes/no question + p, the probability that the answer is yes.
``extract``  state + the extraction schema + the values pulled out of it.

Common to all three: ``id``, ``kind``, ``source``, ``group``, ``label_source``.

``group`` is the leakage key: rows sharing it must never straddle a split.
``label_source`` is honest provenance — ``gold`` means a human-vetted frozen file
supplied the label, ``observed`` means the log recorded only which op was
executed, which is not the same claim as "this was the right op".

The validator reports every problem it finds, not the first, because the repair
loop is "read the list, fix the rows, re-run".
"""
from __future__ import annotations

from typing import Any

from . import criteria as C

KINDS = ("choice", "noul", "extract")
LABEL_SOURCES = ("gold", "observed")

_COMMON = ("id", "kind", "source", "group", "label_source", "state")
_REQUIRED: dict[str, tuple[str, ...]] = {
    "choice": _COMMON + ("criteria_id", "criteria", "label"),
    "noul": _COMMON + ("question", "p"),
    "extract": _COMMON + ("schema", "value"),
}

_OPTIONAL = ("meta", "band", "family", "note", "conf")


def required_keys(kind: str) -> tuple[str, ...]:
    return _REQUIRED[kind]


def check_record(rec: Any) -> list[str]:
    """Every schema violation in one record, as actionable messages."""
    if not isinstance(rec, dict):
        return [f"record is {type(rec).__name__}, expected an object"]

    kind = rec.get("kind")
    if kind not in KINDS:
        return [f"kind={kind!r} is not one of {list(KINDS)}"]

    errs: list[str] = []
    for key in _REQUIRED[kind]:
        if key not in rec:
            errs.append(f"missing required key {key!r} for kind={kind}")

    for key in ("id", "source", "group", "label_source"):
        if key in rec and not _nonempty_str(rec[key]):
            errs.append(f"{key} must be a non-empty string, got {rec[key]!r}")

    if rec.get("label_source") not in LABEL_SOURCES and "label_source" in rec:
        errs.append(f"label_source={rec.get('label_source')!r} is not one of "
                    f"{list(LABEL_SOURCES)}")

    errs.extend(_check_state(rec))

    if kind == "choice":
        errs.extend(_check_choice(rec))
    elif kind == "noul":
        errs.extend(_check_noul(rec))
    elif kind == "extract":
        errs.extend(_check_extract(rec))

    return errs


def check_corpus(records: list[Any]) -> list[str]:
    """Per-record violations plus corpus-wide duplicate ids."""
    errs: list[str] = []
    seen: dict[str, int] = {}
    for i, rec in enumerate(records):
        rid = rec.get("id") if isinstance(rec, dict) else None
        for msg in check_record(rec):
            errs.append(f"record[{i}] id={rid!r}: {msg}")
        if _nonempty_str(rid):
            if rid in seen:
                errs.append(f"record[{i}] id={rid!r}: duplicate id, first seen at "
                            f"record[{seen[rid]}] (ids must be content-derived and unique)")
            else:
                seen[rid] = i
    return errs


def _check_state(rec: dict[str, Any]) -> list[str]:
    if "state" not in rec:
        return []
    state = rec["state"]
    if isinstance(state, str):
        return [] if state.strip() else ["state is an empty string (no decision can be "
                                        "made from an empty state)"]
    if isinstance(state, (dict, list)):
        return [] if state else [f"state is an empty {type(state).__name__} "
                                 "(no decision can be made from an empty state)"]
    return [f"state must be a string or an object/array, got {type(state).__name__}"]


def _check_choice(rec: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    cid = rec.get("criteria_id")
    if "criteria_id" in rec:
        if cid not in C.CRITERIA:
            errs.append(f"criteria_id={cid!r} is not a known criteria set "
                        f"{sorted(C.CRITERIA)}")
        elif rec.get("criteria") != C.CRITERIA[cid]:
            errs.append(f"criteria does not match criteria_id={cid!r} — the option text "
                        "must be the frozen set for that task")

    crit = rec.get("criteria")
    if "criteria" in rec:
        if not isinstance(crit, dict) or not crit:
            errs.append("criteria must be a non-empty object of option id -> description")
            crit = None
        else:
            for opt, desc in crit.items():
                if not _nonempty_str(opt):
                    errs.append(f"criteria option id {opt!r} must be a non-empty string")
                if not _nonempty_str(desc):
                    errs.append(f"criteria[{opt!r}] description must be a non-empty "
                                f"string, got {desc!r}")

    if "label" in rec:
        label = rec["label"]
        if not _nonempty_str(label):
            errs.append(f"label must be a non-empty string, got {label!r}")
        elif isinstance(crit, dict) and label not in crit:
            errs.append(f"label={label!r} is not one of the criteria options "
                        f"{sorted(crit)} — the answer must be an option of its own "
                        "question")
    return errs


def _check_noul(rec: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    if "question" in rec and not _nonempty_str(rec["question"]):
        errs.append(f"question must be a non-empty string, got {rec['question']!r}")
    if "p" in rec:
        p = rec["p"]
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            errs.append(f"p must be a number, got {type(p).__name__}")
        elif not 0.0 <= float(p) <= 1.0:
            errs.append(f"p={p!r} is outside [0, 1]")
    return errs


def _check_extract(rec: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    if "schema" in rec and (not isinstance(rec["schema"], dict) or not rec["schema"]):
        errs.append("schema must be a non-empty object describing the fields to extract")
    if "value" in rec and not isinstance(rec["value"], dict):
        errs.append(f"value must be an object of extracted fields, got "
                    f"{type(rec['value']).__name__}")
    return errs


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())
