"""Ingest — normalise the source logs and frozen files into one record schema.

The rule, stated once and applied by code
-----------------------------------------
``hops.jsonl`` is an **execution trace**, not a decision log. It records one hop
per executed step, and the op on a hop is what *ran*, not what was *correct*. A
hop therefore becomes a candidate ``choice`` record only if all of:

1. ``soft.intent_context`` is a non-empty string — that is the only field in the
   record that can serve as a state. 373 of the decide_pick hops have no intent,
   so nothing exists to make a decision *about*.
2. ``op`` is one of the 8 routable ops — otherwise the label is not an option of
   the routing criteria set and the row is unanswerable.
3. the op is a *consistent* function of the intent across the whole log. An
   intent that was executed as ``filesys_glob`` once and ``filesys_read`` another
   time has no single correct answer in the data, so it is dropped rather than
   resolved by majority vote (a vote would invent a label the log never asserts).

Surviving hops are then deduplicated to one record per distinct intent: 709
repetitions of the same intent are one decision observed 709 times, not 709
independent decisions. Surviving records are ``label_source="observed"``.

``live.jsonl`` is session-level. One session line is one routing decision — the
first hop — and every later hop in that session is execution trace. Sessions that
errored before a hop, or whose first hop is not routable, yield nothing.

The frozen files under ``tests/data/`` are the only **gold** labels in the repo:
human-vetted utterance -> label pairs. They are ingested as written, with the
band the file declares.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import criteria as C
from .paths import ROOT
from .paths import logs_dir as default_logs_dir

HOPS = "hops.jsonl"
LIVE = "live.jsonl"

FROZEN: tuple[tuple[str, str, str, str], ...] = (
    # (source name, filename, criteria_id, label field)
    ("utterances_v1", "utterances.json", C.ROUTING_OPS, "label"),
    ("utterances_v2", "utterances_v2.json", C.ROUTING_OPS, "label"),
    ("shape_expectations", "shape_expectations.json", C.SHAPE, "shape"),
    ("heldout_phase3", "heldout_phase3.json", C.SHAPE, "expect"),
    ("heldout_phase4", "heldout_phase4.json", C.SHAPE, "expect"),
)

_WS = re.compile(r"\s+")


def norm(text: str) -> str:
    """Utterance identity used for dedupe, ids and the leakage group key."""
    return _WS.sub(" ", text.strip().lower())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                out.append(rec)
    return out


def ingest(logs: Path | None = None, data: Path | None = None) -> tuple[list[dict], dict]:
    """Build the corpus. Returns (records, report); never raises on missing input."""
    logs = Path(logs) if logs is not None else default_logs_dir()
    data = Path(data) if data is not None else _default_data_dir()
    records: list[dict] = []
    report: dict[str, Any] = {"logs_dir": _portable(logs), "data_dir": _portable(data),
                              "sources": {}, "skipped": {}}

    records.extend(_ingest_hops(logs / HOPS, report))
    records.extend(_ingest_live(logs / LIVE, report))
    for source, filename, criteria_id, label_field in FROZEN:
        records.extend(_ingest_frozen(data / filename, source, criteria_id,
                                      label_field, report))

    seen: set[str] = set()
    unique: list[dict] = []
    for rec in records:
        if rec["id"] in seen:
            report["skipped"]["duplicate_id"] = report["skipped"].get("duplicate_id", 0) + 1
            continue
        seen.add(rec["id"])
        unique.append(rec)
    report["total"] = len(unique)
    return unique, report


def _ingest_hops(path: Path, report: dict) -> list[dict]:
    if not path.is_file():
        report["sources"]["hops"] = {"present": False, "records": 0}
        return []
    hops = read_jsonl(path)
    per_op = Counter(h.get("op") for h in hops)
    with_intent = Counter()
    records_per_op = Counter()
    skipped: Counter[str] = Counter()
    by_intent: dict[str, Counter] = defaultdict(Counter)
    first_text: dict[str, str] = {}

    for hop in hops:
        op = hop.get("op")
        intent = (hop.get("soft") or {}).get("intent_context")
        if not isinstance(intent, str) or not intent.strip():
            skipped["no_intent_context (pure execution trace)"] += 1
            continue
        with_intent[op] += 1
        if op not in C.OP_DESCRIPTIONS:
            skipped[f"op {op!r} not a routable option"] += 1
            continue
        key = norm(intent)
        by_intent[key][op] += 1
        first_text.setdefault(key, intent.strip())

    records: list[dict] = []
    for key, ops in by_intent.items():
        if len(ops) > 1:
            skipped["contradictory intent (executed as more than one op)"] += sum(ops.values())
            continue
        op = next(iter(ops))
        records_per_op[op] += 1
        records.append(_record(
            kind="choice", source="hops", group=f"u:{key}", label_source="observed",
            state=first_text[key], criteria_id=C.ROUTING_OPS, label=op,
            meta={"observations": sum(ops.values())},
        ))

    report["sources"]["hops"] = {
        "present": True, "records": len(records), "hop_lines": len(hops),
        "per_op": {op: {"hops": n, "with_intent": with_intent[op],
                        "records": records_per_op[op]}
                   for op, n in per_op.most_common()},
        "distinct_intents": len(by_intent),
        "distinct_decisions_after_dedupe": len(records),
    }
    report["skipped"].update(skipped)
    return records


def _ingest_live(path: Path, report: dict) -> list[dict]:
    if not path.is_file():
        report["sources"]["live"] = {"present": False, "records": 0}
        return []
    sessions = read_jsonl(path)
    skipped: Counter[str] = Counter()
    records: list[dict] = []

    for i, session in enumerate(sessions):
        intent = session.get("intent")
        hops = session.get("hops") or []
        if not isinstance(intent, str) or not intent.strip():
            skipped["no intent"] += 1
            continue
        if not hops:
            skipped["no hops (errored before routing)"] += 1
            continue
        op = (hops[0] or {}).get("op")
        if op not in C.OP_DESCRIPTIONS:
            skipped[f"first hop {op!r} not a routable option"] += 1
            continue
        records.append(_record(
            kind="choice", source="live", group=f"s:{session.get('ts', i)}",
            label_source="observed", state=intent.strip(), criteria_id=C.ROUTING_OPS,
            label=op,
            meta={"hops": [h.get("op") for h in hops if isinstance(h, dict)],
                  "status": session.get("status")},
        ))

    report["sources"]["live"] = {"present": True, "records": len(records),
                                 "session_lines": len(sessions)}
    for reason, n in skipped.items():
        report["skipped"][f"live: {reason}"] = n
    return records


def _ingest_frozen(path: Path, source: str, criteria_id: str, label_field: str,
                   report: dict) -> list[dict]:
    if not path.is_file():
        report["sources"][source] = {"present": False, "records": 0}
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        report["sources"][source] = {"present": True, "records": 0, "error": "unreadable"}
        return []

    records: list[dict] = []
    skipped = Counter()
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            skipped["not an object"] += 1
            continue
        utterance = row.get("utterance")
        label = row.get(label_field)
        if not isinstance(utterance, str) or not utterance.strip():
            skipped["no utterance"] += 1
            continue
        if label not in C.options(criteria_id):
            skipped[f"label {label!r} not an option of {criteria_id}"] += 1
            continue
        meta = {k: row[k] for k in ("family", "note") if isinstance(row.get(k), str)}
        if isinstance(row.get("tier_target"), str):
            meta["tier_target"] = row["tier_target"]
        records.append(_record(
            kind="choice", source=source, group=f"u:{norm(utterance)}",
            label_source="gold", state=utterance.strip(), criteria_id=criteria_id,
            label=label, band=row.get("band"), meta=meta or None,
        ))

    report["sources"][source] = {"present": True, "records": len(records),
                                 "rows": len(rows) if isinstance(rows, list) else 0}
    for reason, n in skipped.items():
        report["skipped"][f"{source}: {reason}"] = n
    return records


def _record(*, kind: str, source: str, group: str, label_source: str, state: Any,
            criteria_id: str, label: str, band: Any = None,
            meta: dict | None = None) -> dict:
    rec: dict[str, Any] = {
        "id": _record_id(kind, source, criteria_id, group, label, state),
        "kind": kind, "source": source, "group": group,
        "label_source": label_source, "criteria_id": criteria_id,
        "state": state, "criteria": dict(C.options(criteria_id)), "label": label,
    }
    if isinstance(band, str) and band:
        rec["band"] = band
    if meta:
        rec["meta"] = meta
    return rec


def _record_id(kind: str, source: str, criteria_id: str, group: str, label: str,
               state: Any) -> str:
    blob = json.dumps([kind, source, criteria_id, group, label, state],
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "tests" / "data"


def _portable(path: Path) -> str:
    """Repo-relative or ``~``-relative, never a machine-personal absolute path.

    The report is written to disk and therefore scanned by
    ``tests/test_no_personal_paths.py`` like any other portable artefact.
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix() or "."
    except ValueError:
        pass
    home = Path(os.path.expanduser("~")).resolve()
    try:
        return "~/" + resolved.relative_to(home).as_posix()
    except ValueError:
        return resolved.as_posix()
