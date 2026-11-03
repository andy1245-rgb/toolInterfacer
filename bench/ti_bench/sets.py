"""sets — the frozen utterance sets, and the stable row id every report carries.

A row id has to survive two things: a report being joined to another report
(``compare``), and the set file being read by someone who does not have the
positional index. So each row carries

  ``id``   ``"<set>:<idx>"`` — the row's position in the frozen file, which is
           how docs/model-sweep-consolidated.md and bench/band_summary.py
           reference a row ("idx 30")
  ``uid``  a short hash of the utterance text, which is stable across sets: the
           same sentence in v1 and v2 keeps its uid, and a reordered file still
           pairs up correctly

v1 has no ``band`` field; its rows report band ``"all"`` so per-band aggregation
is one code path rather than a special case.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# set name -> repo-relative file. Paths are relative so the harness is portable.
SETS = {
    "v1": "tests/data/utterances.json",
    "v2": "tests/data/utterances_v2.json",
}

NO_BAND = "all"


def uid(utterance: str) -> str:
    """Stable 12-hex id for one utterance (sha1 is fine: identity, not security)."""
    return hashlib.sha1(utterance.encode()).hexdigest()[:12]


def load(name: str) -> list[dict]:
    """Rows of a frozen set: id / uid / idx / band / utterance / expected."""
    if name not in SETS:
        raise SystemExit(f"unknown set {name!r} (known: {', '.join(sorted(SETS))})")
    raw = json.loads((ROOT / SETS[name]).read_text())
    rows = []
    for idx, row in enumerate(raw):
        rows.append({"id": f"{name}:{idx:03d}", "uid": uid(row["utterance"]), "idx": idx,
                     "band": row.get("band") or NO_BAND, "utterance": row["utterance"],
                     "expected": row["label"]})
    return rows


def bands(rows: list[dict]) -> dict[str, int]:
    """Band -> row count, in band order."""
    out: dict[str, int] = {}
    for row in rows:
        out[row["band"]] = out.get(row["band"], 0) + 1
    return out


def path_of(name: str) -> str:
    """The set file as a repo-relative string (portable in a report)."""
    return SETS[name]
