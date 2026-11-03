"""corpus — what a distillation row IS, and the guard that keeps it out of the eval.

The row shape is 01 §2's, narrowed to the one decision this corpus supervises: a
single-hop routing choice over the service's own candidate block. One JSON object
per line, ``target`` separate from ``candidates`` so a re-serializer change does
not rewrite the labels, and every field the eval needs to slice on present on
every row: ``band`` (t1/t2/t3), ``source`` (template/session), ``family`` (the
split unit) and ``split`` (train/val only).

The leakage guarantee is a refusal, not a review step. ``tests/data/
utterances_v2.json`` is the held-out eval set; a generated row may never restate
one of its 150 utterances, and a row may never be split into it. Both are checked
in three places: at generation (a colliding candidate is dropped and counted), at
label time (``assert_not_frozen`` runs again on the row about to be written, so a
hand-edited candidate file cannot smuggle one in), and in ``stats.py`` over the
finished corpus (the guarantee is reported, not assumed).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import pathlib
from typing import Any, Iterable, Iterator

from . import catalog, similarity

ROOT = pathlib.Path(__file__).resolve().parents[2]
FROZEN_PATH = ROOT / "tests" / "data" / "utterances_v2.json"
OUT_DIR = ROOT / "var" / "distill"

BANDS = ("t1", "t2", "t3")
SOURCES = ("template", "session")
SPLITS = ("train", "val")
VAL_SHARE = 0.15

REQUIRED = ("id", "schema_ver", "catalog_sha", "band", "tier_target", "source",
            "family", "split", "utterance", "instructions", "state", "candidates",
            "intended", "target", "teacher", "cost_usd", "label_method", "ts")


class LeakageError(AssertionError):
    """A row that would put eval content into train/val."""


class RowError(AssertionError):
    """A row that does not match the schema."""


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def frozen() -> similarity.Corpus:
    """The held-out 150. Read on every call: cheap, and it cannot go stale."""
    return similarity.Corpus.from_json(FROZEN_PATH, label="utterances_v2")


def assign_split(family: str) -> str:
    """Split by family, never by row: paraphrases of one frame stay on one side.

    01 §3.6 and 03 §3 both land here — an item-level split leaks, because two
    surface forms of the same intent land on both sides and the val number then
    measures memorisation. ``family`` is the frame (template rows) or the session
    (mined rows), so the hash is a stable coin per family.
    """
    digest = hashlib.sha256(family.encode()).digest()
    return "val" if digest[0] / 255.0 < VAL_SHARE else "train"


def row_id(index: int) -> str:
    return f"rt-{index:04d}"


def build(*, band: str, source: str, family: str, utterance: str,
          instructions: str, state: dict[str, Any], candidates: dict[str, str],
          intended: str | None = None, tier_target: str | None = None,
          index: int = 0) -> dict[str, Any]:
    """One unlabelled row. ``target``/``teacher``/``cost_usd`` arrive at label time."""
    row = {
        "id": row_id(index),
        "schema_ver": catalog.SCHEMA_VER,
        "catalog_sha": catalog.catalog_sha(),
        "band": band,
        "tier_target": tier_target,
        "source": source,
        "family": family,
        "split": assign_split(family),
        "utterance": utterance,
        "instructions": instructions,
        "state": state,
        "candidates": candidates,
        "intended": intended,
        "target": None,
        "teacher": None,
        "cost_usd": None,
        "label_method": None,
        "ts": None,
    }
    assert_row(row)
    return row


def assert_row(row: dict[str, Any]) -> None:
    """Structural validation. Runs before a row is written and after it is read back."""
    missing = [k for k in REQUIRED if k not in row]
    if missing:
        raise RowError(f"row {row.get('id', '?')}: missing fields {missing}")
    if row["band"] not in BANDS:
        raise RowError(f"row {row['id']}: band {row['band']!r} not in {BANDS}")
    if row["source"] not in SOURCES:
        raise RowError(f"row {row['id']}: source {row['source']!r} not in {SOURCES}")
    if row["split"] not in SPLITS:
        raise RowError(
            f"row {row['id']}: split {row['split']!r} not in {SPLITS} — the frozen "
            f"{FROZEN_PATH.name} set is the eval split and no generated row may claim it")
    text = row["utterance"]
    if not isinstance(text, str) or not 8 <= len(text) <= 120:
        raise RowError(f"row {row['id']}: utterance length {len(str(text))} outside 8..120")
    cands = row["candidates"]
    if not isinstance(cands, dict) or not 2 <= len(cands) <= 40:
        raise RowError(f"row {row['id']}: candidate block must hold 2..40 ops")
    if row["band"] == "t3":
        if not row["tier_target"]:
            raise RowError(f"row {row['id']}: a t3 row must name its rival (tier_target)")
        if row["tier_target"] not in cands:
            raise RowError(f"row {row['id']}: rival {row['tier_target']!r} is not offered "
                           "as a candidate, so the boundary it claims is not tested")
        if row["tier_target"] == row["intended"]:
            raise RowError(f"row {row['id']}: rival equals the intended op")
    elif row["tier_target"] is not None:
        raise RowError(f"row {row['id']}: tier_target present on a {row['band']} row")
    if row["intended"] is not None and row["intended"] not in cands:
        raise RowError(f"row {row['id']}: intended op {row['intended']!r} not in candidates")
    target = row["target"]
    if target is not None:
        if not isinstance(target, dict) or target.get("tool") not in cands:
            raise RowError(f"row {row['id']}: target tool is not one of the candidates")
        conf = target.get("conf")
        if not isinstance(conf, (int, float)) or not 0.0 <= float(conf) <= 1.0:
            raise RowError(f"row {row['id']}: confidence {conf!r} outside 0..1")
        if row["teacher"] is None or row["ts"] is None:
            raise RowError(f"row {row['id']}: a labelled row must carry teacher + ts")
    else:
        if row["teacher"] is not None or row["cost_usd"] is not None:
            raise RowError(f"row {row['id']}: teacher/cost present without a target")


def assert_not_frozen(utterance: str, reference: similarity.Corpus | None = None) -> None:
    """Refuse anything that restates the held-out eval set."""
    ref = reference or frozen()
    match, hit = ref.match(utterance)
    if match.leaks:
        raise LeakageError(
            f"utterance collides with the frozen eval set ({match.rule}: "
            f"jaccard={match.jaccard:.2f} ratio={match.ratio:.2f}): "
            f"{utterance!r} ~ {hit!r}")


def assert_labelled(row: dict[str, Any]) -> None:
    if row.get("target") is None:
        raise RowError(f"row {row.get('id')}: not labelled")
    assert_row(row)


class Gate:
    """The refusal point every generated utterance passes: dedup, then the frozen set.

    One instance is shared across the whole generation run, so a mined row is
    checked against the template rows generated before it and vice versa — the
    corpus is deduplicated as a single set, not per source.
    """

    def __init__(self, reference: similarity.Corpus | None = None) -> None:
        self.frozen = reference or frozen()
        self.generated = similarity.Corpus(label="generated")
        self.dropped: dict[str, int] = {}
        self.rejected: list[dict[str, str]] = []

    def offer(self, text: str) -> str:
        """"" when `text` may join the corpus, else the reason it may not."""
        match, hit = self.generated.match(text)
        if match.duplicate:
            return self.drop("dedup", text, f"restates an earlier row ({match.rule}): {hit!r}")
        match, hit = self.frozen.match(text)
        if match.leaks:
            return self.drop("frozen", text,
                             f"collides with the frozen eval set ({hit!r})")
        self.generated.add(text)
        return ""

    def drop(self, kind: str, text: str, why: str) -> str:
        self.dropped[kind] = self.dropped.get(kind, 0) + 1
        if len(self.rejected) < 40:
            self.rejected.append({"kind": kind, "utterance": text, "why": why})
        return why


def leakage_report(rows: Iterable[dict[str, Any]],
                   reference: similarity.Corpus | None = None) -> dict[str, Any]:
    """Post-hoc verification over a finished corpus — reported, not assumed."""
    ref = reference or frozen()
    collisions: list[dict[str, Any]] = []
    worst = (0.0, "")
    n = 0
    for row in rows:
        n += 1
        match, hit = ref.match(row["utterance"])
        if match.leaks:
            collisions.append({"id": row.get("id"), "utterance": row["utterance"],
                               "matched": hit, **match.as_dict()})
        if match.ratio > worst[0]:
            worst = (match.ratio, row.get("id", ""))
    return {"frozen_rows": len(ref), "frozen_sha": ref.sha, "checked_rows": n,
            "collisions": collisions, "ok": not collisions,
            "max_ratio": round(worst[0], 3), "max_ratio_row": worst[1],
            "method": similarity.METHOD}


def read_jsonl(path: pathlib.Path | str) -> list[dict[str, Any]]:
    p = pathlib.Path(path)
    if not p.exists():
        return []
    rows = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: pathlib.Path | str) -> Iterator[dict[str, Any]]:
    return iter(read_jsonl(path))


def append_jsonl(path: pathlib.Path | str, row: dict[str, Any]) -> None:
    """Append one row and make it durable — a crash must not cost the run."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def write_jsonl(path: pathlib.Path | str, rows: Iterable[dict[str, Any]]) -> int:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            n += 1
        fh.flush()
        os.fsync(fh.fileno())
    return n


def write_json(path: pathlib.Path | str, obj: Any) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: pathlib.Path | str, default: Any = None) -> Any:
    p = pathlib.Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))
