"""Export — SFT chat JSONL, the label legend, and the eval file the bench reads.

Two consumers, two formats, and only one of each:

``sft.jsonl`` / ``manifest.jsonl`` / ``labels.json``
    The fine-tuning side. One JSON object per line, ``{"messages": [...]}`` and
    nothing else, so a trainer that only knows chat records can read it. The
    per-row metadata goes in ``manifest.jsonl`` keyed by ``id``, and the
    letter -> option legend in ``labels.json``. The assistant message is the bare
    option letter, which is what makes the task a one-token classification.

``eval_<split>.json``
    The eval side. ``bench/live_probe.py`` reads ``--utterances`` with
    ``json.loads(path)`` and then indexes ``row["utterance"]`` and
    ``row["label"]``, so the file is a JSON **array** of objects with exactly
    those keys. The frozen files are in that shape already; this reproduces it
    rather than inventing a second eval format. Only ``routing.ops_v1`` records
    can go here — the harness compares ``choice == row["label"]`` against op
    names, so a two-option shape row would score as a miss forever.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import criteria as C

SFT_NAME = "sft.jsonl"
MANIFEST_NAME = "manifest.jsonl"
LABELS_NAME = "labels.json"
EVAL_TEMPLATE = "eval_{split}.json"

_MANIFEST_FIELDS = ("id", "split", "kind", "source", "label_source", "criteria_id",
                    "label", "letter", "band", "group", "state")


def messages_for(rec: dict[str, Any]) -> list[dict[str, str]]:
    criteria_id = rec["criteria_id"]
    return [
        {"role": "system", "content": C.system_prompt(criteria_id)},
        {"role": "user", "content": C.state_text(rec["state"])},
        {"role": "assistant", "content": C.letter_for(criteria_id, rec["label"])},
    ]


def sft_rows(records: Iterable[dict[str, Any]], criteria_id: str = C.ROUTING_OPS,
             include_observed: bool = False) -> tuple[list[dict], list[dict], int]:
    """(sft lines, manifest lines, skipped count) for the SFT side."""
    sft: list[dict] = []
    manifest: list[dict] = []
    skipped = 0
    for rec in records:
        if rec.get("kind") != "choice" or rec.get("criteria_id") != criteria_id:
            skipped += 1
            continue
        if rec.get("label_source") != "gold" and not include_observed:
            skipped += 1
            continue
        entry = dict(rec)
        entry["letter"] = C.letter_for(criteria_id, rec["label"])
        sft.append({"messages": messages_for(entry)})
        manifest.append({k: entry.get(k) for k in _MANIFEST_FIELDS})
    return sft, manifest, skipped


def eval_rows(records: Iterable[dict[str, Any]],
              include_observed: bool = False) -> list[dict[str, Any]]:
    """Rows for the bench harness: the keys live_probe indexes, plus the band.

    Gold labels only by default. The harness scores ``choice == row["label"]``, so
    an executed-op label in the eval file would put unverified rows into a number
    that is supposed to measure routing quality against vetted answers.
    """
    rows: list[dict[str, Any]] = []
    for rec in records:
        if rec.get("kind") != "choice" or rec.get("criteria_id") != C.ROUTING_OPS:
            continue
        if rec.get("label_source") != "gold" and not include_observed:
            continue
        row: dict[str, Any] = {"utterance": C.state_text(rec["state"]),
                               "label": rec["label"]}
        if isinstance(rec.get("band"), str) and rec["band"]:
            row["band"] = rec["band"]
        if isinstance(rec.get("meta"), dict) and isinstance(rec["meta"].get("tier_target"), str):
            row["tier_target"] = rec["meta"]["tier_target"]
        rows.append(row)
    return rows


def legend(criteria_id: str = C.ROUTING_OPS) -> dict[str, Any]:
    return {
        "criteria_id": criteria_id,
        "instructions": C.INSTRUCTIONS,
        "letters": {letter: opt for opt, letter in C.letters(criteria_id).items()},
        "descriptions": dict(C.options(criteria_id)),
    }


def write_export(splits: dict[str, Sequence[dict[str, Any]]], out: Path,
                 criteria_id: str = C.ROUTING_OPS, include_observed: bool = False,
                 eval_splits: Sequence[str] = ("val", "test")) -> dict[str, Any]:
    """Write every artefact; returns a summary of what was written and why."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ordered = [rec for name in ("train", "val", "test") for rec in splits.get(name, [])]

    sft, manifest, skipped = sft_rows(ordered, criteria_id=criteria_id,
                                     include_observed=include_observed)
    _write_jsonl(out / SFT_NAME, sft)
    _write_jsonl(out / MANIFEST_NAME, manifest)
    (out / LABELS_NAME).write_text(json.dumps(legend(criteria_id), indent=2) + "\n",
                                   encoding="utf-8")

    written: dict[str, Any] = {"sft": len(sft), "skipped": skipped, "eval": {}}
    for name in eval_splits:
        rows = eval_rows(splits.get(name, []), include_observed=include_observed)
        path = out / EVAL_TEMPLATE.format(split=name)
        path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        written["eval"][name] = len(rows)
    return written


def read_sft(path: Path) -> list[dict[str, Any]]:
    return _read_jsonl(Path(path))


def read_manifest(path: Path) -> list[dict[str, Any]]:
    return _read_jsonl(Path(path))


def load_legend(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
