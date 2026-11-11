#!/usr/bin/env python3
"""Build the classical tier's training corpus from the v3 TRAIN split only.

Why this exists: the shipped classical backend pointed at `bench/registry_382.json`, a registry
of 17 voice-assistant tools, while the router asks it to choose among eight code/file ops. The
label sets shared no names, so the fitted model could name none of the candidates and abstained
on every request it has ever received. A path was configured; nothing checked that the vocabulary
on the other end could answer the question being asked.

So the corpus is built rather than curated: one flat `{"utterance", "label"}` row per v3 training
row, and nothing from the eval split. Training on the eval rows would produce a number that looks
better the more it leans on the rows it is scored against.

    python3 bench/build_classical_corpus.py            # writes the file
    python3 bench/build_classical_corpus.py --check    # exits 1 if the file is stale

The generator refuses to write a corpus that cannot answer the eight ops, which is the check that
was missing: a corpus whose labels do not intersect the option set is not a model, it is a
permanent abstention with a fit step in front of it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "bench/data/heldout_v3.json"
TARGET = ROOT / "service/ti/data/classical_v3_train.json"
OPS = ("filesys_read", "filesys_glob", "filesys_grep", "decide_pick", "decide_extract",
       "web_search", "web_fetch", "reddit_digest")


def build() -> list[dict]:
    rows = json.loads(SOURCE.read_text(encoding="utf-8"))
    train = [r for r in rows if r.get("split") == "train"]
    if not train:
        raise SystemExit(f"{SOURCE}: no row has split == 'train'")
    corpus = [{"utterance": r["utterance"], "label": r["label"]} for r in train]
    labels = sorted({c["label"] for c in corpus})
    unknown = [lab for lab in labels if lab not in OPS]
    if unknown:
        raise SystemExit(
            f"{SOURCE}: labels {unknown} are not among the eight ops — a corpus cannot be built "
            f"from a vocabulary the router never offers")
    if not set(labels) & set(OPS):
        raise SystemExit("the corpus shares no label with the option set: refusing to write it")
    return corpus


def main(argv: list[str]) -> int:
    corpus = build()
    text = json.dumps(corpus, indent=1) + "\n"
    if "--check" in argv:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if current != text:
            print(f"{TARGET} is stale; re-run without --check")
            return 1
        print(f"{TARGET} is current ({len(corpus)} rows)")
        return 0
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(text, encoding="utf-8")
    per_label: dict[str, int] = {}
    for row in corpus:
        per_label[row["label"]] = per_label.get(row["label"], 0) + 1
    print(f"wrote {TARGET.relative_to(ROOT)}: {len(corpus)} rows, "
          f"{len(per_label)} labels, train split only")
    for label in sorted(per_label):
        print(f"  {label:<16}{per_label[label]:>4}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
