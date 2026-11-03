"""catalog — the candidate tool set a routing row is scored against, derived, not invented.

A row supervises a decision over a candidate set. If that set is written down here
it drifts from what the service actually runs, and the row teaches a vocabulary
nothing can emit. So the names come from the op registry: ``ti.ops.op_api``
registers every op module on import, and ``all_ops()`` is what the service will
execute. This module imports it and nothing else about the ops.

Two things are authored here, and both are claims about the derived set:

``DESCRIPTIONS``  the one-line candidate description the decision model reads —
                  "name + one-line description" is the representation 01 §1 calls
                  the practical default, so this is the text the model routes on.
``CONFUSABLE``    the hard-negative neighbourhoods: pairs two ops can be
                  mistaken for, taken from the t3 boundary-pair table in
                  ``tests/data/utterances_v2_NOTES.md``, which records the pairs
                  the run-2 misroutes actually landed on.

Both are asserted equal to the derived tool set in both directions (``check``):
a new op fails until it is described and placed, and a description for an op
that no longer exists fails too. ``python3 tools/distill/catalog.py`` prints the
catalog as the assertion sees it.
"""
from __future__ import annotations

import hashlib
import pathlib
import random
import sys
from typing import Any, Iterable

ROOT = pathlib.Path(__file__).resolve().parents[2]
_SERVICE = str(ROOT / "service")
if _SERVICE not in sys.path:
    sys.path.insert(0, _SERVICE)

SCHEMA_VER = "tools.v1"

# One line per op, and it has to be the difference, not a restatement of the name:
# a description that reads "glob files by pattern" teaches the model the word
# "glob" and not that the answer is a path list.
DESCRIPTIONS: dict[str, str] = {
    "filesys_glob": "List file paths matching a name pattern under a directory",
    "filesys_read": "Read one file's contents by path",
    "filesys_grep": "Search file contents for a pattern and return the matching lines",
    "web_search": "Run search queries against the web and return a deduplicated URL list",
    "web_fetch": "Download the pages at a list of URLs and extract their text",
    "reddit_digest": "Search reddit for a query and project the threads into a digest",
    "decide_extract": "Fill a list of literal values (queries, keywords, keys) out of an intent",
    "decide_pick": "Choose the top-k items out of a source artifact already produced",
}

# Hard negatives. Both directions are registered: the neighbourhood is symmetric,
# the row's gold is not.
CONFUSABLE: tuple[tuple[str, str], ...] = (
    ("filesys_glob", "filesys_grep"),
    ("filesys_grep", "filesys_read"),
    ("filesys_read", "filesys_glob"),
    ("filesys_read", "decide_extract"),
    ("filesys_glob", "decide_pick"),
    ("decide_pick", "decide_extract"),
    ("decide_extract", "web_search"),
    ("decide_extract", "filesys_glob"),
    ("web_search", "web_fetch"),
    ("web_search", "reddit_digest"),
    ("web_fetch", "reddit_digest"),
    ("filesys_grep", "web_search"),
    ("filesys_grep", "web_fetch"),
)


def ops() -> list[str]:
    """The ops the service exposes, straight from the registry."""
    from ti.ops import op_api

    op_api.import_all()
    return op_api.all_ops()


def op_params(name: str) -> dict[str, list[str]]:
    """Declared param schema for one op (required/optional), for the row's context."""
    from ti.ops import op_api

    op_api.import_all()
    d = op_api.get(name)
    return {"required": list(d.required), "optional": list(d.optional), "kind": [d.kind]}


def catalog() -> dict[str, str]:
    """name -> one-line description, sorted; the candidate block a row is built from."""
    names = ops()
    missing = [n for n in names if n not in DESCRIPTIONS]
    stale = [n for n in DESCRIPTIONS if n not in names]
    if missing or stale:
        raise AssertionError(
            "catalog drift: undocumented ops " + repr(missing) +
            ", descriptions for ops that no longer exist " + repr(stale) +
            " — give each op a one-line description in DESCRIPTIONS and a "
            "neighbourhood in CONFUSABLE")
    return {n: DESCRIPTIONS[n] for n in sorted(names)}


def catalog_sha() -> str:
    """Fingerprint of the catalog a row was built against."""
    blob = "\n".join(f"{n}\t{d}" for n, d in catalog().items())
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def rivals(tool: str) -> list[str]:
    """Ops this one can be mistaken for, nearest first (declaration order)."""
    out: list[str] = []
    for a, b in CONFUSABLE:
        if a == tool and b not in out:
            out.append(b)
    for a, b in CONFUSABLE:  # a pair declared the other way round still counts
        if b == tool and a not in out:
            out.append(a)
    return out


def rival_pairs() -> list[tuple[str, str]]:
    """Undirected confusable pairs, one entry each, first-seen order."""
    seen: dict[tuple[str, str], None] = {}
    for a, b in CONFUSABLE:
        seen.setdefault(tuple(sorted((a, b))), None)
    return list(seen)


def candidate_set(gold: str, seed: str, size: int = 5,
                  must_include: Iterable[str] = ()) -> dict[str, str]:
    """A candidate block for one row: the gold, its rivals, then fillers.

    01 §3.2: one gold, 2–5 hard negatives from the same functional neighbourhood,
    the rest filler. Rivals come first (they are the discriminating contrast), and
    the block is sorted by name so position carries no signal (01 §1). ``seed``
    makes the filler selection deterministic per row, so regenerating the corpus
    produces the same blocks.
    """
    cat = catalog()
    if gold not in cat:
        raise KeyError(f"candidate_set: {gold!r} is not an op in the catalog")
    picked: list[str] = [gold]
    for name in list(must_include) + rivals(gold):
        if name in cat and name not in picked:
            picked.append(name)
    if not set(must_include) <= set(picked):
        raise KeyError(f"candidate_set: rivals of {gold!r} cannot include {must_include!r}")
    pool = [n for n in sorted(cat) if n not in picked]
    rng = random.Random(f"{seed}|{gold}|{size}")
    rng.shuffle(pool)
    for name in pool:
        if len(picked) >= size:
            break
        picked.append(name)
    return {n: cat[n] for n in sorted(picked[:max(size, len(picked))]) if n in cat}


def check() -> dict[str, Any]:
    """The catalog as the assertions see it — the report ``__main__`` prints."""
    cat = catalog()
    return {"schema_ver": SCHEMA_VER, "catalog_sha": catalog_sha(),
            "tool_count": len(cat), "tools": cat,
            "rivals": {n: rivals(n) for n in cat}}


if __name__ == "__main__":
    import json

    print(json.dumps(check(), indent=2, sort_keys=True))
