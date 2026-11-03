"""Split — deterministic, seeded, leakage-aware.

Leakage key
-----------
Rows are grouped by ``record["group"]`` and a whole group is assigned to one
split. The key is one of two things, never both:

``u:<normalised utterance>``
    Every row derived from an utterance — v1, v2, the shape annotations, the
    held-out sets, and the hop log's intents. Normalisation is lowercase plus
    whitespace collapse, so "List the files" and "list  the files" are one group.
    This is also what keeps the two *tasks* over the same utterance together: the
    v2 row and its ``shape_expectations`` row carry the same utterance, so they
    share a group and cannot straddle the split.

``s:<session id>``
    Rows derived from one live session (``live.jsonl`` ts). Everything a single
    session produced stays on one side.

The hop log has no session id — no plan id, no request id, only ``ts`` — so
intent identity is the finest defensible key, and it is also the coarsest safe
one: all hops for one intent land together.

Residual leakage this key does NOT prevent, stated so it is not mistaken for
coverage: paraphrase families. ``heldout_phase3`` declares a ``family`` such as
``listing-extension``; two members of that family can land in different splits
with different utterances and the same structure. Splitting on family instead
would collapse the held-out sets to ~5 groups and make the split meaningless, so
utterance identity is the chosen trade and family-level drift remains possible.

Algorithm
---------
Groups are bucketed by ``(source, label)`` so every split sees every op, each
bucket sorted then shuffled with a ``random.Random`` seeded from
``(seed, bucket)``, and each bucket's rows are dealt to the splits by
**largest-remainder quotas** (80/10/10 of that bucket, rounded to whole rows).

The fractional part left over by a bucket's rounding is carried into the next
bucket instead of being dropped. Without the carry, rounding ten ``(source,
label)`` buckets up into val compounds and the corpus lands at 80.7/11.5/7.8;
with it the rounding cancels and the totals stay at the requested ratio.

Two things this replaced, both measured: walking sources alphabetically put every
row of the last-processed source into val/test (train had already hit its count),
and interleaving buckets by index still starved a label whose bucket ran dry
before train filled (``web_fetch`` ended at 20/0/0). Per-bucket quotas give 2/2/1
rather than 5/0/0 for a five-row label.

A bucket with fewer rows than splits cannot cover all three: a one-row label goes
to train and the other splits have none of it, which is arithmetic, not a bug. The
smallest label here has 20 rows, so every op appears in every split.

Deterministic for a given seed, stable under input order, and never thresholding a
hash (which leaves val/test empty at small n).
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_SEED = 13
SPLITS = ("train", "val", "test")
DEFAULT_RATIOS: tuple[tuple[str, float], ...] = (("train", 0.8), ("val", 0.1), ("test", 0.1))


def grouping_key(rec: Mapping[str, Any]) -> str:
    return str(rec["group"])


def assign_groups(records: Sequence[Mapping[str, Any]], seed: int = DEFAULT_SEED,
                  ratios: Iterable[tuple[str, float]] = DEFAULT_RATIOS) -> dict[str, str]:
    """Map every group key to a split name. Pure function of (records, seed, ratios)."""
    ratios = tuple(ratios)
    names = [name for name, _ in ratios]
    size: dict[str, int] = defaultdict(int)
    bucket_of: dict[str, tuple[str, str]] = {}
    for rec in records:
        key = grouping_key(rec)
        size[key] += 1
        bucket_of.setdefault(key, (str(rec["source"]), str(rec.get("label"))))

    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key in sorted(bucket_of):
        buckets[bucket_of[key]].append(key)

    out: dict[str, str] = {}
    carry = {name: 0.0 for name in names}
    for name in sorted(buckets):
        keys = buckets[name]
        random.Random(f"{seed}:{name[0]}:{name[1]}").shuffle(keys)
        count = sum(size[k] for k in keys)
        quota, carry = _quotas(count, ratios, carry)
        placed = {s: 0 for s in names}
        for key in keys:
            split = max(names, key=lambda s: (quota[s] - placed[s], -names.index(s)))
            out[key] = split
            placed[split] += size[key]
    return out


def _quotas(total: int, ratios: tuple[tuple[str, float], ...],
            carry: dict[str, float]) -> tuple[dict[str, int], dict[str, float]]:
    """Whole-row split counts for one bucket, carrying the rounding remainder."""
    names = [name for name, _ in ratios]
    exact = {name: total * share + carry.get(name, 0.0) for name, share in ratios}
    quota = {name: max(0, int(exact[name])) for name in names}
    leftover = total - sum(quota.values())
    order = sorted(names, key=lambda n: (-(exact[n] - int(exact[n])), names.index(n)))
    step = 1 if leftover > 0 else -1
    for name in (order if step > 0 else list(reversed(order)))[:abs(leftover)]:
        quota[name] = max(0, quota[name] + step)
    return quota, {name: exact[name] - quota[name] for name in names}


def split_records(records: Sequence[Mapping[str, Any]], seed: int = DEFAULT_SEED,
                  ratios: Iterable[tuple[str, float]] = DEFAULT_RATIOS
                  ) -> dict[str, list[dict[str, Any]]]:
    """Split records into {split name: rows}, each row carrying a ``split`` field."""
    assignment = assign_groups(records, seed=seed, ratios=ratios)
    out: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in ratios}
    for rec in records:
        row = dict(rec)
        row["split"] = assignment[grouping_key(rec)]
        out[row["split"]].append(row)
    return out


def check_leakage(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[str]:
    """Any group appearing in two splits, as actionable messages."""
    where: dict[str, set[str]] = defaultdict(set)
    for name, rows in splits.items():
        for rec in rows:
            where[grouping_key(rec)].add(name)
    return [f"group {key!r} straddles splits {sorted(names)}"
            for key, names in sorted(where.items()) if len(names) > 1]
