#!/usr/bin/env python3
"""gen_heldout_v3 — an evaluation set whose split is decided before its text exists.

`tests/data/utterances_v2.json` cannot be split by family: 95 of its 150 rows are
atomic, so there is no holdout that withholds a phrasing rather than a row, and every
number measured on it carries an optimism of unknown size. This module builds the set
that does not have that defect, out of the one inventory in the repo that can produce
one: the 384 authored frames in `tools/distill/frames.py`.

The order of operations is the whole point, and it is enforced by the code order:

1. **inventory** — every authored frame, with the number of distinct utterances it can
   yield (its slot capacity; a frame with no `{slot}` yields exactly one).
2. **assign** — each frame is placed on the train or test side *before any utterance
   is rendered*. A frame is a **family**: all slot-filled variants of one authored
   sentence share a skeleton, so they go to one side together or not at all. The seed
   and the complete assignment are written to `heldout_v3_families.json`.
3. **render** — only now are slots filled, and a rendered row inherits the side its
   family was given. Nothing in step 3 can move a row across the split.

Why a frame is the right unit: the inflation this repo has already measured (92.1%
row-wise vs 68.6% template-held-out on `routing_382`) comes from the same skeleton
with a different object. Nothing else in the pipeline varies more than the object.

What this set can and cannot measure is stated in `bench/data/heldout_v3_NOTES.md`.
The generator does not tune: the selection order is a seeded hash of the frame id, so
a frame is never chosen for being easy, and only two things bias the choice at all —
slot-bearing frames first (so families are large enough to *be* families) and, for
t3, round-robin across rival pairs (so the boundary coverage survives the subset).

Usage:
    python3 bench/gen_heldout_v3.py            # regenerate both committed files
    python3 bench/gen_heldout_v3.py --check    # report drift against what is committed
    python3 bench/gen_heldout_v3.py --report   # counts only, no write
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from distill import frames, similarity, templates  # noqa: E402

OUT = ROOT / "bench" / "data" / "heldout_v3.json"
FAMILIES_OUT = ROOT / "bench" / "data" / "heldout_v3_families.json"
V2_PATH = ROOT / "tests" / "data" / "utterances_v2.json"
DISTILL_DIR = ROOT / "data" / "distill"

# The seed the split and the ordering are derived from. Recorded in the manifest, and
# the only knob that changes the assignment: it is a hash key, not an RNG state, so a
# different Python cannot reorder the set.
SEED = 20261104

# Rows per class per band, matching the frozen 150's band proportions (42.7 / 20.7 /
# 36.7 on v2, 42.3 / 19.2 / 38.5 here). The proportions are copied on purpose: the
# set is meant to be a proxy for v2, and a band mix that drifted toward t1 would
# measure the easy band instead.
BAND_ROWS = {"t1": 22, "t2": 10, "t3": 20}
MAX_VARIANTS = 6          # a family larger than this stops adding information
MIN_ROWS_PER_CLASS = 30   # the floor a usable per-class interval needs

_SLOT = re.compile(r"\{(\w+)\}")
OPS = ("decide_extract", "decide_pick", "filesys_glob", "filesys_grep",
       "filesys_read", "reddit_digest", "web_fetch", "web_search")


# --------------------------------------------------------------------------- #
# the inventory
# --------------------------------------------------------------------------- #

@dataclass
class Frame:
    """One authored frame, its family id, and how many utterances it can yield."""

    band: str
    label: str
    rival: str | None
    index: int
    text: str

    @property
    def family(self) -> str:
        # same grammar as tools/distill/templates.py::_row, so a v3 family id is a
        # join key against the distill corpus's `family` column
        if self.rival:
            return f"{self.band}:{self.label}->{self.rival}:{self.index:02d}"
        return f"{self.band}:{self.label}:{self.index:02d}"

    @property
    def capacity(self) -> int:
        """Distinct utterances this frame can produce.

        `templates.fill` rotates one value per slot in lockstep, so the number of
        variants is the widest slot list — not the product of them. A frame with no
        slots is a family of exactly one, which is a fact about the inventory and is
        reported as such rather than papered over.
        """
        names = _SLOT.findall(self.text)
        return max((len(frames.SLOTS[n]) for n in names), default=1)


def _renders_broken(template: str) -> bool:
    """A frame whose literal punctuation contradicts its slot values.

    `find files named {name}.{ext}` is written against a `ext` table whose values
    already carry the dot (`.sh`, `.md`), so every render of it reads `probe..sh`.
    That is a defect in the frame, not a hard row, and it is dropped at the inventory
    rather than filtered out of the text afterwards — a filter on rendered words would
    be a rule about wording, and this is a rule about the slot table.
    """
    for match in _SLOT.finditer(template):
        before = template[match.start() - 1] if match.start() else ""
        values = frames.SLOTS[match.group(1)]
        if before == "." and all(v.startswith(".") for v in values):
            return True
    return False


def inventory() -> tuple[list[Frame], list[dict]]:
    """Every authored frame, minus the ones that cannot contribute a valid row.

    Two frames can be the same sentence: `t1:filesys_grep` and
    `t3:filesys_grep->filesys_glob` both carry "which files mention {topic}?", and
    `reddit_digest` carries one template in both t1 and t3. If both are kept they render
    the same utterance twice and belong to two different families — which is a family
    straddling the split waiting to happen, whichever side each is given. The duplicate
    is therefore removed from the *inventory*, before the split exists, and the first
    declaration wins (t1 before t2 before t3 — the more direct claim).
    """
    out: list[Frame] = []
    seen: dict[str, str] = {}
    dropped: list[dict] = []
    for band, table in (("t1", frames.T1), ("t2", frames.T2)):
        for op, texts in table.items():
            for i, t in enumerate(texts, 1):
                frame = Frame(band, op, None, i, t)
                norm = similarity.normalize(t)
                if norm in seen:
                    dropped.append({"family": frame.family, "template": t,
                                    "why": f"same template as {seen[norm]}"})
                    continue
                if _renders_broken(t):
                    dropped.append({"family": frame.family, "template": t,
                                    "why": "literal punctuation contradicts the slot table"})
                    continue
                seen[norm] = frame.family
                out.append(frame)
    for label, rival, texts in frames.T3:
        for i, t in enumerate(texts, 1):
            frame = Frame("t3", label, rival, i, t)
            norm = similarity.normalize(t)
            if norm in seen:
                dropped.append({"family": frame.family, "template": t,
                                "why": f"same template as {seen[norm]}"})
                continue
            if _renders_broken(t):
                dropped.append({"family": frame.family, "template": t,
                                "why": "literal punctuation contradicts the slot table"})
                continue
            seen[norm] = frame.family
            out.append(frame)
    return out, dropped


def _key(*parts: str) -> str:
    return hashlib.sha256("|".join((str(SEED),) + parts).encode()).hexdigest()


def _selection_order(pool: list[Frame]) -> list[Frame]:
    """Slot-bearing frames first, each group in seeded hash order.

    Slot-bearing means capacity >= 2: only those frames can carry a multi-row family,
    and a set whose families are all singletons cannot be split. Within a group the
    order is `sha256(SEED|band|family)` — arbitrary but not chosen, which is the most
    this can honestly claim.
    """
    return (sorted((f for f in pool if f.capacity >= 2), key=lambda f: _key(f.band, f.family))
            + sorted((f for f in pool if f.capacity == 1), key=lambda f: _key(f.band, f.family)))


def _interleaved(pool: list[Frame]) -> list[Frame]:
    """`_selection_order` per rival pair, then round-robin across pairs.

    Only t3 has the pair structure, and only some pairs hold slot-bearing frames
    (every `decide_extract -> filesys_glob` frame is slotless, for instance). Taking
    the pool in plain capacity order would therefore fill the t3 budget from one pair
    and drop the others — which would remove the very boundaries the set exists to
    probe. Round-robin keeps every pair in play and spends the extra rows on whichever
    pair can carry them.
    """
    buckets: dict[str, list[Frame]] = defaultdict(list)
    for f in pool:
        buckets[str(f.rival)].append(f)
    ordered = {rival: _selection_order(v) for rival, v in buckets.items()}
    out: list[Frame] = []
    for i in range(max((len(v) for v in ordered.values()), default=0)):
        for rival in sorted(ordered):
            if i < len(ordered[rival]):
                out.append(ordered[rival][i])
    return out


# --------------------------------------------------------------------------- #
# step 2 — assign families (before any utterance exists)
# --------------------------------------------------------------------------- #

@dataclass
class Family:
    frame: Frame
    size: int = 0
    split: str = ""

    @property
    def id(self) -> str:
        return self.frame.family


def plan_class_band(pool: list[Frame], rows: int) -> list[Family]:
    """Split a class-band budget over the fewest frames that can carry it.

    Grow the frame count until the selected frames' capacity covers `rows`, then hand
    each one `min(capacity, MAX_VARIANTS)` and trim the overhang off the tail. Growing
    the count only when capacity runs out is what makes families as large as the
    inventory allows — a 22-row band backed by four 8-variant frames is four real
    families, not twenty-two fake ones.
    """
    taking: list[Frame] = []
    capacity = 0
    for frame in pool:
        if capacity >= rows:
            break
        taking.append(frame)
        capacity += min(frame.capacity, MAX_VARIANTS)
    if capacity < rows:
        raise SystemExit(
            f"inventory cannot carry {rows} rows for {pool[0].band}/{pool[0].label}: "
            f"capacity {capacity} over {len(pool)} frames")

    sizes = [min(f.capacity, MAX_VARIANTS) for f in taking]
    overhang = sum(sizes) - rows
    for i in range(len(sizes) - 1, -1, -1):
        if overhang <= 0:
            break
        give = min(overhang, sizes[i] - 1)
        sizes[i] -= give
        overhang -= give
    assert overhang <= 0 and sum(sizes) == rows
    return [Family(f, s) for f, s in zip(taking, sizes)]


def assign_sides(families: list[Family]) -> None:
    """Place whole families on train/test: balanced by rows, and band-complete.

    Two properties have to hold at once. Every band of every class must appear on both
    sides, or a per-band comparison is impossible; and the two sides must carry the same
    number of rows, or a per-class interval is not comparable between them. Balancing
    per band achieves the first and drifts on the second (the small bands round badly),
    so this does both: the two largest families of each band seed the two sides, and the
    rest are placed greedy largest-first against the row total.
    """
    rows = {"train": 0, "test": 0}
    by_band: dict[str, list[Family]] = defaultdict(list)
    for fam in families:
        by_band[fam.frame.band].append(fam)
    rest: list[Family] = []
    for band in sorted(by_band):
        ordered = sorted(by_band[band], key=lambda f: (-f.size, _key("split", f.id)))
        if len(ordered) < 2:
            raise SystemExit(f"{band}/{ordered[0].frame.label}: one family cannot be split")
        seed, second, rest_here = ordered[0], ordered[1], ordered[2:]
        first, other = ("train", "test") if _key("coin", seed.id)[0] in "01234567" \
            else ("test", "train")
        seed.split, second.split = first, other
        rows[first] += seed.size
        rows[other] += second.size
        rest += rest_here
    for fam in sorted(rest, key=lambda f: (-f.size, _key("split", f.id))):
        side = "train" if rows["train"] <= rows["test"] else "test"
        fam.split = side
        rows[side] += fam.size


def band_budget(pools: dict[str, list[Frame]]) -> dict[str, int]:
    """The per-band row budget for one class, clamped to what its inventory can carry.

    `decide_extract` t1 is the one cell that cannot reach 22: twelve of its fourteen t1
    frames are slotless sentences and the other two are `{n}` frames with four values
    each, so the band tops out at 20.
    The shortfall spills into the band with room (t3), never into t2 — veiled rows are
    the scarcest and cheapest-to-get-wrong part of the mix, and inflating them would
    move the set away from the band proportions it exists to mirror.
    """
    cap = {band: sum(min(f.capacity, MAX_VARIANTS) for f in pools[band]) for band in BAND_ROWS}
    budget = {band: min(BAND_ROWS[band], cap[band]) for band in BAND_ROWS}
    short = sum(BAND_ROWS[band] - budget[band] for band in BAND_ROWS)
    for band in ("t3", "t2", "t1"):
        room = cap[band] - budget[band]
        add = min(room, short)
        budget[band] += add
        short -= add
        if not short:
            break
    if short:
        raise SystemExit(f"inventory cannot carry the row budget: {short} rows short")
    return budget


def plan() -> tuple[list[Family], dict[str, dict[str, int]], list[dict]]:
    inv, dropped = inventory()
    plan_rows: list[Family] = []
    budgets: dict[str, dict[str, int]] = {}
    for op in OPS:
        pools = {band: [f for f in inv if f.label == op and f.band == band]
                 for band in ("t1", "t2", "t3")}
        budgets[op] = band_budget(pools)
        per_class: list[Family] = []
        for band in ("t1", "t2", "t3"):
            order = _interleaved(pools[band]) if band == "t3" else _selection_order(pools[band])
            per_class += plan_class_band(order, budgets[op][band])
        assign_sides(per_class)
        plan_rows += per_class
    return plan_rows, budgets, dropped


# --------------------------------------------------------------------------- #
# step 3 — render, only now
# --------------------------------------------------------------------------- #

def _frozen_norms() -> dict[str, str]:
    rows = json.loads(V2_PATH.read_text(encoding="utf-8"))
    return {similarity.normalize(r["utterance"]): r["utterance"] for r in rows}


def render(families: list[Family]) -> tuple[list[dict], list[dict]]:
    """Render each family's slots into rows, keeping its assigned side.

    Dedup is by normalised text, and it is a refusal: a rendered utterance may not
    restate another rendered utterance (two frames can be the same sentence — `t1`
    and `t3` both carry "which files mention {topic}?"), and it may not be a verbatim
    restatement of a frozen v2 row. A refused variant is skipped and the next slot
    rotation is tried, so a family keeps the size the split gave it.
    """
    frozen = _frozen_norms()
    taken: dict[str, str] = {}
    rows: list[dict] = []
    refused: list[dict] = []

    for fam in families:
        salt = int(_key("salt", fam.id)[:8], 16)
        texts: list[str] = []
        for candidate in templates.fill(fam.frame.text, salt):
            if len(texts) == fam.size:
                break
            norm = similarity.normalize(candidate)
            if norm in taken or norm in frozen:
                refused.append({"family": fam.id, "utterance": candidate,
                                "why": "restates a rendered row" if norm in taken
                                       else "restates a frozen v2 row"})
                continue
            taken[norm] = fam.id
            texts.append(candidate)
        if len(texts) < fam.size:
            refused.append({"family": fam.id, "utterance": "",
                            "why": f"only {len(texts)} distinct of {fam.size} wanted"})
        for i, text in enumerate(texts):
            rows.append({
                "utterance": text,
                "label": fam.frame.label,
                "band": fam.frame.band,
                "tier_target": fam.frame.rival,
                "family": fam.id,
                "split": fam.split,
                "_order": _key("order", fam.id, str(i)),
            })

    rows.sort(key=lambda r: r.pop("_order"))
    for i, row in enumerate(rows, 1):
        row["id"] = f"v3-{i:04d}"
    return [dict({"id": r.pop("id")}, **r) for r in rows], refused


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def counts(rows: list[dict]) -> dict:
    per_class = {op: {"n": 0, "train": 0, "test": 0} for op in OPS}
    per_band: dict[str, int] = defaultdict(int)
    for r in rows:
        per_class[r["label"]]["n"] += 1
        per_class[r["label"]][r["split"]] += 1
        per_band[r["band"]] += 1
    return {"rows": len(rows), "per_class": per_class, "per_band": dict(sorted(per_band.items()))}


def family_report(rows: list[dict]) -> dict:
    sizes: dict[str, int] = defaultdict(int)
    side: dict[str, str] = {}
    label: dict[str, str] = {}
    band: dict[str, str] = {}
    for r in rows:
        sizes[r["family"]] += 1
        side[r["family"]] = r["split"]
        label[r["family"]] = r["label"]
        band[r["family"]] = r["band"]
    per_side = {s: 0 for s in ("train", "test")}
    for fam, s in side.items():
        per_side[s] += 1
    multi = sum(n for n in sizes.values() if n > 1)

    def _slice(group: dict[str, str]) -> dict:
        """Each value's family structure — a class whose rows are all singletons has a
        row split wearing a family split's clothes, and the number has to be visible."""
        out: dict[str, dict] = {}
        for fam, n in sizes.items():
            s = out.setdefault(group[fam], {"families": 0, "singleton_families": 0,
                                            "rows": 0, "rows_in_multi_row_families": 0})
            s["families"] += 1
            s["rows"] += n
            if n == 1:
                s["singleton_families"] += 1
            else:
                s["rows_in_multi_row_families"] += n
        return dict(sorted(out.items()))

    return {
        "families": len(sizes),
        "families_per_side": per_side,
        "singleton_families": sum(1 for n in sizes.values() if n == 1),
        "singleton_rows": sum(n for n in sizes.values() if n == 1),
        "multi_row_families": sum(1 for n in sizes.values() if n > 1),
        "rows_in_multi_row_families": multi,
        "rows_in_multi_row_families_pct": round(100 * multi / len(rows), 1) if rows else 0.0,
        "largest_family": max(sizes.values()) if sizes else 0,
        "size_histogram": {str(k): sum(1 for n in sizes.values() if n == k)
                           for k in sorted(set(sizes.values()))},
        "by_class": _slice(label),
        "by_band": _slice(band),
    }


def leakage_report(rows: list[dict]) -> dict:
    """How close the new set sits to the frozen 150, measured rather than assumed."""
    frozen = _frozen_norms()
    corpus = similarity.Corpus.from_json(V2_PATH, label="v2")
    worst = (0.0, "", "")
    exact = 0
    for r in rows:
        if similarity.normalize(r["utterance"]) in frozen:
            exact += 1
        match, hit = corpus.match(r["utterance"])
        if match.ratio > worst[0]:
            worst = (match.ratio, r["utterance"], hit)
    return {"exact_restatements_of_v2": exact,
            "max_ratio_to_v2": round(worst[0], 3),
            "max_ratio_row": worst[1], "max_ratio_nearest_v2": worst[2]}


def distill_overlap(rows: list[dict]) -> dict:
    """Rows whose family the committed distill corpus already trains on.

    Both artefacts are generated from the same frames, so this number is not zero
    and cannot be: it is the join a consumer needs in order to score a model trained
    on `data/distill` on the part of this set that corpus never saw.
    """
    seen: dict[str, str] = {}
    for name in ("train.jsonl", "val.jsonl"):
        path = DISTILL_DIR / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                seen[rec["family"]] = name
    rows_seen = sum(1 for r in rows if r["family"] in seen)
    return {"families_in_distill": len(seen),
            "resolved_rows": rows_seen,
            "resolved_test_rows": sum(1 for r in rows if r["family"] in seen and r["split"] == "test"),
            "unresolved_rows": len(rows) - rows_seen,
            "note": "a model trained on data/distill has seen these families; drop them "
                    "by family join to score it on unseen phrasings"}


def build() -> tuple[list[dict], dict, list[dict]]:
    families, budgets, dupes = plan()
    rows, refused = render(families)
    rendered: dict[str, int] = {}
    for row in rows:
        rendered[row["family"]] = rendered.get(row["family"], 0) + 1
    # a family whose every variant was refused (one frame in this inventory is a
    # verbatim frozen v2 row) is not part of the set, so it is not part of the record
    dropped = [{"family": fam.id, "planned_rows": fam.size, "split": fam.split,
                "why": "every rendered variant was refused"}
               for fam in families if fam.id not in rendered]
    c = counts(rows)
    short = [op for op in OPS if c["per_class"][op]["n"] < MIN_ROWS_PER_CLASS]
    argue = [f"{op}:{band}" for op in OPS for band in BAND_ROWS
             if not ({"train", "test"} <= {r["split"] for r in rows
                                           if r["label"] == op and r["band"] == band})]
    if short or argue:
        raise SystemExit(f"plan is not usable: short classes {short}, "
                         f"class-bands missing a side {argue}")
    manifest = {
        "schema": "heldout_v3_families/1",
        "seed": SEED,
        "generator": "bench/gen_heldout_v3.py",
        "frames": {"module": "tools/distill/frames.py",
                   "sha256": hashlib.sha256(
                       (ROOT / "tools/distill/frames.py").read_bytes()).hexdigest()[:16],
                   "frames": len(inventory()[0]),
                   "duplicate_templates_dropped": len(dupes), "duplicates": dupes},
        "rule": ("family = one authored frame (all slot-filled variants of one sentence "
                 "skeleton); a family is assigned to train or test before any utterance "
                 "is rendered. Selection: slot-bearing frames first, seeded-hash order "
                 "within groups, t3 round-robin across rival pairs. Balance: greedy "
                 "largest-first per class-band, ties by seeded hash."),
        "rows_per_class_band": budgets,
        "max_variants_per_family": MAX_VARIANTS,
        "counts": c,
        "families": family_report(rows),
        "leakage": leakage_report(rows),
        "distill_overlap": distill_overlap(rows),
        "refused": {"n": len(refused), "examples": refused[:12]},
        "dropped_families": dropped,
        "assignment": {fam.id: {"label": fam.frame.label, "band": fam.frame.band,
                                "rival": fam.frame.rival, "split": fam.split,
                                "rows": rendered.get(fam.id, 0),
                                "planned_rows": fam.size,
                                "capacity": fam.frame.capacity,
                                "template": fam.frame.text}
                       for fam in families if fam.id in rendered},
    }
    return rows, manifest, refused


def _render_table(manifest: dict) -> str:
    c, fam, leak, over = (manifest["counts"], manifest["families"], manifest["leakage"],
                          manifest["distill_overlap"])
    lines = ["| label | n | train | test |", "|---|---:|---:|---:|"]
    for op, s in c["per_class"].items():
        lines.append(f"| `{op}` | {s['n']} | {s['train']} | {s['test']} |")
    lines.append(f"| **total** | **{c['rows']}** | "
                 f"**{sum(s['train'] for s in c['per_class'].values())}** | "
                 f"**{sum(s['test'] for s in c['per_class'].values())}** |")
    lines += ["", f"bands: {c['per_band']}",
              f"families: {fam['families']} ({fam['families_per_side']}), "
              f"multi-row {fam['multi_row_families']} covering "
              f"{fam['rows_in_multi_row_families']} rows "
              f"({fam['rows_in_multi_row_families_pct']}%), largest {fam['largest_family']}",
              f"sizes: {fam['size_histogram']}",
              "", "| class | rows | families | singleton families | rows in multi-row families |",
              "|---|---:|---:|---:|---:|"]
    for op, s in fam["by_class"].items():
        lines.append(f"| `{op}` | {s['rows']} | {s['families']} | "
                     f"{s['singleton_families']} | {s['rows_in_multi_row_families']} |")
    lines += ["", f"vs v2: exact restatements {leak['exact_restatements_of_v2']}, "
              f"max ratio {leak['max_ratio_to_v2']}",
              f"in data/distill: {over['resolved_rows']} rows "
              f"({over['resolved_test_rows']} of them on the test side)"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="compare a fresh build against the committed files")
    ap.add_argument("--report", action="store_true", help="print counts, write nothing")
    args = ap.parse_args()

    rows, manifest, refused = build()
    if args.report:
        print(_render_table(manifest))
        return
    if args.check:
        drift = []
        if not OUT.exists() or json.loads(OUT.read_text()) != rows:
            drift.append(OUT.relative_to(ROOT))
        if not FAMILIES_OUT.exists() or json.loads(FAMILIES_OUT.read_text()) != manifest:
            drift.append(FAMILIES_OUT.relative_to(ROOT))
        print("in sync with the committed set" if not drift
              else "DRIFT: " + ", ".join(str(p) for p in drift))
        raise SystemExit(1 if drift else 0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    FAMILIES_OUT.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n",
                            encoding="utf-8")
    print(_render_table(manifest))
    print(f"\nwrote {OUT.relative_to(ROOT)} ({len(rows)} rows) and "
          f"{FAMILIES_OUT.relative_to(ROOT)}")
    if refused:
        print(f"refused variants: {len(refused)}")


if __name__ == "__main__":
    main()
