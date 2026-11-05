# heldout_v3 — the evaluation set that can be split by construction

Artefacts:

| file | what it is |
|---|---|
| `bench/data/heldout_v3.json` | 415 rows, 8 ops, `family` + `split` on every row |
| `bench/data/heldout_v3_families.json` | the assignment record: seed, per-family side, counts, exclusions |
| `bench/gen_heldout_v3.py` | the generator — inventory → assign → render, in that order |
| `bench/report_v3.py` | the comparison table, recomputed offline from recorded runs |
| `bench/results/p2-heldout/v3-d1-3B-Q4.json` | d1-3B-Q4_K_M over the whole set, per row |

It exists because `tests/data/utterances_v2.json` cannot be split by family. 95 of its
150 rows are atomic, so the only holdout available there withholds *rows*, and a number
measured under it carries an optimism of unknown size. On the one dataset in this repo
that can be family-split (`routing_382`, 110 reconstructed templates) the same model
class loses **23.5 points**: 92.1% row-wise, 68.6% template-held-out. Nothing here can
say how much of v2's 74.0% is that leak, and this set is built to make the question
answerable for the 8-op label space.

## The defect this set does not have

| | v2 | v3 |
|---|---|---|
| rows | 150 | 415 |
| rows in a family of one | 95 (63%) | 59 (14%) |
| families | 91 resolved from 150 rows | 133 |
| largest family | 2 (inside a declared pair) | 6 |
| family split available | no — the enumeration is atomic | yes, declared before the text |

v2's *only* grouping is the 11 declared `tier_target` boundary pairs, which group
boundary rows by which rival they straddle — not by phrasing. Everything else is a row.

## Order of operations (this is the guarantee)

1. **Inventory.** Every authored frame in `tools/distill/frames.py` — 208 t1/t2 frames
   plus 176 t3 frames in 22 rival directions — with its slot capacity. Four frames are
   dropped here, before any split exists: three t1/t3 declarations are the same template
   as another frame (`which files mention {topic}?`, `list all files under {dir} matching
   {name}*`, `check the {sub} subreddit for {topic}`), and one t1 frame cannot render
   (`find files named {name}.{ext}` against an `ext` table whose values already carry the
   dot, which yields `probe..sh`). 380 frames reach the planner.
2. **Assign.** Each frame is placed on train or test *before any utterance is rendered*.
   A frame is the family: all slot-filled variants of one authored sentence share a
   skeleton, and the leak this set is built to avoid is exactly "same skeleton, different
   object". The seed (`20261104`) and the complete assignment are in
   `heldout_v3_families.json`.
3. **Render.** Slots are filled last, and a rendered row inherits its family's side.
   Nothing in step 3 can move a row across the split — `tests/test_heldout_v3.py`
   asserts that the recorded side is the planned side for every family.

Two biases are in the selection, and only two: slot-bearing frames first (a family must
be able to hold more than one row), and for t3 a round-robin across rival pairs (so a
subset keeps every boundary the set exists to probe). Within those, frames are taken in
`sha256(seed|family)` order — arbitrary, but not chosen for difficulty. No utterance was
edited after the model output was seen.

Size targets are per class and per band, matching v2's band proportions (t1 42% / t2
19% / t3 39% vs v2's 43% / 21% / 37%) so that the difficulty mix is the one being
proxied, not a new one. `decide_extract`'s t1 band tops out at 20 rows (twelve of its
fourteen t1 frames are slotless sentences and the other two are `{n}` frames with four
values each), so the shortfall spills into t3; that is why one class has 51 rows and the
rest have 52.

## Counts

Per class and side:

| class | rows | train | test |
|---|---:|---:|---:|
| `decide_extract` | 51 | 25 | 26 |
| `decide_pick` | 52 | 26 | 26 |
| `filesys_glob` | 52 | 26 | 26 |
| `filesys_grep` | 52 | 26 | 26 |
| `filesys_read` | 52 | 27 | 25 |
| `reddit_digest` | 52 | 26 | 26 |
| `web_fetch` | 52 | 26 | 26 |
| `web_search` | 52 | 26 | 26 |
| **total** | **415** | **208** | **207** |

Bands: t1 173, t2 80, t3 162. **133 families: 65 on train, 68 on test, no family on
both.** 356 of 415 rows (85.8%) sit in a multi-row family; the size histogram is
`{1: 59, 2: 5, 3: 1, 4: 27, 5: 11, 6: 30}`.

Family structure is not uniform across classes, and the difference matters:

| class | families | rows in multi-row families |
|---|---:|---:|
| `filesys_glob`, `filesys_grep`, `filesys_read`, `reddit_digest`, `web_search` | 10 each | 52/52 |
| `web_fetch` | 18 | 46/52 |
| `decide_pick` | 31 | 28/52 |
| `decide_extract` | 34 | 22/52 |

`decide_pick` and `decide_extract` are the two classes whose authored frames do not name
an object ("pick the best of these", "pull the keywords"), so most of them are
single-form sentences with a family of one. For those two classes a family split is
close to a phrasing split, and a held-out number on them is weaker evidence than the
same number on the six noun-bearing ops. This is a property of the inventory, not of the
split rule, and it is reported rather than patched.

## Validation — is this a proxy, or a new task?

The acceptance test is whether the set reproduces what is already known about the model.
d1-3B-Q4_K_M was run once over all 415 rows (`bench/live_probe.py`, the frozen
`ti_bench.prompts` instructions and option set, the same protocol as the archived v2
run), and the v2 side is the archived recording — no re-run, since the endpoint is
deterministic (0/150 argmax disagreement across two llama.cpp builds).

| set | n | accuracy (Wilson 95%) | how the split is drawn |
|---|---:|---|---|
| v3, all rows | 415 | **80.5% [76.4, 84.0]** | declared, family-disjoint |
| v3, train side | 208 | 79.3% [73.3, 84.3] | families declared train |
| v3, test side | 207 | 81.6% [75.8, 86.3] | families declared test |
| v2 (archived) | 150 | 74.0% [66.4, 80.4] | 95/150 atomic: no family split available |

`python3 bench/report_v3.py` regenerates every number below from the two recordings.

### The target was missed: 80.5%, not ~74%

Stated plainly, because the acceptance test asked for it plainly. The two thresholds
that would mean "this set is measuring something else" (95%, or 40%) were not tripped,
and the overall intervals overlap ([76.4, 84.0] against [66.4, 80.4]) — but 6.5 points
is not "near", and the gap is not noise: it is localised, and it is structural. It was
not revised after seeing this, for the reason the brief gives: text adjusted until the
3B scores 74% would be a mirror. What follows is where the points are.

### The failure mechanism reproduced; its depth did not

| class | v3 n | v3 | v2 n | v2 | Δ |
|---|---:|---|---:|---|---:|
| `decide_extract` | 51 | 39.2% [27.0, 52.9] | 24 | 25.0% [12.0, 44.9] | +14.2 |
| `filesys_grep` | 52 | 69.2% [55.7, 80.1] | 22 | 36.4% [19.7, 57.0] | +32.9 |
| `reddit_digest` | 52 | 80.8% [68.1, 89.2] | 15 | 100.0% [79.6, 100.0] | −19.2 |
| `filesys_read` | 52 | 82.7% [70.3, 90.6] | 17 | 70.6% [46.9, 86.7] | +12.1 |
| `filesys_glob` | 52 | 84.6% [72.5, 92.0] | 21 | 95.2% [77.3, 99.2] | −10.6 |
| `web_search` | 52 | 94.2% [84.4, 98.0] | 18 | 100.0% [82.4, 100.0] | −5.8 |
| `web_fetch` | 52 | 94.2% [84.4, 98.0] | 15 | 93.3% [70.2, 98.8] | +0.9 |
| `decide_pick` | 52 | 98.1% [89.9, 99.7] | 18 | 100.0% [82.4, 100.0] | −1.9 |

**`decide_extract` is still the worst class and `filesys_grep` is still second worst**,
which is the qualitative half of the acceptance test, and the misses are the same shape
phase 1 documented — the answer follows the object noun and ignores the operation:

| | v3 misses | v2 misses |
|---|---|---|
| `filesys_grep` | `filesys_glob` 6, `web_fetch` 5, `web_search` 3, `decide_pick` 1, `decide_extract` 1 | `web_fetch` 6, `filesys_glob` 4, `decide_extract` 2, `filesys_read` 1, `web_search` 1 |
| `decide_extract` | `filesys_read` 13, `filesys_glob` 7, `filesys_grep` 4, `web_search` 4, `decide_pick` 3 | `web_search` 8, `filesys_glob` 8, `decide_pick` 2 |

The split is the same; the rival that wins differs by which noun the row happens to
carry. On v3, `decide_extract`'s dominant loss is to **`filesys_read`** (13), from the
t3 families that hand a path to an extract request — "what terms are inside
docs/architecture.md?", "pull the values out of {path}, don't paste the file". v2 has no
rows of that shape, so its extract losses split between the path→glob and term→search
directions instead. The noun-routing explanation predicts both.

### Where the 6.5 points are

| subset | v3 | v2 |
|---|---|---|
| `filesys_grep` + `decide_extract` | 54.4% [44.8, 63.7] | 30.4% [19.1, 44.8] |
| the other six classes | 89.1% [85.2, 92.1] | 93.3% [86.8, 96.7] |

| band | v3 | v2 |
|---|---|---|
| t1 | 92.5% [87.6, 95.6] | 81.2% [70.0, 88.9] |
| t2 | 50.0% [39.3, 60.7] | 71.0% [53.4, 83.9] |
| t3 | 82.7% [76.2, 87.8] | 67.3% [54.1, 78.2] |

Three structural facts, each visible in the inventory rather than in the model:

1. **The weak classes are 24 points easier here, and that is most of the gap.** The
   selected grep families use `{sym}`, `{pat}`, `{topic}` and `{path}` — and **not one
   `{mod}` frame**. `{mod}` is the package-name column (`urllib`, `requests`,
   `fastapi`), and a package noun is exactly what sent 6 of v2's 14 grep misses to
   `web_fetch`. The inventory has several such frames ("does any file import {mod}?",
   "what files make use of {mod}?", "grep for {mod} in the codebase, offline", …) and
   the seeded hash order took a prefix that excluded all of them.
2. **The t1 band is 11 points easier because the inventory's t1 is a sanity floor.**
   `bands.py` defines t1 as *direct intent, signature vocabulary allowed*; v2's t1 rows
   were written to be direct *and* to carry traps ("which files mention X" labelled
   grep). A floor is a floor; it is not the same band.
3. **The t2 band is 21 points harder, and `reddit_digest` scores 0/10 there.** All ten
   of those misses go to `web_search`. The inventory's veiled frames strip the operation
   cue and lean on an abstract object ("the crowd opinion on {topic} — what is it?"),
   where v2's veiled rows keep a concrete, local object ("pull up the cron config"). A
   model that routes on nouns has nothing to hold in the first case.

Decomposed at this set's class weights: the two weak classes being 24 points easier is
worth **+5.9** points overall, the other six being 4 points harder is worth **−3.1**, and
the remaining **+3.7** is composition — v2 reserves 31% of its rows for the two weak
classes and this set, being balanced by class, reserves 25%. The band table is the same
gap seen from the other side: t1 and t3 are easier here, t2 is harder, and the three
almost cancel.

### Why it was not revised

The obvious repair — stratify grep/extract frame selection by slot column so the `{mod}`
frames are represented, and rebalance the t1/t2 depth — is a change to the *selection
rule*, derived from the inventory, and it would be defensible on coverage grounds alone
(a grep sample with no package-noun object is not a sample of grep). It is nevertheless
**not applied here**, because it was found by looking at the model's answers, and the
brief forbids building the mirror: after this result, any selection change is
indistinguishable from tuning toward 74%. It is written down as the next revision's
hypothesis and left untested. The committed set is the one that was measured.

### What this validation does and does not show

d1-3B is **frozen**: it saw neither side of the split, so the family split cannot move
its score, and the 79.3% / 81.6% train-versus-test difference is sampling noise, not a
finding. What the run establishes is that the set asks the model the same question at a
comparable level of difficulty, with the same failure structure. What it cannot establish
is that the split removes optimism, because there was no fitting to be optimistic about.
That property belongs to a model trained on v3-train, and it is asserted structurally
instead: no family crosses the split (`tests/test_heldout_v3.py`), 85.8% of rows sit in a
multi-row family, and `load_v3(split="row")` refills the same row counts with families
crossed — so the mirage-versus-honest gap is measurable on this set rather than
undefined, which is exactly what v2 cannot offer.

## What this instrument can and cannot measure

**It can** give a family-held-out accuracy for the 8-op label space that does not depend
on a split being reconstructed from text after the fact: the split is a property of the
data, not of the analysis. It can measure the mirage-versus-honest gap on those ops, per
class and per band. At 26 rows per side per class a per-class difference smaller than
roughly 20 points is not resolvable, and the overall figure resolves about 8 points per
side — that is what 415 rows buys, and it is the reason the size target exists. It can
hold out a whole skeleton: paraphrase variants of one authored sentence never straddle
the split, so a test row's *shape* is genuinely unseen by a model fitted on the train
side.

**It cannot** say anything about real user phrasing beyond these frames. Every sentence
in it is a slot-filled variant of something a human wrote in `tools/distill/frames.py`:
384 authored sentences in three registers, not an open-ended sample of how people ask. A
model that scores 90% here has shown it can route *these* phrasings; the distance from
here to a session transcript is unmeasured, and no number in this file closes it. It also
cannot be treated as calibrated to v2: it overstates v2's level by 6.5 points, for the
three structural reasons above. And for `decide_pick` / `decide_extract`, most rows sit
in families of one, so a held-out number on those two classes is closer to a phrasing
split than to a substitution split.

**Overlap with the distill corpus is real and reported, not removed.** 295 of 415 rows
carry a family that `data/distill/train.jsonl` or `val.jsonl` also trains on — both
artefacts are generated from the same frames, so this cannot be zero. It does not affect
the split inside this set (v3-train and v3-test never share a family), but a model trained
on `data/distill` has seen 160 of the 207 test-side rows' skeletons. Score it on the other
47, or join on `family` and drop the rest — the family id uses the same grammar in both
artefacts precisely so that join is mechanical.

**Its closest approach to the frozen 150 is one frame with a different noun**: "get me
whatever lives at that location" (v3) against "get me whatever lives at that address"
(v2), character ratio 0.83. No v3 row restates a v2 row exactly, and the two sets are
independent instruments, so this is register overlap rather than contamination.

## Reproducing

```bash
python3 bench/gen_heldout_v3.py --report      # counts, no write
python3 bench/gen_heldout_v3.py --check       # drift against the committed set
python3 bench/gen_heldout_v3.py               # regenerate both files
python3 -m pytest tests/test_heldout_v3.py -q # the split guards
python3 bench/report_v3.py                    # the validation tables, offline
```

The generator is deterministic: the seed is a hash key (`sha256("20261104|…")`), not an
RNG state, so a different Python or a different hash seed cannot reorder the set.

---

## Note appended 2026-11-05: 80.5% is a run, not a level

The validation table above reports **80.5% [76.4, 84.0]** for v3's all-rows baseline. That
figure was a single run of `bench/live_probe.py` on 2026-11-04 ending 01:50:47, at a p50
latency of 8 225 ms. The same instrument on the same 415 rows, re-run the same night at
02:36:09 → 03:14:16 at p50 **3 959 ms**, gives **325/415 = 78.31%** — **9 rows lower**, and the
difference is not spread evenly: ten of the eleven changed rows are `reddit_digest` and one is
`decide_pick`, in three authored families. Every other class is identical.

Two consequences for anyone quoting this set:

* **Quote the level with its run.** "80.5%" and "78.3%" are the same set, the same build and
  the same prompt 45 minutes apart. Neither is wrong; both are run-level. The interval
  [76.4, 84.0] covers both, which is the honest reading — the level is resolved to about ±4
  points, not to a digit.
* **Paired comparisons remain the right use of this set**, and they need both arms in one
  session. A live-vs-archive pairing on this instrument measures drift as much as effect, and
  it has now done so once: the description-rewrite verdict was built on this 80.5% and had to
  be corrected when the live baseline arm was completed
  (`bench/results/p3-transfer/rows.jsonl`, `arm: "baseline"`). See the appended 2026-11-05
  section of `docs/rewrite-transfer-2026-11-04.md` for the drift ledger and the protocol.

The note above — *"the overall figure resolves about 8 points per side"* — should be read as
**an** 8-point resolution with a session-to-session floor on top of it; the 9-row spread between
these two runs is what that floor looks like on *this* set.
