# The evaluation harness, and what the local numbers become under an honest split

2026-11-04. Builds `bench/eval_harness.py` — group-aware splits, repeated splits,
cluster-bootstrap intervals, one registerable interface — and re-measures every
local signal in this repo through it. Branch `p1-harness` @ `0939140`.

## The finding that reframes the local claims

**The 150-row set that every published local number was measured on has no
phrasing families, so it cannot be split by family at all.** 63.3% of its rows
(95 of 150) are atomic — they carry no group. Holding families out therefore
holds out rows, and the harness measures the consequence directly: the
family-held-out number and the row-wise number agree to within **±4 points and in
both directions**, which is smaller than the seed-to-seed spread of the splits
themselves.

The one set that *can* be split by family is the 382-row set, and there the same
model class loses **23.5 points**:

| set | families | row-wise | family-held-out | gap |
|---|---|---:|---:|---:|
| 382 rows, 18-way | 110 reconstructed templates | 92.1% | **68.6%** | **−23.5 pts** |
| 150 rows, 8-way | 11 declared boundary pairs + 95 atomic rows | 66.7% | 67.1% | **−0.4 pts** |

Read together: the local classifiers' true generalisation penalty is about 24
points, and this repo's own eval set is structurally incapable of showing it.
Every v2 figure in `docs/` — 78.7%, 74.0%, 61.3%, 50.0%, 84.7% — is a row-wise
figure wearing a group-shaped label. That is not a correction of the arithmetic
in those documents (it all reproduces, exactly); it is a statement about what
their split could not test.

## 1. Which dataset can support an honest split

Both sets were confirmed by reading their keys, not by trusting the docs:

| file | n | keys per row | family field |
|---|---:|---|---|
| `tests/data/utterances_v2.json` | 150 | `utterance`, `label`, `band` | **none** |
| `tests/data/routing_382.json` | 382 | `text`, `tool`, `family` | yes, but see §2 |

The 150-row set cannot be family-split. `band` is a difficulty grade
(t1/t2/t3), not a phrasing family; the only grouping it declares is
`tier_target` on the 55 t3 rows, which names the confusable rival op and
therefore does define a boundary-pair family for those rows and no others.

`tests/test_eval_harness.py::test_v2_has_no_family_field_and_yields_the_documented_groups`
pins this so a later change to the file cannot quietly invalidate the claim.

## 2. Reconstructing families on the 150 — attempted, and it fails

Three routes were tried. Two fail, one yields a real but limited grouping. The
distinction matters, because inventing a family grouping and presenting it as the
dataset's would manufacture the very result this harness exists to prevent.

**Route A — the distill frame inventory (`tools/distill/frames.py`).** This file
holds 208 authored frames with `{slot}` holes across `T1`/`T2`/`T3`, and
`tools/distill/templates.py:127` derives the family label
`f"{band}:{op}->{rival}:{frame_index:02d}"` that the task brief cites. Matching
each v2 row back to a frame by literal prefix/suffix — the same reconstruction
`bench/classical_baseline.py::template_key` uses for the 382 — resolves **9 of
150 rows (6%)**. The two are separate authoring passes over the same eight ops,
not one generated from the other: `frames.T1["filesys_read"]` offers
`"open {path} and paste it back"`, while v2 ships `"open service/ti/registry.py"`,
which matches no frame. **Route A is unusable.**

**Route B — `var/distill/labels.jsonl`.** `var/` is gitignored build output
(`.gitignore`: "distill pipeline output: generated rows, labels, audit CSVs
(never committed)") and **does not exist in this worktree**, so the file cannot
be read at all. It would not have helped even if present: it labels the rows the
distill pipeline generates from those frames, not v2's hand-written rows.

**Route C — normalise and cluster near-duplicate phrasings.** Attempted, and it
recovers almost nothing. Single-link clustering on char-3-gram Jaccard ≥ 0.25,
one label at a time (the same construction `bench/local_clf.py::build_groups`
uses, reproduced exactly and asserted in `test_v2_surface_matches_local_clf_build_groups`):

| grouping | groups over 150 rows | singleton groups | atomic row fraction |
|---|---:|---:|---:|
| declared t3 boundary pairs only | 106 | 95 | **63.3%** |
| + char-3gram near-duplicate clusters on t1/t2 | 102 | 87 | **58.0%** |
| every row its own group | 150 | 150 | 100% |

95 t1/t2 rows resolve to 91 clusters: four pairs merge and nothing else does.
The t1/t2 rows are not paraphrases of each other. **Confidence in Route C as a
family grouping: low.** It finds 4 genuine near-duplicate pairs out of 95 rows;
its only real structure is the t3 pairs, which the dataset declares itself.

**What survives.** The 55 t3 rows in 11 boundary-pair families
(`sorted((label, tier_target))`) is a defensible grouping — the dataset names it,
and it holds out both directions of one confusable boundary at a time. That is
the `boundary` grouping used for every 150-row number below. It leaves 63.3% of
the set atomic, which is exactly why it changes the answer so little.

**Consequence.** 150 is reported as "not evaluable by family" and every honest
family-held-out number comes from 382.

## 3. The harness

`bench/eval_harness.py` — stdlib only, ~580 lines including docstrings, no numpy,
no service imports. `bench/eval_signals.py` is the driver that registers this
repo's five signals.

```python
from eval_harness import Predictor, load_v2, measure, render_table

class MyPredictor(Predictor):
    name = "mine"
    def fit(self, train):        # train: list[Example]; Example has .text/.label/.group
        ...
    def predict(self, text):      # -> str | None (None is scored as a miss)
        ...

ds = load_v2()                          # or load_382()
grouped = measure(ds, [MyPredictor()], split="group")   # honest
rowwise = measure(ds, [MyPredictor()], split="row")     # the mirage
print(render_table(grouped + rowwise))
```

What it guarantees:

- **`split` has no hidden default.** `"group"` holds whole families out;
  `"row"` is stratified per label and ignores groups. Any other value raises.
  Every `Result` records its split, so a bare accuracy is not producible.
- **Repeated splits.** `DEFAULT_SEEDS = (0,1,2,3,4)`; the seed changes which
  group lands in which fold, so a number is a distribution. `Result.per_seed`,
  `.acc_min`, `.acc_max`, `.acc_sd` expose it.
- **Cluster-bootstrap CI.** Whole groups are resampled with replacement, so the
  interval respects dependence inside a family. On 382's template split this is
  **2.35× wider** than the Wilson interval the published documents use (§5).
- **`frozen = True`** for a served model or a recorded run: no refit, scored once,
  still with a CI. Resplitting a frozen predictor cannot change its predictions,
  so the harness does not pretend it did.
- **Per-row exports** (`Result.outcomes`, `.preds_per_seed`, `.folds_per_seed`)
  so a caller can pair two signals — union ceilings, per-class selection,
  McNemar — without re-running anything.
- **Family reconstruction is shared, not reinvented.** `r382_template_groups`
  reproduces `classical_baseline.template_key` and `v2_surface_groups` reproduces
  `local_clf.build_groups`; both are asserted equal by test, because a third
  private family definition would make the harness's numbers incomparable with
  the published ones.

`tests/test_eval_harness.py` (15 cases, 0.7 s) pins: no group straddles two
folds on any dataset or seed; a row split *does* straddle; different seeds give
different splits; the cluster interval is wider than the independent one on
clustered data and collapses to it when every row is its own group; and a model
with no capacity beyond the opening three tokens scores under a row split and
collapses under a template-held-out split, which is the sanity check that the
group split is actually grouping.

## 4. The honest baseline — every local signal, one table

5-fold, 5 seeds, cluster-bootstrap 95% CI over groups. `n` and the number of
groups are in the table because a number without them is the defect this task
exists to fix.

### The 150-row set, 8-way (this repo's own task)

`split=group` holds out the 11 declared t3 boundary families (106 groups, 63.3%
of rows atomic). `split=row` is stratified per label.

| signal | split | n | groups | atomic | acc | 95% CI | seeds (min–max) |
|---|---|---:|---:|---:|---:|---|---|
| `rules` (keyword, **contaminated**) | group | 150 | 106 | 63% | **78.7%** | 72.3–85.3 | — |
| `rules` | row | 150 | 150 | 100% | **78.7%** | 72.0–85.3 | — |
| `lr` (tf-idf + softmax) | group | 150 | 106 | 63% | **67.1%** | 59.6–73.9 | 65.3–70.0 |
| `lr` | row | 150 | 150 | 100% | **66.7%** | 59.9–73.3 | 65.3–68.7 |
| `1nn` (char 3–5, cosine) | group | 150 | 106 | 63% | **51.3%** | 44.4–57.7 | 48.0–56.0 |
| `1nn` | row | 150 | 150 | 100% | **55.1%** | 48.5–61.9 | 54.0–56.7 |
| `classical_svm` (tf-idf + LinearSVC) | group | 150 | 106 | 63% | **65.7%** | 58.2–72.8 | 64.0–67.3 |
| `classical_svm` | row | 150 | 150 | 100% | **62.5%** | 55.7–69.7 | 60.0–67.3 |
| `d1_3B` Q4_K_M (frozen) | either | 150 | — | — | **74.0%** | 66.5–81.3 | — |

`d1_3B` is frozen: its *point estimate* is split-invariant, because a model that
is never refitted cannot be helped by a leaky fold. Its *interval* is not — the
cluster bootstrap resamples the groups of whatever grouping is in force, so the
same 74.0% carries [66.5, 81.3] under the boundary grouping and [67.3, 80.7]
under the row grouping. Both of the recorded runs contain **byte-identical
per-row choices** (0 of 150 disagree), so they demonstrate the server is
deterministic; they are one measurement, not two. A live cross-check against the
shared `127.0.0.1:8080` server reproduced the recorded choice on every row
probed.

### The gap, side by side

| signal | row-wise | family-held-out | gap |
|---|---:|---:|---:|
| `rules` | 78.7% | 78.7% | +0.0 pts |
| `lr` | 66.7% | 67.1% | **−0.4 pts** |
| `1nn` | 55.1% | 51.3% | +3.7 pts |
| `classical_svm` | 62.5% | 65.7% | **−3.2 pts** |
| `d1_3B` (frozen) | 74.0% | 74.0% | +0.0 pts |

With the t1/t2 near-duplicate clusters added to the grouping (`surface`, 102
groups), nothing changes materially: `lr` 66.7% (gap 0.0), `1nn` 50.1% (+4.9),
`classical_svm` 63.6% (−1.1).

**Two gaps are negative and one is a third of a point.** That is the tell: on
this set the split method is not identifiable. `lr`'s seed spread alone is 8.7
points (63.3–72.0), several times any row-versus-group difference measured here.

### The 382-row set, 18-way (where families are real)

| signal | split | n | groups | atomic | acc | 95% CI | seeds (min–max) |
|---|---|---:|---:|---:|---:|---|---|
| `classical_svm` | group (template) | 382 | 110 | **7%** | **68.6%** | 57.3–79.1 | 66.5–71.2 |
| `classical_svm` | row | 382 | 382 | 100% | **92.1%** | 89.4–94.6 | 90.8–92.9 |
| `classical_refs` (93.19% protocol) | group (template) | 382 | 110 | 7% | **93.2%** | 87.8–97.9 | — |
| `classical_refs` (93.19% protocol) | row | 382 | 382 | 100% | **93.2%** | 90.6–95.6 | — |

| signal | row-wise | template-held-out | gap |
|---|---:|---:|---:|
| `classical_svm` | 92.1% | 68.6% | **+23.5 pts** |
| `classical_refs` (fit on 200 registry references) | 93.2% | 93.2% | +0.0 pts |

On this set the same architecture loses 23.5 points when phrasing families are
held out, the atomic-row fraction is 7% instead of 63%, and the identity
disappears. That is the difference between a set with families and one without.

### What the honest baseline is

For this repo's own 8-way task the highest **deployable, uncontaminated** number
in the table is **`d1_3B` at 74.0% [66.5, 81.3], n=150** — frozen, identical
across both recorded runs, and its point estimate is unaffected by the split. Behind it: `lr` at ~67%, `classical_svm` at
~62–66%, `1nn` at ~51–55%.

`rules` at 78.7% is *higher* than the 3B and **must not be used as a baseline**:
`local_clf.RULES` was written after reading these 150 utterances, which
`docs/local-classifier-cascade-2026-11-03.md` §7.3 already states. It is also
frozen, so both of its rows are the same number by construction — the split
cannot flatter a table that never learned anything, and contamination is the only
way a hand-written keyword table reaches 78.7% here.

## 5. Three corrections to the numbers as they are currently quoted

**(a) The published intervals assume independent rows.** Every 95% CI in
`docs/` is a Wilson interval, which models rows as independent draws. Under a
family split they are not, and the cluster bootstrap is wider — by 2.35× on the
382 template split and 1.96× on the reference-transfer number:

| signal | set / split | acc | Wilson (published form) | cluster bootstrap | width ratio |
|---|---|---:|---|---|---:|
| `classical_svm` | 382 template-held-out | 68.6% | [63.8, 73.0] | [57.3, 79.1] | **2.35×** |
| `classical_refs` | 382 reference transfer | 93.2% | [90.2, 95.3] | [87.8, 97.9] | **1.96×** |
| `lr` | 150 family-held-out | 67.1% | [59.5, 74.3] | [59.6, 73.9] | 0.96× |
| `1nn` | 150 family-held-out | 51.3% | [43.4, 59.2] | [44.4, 57.7] | 0.84× |

The 150-row ratios sit at ~1.0× for the same reason the gaps do: there is no
dependence structure to widen an interval around. The 382-row ratios near 2×
are the honest width of those claims. Notably 93.19% becomes **93.2%
[87.8, 97.9]** — the `__none__` block is a single 25-row group, so resampling it
moves the number a lot.

**(b) `nested_selection_acc = 84.7%` is an oracle, applied on the true label.**
`bench/local_clf.py:1085` applies the per-class map with `choice_map[labels[i]]`
— the ground-truth class of the row being routed. At inference the class is the
unknown being predicted, so the number is not reachable by any router. Two
further facts pin it:

- the ceiling for *any* rule that picks between `lr` and `d1` row by row is the
  union, measured here at **92.8% [88.5–96.6] family-held-out** and 93.5%
  row-wise (the repo's own §5 table reports 92.7% for its nested-hyperparameter
  `lr`; 92.8% here is the same quantity at the fixed config);
- its two inputs are noise-level apart on 150 rows (`lr` vs `d1`: 33 rows where
  only `lr` is right, 55 where only `d1` is right, exact two-sided p = 0.0246
  family-held-out, p = 0.0042 row-wise), and `d1` is the better of the two under
  both splits.

**(c) The repo's own 8-point v2 "leak" is not reproducible as a split effect.**
`docs/local-classifier-cascade-2026-11-03.md` §2 reports 61.3% grouped against
69.3% random for `lr` — 8.0 points. Reproduced here exactly (the full
`bench/local_clf.py` run lands on 61.3 / 69.3 / 50.0 / 78.7 / 84.7, unchanged).
But that document's own fold table shows `lr` per-fold at 40.0, 60.0, 66.7, 66.7,
73.3 — a **33-point fold spread on 30-row folds**. An 8-point difference
measured across one fold assignment, from folds spanning 33 points, is a draw
from that spread, not a measurement of the split. Across 5 seeds through the
harness, the same comparison is **0.0 points** (66.7% vs 66.7%). The direction
the document asserts — group-aware splits cost something on v2 — is right, and
`1nn` shows it (+3.7 to +4.9 points, consistently), but the magnitude it quotes
is roughly 3× what repeated splits support.

## 6. What this settles, and what it does not

**Settled.** (1) 150 has no family field and cannot be family-split; 63.3% of its
rows are atomic, and every grouping route that looked promising failed —
frame reconstruction resolves 6%, `var/distill/labels.jsonl` does not exist, and
surface clustering finds 4 pairs in 95 rows. (2) The row-versus-group difference
on 150 is smaller than the seed-to-seed noise, so **no 150-row number in this
repo can be quoted with a split method that changes it**. (3) On 382, where 110
real families exist, the same model loses 23.5 points and its interval doubles in
width. (4) 84.7% is an oracle and 92.8% is the union ceiling above it.

**Not settled.** (1) *How* optimistic the v2 numbers are. The measured penalty
for this model class is 23.5 points on 382 and 26 points on the earlier
93.19→66.23 comparison, but neither is v2's own penalty; v2 cannot measure it.
Treat 61–67% as carrying an unquantified optimism of that order, not as a known
bound. (2) Whether v2 rows even *have* families in the wild — if the set is
genuinely 150 distinct phrasings, then a row split is correct for it and its
numbers were never inflated, only unverifiable. (3) `rules` 78.7% remains
contaminated regardless of split. (4) Nothing here is tuned: `lr` is fixed at
`min_df=1, l2=1e-3` (mid-range of `local_clf.GRID`, chosen a priori), which is
why it reads 67.1% rather than the published nested 61.3% — the same
architecture, a different hyperparameter, no selection on the test fold.

## 7. Reproduce

```
python3 bench/eval_signals.py --json /tmp/eval.json     # every table above, ~12 min
python3 bench/eval_signals.py --only v2 --seeds 1        # fast smoke
python3 -m pytest tests/test_eval_harness.py -q         # the harness contract
python3 bench/local_clf.py --json /tmp/lc.json          # the published 61.3% etc.
```

`bench/results/eval-harness-2026-11-04.json` holds all 24 result rows plus each
block's group report. Every row there carries `split`, `n`, `groups`, `seeds`
and `ci95`.
