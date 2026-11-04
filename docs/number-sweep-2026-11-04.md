# The number sweep — every instance of the four retired patterns

2026-11-04. Phase 1 retired four claims (`docs/eval-harness-2026-11-04.md`). This
document records the sweep for **every other place the same four patterns appear**
— corrected, or judged already correct, with the reason. It exists so the sweep
does not have to be done again from memory: the list is the deliverable, and a
number that is quoted on a GitHub page without its condition is how the four
claims got published in the first place.

Re-run it with:

```
grep -rn --include=*.md -F "84.7" docs README.md CHANGELOG.md TODO.md RELEASE.md bench data tests
# and the same for 93.19, 66.23, 62.7, 74.0 — plus 91.36, 92.41, 82.0, 61.3, 69.3
```

Line numbers below are as of this sweep, after the corrections were appended;
where a line sits inside a correction section it is marked *(appendix)* and needs
no verdict of its own.

## The four patterns

| id | pattern | what it looks like | why it is wrong |
|---|---|---|---|
| **P1** | an **oracle** presented as a result | "per-class routing gets 84.7% against the 3B's 74.0%" | the per-class map is applied on the row's true label (`bench/local_clf.py:1085`, `:1063`, `:1097`); no router can compute it |
| **P2** | a **different task** presented as comparable | "93.19% vs 62.7%", "74.7% until a classical model at 93.19% is on the same board" | `routing_382` is 18 smart-home classes with its own registry and an **empty** intersection with this repo's 8 ops |
| **P3** | a **split that measured nothing** | "the leak is 8.0 points", "grouped 60.0% vs random 62.0%" | `utterances_v2` is 63.3% atomic rows; holding families out moves accuracy −0.4 to +3.7 points, in both directions, inside the seed spread |
| **P4** | an **in-sample** improvement read as a measurement | "74.0% → 82.0% by rewriting two descriptions" | the two strings were written by reading the errors of the same 150 rows they are scored on |

Verdict key: **fixed** (a dated correction was appended, named) · **clean** (the
instance states its condition and needs nothing) · **coincidence** (the digits
match; the quantity does not).

## 84.7 — the oracle

| file | lines | verdict |
|---|---|---|
| `docs/local-classifier-cascade-2026-11-03.md` | 11, 224, 225, 250 | **P1, fixed** — Corrections C1. Line 11 is the verdict, 224/225 the nested and pooled-OOF rows, 250 the §6 claim that per-class routing "is the right device". The same defect at 223 (84.0% static table) and 226 (92.7% union) is named in the same correction |
| `README.md` | 438 | **P1, fixed** — the docs-table row now says the 84.7% was an oracle applied on the true label, names 92.8% as the ceiling, and states that no deployable number is quoted |
| `docs/eval-harness-2026-11-04.md` | 27, 253, 271, 290 | **clean** — the report that establishes the defect |
| `docs/vulkan-igpu-2026-11-03.md` | 204 | **coincidence** — `84.7 ms/tok`, a per-token latency, unrelated |
| appendix lines | local-classifier 351–400; README 448; CHANGELOG 161–167; TODO 134 | *(appendix)* |

## 93.19 / 66.23 — the different task

The pair always travels together; one row per file.

| file | lines | verdict |
|---|---|---|
| `docs/classical-baseline-2026-11-03.md` | 18, 124 | **P2, fixed** — Corrections C1. Line 18 is the summary table that puts a 382-row row directly above a 150-row row; line 124 nominates 66.23% as "the number to quote next to 93.19%", inviting the reading that the pair is this repo's task at two split settings |
| `docs/classical-baseline-2026-11-03.md` | 8, 63, 95, 114, 118, 226, 251, 252, 256, 310, 313 | **clean** — these are statements about the recorded protocol on its own set, and they carry their caveats (§8 "that table measures a different task", §9.1, §9.2). C1 adds the missing fact (empty intersection) without touching them |
| `docs/assets-inventory.md` | 468 | **P2, fixed** — Corrections C1. This is the `Status of this survey` row that runs 93.19 / 66.23 / 62.7 / 74.0 together in one sentence |
| `docs/assets-inventory.md` | 46, 66, 77, 81, 94, 391 | **P2, fixed for 77 and 391** — Corrections C2. 46, 66, 94 are correct quotes *of that benchmark* and are fine; 77 and 391 make the 74.7%-against-93.19% comparison, which pairs a 150-row figure with a 382-row one |
| `docs/roadmap-2026-11.md` | 141, 203 | **P2, fixed** — Correction C1. Finding 1 closes with a disjunction that omits the third possibility (the linear model is right *for that label set*), and the Outcome states the unhedged form |
| `docs/local-classifier-cascade-2026-11-03.md` | 59, 60 | **P2, fixed** — Corrections C3. The report opened by citing 93.19 → 66.23 as a split precedent for *this* task |
| `README.md` | 437 | **P2, fixed** — the docs-table row now names the 382-row set as a different task and says only the 62.7% is comparable |
| `CHANGELOG.md` | 113 | **P2, fixed** — the released bullet is left as published and the `Corrections — 2026-11-04` section retires it by name |
| `CHANGELOG.md` | 145 | **clean** — a reproduction-status row ("reproduces exactly"), which makes no comparability claim |
| `TODO.md` | 135 | **P2, fixed** — the known-limits row now reads "(n=382 — a different task, see above)", and a new row states the empty intersection |
| `docs/distill-corpus-certification-2026-11-03.md` | 81 | **clean** — this is the document that states the empty intersection and says a classifier trained here cannot be scored there |
| `bench/README.md` | 105 | **clean** — 93.19% is described as "template reuse", the correct reading, and the 382-row family split is cited as the harness's demonstration |
| `tests/data/routing_382_PROVENANCE.md` | 16, 65, 76, 96 | **clean** — the provenance record, with a "Caveat carried forward" section naming the 18-way home-assistant task and saying numbers do not transfer |
| `docs/eval-harness-2026-11-04.md` | 207, 208, 249, 294 | **clean** — the report |
| appendix lines | classical-baseline 406–433; assets-inventory 493–538; local-classifier 472–488; README 455; CHANGELOG 180–189 | *(appendix)* |

## 62.7 — the figure that *is* comparable

Every instance was checked because it sits beside the two that are not. All are
clean: 62.7% is measured on `utterances_v2`, this repo's own 8 ops, and each
source names the set.

| file | lines | verdict |
|---|---|---|
| `docs/classical-baseline-2026-11-03.md` | 10, 23, 157, 317, 352 | **clean** — the v2 transfer figure with its n |
| `docs/local-classifier-cascade-2026-11-03.md` | 163, 238 | **clean** — reconciled as the same-task transfer figure |
| `docs/roadmap-2026-11.md` | 205 | **clean** — "on this repo's own set"; the juxtaposition with the two 382-row figures is what Correction C1 addresses |
| `docs/distill-corpus-certification-2026-11-03.md` | 200 | **clean** |
| `docs/doc-review-2026-11-03.md` | 162 | **clean** — dated audit, claim holds, not rewritten |
| `README.md` · `CHANGELOG.md` · `TODO.md` · `data/distill/README.md` | 437 · 113 · 133 · 141 | **clean** |
| appendix lines | classical-baseline 406–433, 467; assets-inventory 495, 497; local-classifier 473, 474; roadmap 258; README 451 | *(appendix)* |

## 74.0 — the 3B on this repo's task

The most-quoted number in the repo and, by inspection, the cleanest: it is
measured on the 150 rows this repo evaluates on, and `docs/eval-harness-2026-11-04.md`
§4 re-measures it as frozen and split-invariant at 74.0% [66.5, 81.3].

| file | lines | verdict |
|---|---|---|
| `bench/results/p1-diagnose/FINDINGS.md` | 21, 33, 160 | **P4, fixed** — a 2026-11-04 correction is appended to the file: the 74.0% → 82.0% gain is in-sample, the Verdict must carry the caveat, and the §3 band breakdown is not a robustness check. 33 and 160 are the baseline column and are correct; 21 is the headline |
| `docs/assets-inventory.md` | 468 | **P2, fixed** — via the same sentence as 93.19 / 66.23 above; 74.0% itself is the right set, the problem was the company it kept |
| `docs/classical-baseline-2026-11-03.md` | 11, 13, 23, 157, 196, 198, 320 | **clean** — same-task comparisons against d1's recorded v2 answer |
| `docs/local-classifier-cascade-2026-11-03.md` | 12, 49, 106, 141, 151, 197, 222, 249, 251 | **clean** — the comparator, the per-label column, the gate and fallback arithmetic |
| `docs/eval-harness-2026-11-04.md` | 27, 171, 176, 191, 222 | **clean** — the report, including the split-invariance argument |
| `docs/model-sweep-consolidated.md` | 49 | **P3, noted** — the table's figures are row-wise on a set with 63.3% atomic rows; the 2026-11-04 correction records that this is a ranking on these rows, not a deployment estimate |
| `docs/roadmap-2026-11.md` | 24, 152, 166, 205 | **clean** |
| `docs/doc-review-2026-11-03.md` | 67, 162 | **clean** — 67 is the audit that *already* caught the cross-set 91.4% statement ("a sibling-project figure on a different, easier 382-row set"), which is the same defect found here for 93.19%. A dated audit; not rewritten |
| `README.md` | 375, 437, 438 | **clean** — the Measured table names the 150-row set; 437 and 438 carry the corrections |
| `RELEASE.md` | 219 | **clean** — the announcement copy names the 150-row set |
| `CHANGELOG.md` | 108, 114, 115, 144 | **clean** |
| `TODO.md` | 35 | **clean** — 79.3% against 74.0%, same task, with McNemar p stated |
| `data/distill/README.md` | 141 | **clean** — and line 150 already says `routing_382` is out of scope |
| appendix lines | classical-baseline 407–433; assets-inventory 497; local-classifier 374–428, 502, 507; model-sweep 255–263; roadmap 258; README 437, 438, 448, 451, 462; CHANGELOG 162, 172, 207; FINDINGS 258, 292 | *(appendix)* |

## Pattern instances that do not use the five numbers

Found by reading the affected documents rather than by grepping, and worth
listing because a future sweep on the digits alone would miss them:

| instance | pattern | action |
|---|---|---|
| `docs/assets-inventory.md:77, 390, 424` — "a 3B at 74.7% … until a classical model at 93.19% is on the same board" / "17 points above the 74.7%" | **P2** | **fixed** (Corrections C2). 74.7% is d1's `logit_read` lane on the **150-row** set; on the 382-row board the same model is 91.36%, so the real same-board difference is **1.83 points**, and the "17 points" spans two datasets and two adapters |
| `docs/local-classifier-cascade-2026-11-03.md:223` (84.0% static table), `:226` (92.7% union) | **P1** | **fixed** in Correction C1 — all three per-class rows are oracle-applied (`bench/local_clf.py:1063,1085,1097`) |
| `docs/local-classifier-cascade-2026-11-03.md:89–93, 241–243` — "the 8.0-point difference is the leak" | **P3** | **fixed** (Correction C2) |
| `docs/classical-baseline-2026-11-03.md` §5 — "62.0% random, 60.0% grouped" read as the set's leak | **P3** | **fixed** (Correction C2) |
| `README.md:438` (pre-correction) — "an 8.0-point leak" | **P3** | **fixed** — the phrase is gone from the row and the correction block states the recomputation |
| `docs/roadmap-2026-11.md` §Outcome — "the premise 'a neural decision model is required' is overstated" | **P2** | **fixed** (Correction C1) |
| `docs/classical-baseline-2026-11-03.md` §8 — "the premise … is overstated for a well-specified closed label set with in-vocabulary phrasing" | **P2** | **clean** — the qualifier is load-bearing and C1 says so explicitly |
| `bench/results/p1-diagnose/FINDINGS.md` §3 — the band breakdown as evidence the rewrite is not boundary-specific | **P3/P4** | **fixed** in the appended correction |
| `docs/cost-reduction-glm-flash.md:34, 37` — `92.7%` and `84.0%` | **coincidence** | no action — those are hosted-endpoint accuracies (139/150, 126/150) on the 150 rows, not the classifier cascade |
| `docs/vulkan-igpu-2026-11-03.md:165, 172, 185, 191` — `84.0%` | **coincidence** | no action — routing accuracy of CPU lanes at n=50, correctly caveated as "two rows at n=50" |
| `docs/dataset-status.md:166` — "residual leakage it does not prevent: paraphrase families" | neither | no action — a different and correct claim about `heldout_phase3`, not about v2's split |

## What does not fit the four

Recorded because a sweep should say what it could not classify:

1. **`docs/doc-review-2026-11-03.md:67` already had P2 right.** It flags
   `spec/1.0-spec.md:10`'s "91.4% tool routing" as a sibling-project figure on a
   different, easier 382-row set. The defect was known in one place on 2026-11-03
   and was not carried into the documents that quote the same set — which is why
   this sweep exists as a file rather than as a paragraph.
2. **`docs/classical-baseline-2026-11-03.md` §7's cascade threshold** (79.3% at
   gate ≥ 0.20) is selected on the set it is scored on. It is caveated in place
   ("do not read +5.3 points as established on 150 rows", "the threshold was
   chosen by looking at this set") and is not one of the four patterns, but it is
   the same family of error and should not be quoted without that sentence.
3. **`docs/local-classifier-cascade-2026-11-03.md` §4's 95.3% hosted reference**
   is an asserted constant with no probe in this tree — already named as such in
   that document's own §4 and §7.2. Not swept further here; it is a provenance
   problem, not a retired claim.
4. **The `rules` 78.7% row** is contaminated in both the harness and the report,
   consistently. Not a pattern above, but it appears in `docs/local-classifier-cascade-2026-11-03.md`
   line 104 and `docs/eval-harness-2026-11-04.md` line 165 and neither should be
   quoted as a baseline.
