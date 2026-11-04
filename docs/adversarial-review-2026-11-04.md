# Adversarial review — 2026-11-04

Reviewer: hostile reviewer, not the repair crew. Branch `v-adversarial` at the HEAD of
`main` (`969218d`). Read-only review of the artefacts, not of the prose: every number
below was recomputed from `bench/results/**`, `bench/data/heldout_v3*.json`,
`tests/data/utterances_v2.json` and the scripts under `bench/`. No model was called, no
server was touched, no network was used. Nothing outside this file was edited.

Baseline, executed:

```
$ python3 -m pytest tests/ -q
826 passed, 551 subtests passed in 169.02s (0:02:49)
```

The brief asks for the strongest case that a headline claim is false, overstated, or
unsupported, plus which one I would bet against. The short version: **two of the five
claims fail, two survive with a defect of framing rather than of arithmetic, and one
holds outright.** The failure mode is not miscalculation — I could reproduce every
published number to the digit. It is that the project computes its statistics
consistently where the correction hurts a claim and inconsistently where it would help
one.

---

## Verdict, ranked by how likely the claim is wrong

| # | claim | my estimate it is wrong as stated | what decides it |
|---|---|---:|---|
| 1 | **C, exception:** the `filesys_grep` line "transfers" (+17.3, 9/0, p = 0.0039) | **75%** | 9 fixed rows are 3 authored frames; on v3's own holdout side it is 3 rows from **1** frame; family-aware p = 0.25 |
| 2 | **C, headline:** "the rewrite fails out of sample" (80.5 → 77.3) | **70%** | p = 0.0725, CI spans zero, design MDE 7.4 pts vs a 3.1-pt effect, and the same-session control reverses the sign |
| 3 | **B:** "79.3% deployable, not separable from 74.0%" | **50%** | "not separable" is true and vacuous (MDE 9.8 vs claimed 5.3); "79.3% deployable" depends on an estimator choice worth 4.4 pts |
| 4 | **D:** ρ = −0.71 "ranks candidates backwards" | **35%** | reproduces exactly (−0.714, permutation p = 0.044) but is one binary contrast wearing seven ranks |
| 5 | **E:** `heldout_v3` is a valid instrument at 80.5% | **25%** | calibration miss is disclosed, located and not patched; holds for the aggregate, fails for per-class use |
| 6 | **A:** `d1_3B` at 74.0% [66.5, 81.3] is the honest baseline | **10%** | Q8_0 scores 73.3% on the same rows; the interval is a real cluster bootstrap (verified), not Wilson |

**I would bet against #1.** Reasoning in §1, and the wager is stated in §8.

---

## 1. The `filesys_grep` line does not have the evidence claimed for it

This is the claim that has been promoted furthest — out of a probe, into `README.md`, and
into an explicit shipping recommendation ("If anything from `p1-diagnose` ships, it is
that one line", `docs/rewrite-transfer-2026-11-04.md` §9). It is also the claim whose
support is thinnest per unit of promotion.

### 1.1 The arithmetic reproduces, and I accept it

Recomputed from `bench/results/p3-transfer/rows.jsonl` against the archived baseline
`bench/results/p2-heldout/v3-d1-3B-Q4.json`, all 52 `filesys_grep` rows:

```
base 36/52 (69.2%)  ->  targeted 45/52 (86.5%)   delta +17.3 pts
fixed 9, broke 0, exact McNemar p = 0.0039
```

Every digit is right. The attack is not on the arithmetic, it is on what the test assumes.

### 1.2 The nine fixes are three sentences

`bench/data/heldout_v3.json` carries a `family` field on every row, and the family *is*
the authored frame — all slot-filled variants of one sentence. The nine fixes:

| family | rows fixed | rows in family | v3 side |
|---|---:|---:|---|
| `t1:filesys_grep:14` | 4 | 5 | **train** |
| `t3:filesys_grep->web_fetch:03` | 3 | 5 | **test** |
| `t3:filesys_grep->filesys_glob:02` | 2 | 6 | **train** |

The exact McNemar p = 0.0039 treats those as 9 independent Bernoulli trials. The dataset's
own design says they are not: all 52 grep rows sit in 10 families (`heldout_v3_NOTES.md`
reports this as a virtue — "`filesys_grep`… 10 families, 52/52 rows in multi-row
families"). Apply the correction the project applies everywhere else — resample whole
families, not rows — and the same data give:

* family-level exact sign test: **3 families improved, 0 worsened, 7 unchanged → p = 0.25**
* `rewrite_transfer.py`'s **own** `cluster_bootstrap()`, called on this slice at its own
  defaults (4 000 reps, seed 20261104): **+17.3 pts [+0.0, +36.5] over 10 units** — the
  lower bound is exactly 0.0. The function the document uses to build the interval on its
  headline cannot exclude no effect on the line it promotes.

So the p-value moves from 0.0039 to 0.25 — a factor of 64 — purely by choosing the
sampling unit the document chose for its own headline delta.

### 1.3 The holdout half of the evidence is three rows from one sentence

v3 has a declared split, made before any utterance was rendered. Fixes by side:

| side | n | fixed | broke | families containing a fix | exact p |
|---|---:|---:|---:|---|---:|
| v3 **test** (held out by construction) | 26 | **3** | 0 | `t3:filesys_grep->web_fetch:03` (**1**) | 0.25 |
| v3 **train** | 26 | 6 | 0 | 2 families | 0.031 |

Two thirds of the "transfer" evidence is in the half of v3 that the instrument designates
as training material. On the side that is a genuine holdout, the entire case for the line
is **three rows rendered from one authored sentence**:

| frame | the utterance | fixed |
|---|---|---:|
| `t3:filesys_grep->web_fetch:03` | `search the tree for {sym} offline` | 3 of 5 |
| `t1:filesys_grep:14` (train) | `is {sym} referenced anywhere else?` | 4 of 5 |
| `t3:filesys_grep->filesys_glob:02` (train) | `list the files that reference {sym}` | 2 of 6 |

That matters for the ship decision: the claim is not "grep improved on 52 rows", it is
"this line transfers to phrasing it has not seen". The holdout side is where that claim
lives, and there the count is 3 rows / 1 frame / p = 0.25. And that one frame is
`search the tree for {sym} offline`, whose two operative words — *search*, and *offline*
against the new text's *local file contents* — nearly restate the replacement description.
The single sentence carrying the holdout half of the evidence is the sentence in the set
with the highest lexical overlap with the string being tested; a lexical-overlap
explanation and an operation-naming explanation predict the same three rows, and nothing
here separates them.

### 1.4 A quarter of the gain is in the band the instrument calls a floor

Four of the nine fixes are `t1:filesys_grep:14`, a 5-row frame in the t1 band.
`heldout_v3_NOTES.md` describes that band in its own words: "`bands.py` defines t1 as
*direct intent, signature vocabulary allowed*… A floor is a floor; it is not the same
band." The rewrite's cleanest evidence is concentrated in the one band the instrument
author says should not be read as difficulty.

### 1.5 The multiple-comparison attack the brief proposed mostly fails — and I will say so

I was asked to compute Bonferroni across the eight classes. It does not kill the claim,
and reporting otherwise would be dishonest:

| correction | adjusted p | verdict |
|---|---:|---|
| none (as published) | 0.0039 | — |
| Bonferroni × 8 classes | **0.0312** | survives α = 0.05 |
| Holm, rank 2 of 8 (after `reddit_digest`) | **0.0273** | survives |
| family = the 2 lines actually changed | 0.0078 | survives |
| family = 10 lines across both variants attempted | 0.0391 | survives, barely |
| family = 2 variants × 8 classes | **0.0625** | fails |

The claim dies only if the family includes the aborted global rewrite (`OPERATION_FIRST`,
70.7% on v2 — `bench/results/p1-diagnose/rewrite.json`), which is a defensible family but
not the one the brief named. **The multiplicity attack is the weaker one. The clustering
attack in §1.2 is the one that lands**, and it lands harder: it is not a penalty for
looking at eight classes, it is the same correction the project accepts when it reports
−8.5 to +1.9 for the overall delta.

### 1.6 The internal inconsistency, stated plainly

`bench/rewrite_transfer.py` computes the overall delta with `cluster_bootstrap()` over 133
families — and its docstring says exactly why: "356 of 415 sit in a multi-row family, so a
row bootstrap would understate the interval". Twelve lines earlier, `by_class()` reports a
per-class `mcnemar_exact(f, br)`, which is a row-level test. The project clusters where
clustering *hurts* the headline (−3.1 becomes [−8.5, +1.9]) and does not cluster where it
*would have hurt the exception* (0.0039 would have become 0.25). That asymmetry, not any
single p-value, is the defect.

---

## 2. "The rewrite fails out of sample" is not what the data say

I accept the half of this that is well supported and attack the half that is not.

**Holds:** the in-sample 82.0% must be retired. It does not reproduce anywhere on v3 —
not on all 415 rows (77.3%), not on the test side (79.7%), not on the train side (75.0%),
and the grep+extract pair goes 54.4% → 68.0% rather than +8 points overall. Anyone
quoting 82.0% as an improvement should quote this instead. That is settled and I would not
bet against it.

**Does not hold:** "the whole rewrite fails out of sample" (`docs/rewrite-transfer…` §
verdict), "that hypothesis is now falsified" (§9). A 3.1-point point estimate whose
interval is [−8.5, +1.9] and whose p is 0.0725 is not a verdict, it is a null result with
a direction. Three computations, each made from committed artefacts:

### 2.1 The design cannot resolve the effect it reports

From the reported CI on the paired delta, SE = (1.9 + 8.5) / (2 × 1.96) = 2.65 points.

| | value |
|---|---:|
| observed effect | **3.1 pts** |
| MDE at 80% power (paired, α = 0.05) | **7.4 pts** |
| MDE at 90% power | 8.6 pts |
| rows needed to detect 3.1 pts at 80% power | ~2 400 (**5.7× this set**) |
| rows needed to detect 5.0 pts | ~920 (2.2×) |

The instrument's own note reaches the same conclusion independently: `heldout_v3_NOTES.md`
says "the overall figure resolves about 8 points per side — that is what 415 rows buys".
The document that owns the instrument says it resolves ~8 points; the document that uses
it reports a 3.1-point difference as a verdict. One of the two is wrong, and it is not the
instrument note.

### 2.2 The comparison arm was not measured under the same conditions

`bench/results/p3-transfer/rows.jsonl` contains **648 rows, of which 233 are a live
re-run of the baseline arm** (rows are keyed `(arm, id)`). The three recordings:

| arm | window | n | latency p50 |
|---|---|---:|---:|
| baseline (archived, `p2-heldout`) | ends 2026-11-04T01:50:47 | 415 | **8 225 ms** |
| targeted (live) | 02:06:14 → 02:36:06 | 415 | 4 079 ms |
| baseline (live re-run) | 02:36:09 → 02:51:07 | 233 | 3 374 ms |

The server got **twice as fast between the two arms**. `docs/rewrite-transfer…` §2
notices ("the server was at 3–5 s/call rather than the expected 14–18 s") and does not
treat it as a confound. It should: latency halving on a llama-server means a restart, a
different thread count, or a different build, and any of those can move argmax.

### 2.3 The control exists, was run, is never reported, and contradicts the verdict

`docs/rewrite-transfer…` §3 promises it: "The v3 baseline arm was re-run live alongside
this experiment as a whole-set check (**below**)". There is no such section below — the
document's ten sections contain no table of the re-run. And the code discards it:
`report()` loads the rows and immediately skips every non-`targeted` record
(`bench/rewrite_transfer.py:374`). §10 of the same document invites a reader to run
`--arm baseline`. The output of that run is committed and unread.

Reading it (233 rows present in both arms):

```
archive 184  ->  live 180  ->  targeted 183
```

* **archive → live, same prompt, 45 minutes apart: −4 rows.** The archive is not
  reproducible against the endpoint it is paired with. Six rows change argmax; four of
  them move a row from *correct* to *wrong* and **not one moves a row the other way** (the
  other two flip wrong→wrong). Four of the six flips are `reddit_digest → web_search`,
  which is exactly the class the rewrite is accused of breaking.
* **live → targeted, same session: +3 rows (+1.2 points).** On the only rows where both
  arms were measured under the same server, **the rewrite gains**.

Run the project's own interval on that overlap — `cluster_bootstrap()` at default
settings, 80 families — and the archive-paired delta there is **−0.4 pts [−6.5, +5.5]**,
i.e. nothing. The −3.1-point headline is the *union* of that nothing and twelve unverified
rows.

Decomposing the headline −13 net rows:

| slice | rows | archive | targeted | net |
|---|---:|---:|---:|---:|
| re-measured (both arms live) | 233 | 184 (live 180) | 183 | **+3 vs live**, −1 vs archive |
| never re-measured | 182 | 150 | 138 | **−12** |

Twelve of the thirteen net rows of loss live in the 182 rows whose baseline is an archive
that was never re-checked — 157 of them the v3 *train* side of the six untouched classes.

### 2.4 Sensitivity: every correction moves toward zero

| baseline used | baseline correct | targeted | net | delta |
|---|---:|---:|---:|---:|
| published (archive, 415) | 334 | 321 | −13 | **−3.1 pts** |
| archive − measured drift only | 330 | 321 | −9 | −2.2 pts |
| archive − drift imputed to all 415 | 327 | 321 | −6 | **−1.4 pts** |

None of this shows the rewrite is good. It shows the study cannot distinguish "the rewrite
costs 3 points" from "the rewrite does nothing", and that the only same-session evidence
in the tree points the other way. **"Fails" is a verdict the design cannot issue.** The
honest sentence is: *on 415 rows the rewrite is 3.1 points down with an interval spanning
zero, and on the 233 rows measured against a same-session baseline it is 1.2 points up.*

Two caveats I owe the reader: 4 one-directional drifts out of 233 is p = 0.125 on its own,
so "the server degraded" is not established either; and the 182 unmeasured rows may
contain a real effect. That is the point — nobody knows, and the document says it does.

---

## 3. Claim B: the null is real, the "deployable" number is not settled

### 3.1 "Not separable" is true — and it is a statement about the design, not the router

`bench/results/router-policy-2026-11-04.json → pairs["router[per-class, A] vs 3b"]`:
Δ = +5.3, CI [−1.4, +12.3], `a_only` 27 / `b_only` 21, p = 0.4709. I accept the number.

Power, from the CI: SE = 3.51 points.

| | value |
|---|---:|
| claimed gain | 5.3 pts |
| power to detect a true 5.3 pts | **0.33** |
| MDE at 80% power | **9.8 pts** |
| MDE at 90% power | 11.4 pts |
| rows for 5.3 pts at 80% power | ~516 (**3.4× the 150**) |
| rows for 3.0 pts at 80% power | ~1 610 (10.7×) |

The sign-test view agrees from the other side: 48 discordant pairs, so p < 0.05 requires a
**16-row net = 10.7 points**, and 80% power requires ~19–20 rows = 13 points. A true
5-point gain would be *invisible* here by construction.

The claim as written ("not separable from 74.0% at n=150") is therefore correct and
uninformative. It should be quoted as **"this design detects only gains above ~10 points,
so it cannot speak to a 5-point one"** — not as a null, and not as "the router cannot claim
them" (commit `0272d36`), which reads like evidence against the gain. The document
`docs/router-policy-2026-11-04.md` is more careful than its own commit message; the commit
is the overreach.

### 3.2 The 79.3% depends on an estimator choice worth as much as the effect

`_class_map` (`bench/router_policy.py`) files each class under `key="true"` — and its own
docstring says of the alternative: `key="pred"` "is the population the choice is actually
applied to".

* `router[per-class, A]`: map fitted on **true** labels, dispatched on the **predicted**
  class → **79.3%**
* `router[per-class, B]`: both on the predicted class → **74.9%**

Both are computable at train time without test labels, so neither is leakage. The spread
is **4.4 points — larger than the 3.1-point effect claim C is built on, and comparable to
the 5.3 the router is trying to claim.** The README states "its deployable replacement is
79.3%"; the code's own vocabulary says the population the choice is applied to is B's.
Pick the other defensible estimator and the router is *below* the baseline it was supposed
to beat (74.9% vs 74.0%).

Worse, the two significance instruments disagree on this very pair:

```
pairs["router[per-class, B] vs router[per-class, A]"]:
  delta -4.4, delta_ci95 [-8.2, -0.9]   <- excludes 0
  a_only 18, b_only 30, exact p 0.1114  <- does not exclude 0
```

A CI that excludes zero beside a p-value of 0.11 means the instrument stack is not
internally consistent at n = 150. Since `table_pairs` says the interval "is the
instrument", the project should say which estimator it considers deployable and that the
answer moves the headline by 4.4 points.

### 3.3 What survives in B

The oracle retraction itself (`oracle[per-class]` keys the dispatch on `lab_te[r]`, the
*true* label — verified in `_fold_pass`) is correct and well founded. But its supporting
test is also uncorrected: Δ 5.2 pts, CI [0.1, 10.6], p = 0.0192, against a table whose own
footnote says "the exact p beside it is unadjusted for the eleven comparisons above". Holm
over those eleven gives 0.19. So the project retires a 84.7% headline on a p of 0.019 that
would not survive the correction it applies to nothing else. The retraction is right for a
structural reason (the dispatch uses a label nobody has at test time), which is stronger
evidence than the test — cite the structure, not the p.

---

## 4. Claim D: ρ reproduces; "ranks backwards" is a 7-costume version of one bit

I recomputed the rank correlation from the seven candidate pairs in
`docs/distill-classifier-2026-11-04.md` §6:

* Spearman **ρ = −0.714** (published −0.71) — exact.
* Exact permutation over all 7! = 5 040 orderings: **p = 0.044**. It survives, barely;
  the critical value at n = 7 is |ρ| > 0.750 for the 2.5% tail, so the published −0.71 is
  inside the rejection region by 0.04.
* Leave-one-out: ρ ∈ [−0.54, −0.89]. Not driven by a single candidate. Kendall τ = −0.52.

So the multiple-comparison-shaped attack on D fails on its own terms, and I will not
pretend otherwise. Three real objections remain:

1. **The seven candidates are not exchangeable.** They are 4 views of one ensemble plus 3
   views of one ridge. The permutation null assumes 7 independent orderings; the effective
   count is closer to 2 architectures × rank-order noise. Within the ridge group the CV
   order is C6 > C5 > C7 and the transfer order is C6 > C7 > C5; within the ensemble group
   CV C1 > C4 > C2 > C3 against transfer C3 > C2 > C1 > C4. The whole ρ is carried by one
   between-architecture contrast — "CV likes ensembles, transfer likes ridges" — dressed
   in seven ranks. One bit of information, and the doc's own §6 says so in different
   words ("the corpus's criterion prefers the four models that memorise").
2. **The decision-relevant gap is not tested.** "CV would have picked C1, costing 8.7
   points" compares C1 60.0% with C6 68.7% on 150 rows. The `paired` block tests C6~d1 and
   C6~rules, not C1~C6. Eight point seven points is 13 rows, and a sign test reaches
   p < 0.05 on a 13-row net only if the pair disagrees on **fewer than ~44** of 150 rows
   (`13/√D > 1.96` ⇒ `D < 44`). For scale, C1 against the 3B disagrees on 77 rows and C6
   against the 3B on 64; two members of the same family are unlikely to disagree on fewer
   than 44. So the 8.7-point selection penalty is probably not significant either, and it
   is the number the selection-defect story rests on.
3. **The negative result is a p = 0.38 read as a decision.** 68.7% against 74.0%,
   28 fixed / 36 broken, p = 0.38 — the doc calls this "no evidence that the corpus-trained
   classifier beats the 3B", which is correct, then §7 heads a section "Do not ship it",
   which is a decision. Claim B's null and claim D's null are the same p and are read in
   opposite directions. The classifier's real support is the p = 0.040 loss to a 35-cue
   keyword table, and that is `rules` versus a fitted model on 150 rows — the same
   resolution problem in a third costume.

D's own doc is the most self-aware artefact in the tree (it flags the post-hoc rule and
the spent benchmark unprompted). I rank it 4th for that reason, not because the number is
good.

---

## 5. Claim E: the instrument holds; the per-class use of it does not

The attack offered was "6.5 points easier, built by an agent that could see the answers".
I checked what could be checked and most of it comes back clean:

* **The gap is located, not unexplained.** t1 is 11 points easier, t2 21 points harder, t3
  15 points easier; decomposed at v3's class weights, +5.9 (two weak classes easier), −3.1
  (six others harder), +3.7 (composition) — which sums to the 6.5. That is a real
  accounting, not a story.
* **The {mod} caveat is answered by computation, and I verified the answer.**
  `heldout_v3_NOTES.md` says the `{mod}` frames were excluded by the hash prefix and
  `docs/rewrite-transfer…` §8 follows it up by measuring the rewrite on v2's package-noun
  rows: **0 fixed, 0 broke** — the rewrite provably does not repair them, so the omission
  does not inflate the grep result. My recomputation agrees (I find 5 package-noun rows in
  v2's grep, not 4 — `yaml.safe_load` is counted by one reader and not the other; `wallet.json`
  is not a package. The substance is unaffected: 0/5 fixed, 0/5 broken).
* **The one thing that does not survive:** v3's grep is **69.2%** where v2's is **36.4%**
  — a 33-point difference on a 22-row base. Any per-class conclusion about grep drawn on
  v3 is drawn at a difficulty the target population does not have, and the two classes
  where v3 is a comparable instrument (`filesys_read`, `web_search`, `decide_pick`) are
  not the two the project needs it for. The instrument is valid for the aggregate and
  weakest exactly where claim C's exception lives.

So E survives, with the caveat that "valid instrument" cannot be extended to per-class
transfer claims without also striking the grep claim — another instance of the same
asymmetry, this time inside one document pair.

---

## 6. Claim A holds, and the obvious attack is refuted by the repo's own files

I expected to find 74.0% was an artefact of `Q4_K_M` on one box. `bench/results/dell/` has
the counter-experiment already recorded:

| model | file | n | accuracy |
|---|---|---:|---:|
| `d1-3B-Q4_K_M` | `v2-d1-3B-Q4-run2.json` | 150 | 74.00% |
| `d1-3B-Q8_0` | `v2-d1-3B-Q8-run2.json` | 150 | **73.33%** |

Q8 is *one row worse*. Quantisation is worth 1 row out of 150 on this set, so the claim
that 74.0% is a property of Q4 rather than of the model is false, and the repository
already contains the disproof. Two further checks:

* The interval `[66.5, 81.3]` is a genuine cluster bootstrap over 106 groups, not a Wilson
  interval mislabelled. I re-ran `eval_harness.cluster_bootstrap_ci` on the archived
  per-row outcomes: `(0.6645, 0.8134)` — exact match, and Wilson would have given
  `(0.6644, 0.8036)`. The lower bounds coincide to four digits, which is a coincidence and
  not a bug.
* The model is asserted against `/props` on every run (`model_assertion`, `checked: true`).

What A does *not* mean is already disclosed in `router-policy-2026-11-04.md`: 63% atomic
rows, so "honest" here means "not an oracle", not "unbiased for generalisation". The claim
is stated at that strength. It holds.

---

## 7. Other instances of the same error shape

The orchestrator flagged one tonight: "7 fixed, 0 broken" quoted from a 87-row stratified
sample where only 39 rows came from the classes that broke. Same shape, found by reading
the artefacts:

1. **`bench/rewrite_transfer.py:374`** — the baseline control is executed, committed, and
   discarded by the report. Coverage is printed for the targeted arm only
   (`coverage: {ran: 415, planned: 415, skipped: 0}`), which is true of one arm of a
   two-arm comparison.
2. **"Whole-set check (below)"** in `docs/rewrite-transfer…` §3 points at a section that
   does not exist. The check was run; the pointer is a dead reference and the result is
   never shown.
3. **Commit `55449e2`: "27 broken against 16 fixed."** 27 is `v3_others.broke` (the six
   untouched classes; `summary.json`), 16 is `v3_all.fixed` (all rows). The all-rows pair
   is 16 / 29. Numerator and denominator from different slices — the same mixing the
   orchestrator caught, in a commit subject.
4. **Determinism.** `heldout_v3_NOTES.md` leans on "0/150 argmax disagreement across two
   llama.cpp builds" to justify pairing a new arm against an archive. The committed live
   re-run shows **6 disagreements in 233 rows (2.6%)** on the identical prompt. The
   determinism claim is true of that one historical comparison and is not true of this
   one; the archive pairing inherits the difference.
5. **`pairs` footnote.** The table states "the exact p beside it is unadjusted for the
   eleven comparisons above", then rows are quoted one at a time in prose and commits
   (`p = 0.0192` for the oracle in `docs/router-policy-2026-11-04.md:143`, and in the
   merge commit "the oracle hid 5.2 points") without the adjustment the footnote just
   promised. Disclosure is not correction.

---

## 8. What I would stake

**I bet against claim C's exception: that the `filesys_grep` description line "transfers".**

Confidence 75%. The falsifiable version of my position:

> If that one line, with `decide_extract` restored to its original text, is run on a
> freshly authored ~400-row grep set, its point gain will be **below half of +17.3**, or
> not significant under a family-aware test, or both.

Why that and not the others:

* Every number in the claim is arithmetically right, so I cannot win by finding an error.
  I win if the effect is smaller than advertised, and the structure of the evidence says it
  should be: 9 rows from 3 sentences, 3 of them from 1 sentence on the holdout side, 4 of
  them in the band the instrument calls a floor, and a p-value computed on an independence
  assumption the same file refuses to make twelve lines earlier.
* It is the claim with the most downstream consequence. The others change what you believe;
  this one changes what ships. The document already recommends shipping it.
* Its support does not improve under any correction, only under *no* correction.

I am least able to attack **A** (10%) and **E** (25%), and I said so above rather than
manufacture objections to them: the Q8 run refutes the quantisation story, the {mod}
caveat is answered with data, and 74.0% is the one number in the tree that is stated at
exactly the strength its evidence supports.

If I am wrong about #1, the most likely reason is that the grep frame families are more
independent than I am treating them as — a reader who believes slot-filled variants of one
authored sentence are 3–6 independent observations rather than 1 will get p = 0.0039 and a
shippable line. That is the disagreement to have, and it is worth having explicitly rather
than in a footnote.

---

## Reproduce

Offline, no model call, no network. From the repo root:

```bash
python3 bench/rewrite_transfer.py --report      # regenerates the published v3 tables
python3 bench/report_v3.py                      # regenerates the instrument validation
python3 -m pytest tests/ -q                     # 826 passed
```

The numbers not produced by those scripts come from the block below, which reads only
`bench/results/p3-transfer/rows.jsonl`, `bench/results/p2-heldout/v3-d1-3B-Q4.json`,
`bench/data/heldout_v3.json` and `bench/results/router-policy-2026-11-04.json`.

```python
import json, collections
from math import comb

v3l = json.load(open("bench/data/heldout_v3.json")); v3 = {r["id"]: r for r in v3l}
arch = json.load(open("bench/results/p2-heldout/v3-d1-3B-Q4.json"))
A = {r["id"]: bool(p["correct"]) for r, p in zip(v3l, arch["per_row"])}
rows = [json.loads(l) for l in open("bench/results/p3-transfer/rows.jsonl")]
T = {r["id"]: bool(r["correct"]) for r in rows if r["arm"] == "targeted"}
L = {r["id"]: bool(r["correct"]) for r in rows if r["arm"] == "baseline"}

# 1. grep: row-level vs family-level
grep = [i for i in T if v3[i]["label"] == "filesys_grep"]
fix = [i for i in grep if not A[i] and T[i]]; brk = [i for i in grep if A[i] and not T[i]]
fam = collections.Counter(v3[i]["family"] for i in grep)
net = {f: sum(T[i] for i in grep if v3[i]["family"] == f)
          - sum(A[i] for i in grep if v3[i]["family"] == f) for f in fam}
pos = sum(1 for d in net.values() if d > 0); neg = sum(1 for d in net.values() if d < 0)
p_fam = min(1.0, 2 * sum(comb(pos + neg, i) for i in range(min(pos, neg) + 1)) / 2 ** (pos + neg))
print(len(fix), len(brk), "fixes by family:", collections.Counter(v3[i]["family"] for i in fix))
print("family sign test p =", round(p_fam, 4))

# 2. drift on the re-measured half
both = sorted(set(T) & set(L))
print("archive", sum(A[i] for i in both), "live", sum(L[i] for i in both),
      "targeted", sum(T[i] for i in both), "of", len(both))

# 3. multiple comparison
for k in (2, 8, 10, 16):
    print(k, "hypotheses ->", round(min(1.0, 0.003906 * k), 4))
```

Power figures use the reported intervals: SE = (hi − lo) / (2 × 1.959964), and
MDE₈₀ = (1.959964 + 0.8416) × SE.
