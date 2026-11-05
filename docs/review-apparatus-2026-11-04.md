# The apparatus, not the arithmetic

2026-11-04. A review of the instrument and of the reasoning behind tonight's four
results — `docs/eval-harness-2026-11-04.md`, `docs/rewrite-transfer-2026-11-04.md`,
`docs/router-policy-2026-11-04.md`, `docs/distill-classifier-2026-11-04.md`, read
against `bench/data/heldout_v3*.json`, `bench/results/p3-transfer/`,
`data/distill/` and `tests/data/utterances_v2.json`. Branch `v-method`.

I did not recheck the arithmetic, and I found none of it wrong. The overall rewrite
verdict reproduces to the point (−3.1 pts, 16 fixed / 29 broken, 133 families), the
cluster intervals reproduce, `Spearman ρ = −0.72` reproduces exactly, the router's 27/21
discordant counts reproduce. Every number I could recompute from a committed artifact
came back the same. That is the point of this document:
the arithmetic is fine and four of the inferences drawn from it are not, and the
gap between the two is the thing worth writing down.

**Every claim below is reproducible offline:**

```
python3 bench/verify_apparatus_review.py            # all sections, ~5 s, stdlib only
python3 bench/verify_apparatus_review.py --only b c # one or more sections a..g
```

| question | position |
|---|---|
| 1. Is `heldout_v3` a valid instrument? | **Valid as a paired instrument, invalid as a level instrument.** The 80.5%-not-74% gap is not what invalidates it; the refusal to revise was right about the bytes and wrong about the principle, and it shipped one known sampling hole. |
| 2. Is the rewrite verdict sound? | **The overall verdict is sound and stronger than the document claims. The published per-class consequence is not** — not because of multiplicity, but because the p is computed at the wrong unit and the ledger is attributed to the wrong object. |
| 3. Is "not separable from the 3B" right? | **Wrong sentence for a correct abstention.** Power 0.17 against the observed effect; the design resolves only ~13 points, not equivalent. |
| 4. Does `utterances_v2` support its CIs? | **No.** 89.6% of its groups are singletons; the "cluster bootstrap" is an i.i.d. bootstrap with a cluster-shaped label, and the dependence it misses runs the other way. |
| 5. Is the corpus's protocol defect correctly diagnosed? | **Defect real, mechanism wrong, correlation is a tie.** |
| 6. One change | **Make provenance part of the number** (below). |

---

## 1. `heldout_v3`: the right call for the wrong reason, and one hole it left open

Start with what is verifiable and came out clean. `bench/data/heldout_v3.json` has
exactly one commit and no later modification; nothing was edited into it after the
model was run. The splitting happened before rendering, and `tests/test_heldout_v3.py`
holds it. **The refusal is honest as executed.** Whatever the agent could see, it did
not touch the text.

The argument it gave for the refusal is where I part company: *"any post-hoc selection
change is indistinguishable from tuning toward 74%."* That is true of one kind of
change and false of the other, and the document conflates them.

- Editing a **rendered utterance** after seeing the model's answers is tuning. No
  defence exists, and none should be attempted.
- Changing a **selection rule over an inventory of authored frames** can be judged
  without the model at all: does every cell of (class × band × object column) appear
  in the sample in proportion to the inventory? That question has an answer that does
  not contain the number 74.

v3 fails that model-free test, and the document knows it: grep's 52 rows carry **zero**
package-noun objects against v2's 4 of 22, while `tools/distill/frames.py` offers two
`{mod}` frames among the 26 t1/t2 grep frames. Of v3's ten grep families, six are t1/t2
and four are t3 rival directions, so the t1/t2 draw is 6 of 26. I checked whether the
zero is a systematic bias or a coin flip, and it is a coin flip: such a draw misses both
`{mod}` frames with probability 0.585. **So the hole is ordinary small-sample variance —
but it is unchecked variance, in the class the document then uses for its one surviving
claim, and the honest repair (stratify the frame draw by slot column) would have been a
coverage argument, not a mirror argument.** The refusal preserved the set's integrity at
the price of shipping a known-undersampled class and calling the undersampling a
hypothesis for the next revision. Both of those are defensible individually. Together
they are the pattern this repo keeps repeating.

**The 6.5-point gap does not invalidate the set.** It disqualifies one specific use —
cross-set comparison of *levels* — and the document already refuses that use almost
everywhere. But it is worth being precise about why, because "uncalibrated" implies a
constant you could subtract and there isn't one. Three of the four causes the document
gives are *structural*, not scalar:

| cause | evidence | why it is not a constant |
|---|---|---|
| the t1 band means something different | v3 t1 92.5% vs v2 t1 81.2% | `bands.py` defines t1 as *signature vocabulary allowed* — a floor. v2's t1 was written to carry traps. Two constructs, one label. No reweighting reconciles them. |
| a whole object column is missing | v3 grep: 0 of 52 package-noun rows; v2: 4 of 22 | it changes the *within-class* difficulty, so it moves per-class deltas, not just the pooled level |
| class balance | 25% in the two weak classes vs v2's 31% | this one *is* scalar — the document's own Oaxaca decomposition gets 1.4–2.7 points out of it, correctly |

So: **the level gap says the two sets are not measurements of one quantity, and that
kills level comparison. It says nothing about whether within-set paired deltas are
valid, and the paired deltas are what the set is for.** The instrument is fit for the
job it was built to do and unfit for the job the acceptance test asked it to do. The
acceptance test was the defect — a new authoring pass over the same eight ops has no
reason on earth to reproduce an older pass's difficulty, and asking it to is a
criterion that *only* an honest set can fail.

Two consequences the document does not draw:

1. **`decide_extract` and `decide_pick` have the v2 defect inside the v3 instrument.**
   22 of 51 extract rows and 28 of 52 pick rows sit in multi-row families; the rest are
   families of one. A family-held-out number on those two classes is a *phrasing* split,
   which is the exact thing v2 is criticised for. The document says this about pick and
   extract in general; it does not say that the rewrite's extract half (+9.8, p = 0.18)
   therefore rests on 29 atomic rows of 51 (57%), i.e. that the half of the verdict which says
   "extract was the fit" is measured on the weakest structure in the set.
2. **v3-test is now spent.** The acceptance run reported per-class test-side accuracy,
   and the NOTES record a future revision's hypothesis ("stratify grep by slot column")
   that was derived from that table. Any agent acting on that hypothesis is now
   selecting frames using the model's answers, and the "indistinguishable from tuning
   toward 74%" objection applies to the *next* revision, not the last one. If the {mod}
   repair happens, it needs a fresh set, not a patched v3.

---

## 2. The rewrite verdict, and the `filesys_grep` claim

Position, stated flatly, because this is the question that matters:

> **The overall verdict is correct and understated. The per-class partition is
> legitimate as description and illegitimate as a published claim. And the specific
> reason is not multiple comparisons — it is that the p-value counts rows where the
> document counts families everywhere else, and that the ledger is kept by label when
> the thing that changed is an option.**

First, what I checked and could not use to defeat the claim, because a review that only
lands punches is not a review:

- **Multiplicity across the eight classes is not the killer.** 0.0039 × 8 = 0.031 < 0.05.
  Even Bonferroni over the class table leaves grep standing. This is *not* the 82.0%
  trap in its usual form.
- **The direction is robust to any single family.** Leave-one-family-out over grep's ten
  families keeps the delta between +10.6 and +19.6. It is not one outlier sentence
  dragging a null into significance.

Now the three things that do defeat it.

### (a) The p-value is computed at a unit the document rejects

`filesys_grep`'s nine fixes live in **three authored sentences**:

| family | rows | side | example |
|---|---:|---|---|
| `t1:filesys_grep:14` | 4 | train | "is handle_stream referenced anywhere else?" |
| `t3:filesys_grep->web_fetch:03` | 3 | **test** | "search the tree for state_max_tokens offline" |
| `t3:filesys_grep->filesys_glob:02` | 2 | train | "list the files that reference resolve_backend" |

The `p = 0.0039` is `mcnemar_exact(9, 0)` — nine independent observations. They are
three. At the unit this same document uses for every aggregate ("a family of 6
contributes 6 rows but only one observation", §6) the result is:

| slice | delta | family cluster 95% CI | families | family-level sign test |
|---|---:|---|---:|---|
| grep, all 52 rows | +17.3 | **[0.0, +36.5]** | 10 | 3–0, p = 0.25 |
| grep, v3-test only | +11.5 | [0.0, +37.5] | 5 | 1–0, p = 1.0 |
| **the two rewritten classes** (pre-specified) | +13.6 | **[+3.4, +25.6]** | 44 | 10–1, p = 0.012 |
| the six untouched classes | −8.7 | [−14.2, −4.0] | 89 | 0–14, p = 0.0001 |
| all 415 | −3.1 | [−8.5, +1.9] | 133 | 10–15, p = 0.42 |

The grep interval *touches zero* at the document's own unit. The single held-out
family that supports "and it is better out of sample than in sample" is one sentence,
"search the tree for {pat} offline", whose surface already contains the operation verb.

### (b) The ledger is by label; the thing that changed is an option

Tabulating the 29 breaks by the option that captured them:

| captured by | rows |
|---|---:|
| `decide_extract` | 20 |
| `web_search` | 6 |
| **`filesys_grep`** | **3** |

Those three are rows labelled `web_search` — "find writing about {topic} outside the
community threads" — that the new grep text captured. The document files them under
`web_search`'s loss column and reports the grep edit as *9 fixed, 0 broken*.
**The grep edit's ledger on v3 is 9 fixed / 3 broken, net +6 rows over 415, and its
damage is booked in someone else's account.**

This is the structural reason a per-class table cannot adjudicate an option-text edit:
the unit that was fitted is an *option*, the unit of accounting is a *label*, and one
edit's break shows up as another class's gain. The document notices the phenomenon for
`decide_extract` (20 of 29 land there, correctly attributed) and misses it in the
mirror direction, where it happens to flatter the claim being made — and the miss
produces a direct contradiction between two sentences of the same document. §6: "the grep
line leaks too … the rewritten grep line claiming a web-shaped search." §9, justifying
shipping the line: "grep's 9 fixes are pure and its 0 breaks are independent of the
extract change." Both are about the same edit. At most one is true.

### (c) The estimate is conditional on the other edit being live

Both strings changed in the targeted arm. The document says so, and lists "whether the
grep line alone is net positive out of sample" under **Not settled** (§9). Then
`README.md:482` and `TODO.md:92` publish it as settled. That escalation — a summary
claiming more than its own source document — is precisely the mechanism by which this
repo's previous three wrong claims were made, and it is the one finding here I would
fix before anything else.

### Is it the same trap as 82.0%?

No, and the difference is worth stating precisely, because it is the difference between
a claim that could be salvaged and one that could not.

**82.0% was double-dipping**: two strings written by reading the errors of the 150 rows
they were then scored on. The data was spent before the number existed.

**+17.3 is fresh data with a post-hoc slice.** The provenance is checkable and clean:
`bench/routing_diagnose.py`'s `TARGETED` was introduced in `606f3f4`, `bench/data/heldout_v3.json`
in `3d1377a`, which comes later in the history. Nothing about v3 was available when the
two strings were written, so every one of the 415 rows is genuinely out of sample and
the slice is the only thing wrong with the claim. What it shares with 82.0% is the *move
at the publication step*: select the cell where the effect appeared, then quote that
cell's precision computed at a unit the repo rejects everywhere else. It is the sibling
of the original error, not the same error, and it is falsifiable by one more run.

**What should have been published instead**, and is fully supported: the pre-specified
pair gained +13.6 [3.4, 25.6] over 44 families while the six untouched classes lost
−8.7 [−14.2, −4.0] over 89 with **zero** families net-positive. That is a cleaner and
stronger negative result than the one in the document, and it needs no post-hoc
partition to survive at the family unit.

---

## 3. "Not separable from the 3B" is an affirmative claim from a negative test

The router's own numbers: 79.3% vs 74.0%, 27 rows where only the router is right
against 21 where only the 3B is, paired CI −1.4 to +12.3, exact p = 0.47.

"Not separable" reads as *we measured and they are the same*. The data says *this
design cannot see a difference this small*. Those are different statements and only one
of them is true here.

At 48 discordant rows in 150 (32%), with α = 0.05 two-sided and 80% power:

| n | minimum detectable difference (80% power) | (50% power) |
|---:|---:|---:|
| **150 (this study)** | **12.9 pts** | 9.1 pts |
| 415 (a v3-sized routing set) | 7.8 pts | 5.4 pts |
| 900 | 5.3 pts | 3.7 pts |
| 1,200 | 4.6 pts | 3.2 pts |

To settle 5.3 points at 80% power: **≈ 900 rows**, or ≈ 1,100 with the family design
effect this repo's own 382-row measurement implies. To settle the raw 4.0-point
discordant difference: ≈ 1,570, or ≈ 1,900 corrected. The study's power against the
observed effect is **0.17**.

Two things follow, one of which is uncomfortable for the roadmap:

1. **"Not separable" should read "the design resolves only ~13 points, and the effect
   is 5."** The abstention is correct; the sentence is not. A CI that spans zero is
   evidence about the design's resolution first and about the parameter second.
2. **The recommended fix is about 2.5× too small.** §7 says "the next step is a held-out
   routing set". At v3's size (415) that set would resolve 7.8 points — still blind to
   the effect at issue. The next step is a routing set of roughly a thousand rows, built
   with families, or the question stays open at any n that fits in one authoring pass.

There is a better-posed version of the same question that this data *can* answer, and
the router document half-finds it: on the 22 rows the 3B is confidently wrong, the
cheap model is right on 16.4 and the router on 14. That is a large, mechanically
selected difference — but it is a selected subpopulation, so it cannot be converted
into an accuracy claim, and the document correctly does not try. The remaining honest
route is decision-theoretic rather than inferential: the router keeps 70% of the 3B's
cost to buy an unproven 5 points, which is a "no" that does not require the interval at
all. That is the argument to publish. "Not separable" is not.

---

## 4. `utterances_v2` cannot support the intervals computed on it

Direct answer: **no, and the group structure is worse than the question assumes.**

The declared grouping is 106 groups: 95 singletons and 11 five-row t3 families.
**89.6% of groups are single rows.** A cluster bootstrap that resamples 106 units, 95 of
which are one row apiece, is an i.i.d. bootstrap over 95 rows plus eleven five-row
blocks. The arithmetic ceiling on how much it could ever widen the interval is
150/106 = 1.42 in variance, i.e. 1.19× in width, and the measured ratios in the harness
document (0.96×, 0.84×) sit *below* i.i.d. rather than near the ceiling.

I checked whether the eleven real families carry the dependence they are supposed to,
using the one frozen signal that exists per row for all 150 (`baseline_live.json`):

| statistic | value |
|---|---|
| ICC of the 3B's outcome inside the declared families | **−0.12** |
| ICC inside only the 11 t3 families | +0.05 |

There is no positive within-family dependence to respect. The cluster bootstrap is
correcting for a structure that, for this signal, does not exist — which is exactly why
its interval comes out the width of an independent one. The harness document reads that
outcome as reassurance ("there is no dependence structure to widen an interval around").
The inference is circular: the ratio ≈ 1 *is* the statement that the declared grouping
contains no dependence, not that the data lacks any.

Meanwhile the dependence that does exist has the wrong shape and the bootstrap never
sees it. Same-label near-duplicate pairs, char-3gram Jaccard:

| threshold | pairs | straddle a declared group | outcome agreement |
|---|---:|---:|---|
| ≥ 0.30 | 5 | 5 | 5/5 |
| ≥ 0.25 | 10 | 9 | 8/9 |
| ≥ 0.20 | 37 | **32** | **29/32** |

Near-duplicate phrasings sit in *different* groups and agree on outcome 91% of the time
against a same-label base rate of 77%. That is positive dependence the declared grouping
cannot express, and it runs in the anti-conservative direction: the true interval is
wider than the one quoted. How much wider, on the 3B's own accuracy:

| grouping of the same 150 rows | 95% interval | width |
|---|---|---:|
| i.i.d. (150 units) | [66.7, 80.7] | 14.0 |
| declared (106 units) | [66.9, 81.5] | 14.6 |
| surface, 3-gram (102 units) | [66.4, 81.7] | 15.3 |

So the honest correction is **~1.05–1.5×**, not 1.0×, and it is a correction of
*unknown* size because the true phrasing families of v2 are unrecoverable — which the
harness document says, and then quotes intervals anyway.

**What this does and does not overturn.** It does not flip any conclusion in these four
documents. The router's paired CI widens from [−1.4, +12.3] to roughly [−2.6, +13.5] and
still spans zero; the classifier's 68.7 vs 74.0 was already null. It does mean that
every "95% CI" on v2 is a 150-row i.i.d. interval and the phrase "cluster-bootstrapped
over 106 groups" is a true description of the computation and a false description of the
inference. The single claim whose significance *could* flip under the correction is the
distill classifier's "significantly worse than the 35-rule cue table" (16 fixed / 31
broken, p = 0.040 over 47 discordant rows) — and `bench/results/distill_transfer*.json`
ships no per-row outcomes, so a reader cannot check it, while `eval_harness.Result`
exports `.outcomes` for exactly this purpose. That is an apparatus gap worth closing
before anyone quotes that p again.

---

## 5. The corpus's protocol defect: real, misdiagnosed, and correlated with a tie

Three separate claims are bundled in §6 and they have three different statuses.

**The defect is real.** An in-corpus family-held-out number cannot select a model for
out-of-corpus transfer. The demonstration is solid and does not depend on the
correlation: `val.jsonl` ranks the candidates the same way family-CV does, so the
certification's recommended protocol inherits the failure; `C3` scores 64.2% on val —
last of seven — and transfers at 66.0%, third; the top CV candidate transfers 8.7 points below
the fourth. Anyone who follows "fit on train, select on val" today loses 8.7 points of
transfer. That is a genuine, expensive, well-evidenced finding and it should keep its
place in the roadmap.

**The correlation is a tie, not a reversal.** ρ = −0.7143 reproduces exactly, and the
exact permutation p over 7! = 5,040 orderings is 0.044 — so the number is real and
mildly significant. But the artifact already contains the per-seed values, and the
document prints only the mean:

| candidate | cv mean | sd | per fold-assignment seed | v2 |
|---|---:|---:|---|---:|
| C1 ens raw | 77.4 | 1.80 | 79.9, 76.0, 76.3 | 60.0 |
| C4 ens decayed | 77.2 | 1.31 | 79.0, 76.0, 76.6 | 59.3 |
| C2 ens both | 77.0 | 1.27 | 78.7, 76.6, 75.7 | 60.7 |
| C6 ridge both | 76.3 | 0.25 | 76.6, 76.3, 76.0 | **68.7** |
| C3 ens abstract | 75.2 | 0.80 | 74.8, 76.3, 74.5 | 66.0 |
| C5 ridge raw | 74.0 | 0.57 | 74.8, 73.6, 73.6 | 63.3 |
| C7 ridge abstract | 73.2 | 0.00 | 73.2, 73.2, 73.2 | 66.7 |

Per seed the correlation is **−0.71, −0.04, −0.61**, and the induced orderings disagree
with each other (ρ(seed 0, seed 1) = +0.43; ρ(seed 1, seed 2) = +0.32): C1 is first,
fourth, second; C3 is fifth, second, fifth. The spread across candidates is 4.2 points,
the within-candidate spread across fold assignments reaches 1.8, and one 66-row fold has
a binomial SE of 5.2. **Averaging three unstable orderings produced a stable-looking
rank correlation.** The defensible statement is "family-held-out CV does not resolve
these seven candidates — it produces an ordering that changes with the fold assignment"
— which is a stronger and more actionable claim than a ρ, because it says the criterion
has no resolution rather than that it has a backwards one.

One anomaly in that table a reader should chase before quoting the protocol at all: `C7`
reports an sd of exactly 0.00 across three fold assignments, and I confirmed
`train.group_folds` does produce three distinct assignments for its seeds. An accuracy
identical to 4 dp (241/329) under three different splits is possible for a stable model
and surprising for this one; either it is real and worth knowing, or the ridge path is
not refitting per seed. Nobody has checked, and the number is load-bearing for the
"bounded models are more stable" half of the argument.

**The mechanism is contradicted by the repo's own experiment.** The diagnosis is that
family-held-out CV leaves slot nouns in the training fold, "so a model that has learned
`tau_lo → grep` is rewarded by the split that was supposed to punish it". I checked the
premise: it is true. 44 of 57 resolvable objects appear in more than one train family;
59 of 81 val rows name an object train also has. But the premise does not produce the
claimed effect, and the document's own third protocol says so:

> `object-CV` ... scores **7.0 points higher** than the family split and every band
> rises. That is the opposite of what a generalisation test should do.

It is the opposite of what the *stated mechanism* predicts, too. If the memorised noun
were the crutch, holding it out would cost accuracy; it gains 7 points. The reason is
visible in the data: the high-traffic objects are shared *across labels* — "the x86 lane
numbers" appears under `web_search`, `reddit_digest` and `filesys_grep` — so the noun
carries no label information to memorise. Of 239 object-bearing train rows, **144 (60%)
name an object that occurs under more than one label**, and 95 name a single-label object
at a median of 2 rows each. The document's own example is literally true — `tau_lo`
appears 9 times in train, all `filesys_grep` — but it is one of 34 low-traffic objects
covering 40% of the object-bearing rows, and a shortcut available on 40% of rows at a
median of two occurrences is not a mechanism that can carry a 1.1-point CV advantage.

What actually separates the seven is boundedness, and it is a **4-vs-3 contrast, not a
graded inverse relation**: four ensembles at CV 75.2–77.4 / transfer 59.3–66.0, three
ridge fits at CV 73.2–76.3 / transfer 63.3–68.7. That is one observation about
over-fitting in-domain versus transferring out-of-domain, dressed as a monotone trend
with a correlation coefficient. The document half-sees this ("the bounded-vs-unbounded
split was decided after C1's transfer number was seen"), which is the right caveat
attached to the wrong statistic.

**Net position on Q5:** the defect stands and the roadmap item stands; the mechanism
should be rewritten as "in-corpus generalisation does not predict cross-corpus
transfer", and `Spearman ρ = −0.71` should be replaced by the per-seed spread that shows
the criterion has no resolution. The item is not made less urgent by this — it is made
more urgent, because "the corpus can't rank models" is a harder fact than "it ranks them
backwards".

---

## 6. The one change

If I could make exactly one methodological change, and had to pick the one with the best
ratio of work saved to publication risk:

> **Make provenance part of the number: an `authored_on` field on every signal, model
> and instrument configuration, naming the rows whose *output* informed it — and a
> harness rule that refuses to render an accuracy as an improvement when
> `authored_on ∩ scored_rows ≠ ∅`, printing instead "in-sample fit, *j* rows" and
> requiring the untouched complement's number beside it.**

Applied to 82.0%: the two description strings carry
`authored_on = {18 rows whose errors they were written from}`. The harness cannot print
82.0% as a gain; it prints "18 of 150 rows are in the authoring set" and puts the 132
untouched rows next to it. The error becomes impossible to state rather than merely
unadvisable to state, and the run that produced 82.0% would have produced a non-claim.

Applied tonight: the grep slice carries `authored_on = {the 150 v2 rows}`, which is
clean, so the harness would *permit* it — and this review's complaint about grep is
about the unit and the ledger, not provenance. That is right: the gate is not a
substitute for the other corrections, it is the one that would have prevented the
specific failure this repo has now repeated three times, which is **a documented caveat
being outrun by a headline.** This repository has shown, twice, that a rule the harness
enforces holds (the overlap guard in `12b2b98`; the no-ground-truth guard in
`tests/test_router_policy.py`) while a caveat written in prose does not. Enforcement is
the only form of honesty this codebase has demonstrated it keeps.

The alternative I would accept, if provenance is too much bookkeeping: **pre-register a
written-on half and a scored-on half of the frozen 150 before authoring anything, and
publish only the scored-on half's number.** Same effect, 20 lines, no new data, and it
also converts the current habit of spending the 150 on every arm into a cost the
pre-registration makes visible.

---

## 7. Everything I found that undermines a conclusion

Ordered by how much it matters.

1. **`filesys_grep`'s published p-value counts 9 rows that are 3 sentences** (Q2a).
   At the document's own family unit: +17.3 [0.0, +36.5], sign test 3–0, p = 0.25. The
   only held-out family behind "it is better out of sample than in sample" is one
   authored sentence.
2. **The grep edit's breaks are booked in another class's column** (Q2b). Measured
   ledger: 9 fixed / 3 broken, net +6 rows over 415 — not 9/0. Three `web_search` rows
   were captured by the new grep text.
3. **`README.md:482` and `TODO.md:92` publish as settled a claim the source document
   lists as not settled** (`docs/rewrite-transfer-2026-11-04.md` §9). This is the exact
   failure mode that produced the previous three wrong claims — and the git history shows
   it happening twice in this one task. Commit `1106ac8` is titled "bench: the rewrite
   transfers, 7 fixed and 0 broken on held-out families". Its own artifact,
   `bench/results/p3-transfer/summary.json` at that commit, reported both numbers:
   `v3_targets` n=52, **7 fixed / 0 broken**, p = 0.0156 *and* `v3_all` n=102,
   **7 fixed / 4 broken**, p = 0.55. The commit message quoted the first and dropped the
   second, while 313 of the 415 planned rows — both train-side strata and most of the
   guard set — were still unrun, and the completed run moved the same quantity to
   16 fixed / 29 broken. The identical move — quote the favourable cell,
   drop the aggregate beside it — is what `README.md:482` is now doing with grep's 9/0.
   A document can carry a dated correction; a commit message cannot, so the log is a
   permanent record of a retired claim, and the only fix that survives this pattern twice
   in one night is enforcement rather than discipline.
4. **"Not separable from the 3B" asserts equivalence from a test that resolves only
   ~13 points** (Q3), and the prescribed remedy — a v3-sized routing set — is 2.5× too
   small to settle the question it is prescribed for.
5. **`utterances_v2`'s cluster intervals are i.i.d. intervals** (Q4): 89.6% singleton
   groups, ICC within the declared families of −0.12, and same-label near-duplicate
   pairs that straddle groups and agree 91% of the time. Every 95% CI in the router and
   classifier documents is ~1.05–1.5× too narrow.
6. **`Spearman ρ = −0.71` is an average of three unstable orderings** (Q5): −0.71,
   −0.04, −0.61 by seed, with the per-seed values sitting unread in the committed
   artifact.
7. **The corpus's stated mechanism is contradicted by its own third protocol** (Q5):
   object-held-out CV scores 7 points *higher*, and the high-traffic nouns are shared
   across labels.
8. **The rewrite document's framing of v3's train/test split is a category error, and it
   weakens its own verdict.** "The test-side number is the one that matters, because it
   is the only slice whose phrasings were held out by construction" — held out from
   *what*? Nothing is fitted on v3. The rewrite saw neither side, the 3B saw neither
   side. All 415 rows are equally out of sample, so the 415-row paired result **is**
   the test, and it is the better one: −3.1 [−8.5, +1.9] is an interval that *excludes*
   the claimed +8.0, which falsifies the original claim outright. Reporting the 207-row
   test slice instead (81.6 → 79.7, p = 0.48) turns a falsification into a shrug. The
   split's real job is to gate the *next* edit, and §9 does say that — the two uses are
   conflated in §4.
9. **`decide_extract`'s out-of-sample number rests on 57% atomic rows** (Q1). The half
   of the verdict that says "extract was the fit" is measured on the weakest structure
   in the instrument built to fix weak structure.
10. **`bench/distill_transfer.py` ships no per-row outcomes** while `eval_harness`
    exports `.outcomes` for exactly this purpose. The one v2 claim that a unit-of-
    analysis correction could flip (68.7 vs 78.7, p = 0.040) is therefore uncheckable
    by a reader.

---

## 8. What is right, and should not be softened in the other direction

A review that only takes is not useful, and three of tonight's four results are better
evidenced than the repo has managed before.

- **Retiring 82.0% is correct, and the evidence is stronger than the document says** —
  see item 8 above.
- **The rewrite's harm is the best-evidenced finding of the four.** 27 broken against 0
  fixed across 14 families, family sign test 0–14 at p = 0.0001, cluster CI
  [−14.2, −4.0], and a jackknife range of −8.8 to −6.9 across 89 single-family deletions.
  Nothing about it depends on a unit choice or a split.
- **The oracle defect was found and fixed properly**, and the guard that keeps it fixed
  perturbs the scored labels and requires the deployable policies to be byte-identical
  while requiring the oracle's decisions to change — a test that cannot pass by
  perturbing nothing. That is the model the rest of the apparatus should follow.
- **The harness's frozen/refit distinction is real and correctly enforced.** The 3B's
  point estimate is split-invariant by construction and the document says so instead of
  taking credit for it.
- **The contamination sweep is among the most rigorous things in this repository**:
  all-pairs maxima rather than nearest-neighbour, a pruning bound that proves no pair is
  skipped, and a second rule for the deletion case the first rule cannot see. Two rows
  removed on a rule stricter than the one in force, and both written down.
- **The certification's refusal to pool without per-band n, and the "t1 teaches keyword
  matching" warning, are both correct and both under-used** — v3's t1 band is 42% of the
  set and scores 92.5%, and the grep fix's largest single family is a t1 frame. The
  certification's own advice ("report t1 and t2/t3 separately, always") would have
  caught both of the per-class problems in §2.

---

## 9. Reproduce

```
python3 bench/verify_apparatus_review.py               # every section, ~5 s
python3 bench/verify_apparatus_review.py --only a      # v3 structure and band mix
python3 bench/verify_apparatus_review.py --only b c    # the grep claim and its ledger
python3 bench/verify_apparatus_review.py --only d      # v2 group structure, ICC, straddlers
python3 bench/verify_apparatus_review.py --only e      # resolution of the router comparison
python3 bench/verify_apparatus_review.py --only f g    # corpus selection and its nouns
python3 -m pytest tests/ -q                            # 826 passed, unchanged by this review
```

Reads only committed artifacts: `bench/data/heldout_v3.json`,
`bench/results/p2-heldout/v3-d1-3B-Q4.json`, `bench/results/p3-transfer/rows.jsonl`,
`bench/results/p1-diagnose/baseline_live.json`,
`bench/results/distill_transfer_sweep.json`, `data/distill/{train,val}.jsonl`,
`tools/distill/frames.py`, `tests/data/utterances_v2.json`, `bench/eval_harness.py`.
No network, no model call, no refit, stdlib only.

---

## Correction appended — 2026-11-05: the live baseline arm, and which of these positions survive it

*Appended, not rewritten. The review above is a record of 2026-11-04 against the artefacts of
that day. The baseline arm that §2 treats as archived has since been completed to all 415 rows
in `bench/results/p3-transfer/rows.jsonl` from the same session as the targeted arm (02:36:09 →
03:14:16, p50 3 959 ms, against the targeted arm's 4 079 ms and the archive's 8 225 ms).*

### §2's position was half right, and the half that was wrong matters

§2 stated: *"The overall verdict is correct and understated."* The first clause is now false.
The verdict was **wrong** — "the whole rewrite fails out of sample" was produced by pairing a
live arm against an archive that had drifted 9 rows, and the corrected live-vs-live result is
**78.3% → 77.4%**, 16 fixed / 20 broken, p = **0.618**, with the held-out test side at
**165 → 165**. A wash, not a failure.

§2's reasoning was not the problem; its premise was. It computed the unit and ledger defects
correctly and then concluded the direction was sound because the arithmetic was sound. This
review's own §7 opens with exactly that pattern as the repo's recurring failure, and this is a
third instance of it: **arithmetic that reproduces against the wrong comparison.**

What survives, precisely:

| §2 claim | status after the live arm |
|---|---|
| (a) The p-value counts rows where the document counts families | **stands.** grep's 9 fixes are still 3 families, 3–0, p = 0.25 on the live pairing. `web_search`'s damage is 8 rows in 4 families, 0–4, p = 0.125. Neither class is significant at the family unit. |
| (b) The ledger is by label; the change is to an option | **confirmed exactly.** Live capture shows grep 9 fixed / **3 broken**, net +6, the 3 booked under `web_search` — the number §2 derived from the 233-row control, reproduced over 415. |
| (c) `README.md` and `TODO.md` published as settled what the source lists as not settled | **was true, now fixed in place.** Both documents now carry the measured effect *and* its price. |
| "What should have been published instead" — the pre-specified pair and the six untouched classes | **stands, and is adopted.** The pair: 56/103 → 70/103, 16 / 2, p = 0.0013, 10 net-positive families against 1. The six untouched: 269/312 → 251/312, 0 / 18, **−5.8 pts**, 13 discordant families, none net-positive. (The review's −8.7 / 27 rows was archive-inflated.) |

### §1's conclusion is strengthened, by a fact that did not exist on 2026-11-04

§1 argued that `heldout_v3` is invalid as a *level* instrument and valid as a *paired* one, and
that the refusal to revise its 80.5% was right about the bytes and wrong about the principle.
The live arm supplies a harder version of that point: **80.5% is not the instrument's level.**
The same instrument on the same 415 rows, re-run 45 minutes later, gives **78.3% (325)** — a
session-to-session spread of 9 rows on a set whose note claims to resolve about 8 points per
side. The number that was argued over is not a property of the set at all; it is a property of
the run. `bench/data/heldout_v3_NOTES.md` carries a pointer to this.

That does not revive the missing ~74%: the paired design is what §1 said it was, and the paired
result is now measured both ways. It does mean a v3 *level* should never be quoted without the
date and the server it was measured on.

### §7's ledger, updated

* Items **1 and 2** (grep's unit and ledger) still stand as written.
* Item **3** (the escalation into `README.md` / `TODO.md`) is fixed in place in both files.
* Item **3**'s comparison — commit `1106ac8` quoting the favourable cell while the aggregate sat
  beside it — is now a documented pattern three times over, and the last one is the published
  verdict itself.
* The **"one change"** in §6, a provenance gate, is untouched by this and remains the
  recommendation. Nothing here argues against it; this correction is an argument for the
  sibling change, that a paired delta also records the conditions of both arms.

Full re-derivation, the eleven-row drift ledger and the protocol are in the appended 2026-11-05
section of `docs/rewrite-transfer-2026-11-04.md`.
