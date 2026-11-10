# Forward plan — four directions adjudicated, and the one measurement that outranks all of them

2026-11-10. Written by an orchestrator that was allowed to read, to dispatch, and to run
in-process experiments, and **not** to write product code. No file outside this one was
modified. No llama-server was started, stopped, or contacted; `127.0.0.1:1113` was not touched.

Five adversarial subagents ran in parallel — one per direction plus one on the harness. Their
job was to attack their own direction, and three of them refuted the premise they were handed.
Everything below is either located at `file:line` or reproduced by a command in §10. Where the
brief this plan was written against was wrong, §2 says so with the evidence.

---

## 1. The answer, in one paragraph

**Direction 3 wins, but not in the form it was proposed.** The members do *not* fail together
(φ = 0.406 on correctness, not ~1), and the fusion of the 3B with a 2-millisecond lexical
member reaches **352/415 = 84.82%** against the 3B's 325/415 = 78.31% — **+27 rows, exact
McNemar p = 0.0002**, holding at 346 and 349 rows across two further fold seeds and holding
separately inside and outside the frame-rendered subset of the corpus. The same gain is
reachable by a mechanism that shares nothing with probability averaging — *gate on the 3B's own
confidence* (AUC 0.858 for correctness) and hand 13.7% of rows to the cheap arm: 345/415 =
83.13%, p = 0.0003. Direction 1's cheap half is what makes Direction 3 pay, so it is ranked
second rather than dropped. Direction 4's instinct was right and its **gate variable was
wrong** — pointed at the classifier's confidence the cascade fires on 0 of 207 rows and buys
exactly zero; pointed at the 3B's confidence it buys 20. Direction 2 is dropped: a *frozen*
22.7M encoder already reaches 78.74–81.64% on the honest split, level with the shipping 3B, so
there is no headroom for a fine-tune to spend 20–30 hours reaching.

| rank | direction | verdict | the number that decides it | cost |
|---|---|---|---|---|
| 1 | **D3 fusion** | **DO** (as a confidence gate or an equal-weight mean, not as a search over modes) | **352/415 = 84.82%, +27 rows, p = 0.0002**; gate form 345/415, p = 0.0003 | ~1 day |
| 2 | **D1 scaffolding** | **DO the cue-feature half / DROP the lexicon half** | cue features **+7.7 pts** (144→160 of 207); slot lexicons **+13.0 on v3, −7.3 on transfer** | 3 h |
| 3 | **D4 product backend** | **DO the 3 h corpus fix / DROP the accuracy claim and the temperature edit** | honest OOF gate: **0 of 207 rows fire, delta 0.0000, p = 1.0000** | 3 h |
| 4 | **D2 fine-tune** | **DROP** | frozen encoder **78.74–81.64%** on 207 test rows vs 3B **79.71%** — at the plateau already | 0 (20–30 h avoided) |

**The harness must be fixed before any of these numbers can be quoted from the repo**: three
gaps make tonight's result non-reproducible through `bench/harness/` (§8).

---

## 2. What was wrong in the numbers we started from

| claim in the brief | truth | source |
|---|---|---|
| "a **two**-of-eight option-description rewrite: buys 10 rows on one option, costs 7 elsewhere, p=0.6291" | The 10/7/p=0.6291 figures belong to `grep_only`, which changes **one** of eight descriptions. The two-of-eight arm (`targeted`) is **16 fixed / 20 broken, −4 rows, p = 0.6177**. | `bench/rewrite_transfer.py:79-85`; recomputed from `bench/results/p3-transfer/rows.jsonl` (§10.2) |
| "plain TF-IDF + linear classifier on the same 415 rows: **60.7%**" | 0.607 is the **stdlib** `linear_text` model fitted on a 50-row corpus (`tests/data/utterances.json`), in-sample on v3. The sklearn arm is **62.2%** word / 63.1% word+char / 62.7% calibrated — and those are **seed 0 of a 62–72% band**. These are three different models and one number cannot stand for them. | `docs/plan-post-rewrite-2026-11-09.md:371,451`; `docs/todos-2026-11-10.md:33`; D1's seed sweep (§10.5) |
| "66.23% with whole phrase-**families** held out" | It is **template**-grouped (110 groups = 109 templates + the 25 atomic `__none__` rows). `data/routing_382.json` has no template field, so that number cannot be reproduced from the file the sentence names. Family-grouping that corpus is **impossible**: 7 families over 18 intents means every held-out family removes its labels and every fold scores 0. | `docs/classical-baseline-2026-11-03.md:120-133`; harness review §Q2 |
| "union ceiling 92.8% [88.5-96.6] family-held-out" | That is **n = 150 (v2)**, between `lr` (67.1%) and a 3B at **74.0%** — and it is an oracle. On v3 the equivalent oracle is **90.34%** (207 test rows) / 91.81% (415 rows, D3) / 87.95% (D1). None is achievable. | `docs/eval-harness-2026-11-04.md:260`; `docs/router-policy-2026-11-04.md:129` |
| "T~0.17 gives 44.6% coverage at the 0.70 gate (precision 0.865)" | The triple reproduces **exactly** — 252/415 = 0.6072, coverage 0.4458, precision 0.8649 — but it is **in-sample on v3**. And "accuracy is invariant across temperature" is a softmax-argmax identity (`linear_text.py:136-145`), not a measurement, so it cannot support a claim in either direction. | D4 §CLAIMS; `docs/plan-post-rewrite-2026-11-09.md:451` |
| "cascade 79.3% at 95 ms vs 74.0% at 169 ms, p=0.152 at n=150 — the best configuration anyone measured" | Correct as recorded, and unusable as a plan: the threshold (0.20) was chosen by looking at the 150 rows it is scored on. Transferred honestly to v3 it is **+3.38 pp at p = 0.2478**; chosen out-of-fold it is **0.0000 at p = 1.0000**. | `docs/classical-baseline-2026-11-03.md:196-206`; D4 §CASCADE |
| "torch is not installed yet" | **Wrong as of tonight.** `torch 2.14.1+cpu`, `transformers 5.20.2`, and `sentence-transformers/all-MiniLM-L6-v2` (22.7M params, 90 868 376 B) is in the HF cache and loads **offline**. Direction 2 was never blocked on an install. | `python3 -c "import torch"`; `~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/` |
| "`docs/confusion…` if present" | It does not exist. | `ls docs/confusion*` → no such file |
| "35-rule cue table 78.7% on the 150-row set, BUT contaminated" | Contaminated **and** it does not survive v3: `rules_predict` on all 415 rows is **259/415 = 62.41%** with 91 abstentions (79.9% among the 324 it answers). It has never been scored on v3 before tonight. | `bench/local_clf.py:243-298`, reproduced §10.3 |

**Verified and correct as stated:** the 3B's 325/415; order reversal moving **34 of 415 argmaxes**,
costing **9 rows**, p = 0.0784; the A/A floor of **1 flip in 415**; `grep_only` 10/7 p = 0.6291;
the 17-vs-8 label mismatch in `bench/registry_382.json`; the 133 families and the clean 208/207
split; the 7 families on the 382 corpus.

---

## 3. Direction 3 — fusion the 3B with a cheap lexical arm. **DO.**

### Why the brief's premise was wrong

The direction was framed as "if both members fail on the same rows, the mix buys nothing".
Measured, they fail together on only **34–52 of 415** rows depending on the member:

| member | member OOF | 2×2 (both right / 3B only / member only / both wrong) | φ | source |
|---|---:|---|---:|---|
| tfidf_word_char + sigmoid (D3) | 314/415 = 75.7% | 258 / 67 / 56 / 34 | **0.165** | D3 `strict.py` |
| frozen MiniLM + logreg (this plan) | 313/415 = 75.4% | 275 / 50 / 38 / 52 | **0.406** | §10.6 |
| tfidf + 8 rule-score features (this plan) | 322/415 = 77.6% | — | — | §10.7 |

Every member is far from redundant with the 3B. The ensemble has something to work with.

### What it actually buys

Equal-weight mean of the 3B's recorded probabilities and the member's out-of-fold
probabilities. **w = 0.5 is the a-priori default, not a tuned maximum** — that distinction is
the whole difference between this and the failed cascade below.

| member | seed | mean@0.5 | vs 3B | McNemar |
|---|---:|---|---:|---|
| tfidf + rule features | 0 | **352/415 = 84.82%** | fixed 39 / broke 12 | **p = 0.0002** |
| tfidf + rule features | 1 | 346/415 = 83.37% | fixed 36 / broke 15 | p = 0.0046 |
| tfidf + rule features | 2 | 349/415 = 84.10% | fixed 36 / broke 12 | p = 0.0007 |
| frozen MiniLM | 0 | 341/415 = 82.17% | fixed 19 / broke 3 | p = 0.0009 |
| frozen MiniLM | 1 | 339/415 = 81.69% | fixed 17 / broke 3 | p = 0.0026 |

### The mechanism, and why it is not a probability-scaling artefact

The 3B's confidence is a **usable correctness signal**: median 0.895 on rows it gets right,
**0.395 on rows it gets wrong**, AUC **0.858** over all 415. (This refutes
`docs/classical-baseline-2026-11-03.md` §6's "d1-3B reports confidences around 0.9 by
construction" — that sentence describes the 18-way ballot, not this arm.)

So the fusion wins by deferring exactly where the 3B is already unsure. A mechanism that shares
nothing with averaging — *if `conf < 0.38`, take the member, else the 3B* — with the threshold
chosen **on the 208 `split=="train"` rows only** and then frozen:

| member | seed | 3B-conf gate@0.38 | escalation | McNemar |
|---|---:|---|---:|---|
| tfidf + rule features | 0 | 345/415 = 83.13% | 13.7% | p = 0.0003 |
| tfidf + rule features | 1 | 344/415 = 82.89% | 13.7% | p = 0.0009 |
| tfidf + rule features | 2 | 346/415 = 83.37% | 13.7% | p = 0.0005 |
| frozen MiniLM | 0 | 339/415 = 81.69% | 15.2% | p = 0.0125 |
| frozen MiniLM | 1 | 339/415 = 81.69% | 15.2% | p = 0.0125 |

Two unrelated combination rules, five fold seeds, two member families: the effect is +5 to +6.5
points and it never falls below p = 0.013. **This is the only arm in the repo that clears the
3-row noise band and the A/A floor at once.**

### It is not a frame-rendering artefact

86.0% of v3's rows contain a literal value from `tools/distill/frames.py:SLOTS` (70 values;
32.7% of v2 does, 38.0% of v1). Splitting on that — the strongest "is this corpus leaking its
generator" test available in-repo:

| subset | n | 3B | member | mean@0.5 | McNemar |
|---|---:|---|---:|---|---|
| frame-rendered (contains a SLOT value) | 357 | 79.83% | 76.19% | **84.59%** (+17) | p = 0.0095 |
| **not** frame-rendered | 58 | **68.97%** | **86.21%** | **86.21%** (+10) | fixed 11 / broke 1, p = 0.0063 |

The gain holds in both halves, and the 3B is *worse* on the 58 rows that carry no generator
fingerprint — which is where the member carries it. This is the sentence that makes the result
credible.

### The honest caveats

1. **The only clean cross-corpus test gives +2.0 points at n = 150, p = 0.375.** Fitting the
   frozen-MiniLM member on v3-train and scoring v2: member 103/150 = 68.67%, mean@0.5
   **114/150 = 76.00%** against the archived 3B's 111/150 = 74.00% (fixed 4 / broke 1). Same
   sign, no power. The same test with the **rules** member reads 92.00% and must be **discarded
   as contaminated** — `local_clf.RULES` was authored by reading v2 (`docs/eval-harness-2026-11-04.md`
   §6(3)). There is no corpus in this repo on which a single member is both available and clean:
   rules are clean on v3 and contaminated on v2; slot lexicons are inflated on v3 and invalid
   elsewhere; only the frozen encoder is clean on both.
2. **`w` is a knob and the curve is not monotone at small n.** On the 207 test rows w=0.4 gives
   83.57% (p = 0.039), w=0.5 gives 82.13% (p = 0.332), w=0.6 gives 85.02% (p = 0.043) — that
   p-value is a function of a knob, which is exactly the error that killed D4's cascade. The
   defence is that **0.5 is the null** and the 415-row sweeps peak at or next to it. Do not
   quote a peak-of-sweep number.
3. **The 3B side is a replay** of `floor_a`'s recorded probabilities, not a live call. A shipping
   version needs `/v1/systemone` to keep returning the 8-way distribution it already returns.
4. **The member's OOF number depends on the fold scheme, by ~4 points.** The harness's
   `assign_folds` is unstratified `sha256(seed|group) % folds` (`metrics.py:220-226`) and its
   folds came out 98/70/80/65/102 rows on seed 0. Frozen MiniLM scores 68.43 / 76.14 / 73.25%
   across three hash seeds and 75.42 / 74.22 / 73.49% across three size-balanced seeds. **No
   single-fold-assignment number on this corpus is worth more than ±4 points.**

### `ensemble.py` cannot express the winner

- `ENSEMBLE_KEYS = ("members","mode","folds")` (`ensemble.py:65`) — **there is no `weights=`**,
  so the equal-weight mean is expressible and any weighted mean is not.
- `vote` with two members is a structural no-op: every disagreement goes to the tie rule and it
  cannot exceed the stronger member. It scored exactly 325/415 in D3's run.
- `stacked` does run a real inner out-of-fold loop and `error_correlation()` does report
  **per-pair** φ, so those two claims in the brief check out.
- No `gated`/`cascade` mode at all (`ensemble.py:64`). The winning rule is not in the file.

---

## 4. Direction 1 — scaffolding features instead of fitting. **DO the cue half, DROP the lexicon half.**

### The free number nobody had taken

`bench/local_clf.py`'s 35-rule cue table has existed since `bdc9a71` (2026-11-03 17:17), before
v3 was authored (`3d1377a`, 2026-11-04 01:56). It has **never** been scored on v3. It scores
**259/415 = 62.41%** — the same band as TF-IDF, with 91 abstentions. So "a hand-written cue table
beats a classifier" was a v2-only statement.

### Where the real gain is

The rules are worth much more as **features** than as a predictor:

| arm (declared split, fit on 208, score 207) | correct | accuracy |
|---|---:|---:|
| TF-IDF word 1–2, logistic | 144/207 | 69.57% |
| TF-IDF **+ the 8 rule scores as features** | **160/207** | **77.29%** |
| …the same, balanced family CV, 3 seeds | 322/317/321 of 415 | 77.59 / 76.39 / 77.35% |

**+7.7 points for ~40 lines and no new authoring**, and it is this member — not the encoder —
that produces D3's best fusion (+27 rows vs +16). D1's version of the same arm — word+char plus
the rule block plus a slot block — reads **85.06%** on v3 family-held-out, and its rule block
alone takes v2 transfer from TF-IDF's 52.67% to **84.00%**. That second figure is inflated
(`RULES` was authored against v2) and D1 says so; it is quoted here only because the honest
version of the comparison is what §4's frame split exists to supply.

### Why the lexicon half must die

Object-noun / slot lexicons buy **+13.0 points on v3 and lose 7.3 on transfer**. The reason is
mechanical: **86.0% of v3 rows contain a literal `frames.py:SLOTS` string** versus 32.7% of v2.
A feature keyed on that vocabulary decodes the generator, and the generator ships in the repo
beside it. This is the single sharpest methodological finding of the night and it constrains
every future feature experiment: *v3 cannot validate a feature that shares a source with
`tools/distill/`.*

### Cost and reversal

3 hours: lift `local_clf.RULES` into a feature block behind the harness's `sklearn_text` spec,
score it OOF, and merge it into the fusion member. **Reverse** if a cue-feature block fails to
beat plain TF-IDF by ≥3 points on a corpus that is not v2 and not v3 — none exists today, so
the check must wait for one.

---

## 5. Direction 4 — fix the classical backend in the product. **DO the corpus fix as observability; DROP it as an accuracy play and DROP the temperature edit.**

### What is confirmed

- `decide.classical.corpus_path = bench/registry_382.json`, whose 17 tool ids share **zero**
  names with the 8 ops; `classical.py:103-108` therefore returns `choice=None` on **415/415
  rows at any temperature**. Confirmed by reading and by D4's replay.
- `decide.backends_in_use.choice` is **`"pair"`** and `cascade.enabled` is **false**
  (`defaults.json:56,67`) — so fixing this changes **nothing shipped**. That is the fact the
  brief's "never wired" hides.
- `__none__` never fires on any 8-op corpus fitted (0/415, 0/207, 0/208). The "learned
  abstention" the module docstring credits is untested on this lane.

### Why the accuracy claim dies

| corpus | eval | gate | cascade | vs 3B | McNemar |
|---|---|---|---:|---|---|
| v3-train (208) | v3-test 207 | 0.25, hand-picked on the evaluated rows | 84.06% | +4.35 pp | 9/0, p = 0.0039 |
| v3-train (208) | v3-test 207 | 0.20, transferred from v2 | 83.09% | +3.38 pp | 17/10, p = 0.2478 |
| v3-train (208) | v3-test 207 | **0.40, chosen out-of-fold** | **79.71%** | **0.0000** | 0/0, **p = 1.0000** |
| `tests/data/utterances.json` (50) | v3-test 207 | 0.20 | 75.36% | **−4.35 pp** | worse |

The honest gate **never fires**, and the sweep is monotone toward "disable the classifier"
(0.4760 → 0.7692 as the gate rises 0.15 → 0.40). The 50-row leak-free corpus **loses**. And the
208-row corpus that wins has **no leak-free load path**: `load_corpus` (`classical.py:143-171`)
has no split filter, so pointing `corpus_path` at `heldout_v3.json` loads all 415 rows and scores
**415/415 = 100.0%**. The one-line path change is either useless or cheating.

### What should ship anyway, and why

**The 3-hour corpus half, for observability, not accuracy.** A backend that answers `None` on
every row cannot be measured, reviewed, or deleted on evidence; one that answers can be. Derive a
208-row 8-op corpus file so `load_corpus` never sees `split=="test"`, amend
`tests/test_classical_backend.py:142-146` (which pins 200 rows / 18 labels on the shipped path),
**keep** `:74-78` (it asserts the None path is *correct* behaviour — this is a conf mismatch, not
a defect), and add a `bench/arms.json` entry with the override, since no classical arm exists
there today. Then it is a bench arm with a number, and D3's gate — which is the cascade this
direction wanted — has something to gate with.

**Drop the temperature edit.** With accuracy argmax-invariant and no held-out calibration set on
the shipped path, `temperature` cannot be fitted honestly; and the 44.6%/0.865 operating point
belongs to a 0.70 gate that nothing in the cascade path consults.

---

## 6. Direction 2 — fine-tune an encoder. **DROP.**

The blocker the brief assumed does not exist: torch and transformers are installed and
`all-MiniLM-L6-v2` loads offline. So the question becomes purely scientific, and the answer is
that the plateau is already reached by a **frozen** encoder with a linear head:

| arm | 207 test rows (fit on 208) | source |
|---|---:|---|
| 3B, `floor_a` | **165/207 = 79.71%** | `rows.jsonl` |
| frozen MiniLM + logreg, C = 1.0 | 163/207 = 78.74% | §10.4 |
| frozen MiniLM + logreg, C = 1.0 (D2's run) | 81.64% | D2 `exp.py` |
| frozen MiniLM on the clean v2 transfer | 103/150 = 68.67% | §10.8 |
| TF-IDF word+char OOF (reported) | 63.1% | `docs/todos-2026-11-10.md:33` |

A frozen 22.7M encoder is worth **+17 points over TF-IDF** and then stops dead, level with a 3B
LLM and level with a 35-rule cue table. Three unrelated families sitting in one 78–82% band is
the definition of a corpus-limited ceiling, and freeing 22.7M parameters on 208 rows is the
`ridge` failure (97.9% in-sample → 68.7% transfer, `docs/distill-classifier-2026-11-04.md:79`)
with more freedom rather than less.

Two further facts close it:

- **The selection loop is poisoned.** `docs/distill-classifier-2026-11-04.md:247-281` shows
  family-held-out CV ranking seven candidates **backwards** against transfer (ρ = **−0.71**), and
  the certification's own "select on `val.jsonl`" protocol inherits the defect. A fine-tune
  produces epochs × LR × early-stop checkpoints with no defensible way to choose between them.
  D2 reproduced the pathology in miniature: the identical protocol, only `C` moved, 76.81% →
  81.64%. My own run of the same config landed at 78.74%. **Two honest runs, 2.9 points apart, on
  a knob nobody can rank.**
- **The headroom is complementarity, not capacity** — which is Direction 3, and Direction 3 is
  cheap.

**Keep the frozen embedding arm** as an optional third member: it is the only member that is
clean on both corpora. **Reverse** if a frozen arm scores below 74.0% on a clean same-rows
protocol (it would then have a real ≥6-point gap to chase).

---

## 7. The ONE next measurement

**What it is:** the pre-registered, harness-native reproduction of the fusion arm on all 415
rows, with per-row out-of-fold records written and the per-label ledger and McNemar computed
against `recorded|arm=floor_a`. It is the artifact the shipping decision needs, it exercises the
two harness fixes every later experiment depends on, and it is the only way this number gets out
of `/tmp`.

**Prerequisites** (both specified to the line by the harness review, ~40 min total):

1. `run.py:193-203` — `metrics.grouped_cv` returns `oof` (`metrics.py:256`) and `run.py` throws it
   away. Write it, and make it **be** the arm's records, so `paired`/`ledger` work between a
   fitted arm and a replayed one. Without this the CLI scores the member **unfitted** and the
   artifact reads `0/415, delta −325, p = 0.0000`.
2. `ensemble.py:65` — add `weights=` (and a `gated` mode) so the winning configuration is
   expressible, or register a small `fusion` predictor instead.

**The run** (after those edits; 415 rows, no server, no network):

```bash
python3 -m bench.harness.run \
  --dataset tool_routing \
  --arm floor_a='recorded|arm=floor_a' \
  --arm fusion='fusion|member=sklearn_text|vectorizer=tfidf_word|clf=logreg|rules=local_clf|weights=0.5' \
  --arm gate='fusion|member=sklearn_text|vectorizer=tfidf_word|clf=logreg|rules=local_clf|gate=3b_conf|t=0.38' \
  --reference floor_a \
  --cv 5 --cv-seeds 0,1,2,4,7 \
  --preregister /tmp/fusion-prereg.json \
  --out /tmp/fusion-2026-11-10
```

**Expected wall clock:** ~40 min of edits, then **≤4 min** for the run (TF-IDF + logistic on 415
short rows × 5 folds × 5 seeds is seconds; the 3B side is a replay). If it exceeds 10 minutes
something is wrong with the fold loop, not with the science.

**Pass/fail, written before the run** — `PASS` requires all four:

```
C1  delta >= +20 rows          # observed 27, 21, 24 across seeds; 20 is the bar, 3x the A/A floor of 1
C2  mcnemar_p < 0.01           # observed 0.0002..0.0046
C3  min over labels of ledger.by_label[*].net >= -3    # no class may be sacrificed
C4  delta > 0 in >= 4 of 5 fold seeds                  # the sign must not be a fold accident
```

`FAIL` on any of them, and the verdict is: the fusion is a recorded curiosity on one fold
assignment, not a result, and the plan reverts to Direction 1 alone.

---

## 8. Harness gaps — three blocking, five dishonest

The harness is good: 110 tests pass, the metric arithmetic is right, and it already replays the
published arms exactly. These are the gaps that would let a wrong number through, ranked by what
they cost.

**Blocking**

1. **`--cv` throws away the out-of-fold records** (`run.py:193-203`, `metrics.py:256`). The
   artifact keeps `oof_accuracy` and `fold_of` and no per-row choice, so *no* paired test between
   a CV'd classifier and a recorded arm is computable from it. This is the single biggest gap and
   it blocks the entire plan.
2. **A fitted arm is scored unfitted, and that pass owns the artifact** (`run.py:140-145`,
   `154-159`). Verified: a logreg arm reads `0/415, delta −325, p = 0.0000` in `summary.json`
   while stdout prints 57.6%.
3. **No `--split` / `--fit-on`** (`run.py:101-112`). `Dataset.slice` (`rows.py:81`) exists and is
   called only from a test. "train = 208 / test = 207" is not expressible, and `--cv` folds by
   `group` only, silently ignoring `meta["split"]`, so test rows train other folds. There is **no
   leakage guard anywhere.**

**Dishonest**

4. **`--order` is inert for every predictor that matters.** `protocol.py:131,135-136,151`
   permutes the options, then calls `predict_for(row)` which discards the permutation — while
   `order_sha` is computed *from* the permuted tuple. The predictor sees the frozen order and the
   record claims otherwise. It hits `recorded`, `llama_systemone` and `ensemble`. And for
   `sklearn_text`/`torch_text` the control is vacuous by construction: order never enters the
   feature space.
5. **`coverage_at_precision` emits a `threshold` that does not define the set it reports**
   (`metrics.py:194-199`): it returns the k-th confidence. Verified — it claims `cov 0.556 /
   prec 1.0 / t 0.95` where gating at 0.95 actually gives `0.667 / 0.833`. So **"0.44 coverage at
   threshold 0.74" is not an operating point**, and the 73.7%/78.3% pair has no threshold beside
   it that means anything.
6. **`gate_verdict` is vacuous for deterministic arms** (`controls.py:173-176`): floor 0 →
   `needed = 0` → a **1-row delta clears**. Verified. With a `recorded` reference and a `fake`
   arm, the A/A gate can never mean anything; McNemar is the only real test.
7. **`assign_folds` is unstratified and unchecked** (`metrics.py:220-226`): no guarantee a
   training fold contains every label, no reported `n_groups / n_labels`, and a 0%-accuracy fold
   is averaged in silently. 7 groups over 7 labels gives five folds at `0.0%` with no flag in the
   returned dict.
8. **Ensemble member failures are invisible** (`ensemble.py:400-409`): an unfitted vote ensemble
   reports `delta +0, p = 1.0000` with 40/40 member failures and nothing in `Record.meta`.
   `_fold_assignment` (`:334-342`) is round-robin, unstratified — the same wall as (7).

Two documentation errors worth fixing in the same pass: `docs/harness.md:171` says 78 tests (113
are collected; 110 pass, 3 skip) and `:139` says scikit-learn and torch are absent (both are
installed).

**Ranked fix order:** (1) write `cv["oof"]` and make it the arm's records — ~15 lines, kills gaps
1, 2 and half of 8. (2) `--split`/`--fit-on` over `meta[...]`, plus refusing a fold whose training
side is missing a label, reporting `n_groups/n_labels` — ~20 lines. (3) pass the permuted options
to `predict_for`, or refuse `--order` for predictors that ignore it, and stop printing a
`threshold` from `coverage_at_precision` — ~10 lines.

---

## 9. Ordered follow-ups

| # | item | why now | cost | depends on |
|---|---|---|---|---|
| 1 | Harness fixes (1) and (2) from §8 | nothing else is quotable without them | 40 min | — |
| 2 | **The §7 run** | the shipping decision | 4 min | 1 |
| 3 | Lift `local_clf.RULES` into a features spec and re-run the fusion with it | it is the member that pays (+27 vs +16) | 3 h | 1 |
| 4 | `weights=`/`gated` in `ensemble.py`, or a `fusion` predictor; delete `vote` for 2 members | the winner is not expressible today and `vote` is a no-op | 2 h | 1 |
| 5 | The 208-row 8-op corpus file + `test_classical_backend.py:142-146` + a `bench/arms.json` entry | the backend is unmeasurable while it abstains | 3 h | — |
| 6 | Fold-seed policy: report 3 seeds and `n_groups/n_labels` in every grouped-CV artifact | ±4 points of unreported noise today | 2 h | 1 |
| 7 | A **third** corpus, hand-written, no `frames.py` slot values, ≥150 rows, family-tagged | the only thing that can settle the fusion | 1 day | — |
| 8 | Frozen-MiniLM as an optional third member (clean on both corpora) | the only uncontaminated member | 2 h | 1 |
| 9 | Do **not** do: fine-tune; build object-noun lexicons; weight-of-sweep tuning; the temperature edit | §4, §5, §6 | 0 | — |

---

## 10. Appendix — every number, and how to reproduce it

Scratch scripts live under `/tmp/forward-plan/` and are **not** part of the repo; each row below
gives the command that reconstructs it. Read-only throughout.

**10.1 The recorded arms.** `python3 - <<'EOF'` over `bench/results/p3-transfer/rows.jsonl`:
`floor_a 325/415`, `floor_b 326/415` (1 flip), `grep_only 328` (10 fixed / 7 broke, p = 0.6291),
`grep_control 321` (10/14, p = 0.5413), `targeted 321` (16/20, p = 0.6177), `order_rev 316 (34
argmax flips, 15 broke / 6 fixed, p = 0.0784)`.

**10.2 The arm definitions.** `bench/rewrite_transfer.py:79-85` — `targeted` differs from
`OP_DESCRIPTIONS` in **2** of 8 keys, `grep_only` and `grep_control` in **1**, `order_rev` is the
same multiset in reversed key order.

**10.3 The cue table on v3.**
`python3 -c "import sys,json;sys.path.insert(0,'bench');import local_clf as lc;r=json.load(open('bench/data/heldout_v3.json'));p=[lc.rules_predict(x['utterance'])[0] for x in r];print(sum(1 for a,b in zip(p,r) if a and a==b['label']),len(r))"`
→ `259 415`; abstentions 91, accuracy among answered 259/324.

**10.4 Frozen MiniLM + logistic, declared split.** `/tmp/forward-plan/mine/encode.py` then
`frozen_probe.py`: embed 415 rows (6.7 s, `all-MiniLM-L6-v2`, mean-pooled, L2-normalised) → fit
`LogisticRegression(C=1.0)` on `split=="train"` → **163/207 = 78.74%** test; balanced family CV
**313/415 = 75.42%**, **308/415 = 74.22%**; hash folds **68.43 / 76.14 / 73.25%**.

**10.5 The fold-assignment spread.** Same script, three seeds each: hash folds
`68.43/76.14/73.25`; size-balanced folds `75.42/74.22/73.49`. TF-IDF word 1–2: `62.17%` on seed 0
(reproducing `docs/todos-2026-11-10.md`'s 62.2%), and 62.17–66.99 across seeds 0–4 with
`GroupKFold(5) = 71.81%` (D1).

**10.6 Error overlap and the mean, frozen-MiniLM member.** `/tmp/forward-plan/mine/ens415.py`.
2×2 on seed 0: `275 / 50 / 38 / 52`, φ = **0.406**; mean@0.5 **341/415, p = 0.0009**.

**10.7 The winning member and its fusion.** `/tmp/forward-plan/mine/cheap_member.py`:
TF-IDF word 1–2 (`sublinear_tf`) + the 8 normalised rule scores + `LogisticRegression(C=10)`,
OOF on size-balanced family folds → member **322/317/321 of 415**; mean@0.5 **352/346/349**
(p = 0.0002 / 0.0046 / 0.0007); gate@0.38 **345/344/346** (p = 0.0003 / 0.0009 / 0.0005,
13.7% escalation). Total CPU: 71 s for all three seeds.

**10.8 Transfer, clean member.** `/tmp/forward-plan/mine/transfer_v2.py` — fit on v3 `train` only,
score `tests/data/utterances_v2.json` (150 rows), join `bench/results/dell/v2-d1-3B-Q4.json`
`per_row` (150 rows, `conf` + `probabilities`): 3B 111/150 = 74.00%, member **103/150 = 68.67%**,
mean@0.5 **114/150 = 76.00%** (fixed 4 / broke 1, p = 0.375), gate@0.38 114/150 (p = 0.58).

**10.9 Transfer, contaminated member — recorded to be disowned.**
`/tmp/forward-plan/mine/cheap_transfer_v2.py`: member 124/150 = 82.67%, mean@0.5 138/150 =
92.00%. **Invalid**: `local_clf.RULES` was authored against v2.

**10.10 The frame split.** `/tmp/forward-plan/mine/slot_split.py` — 357/415 rows contain a
literal `tools.distill.frames.SLOTS` value (86.0%; v2 32.7%, v1 38.0%). Gains hold in both
halves (§3).

**10.11 The 3B's confidence is informative.** From `floor_a`: median 0.895 on correct rows,
0.395 on wrong rows, AUC **0.858**.

**10.12 The gate on the wrong variable.** `/tmp/forward-plan/mine/gate3b.py` — the classifier-
confidence gate never commits usefully (t = 0.20 commits 96.6% and loses); the 3B-confidence gate
at t = 0.38 chosen on `split=="train"` rows alone gives **339/415 = 81.69%**, fixed 21 / broke 7,
p = 0.0125 on two independent fold seeds.

**10.13 Direction-brief numbers reproduced, in the subagents' scratch:** D1
`/tmp/forward-plan/d1/` (rules-on-v3, slot-leak transfer, seed band, 84–85% feature result);
D2 `/tmp/forward-plan/d2/` (offline encoder load, C-sensitivity 76.81→81.64, ρ = −0.71 citation);
D3 `/tmp/forward-plan/d3/` (φ = 0.165, mean@0.5 356/415 with the tfidf_word_char member, strict
grouping 347/415, bootstrap CI [+0.000, +0.108]); D4 `/tmp/forward-plan/d4/` (cascade table,
`load_corpus` leak, 0.607/0.446/0.865 in-sample); harness `/tmp/forward-plan/harness/` (the four
verified defects at 4, 5, 6, 7 above).

---

## 11. What would make me reverse each verdict

| verdict | reversal condition |
|---|---|
| **D3 do** | The §7 run fails C1–C4; or a third corpus (follow-up 7) shows a delta ≤ +3 rows; or a live 3B re-run shows its confidences are no longer 0.395-median on wrong rows (the whole mechanism is that number). |
| **D1 cue half do** | A cue-feature block fails to beat plain TF-IDF by ≥3 points on a corpus that is neither v2 (where the rules were authored) nor v3 (which shares `frames.py` with the lexicons). |
| **D1 lexicon half drop** | A slot/object-noun feature block that beats TF-IDF by ≥5 points **on v2 transfer** — today it is −7.3. |
| **D4 accuracy drop** | A gate chosen out-of-fold that fires on ≥15% of rows and yields ≥83% on 415 rows with p < 0.05. Today: fires on 0 rows, delta 0.0000, p = 1.0. |
| **D2 drop** | A frozen encoder below 74.0% on a clean same-rows protocol, which would leave a real ≥6-point gap; or a selection criterion that ranks fine-tune checkpoints correctly (ρ = −0.71 today). |
| **harness fixes** | Someone shows a `--cv`-produced artifact from which a paired ledger against `recorded|arm=floor_a` is computable. It is not, today (`run.py:193-203`). |

---

## 12. Disagreements we did not settle

1. **Which member is best.** D3's `tfidf_word_char` + sigmoid reached 356/415 in the mean while my
   `tfidf + rules` reached 352 and my frozen MiniLM only 341 — yet all three are within 3.5 points
   as *members*. I could not run them head-to-head under one fold scheme in the time available,
   so "the best member" is open. The ordering is stable in the direction that matters (any decent
   member beats no member by 14–31 rows), not in the second decimal.
2. **How much of the 84–85% feature result is real.** D1 reports 84–85% family-held-out for
   word+char+rule+slot features; I reproduce 77.3–77.6% without the slot block. Both are correct
   for what they measure; neither is the "true" number, and the gap is exactly the slot leak that
   the frame split (10.10) exists to expose. Resolving it needs follow-up 7's corpus, not a rerun.
3. **`w = 0.5` versus a fitted weight, and gate versus mean.** The mean at 0.5 edges the gate on
   both members (352 vs 345 on rules, 341 vs 339 on MiniLM) — a difference inside the noise of
   either. The gate is interpretable (13.7% escalation, one scalar the 3B already reports); the
   mean needs one line and no threshold. I recommend the mean and would not argue with the gate.
   The **stacked** meta-learner was not fitted at all: its headroom over a weighted mean is
   unattested, and `docs/harness-spec-2026-11-10.md:163-165` already concedes that
   stacked-plus-`--cv` is double-nested with no separate accounting.
4. **Whether the 382 corpus should be re-grouped at all.** Harness gap 7 says grouping by family
   makes it unscorable; the spec's open question 5 offers "group by intent (18 groups)". I think
   that trades one defect for another (same-intent paraphrases then straddle), and the right answer
   is a template key that the file does not carry. Unsettled; unblocked by nothing in this plan.
5. **The 3B's confidence on a live server.** Every number in §3 uses `floor_a`'s **recorded**
   probabilities. The mechanism depends on the live readout producing the same shape. Nobody
   checked tonight, because nobody was allowed to start a server.
6. **Whether `docs/grep-arm-2026-11-10.md`'s guidance still holds.** It recommends a paired
   rewrite of `filesys_grep` + `web_search`. Nothing here contradicts it; nothing here tested it
   either. It stays queued behind the fusion work, not because it is wrong but because it is a
   description edit on a prompt whose readout is the layer that is already 78–85% accurate.
