# A local op classifier as a gate against a hosted fallback

2026-11-03. Measures a numpy-only 8-way op classifier under **grouped** CV on this
repo's frozen 150-row set, prices it against a hosted 95.3% fallback, and reports
what a local commit buys. Branch `bench/local-classifier` @ `c6b37ac`, adapted to
this tree (§9).

**Verdict: the local model is not a gate.** Its committed-row precision only
reaches the hosted accuracy at **2.7% coverage — 4 of 150 rows** — and every
coverage above that lowers blended accuracy. What survives is not the gate but
the *shape*: per-class routing, decided on training rows, gets **84.7%** against
the 3B's **74.0%**, and the local model's confidence is the reason the per-row
gate fails while the 3B's is the reason its gate works.

This document is deliberately **not** a correction to
[`docs/classical-baseline-2026-11-03.md`](classical-baseline-2026-11-03.md). It
measures the same 150 rows and the same 8 labels and asks a different question —
§1 — and its accuracy figure **corroborates** that document's rather than
competing with it.

## Contents

1. [What this measures, and how it differs from the classical baseline](#1-what-this-measures-and-how-it-differs-from-the-classical-baseline)
2. [The split, and the leak it removes](#2-the-split-and-the-leak-it-removes)
3. [The numbers](#3-the-numbers)
4. [Against the numbers already on this tree](#4-against-the-numbers-already-on-this-tree)
5. [The gate: local confidence does not discriminate](#5-the-gate-local-confidence-does-not-discriminate)
6. [What this settles](#6-what-this-settles)
7. [What this does not settle](#7-what-this-does-not-settle)
8. [Reproduce](#8-reproduce)
9. [Defects found merging this onto the tree](#9-defects-found-merging-this-onto-the-tree)

## 1. What this measures, and how it differs from the classical baseline

**The task is the same.** Both files classify `tests/data/utterances_v2.json`:
150 rows, **8 op classes** (`filesys_read`, `filesys_glob`, `filesys_grep`,
`decide_pick`, `decide_extract`, `web_search`, `web_fetch`, `reddit_digest`),
bands t1=64 / t2=31 / t3=55, no `__none__` class. This is op selection. It is
**not** intent routing, and it is not a different label space from the classical
baseline's v2 half.

**The question is not.** They differ in three ways, and the difference is why
both belong in `docs/`:

| | classical baseline (main) | this branch |
|---|---|---|
| decision it informs | should a linear model **replace the 3B** as the shipped `choice` backend? | how much traffic can a **free** local model take off a **paid hosted** model, and what does it buy per 1k decisions? |
| local model | the **production** backend — `ti.decide.backends.classical` over `service/ti/decide/linear_text.py` (TF-IDF + LinearSVC) | a **standalone** re-implementation: word 1–2 + char 3–5 tf-idf, multinomial softmax by full-batch GD, plus a char-ngram 1-NN and a keyword table as controls |
| comparator | **d1-3B Q4_K_M, 74.0%**, measured in-repo and recorded under `bench/results/dell/` | **`z-ai/glm-5.3-flash`, 95.3%**, an inherited constant with no probe in this tree (§4) |
| cascade shape | classical **in front of** the 3B, measured end-to-end with real latencies | local **commits else** hosted, computed algebraically from a coverage/precision curve |
| training protocol | transfer from the 50-row v1 set, plus CV within v2 | 5-fold grouped CV within v2, nested inner CV for hyperparameters |

So there are **two answers to two questions**, not two competing answers to one.
The one place they can be laid side by side is accuracy on the same 150 rows, and
there they agree: see §3 and §4.

## 2. The split, and the leak it removes

`docs/classical-baseline-2026-11-03.md` §4 learned this the expensive way: 93.19%
row-wise against **66.23%** when whole phrasing families are held out. A high
number with no stated split is the defect this repo already caught once, so this
is the first thing to check and it is stated in three places — the module
docstring, the printed report, and the commit message.

**The split is grouped, and it is stricter than main's v2 CV.** `build_groups()`
assigns group ids two ways:

- **t3 rows (55)** group by **boundary family** — the sorted `(label,
  tier_target)` pair the row was constructed inside, from both directions. 11
  families.
- **t1/t2 rows (95)** group by **single-link near-duplicate paraphrase cluster**
  within one label, char-3-gram Jaccard ≥ 0.25. 91 clusters (4 pairs merged,
  largest cluster 2).

102 groups over 150 rows, folded 5 ways at 30 rows each, stratified by label with
the largest groups placed first. Hyperparameters are chosen by a **3-fold inner
grouped CV on the training portion of each outer fold**, so the test fold is never
consulted for model selection.

Main's v2 CV grouped only the **t3 twin families**. This file groups t1/t2
paraphrase siblings too, and that is what the leak audit shows it buys:

| split | group straddle | mean max Jaccard | median | same-label dup ≥ 0.25 |
|---|---:|---:|---:|---:|
| grouped | **0.0%** | 0.20 | 0.19 | **9%** |
| random | **42.0%** | 0.21 | 0.19 | **16%** |

Same pipeline, same model, random rows instead of groups: **69.3%** (104/150).
Grouped: **61.3%** (92/150). The **8.0-point** difference is the leak, and it is
reported as a number in the table above rather than asserted. The corresponding
figure for main's shallower grouping was 2.0 points (62.0% random vs 60.0%
grouped) — not a contradiction, a consequence of holding out more. On this set the
t1/t2 paraphrase siblings carry roughly three times the leak the t3 twins do.

## 3. The numbers

All 150 rows, 8-way, 5-fold grouped CV, inner 3-fold for hyperparameters.
Wilson 95% intervals.

| method | pooled | 95% CI | t1 (64) | t2 (31) | t3 (55) | fold min..max |
|---|---:|---:|---:|---:|---:|---:|
| **lr (softmax, nested)** | **61.3%** (92/150) | [53.3, 68.8] | 54.7% | 58.1% | **70.9%** | 40.0..73.3 |
| 1-NN (char 3–5) | 50.0% (75/150) | [42.1, 57.9] | 45.3% | 51.6% | 54.5% | 36.7..60.0 |
| rules (keyword) — **contaminated, see §7** | 78.7% (118/150) | [71.4, 84.5] | 85.9% | 67.7% | 76.4% | 73.3..93.3 |
| lr, **random** split (control) | 69.3% (104/150) | — | — | — | — | — |
| d1-3B Q4_K_M (recorded, not re-run) | 74.0% (111/150) | [66, 80] | — | — | — | — |
| `glm-5.3-flash` (asserted reference, §4) | 95.3% (143/150) | — | — | — | — | — |

Per label — the column that carries the result:

| label | n | lr | 1-NN | rules | d1-3B | Δ lr−3B |
|---|---:|---:|---:|---:|---:|---:|
| filesys_read | 17 | 64.7% | 47.1% | 82.4% | 70.6% | −6 |
| filesys_glob | 21 | 66.7% | 61.9% | 71.4% | 95.2% | −29 |
| **filesys_grep** | 22 | 50.0% | 50.0% | 72.7% | 36.4% | **+14** |
| decide_pick | 18 | 77.8% | 38.9% | 94.4% | 100.0% | −22 |
| **decide_extract** | 24 | 79.2% | 70.8% | 83.3% | 25.0% | **+54** |
| web_search | 18 | 38.9% | 22.2% | 72.2% | 100.0% | −61 |
| web_fetch | 15 | 46.7% | 46.7% | 73.3% | 93.3% | −47 |
| reddit_digest | 15 | 60.0% | 53.3% | 80.0% | 100.0% | −40 |

The local model wins exactly two classes and loses six, and it wins on the two
where the ask **carries its own vocabulary** (`decide_extract` wants phrasings and
keywords named in the utterance; `filesys_grep` mentions callsites and symbols).
It is destroyed on `web_search` (−61) and `web_fetch` (−47), where the op is
implied by the target rather than stated. That is the same shape main already
recorded — a lexical model is a boundary-pair specialist and an intent-veiling
failure — reproduced by an independently written model.

Latency (indicative, single-shot on a shared laptop, 5763 features):
**p50 0.22 ms**, p95 0.61 ms, batch 0.343 ms/row, against 588 ms asserted for the
hosted model. Single-shot timings at this scale move 3× between runs — the first
run of this exact script gave p50 0.24 ms and p95 0.29 ms — so quote the scale
(~0.2–0.3 ms), not the digit.

## 4. Against the numbers already on this tree

Every figure was reconciled against `main`. **No figure on `main` is
contradicted.** Two need care:

- **d1-3B 74.0% — agrees exactly.** 111/150, matching
  `docs/model-sweep-consolidated.md` and
  `docs/classical-baseline-2026-11-03.md` §5. This column is read from
  `bench/results/dell/v2-d1-3B-Q4.json`, which is on this tree.
- **`glm-5.3-flash` 95.3% — new, and unbacked.** The string `glm-5.3-flash`
  appears nowhere on `main`: not in `docs/`, not in `service/`, not in
  `bench/results/`. The script says so itself — *"reference points supplied with
  the task, not re-derived here."* So 95.3%, `$0.0221`/1k decisions,
  `$0.0232`/1k correct and 588 ms p50 are **asserted constants**. Every cascade
  and break-even number in this document inherits them. If 95.3% is wrong, the
  sign of §5 inverts: at a 74.0% fallback the same local model is a *good* gate.
  That sensitivity is the largest unquantified risk here, and it is worth stating
  because the repo's own rule in
  [`docs/confidence-sources.md`](confidence-sources.md) is that an **asserted**
  number is a suspect and must carry its provenance. It now does.
- **`61.3%` is numerically adjacent to `61.34%` on `main` and unrelated.**
  `docs/assets-inventory.md` records needle3 at 61.34% over **357 rows** on the
  **382-row, 18-way** set. Different model, different task, different n. Do not
  conflate them.
- **Main's v2 CV grouped was 60.0%; this file's is 61.3%.** Different models
  (production LinearSVC vs this standalone softmax) over the same rows, with
  different fold construction. Two local methods landing at ~60–61% is
  corroboration, and the 62.7%-transfer figure from main sits in the same band.
  Three protocols, 60–63%, now measured three ways.

## 5. The gate: local confidence does not discriminate

The economics are arithmetic, not measurement, and the algebra is the part that
generalises. Local inference is free, so a commit only helps when
`precision > hosted_accuracy`:

```
blended_acc − hosted_acc = coverage × (precision − 0.953)
```

Measured frontier — rows taken in order of the model's own max probability, out
of fold:

| coverage | n | precision | 95% CI | blended | Δ acc | wrong/1k |
|---|---:|---:|---:|---:|---:|---:|
| 5.3% | 8 | 87.5% | [53, 98] | 94.9% | **−0.4** | 6.7 |
| 10.0% | 15 | 86.7% | [62, 96] | 94.4% | −0.9 | 13.3 |
| 14.7% | 22 | 86.4% | [67, 95] | 94.0% | −1.3 | 20.0 |
| 20.0% | 30 | 83.3% | [66, 93] | 92.9% | −2.4 | 33.3 |
| 25.3% | 38 | 86.8% | [73, 94] | 93.2% | −2.1 | 33.3 |
| 30.0% | 45 | 84.4% | [71, 92] | 92.0% | −3.3 | 46.7 |
| 50.0% | 75 | 81.3% | [71, 89] | 88.3% | −7.0 | — |
| 100% | 150 | 61.3% | [53, 69] | 61.3% | −34.0 | — |

**Every cell is negative except 4 rows.** The largest coverage whose committed
precision still clears 95.3% is **2.7% (4 rows)**, and at n=4 the interval is
[51, 100] — that cell certifies nothing.

Two things are wrong with the confidence itself, not just the accuracy:

1. **It barely lifts.** Pooled 61.3% → 86.7% at the top 10%. Compare the 3B,
   whose max probability goes 74.0% → **97.6%** at 28% coverage.
2. **It is not monotone.** At 20% coverage precision is 83.3%; adding five more
   rows at 25.3% coverage *raises* it to 86.8%. A confidence that ranked its own
   correctness could not do that.

The 3B's curve, same rows, for contrast — this is the only gate here with a
margin:

| τ | coverage | n | precision | 95% CI | blended |
|---|---:|---:|---:|---:|---:|
| 0.80 | 47.3% | 71 | 91.5% | [83, 96] | 93.5% |
| **0.90** | **28.0%** | **42** | **97.6%** | [88, 100] | **95.9%** |
| 0.95 | 12.7% | 19 | 100.0% | [83, 100] | 95.9% |

`+0.6` points at 28% coverage is not a result on 42 rows — the CI is [88, 100] —
but the *ordering* is: at matched coverage the 3B beats the local model at every
point (10%: 100.0 vs 86.7; 50%: 90.7 vs 81.3), and its curve is monotone.

The per-class device is the one that works where the per-row gate does not.
Selection decided **on the training portion of each fold** (never the scored
rows):

| routing | accuracy | n |
|---|---:|---:|
| lr alone | 61.3% | 92/150 |
| d1-3B alone | 74.0% | 111/150 |
| per-class choice, static prior table (no fitting) | 84.0% | 126/150 |
| **per-class choice from training portion (nested)** | **84.7%** | **127/150** |
| per-class choice from pooled OOF (optimistic bound) | 84.7% | 127/150 |
| oracle union lr \| 3B | 92.7% | 139/150 |
| oracle union lr \| 3B \| rules | 97.3% | 146/150 |

The nested choice keeps **57%** of the `lr|3b` oracle gap over the 3B alone. The
static table written from prior knowledge and the fitted selection agreeing on
**8 of 8** classes is the only reason to trust the shape at 15–24 rows per class;
a class-level difference under ~15 points is noise here and 3 of 8 classes flip on
a single row.

## 6. What this settles

- **A third independent local method lands at ~61% on this set.** 61.3% grouped,
  against main's 60.0% grouped and 62.7% transfer. The band shape reproduces too:
  t3 is the best band (70.9%), matching the classical baseline's 69.1% and the
  observation that a 0.2 ms lexical model beats the 3B on boundary pairs.
- **The leak on this set is 8.0 points, and it is not only in t3.** Main's v2 CV
  held out t3 twin families and found 2.0 points; holding out t1/t2 paraphrase
  clusters as well finds 8.0. Any future v2 CV here should group both.
- **The break-even algebra.** `Δacc = coverage × (precision − hosted)`, and with
  local inference free the cost side never binds — precision does. This is
  arithmetic and holds for any dataset.
- **Gate quality is a property of the model, not the features.** A model whose
  max probability discriminates can front a cascade; one whose does not cannot,
  at any threshold. 3B: 74.0% → 97.6%. Local: 61.3% → 86.7%, non-monotone.
- **Per-class routing is the right device for this pair.** 84.7% nested against
  74.0%, with the local model contributing exactly the two classes its features
  suit (`decide_extract` +54, `filesys_grep` +14).
- **How many rows it would take to certify a gate.** One-sided binomial,
  α=0.05, power 0.8, against a 95.3% baseline: 96% precision → **5,372**
  committed rows; 97% → 837; 98% → 298; 99% → 137; 99.5% → 95. At the measured
  operating point ~59 committed rows per 400 live rows, so certifying this gate
  is a large eval, not a spot check.

## 7. What this does not settle

1. **This is not a measurement of live traffic.** 150 rows, one author, written
   adversarially against a specific 3B.
2. **The 95.3% hosted reference is asserted, not measured** (§4). Everything in
   §5 is conditional on it. The repo has no probe of `glm-5.3-flash` on these
   rows.
3. **The rules row is contaminated and is not evidence.** The cue table was
   written after reading these 150 utterances, so 78.7% is partly the author
   having seen the test data. It is in the table as a reference for what a hand
   table reaches on the set it was written against, and for nothing else. Its
   `filesys_read` 82.4% and `decide_pick` 94.4% are memorisation.
4. **The comparison is asymmetric.** v2 was built from the **3B's** misses, so
   the 3B column is pessimistic by construction and the local columns are the
   most flattered — features, hyperparameter grid and cue table were all chosen
   with these 150 rows in view.
5. **No specific coverage number survives.** 14.7% at a 2-point budget is a
   property of 150 rows; live traffic moves both coverage and precision in a way
   that is not modellable from here.
6. **Committed-row error is unpriced.** A wrong local commit runs a wrong tool
   on the user's machine. `wrong/1k` counts it; no column values it.
7. **Band cells are not comparable to the classical baseline's.** Main's v2 bands
   come from transfer out of a 50-row v1 set; these come from 5-fold CV with 120
   training rows per fold. Same rows, different protocol, so the t1/t2 gap
   between the two tables is training-set size, not model quality.
8. **Hyperparameters were chosen by an inner grouped CV, and the grid is one
   author's.**   `min_df ∈ {1,2}` × `l2 ∈ {3e-4 … 1e-2}`; five outer folds chose
   three distinct configs and one fold's inner accuracy was 0.575, which is itself
   evidence the surface is flat rather than peaked.

## 8. Reproduce

```
python3 bench/local_clf.py --json /tmp/local_clf_v2.json    # every number above
```

**The `--json` flag is not optional if you want to keep the recording.**
`--json` defaults to `bench/results/local_clf_v2.json`, so a bare run overwrites
the artifact that holds the numbers being reproduced — the same hazard
`docs/classical-baseline-2026-11-03.md` §10 documents for `classical_baseline.py`.
This reproduction was run with `--json /tmp/…` and the recorded file regenerated
deliberately (§9).

Runtime ~80 s on this box (all outer folds, the inner grid, three models,
and 430 latency draws). Every accuracy, frontier and break-even figure is
bit-identical across runs; only the latency rows move (p50 0.22 / 0.30 ms on two
runs of the same code), which is why §3 quotes the scale.
Dependencies: stdlib + numpy. No network, no service imports.

## 9. Defects found merging this onto the tree

The branch was written at `c6b37ac` in a different worktree and did not run on
`main` HEAD. Three defects, all fixed here; none changes what is measured, and
the regenerated artifact differs from the branch's only in the fields named.

- **It crashed.** `--heldout` defaults to `tests/data/heldout_phase3.json`, which
  did not exist where this was authored (the recording says `"reason": "file
  absent"`). On this tree the name **is** taken — by a **76-row shape** set
  (`singular`/`enumerative`, keyed `expect`/`family`, per
  `docs/dataset-status.md`), so the held-out block raised
  `KeyError: 'label'` after ~40 s of work. The block now validates that the rows
  carry this file's 8-way op labels and reports
  `heldout_phase3.json carries no 8-way op labels` instead of scoring a different
  task's data. **The held-out measurement was never part of this result** and
  still is not.
- **It recorded two machine-personal absolute paths.** `bench/results/local_clf_v2.json`
  held a machine-personal absolute path (`/home/<user>/…`) twice — for `data.path` and
  `heldout.path` — and
  `tests/test_no_personal_paths.py` failed on exactly those two, which is how this
  was found. The root cause was the recorder: `str(args.data)` absolutises
  whatever worktree it runs in, so it would re-leak on every run. A `repo_rel()`
  helper now records the path repo-relative when it is inside the tree and the
  bare filename otherwise.
- **The artifact was stale.** It was regenerated on this tree. Against the
  branch's copy the only differences are the two paths, the held-out `reason`,
  and three latency values (p50 0.244→0.220 ms, p95 0.289→0.606 ms,
  0.309→0.343 ms/row — the 3× single-shot noise this repo already documents).
  **Every measured and every analytic figure is identical.**

Artifact: `bench/results/local_clf_v2.json` — CV with intervals and per-band
cells, the leakage audit for both splits, the local and 3B coverage/precision
curves, the frontier and budget ladder, per-band cells at the operating point,
the per-class hybrid, `rows_to_certify`, latency, and the held-out disposition.
