# The classical routing baseline, measured in-repo

2026-11-03. Reproduces the sibling benchmark's TF-IDF + linear-SVM result as a
real `choice` backend, checks what that number is worth, and answers the
question it raises: should a linear model replace the 3B as the shipped
`choice` backend?

**Verdict: no — not yet.** The recorded 93.19% is real, reproduces to the row,
and is genuinely held out. But it was measured on a task this repo does not
have, and on this repo's own frozen set the same model scores **62.7% against
d1-3B's 74.0%**. What the evidence does support is the linear model as a *front
gate*: measured on the same 150 rows, `classical` in front of d1-3B gives
**79.3% at 95 ms mean** against d1 alone at **74.0% at 169 ms** — a direction
worth chasing, at 16 rows fixed against 8 broken, not yet a result.

| | recorded 2026-10 | in-repo 2026-11-03 |
|---|---:|---:|
| TF-IDF + LinearSVC, 382 rows, 18-way | 93.19% | **93.19%** |
| none-detection | 0.60 | **0.60** |
| false-none rate | 0.00 | **0.00** |
| p50 latency | 0.61 ms | **0.04–0.14 ms** (run-dependent) |
| d1-3B Q4_K_M, same rows | 91.36% | — (not re-run) |
| v2 set, 150 rows, 8-way | — | **62.7%** (d1-3B **74.0%**) |

## 1. What was built, and why it is not scikit-learn

`python3 -c "import sklearn; print(sklearn.__version__)"` raises
`ModuleNotFoundError` on this box (Python 3.10.12), and `pyproject.toml` declares
`dependencies = []` with no third-party imports anywhere under `service/`. So
scikit-learn was **not** added; the vectoriser and the solver were re-implemented
in stdlib Python and the recorded hyper-parameters were copied exactly.

| file | what it is |
|---|---|
| `service/ti/decide/linear_text.py` | `TfidfVectorizer` (word 1–2-grams, `min_df=1`, sublinear tf, smoothed idf, L2 rows — sklearn's defaults) and a one-vs-rest squared-hinge linear SVM trained by SGD with Polyak averaging and an `eta0/(1+eta0*l2*t)` schedule. `log` loss (multinomial logistic) is the alternative. ~300 lines, stdlib only. |
| `service/ti/decide/backends/classical.py` | The `choice` backend. Fits lazily from a corpus resolved out of `decide.classical.corpus_path`, caches per (config, corpus), reports a full `probabilities` dict and `confidence`. |
| `service/ti/conf/defaults.json` | `decide.classical` — corpus path, loss, `C`, epochs, step, n-grams, seed, abstain label, temperature. Registration is by config + the package import in `backends/__init__.py`; `primitives.make_backend` was not touched. |
| `tests/data/routing_382.json` | The 382 rows, verbatim. |
| `bench/registry_382.json` | The 17-tool + `none_route` registry, verbatim — this is the **training** corpus the recorded number was fitted on. |
| `tests/data/routing_382_PROVENANCE.md` | Origin, digests, label distribution, the exact command. |
| `bench/classical_baseline.py` | Every number in this document. Writes `bench/results/classical_382.json` and `classical_v2.json` — *by default, over the recorded copies*; pass `--out-dir` unless replacing them is what you mean. See §10. |

**Probabilities are a softmax of decision values, not `predict_proba`.** A linear
SVM's margins are uncalibrated, so the distribution is the softmax of those
margins over the ballot (the caller's `criteria` plus the fitted abstain class)
at `temperature = 1.0`, and `confidence` is its peak — because `policy.gate_route`
is expressed in p_max units (`policy/confidence.py`). Section 5 reports what that
does to the gate, which is not pretty.

**Abstention is learned, not bolted on.** `__none__` is a real class in the
corpus, as in the original. It is on the ballot of every call even though callers
never list it in `criteria`; when it wins, `choice` is `None` — the same "no
answer" `resolve.FastPathBackend` and `cascade` already branch on. none-detection
therefore means what the original meant: of the utterances that really are
off-scaffold, how many route to `__none__`.

## 2. The reproduction is the same decision function, not a similar score

Fit on the registry's 200 references (18 classes), scored on the 382 rows.

| metric | recorded | in-repo | n |
|---|---:|---:|---:|
| accuracy | 93.19% | **93.19%** | 356/382 |
| none-detection | 0.60 | **0.60** | 15/25 |
| false-none rate | 0.00 | **0.00** | 0/357 |
| p50 latency | 0.61 ms | 0.04–0.14 ms | 382 |
| p95 latency | 1.13 ms | 0.05–0.18 ms | 382 |

Two independent checks say this is the same model rather than a coincidental
accuracy:

- **Per-class accuracy is identical to three decimals** on all five worst classes:
  `__none__` 0.60, `light_query` 0.80, `projects_query` 0.833,
  `thermostat_offset` 0.87, `light_set` 0.92.
- **The top five confusions are identical in class and count**:
  `light_query→light_set` 5, `__none__→light_query` 4,
  `projects_query→shoppinglist_view` 3, `thermostat_offset→light_set` 3,
  `light_set→light_query` 2.

The p50 gap is measurement, not speed: the recorded 0.61 ms is
`clf.predict(vec.transform([txt]))` through sklearn, this is the same transform
plus one 18-class dot product in a plain loop. Single-shot timings at this scale
move by 3x between runs on this box, which is why the row is a range, and both
numbers sit far below the per-hop overhead that matters (`decide.cache` hits cost
~50 ms). Latency here is single-shot per row on the local box; it is not a
throughput claim.

The stability check, because a single accuracy can be solver luck: **18 fits —
6 (epochs, eta) budgets x 3 seeds — all land on 355 or 356 of 382**, and all 18
give none-detection exactly 15/25 and false-none exactly 0. 356/382 is the
objective's answer, not the schedule's. Shipped config: `C=1.0`, `epochs=100`,
`eta=0.3`, `seed=42`, model fingerprint
`2eff58ac9806fd8b25bd4d9e32582569fa518811dc8dba09c52a0a4fa37f09a3`.

## 3. Was the original 93.19% in-sample? No.

This was the leading hypothesis for a suspiciously high number, and the check
falsifies it.

`exp_classic.py` fits on `registry["tools"][*]["references"]` +
`registry["none_route"]["references"]` — **200 strings** — and scores
`data/test.json` — **382 rows**. `gen_data.py` builds those 382 rows by expanding
registry *templates* over slot vocabularies and explicitly dropping any utterance
that is verbatim a reference. Confirmed on the imported copies:

- verbatim train/test overlap: **0 of 382**;
- in-sample contrast: fit on the 382 and score the same 382 → **382/382 = 100.0%**,
  so a genuine in-sample run on this data would not have printed 93%.

The number is a transfer measurement. It is nonetheless transfer *to slot
fillings of phrasings an author already wrote references for*, which is a weaker
claim than "routes new phrasing" — see the next section.

## 4. What 93.19% actually measures

| protocol | accuracy | n | reads as |
|---|---:|---:|---|
| references → 382 (**the recorded protocol**) | **93.19%** | 356/382 | generalises across phrasings |
| 5-fold on the 382, random rows | 92.41% | 353/382 | same thing, with in-distribution supervision |
| 5-fold on the 382, **template-grouped** | **66.23%** | 253/382 | generalises **badly** to a phrasing family it has not seen |

110 groups (109 templates + the 25 hand-written `__none__` rows, which have no
template and so are atomic in the split). Holding out whole templates costs
**26 points**. That is the number to quote next to 93.19% whenever the deployment
question is "will it route something phrased differently", because the 382 rows
are one authoring pass of one template set and the references were written
against those same templates.

A softer view of the same thing — TF-IDF cosine from each test row to its
nearest training reference — says the score is *not* surface lookup:

| nearest-reference cosine | n | accuracy |
|---|---:|---:|
| < 0.80 (far) | 357 | 92.72% |
| ≥ 0.80 (near) | 25 | 100.0% |

Median nearest-reference cosine is 0.578 and only 3 rows sit at ≥ 0.95, so 93% is
carried by 357 rows that share little surface form with any reference. The
model learned the template *skeletons* ("set a timer for …", "cross … off the
shopping list"), which is exactly what the grouped CV then shows it cannot
transfer to a new skeleton.

## 5. This repo's own task: the linear model loses

`tests/data/utterances_v2.json` is 150 rows, 8 op classes, banded t1/t2/t3. It
has **no train split** — it is the frozen eval set for `bench/live_probe.py` and
`bench/band_summary.py` — so the classical model was trained on the only
disjoint in-repo corpus, `tests/data/utterances.json` (50 rows, 8 classes, zero
verbatim reuse in v2, per `utterances_v2_NOTES.md`). Candidates were the same
eight `OP_DESCRIPTIONS` d1-3B sees.

| band | classical (train v1, 50 rows) | d1-3B systemone | delta |
|---|---:|---:|---:|
| t1 unambiguous (64) | 67.2% | 81.2% | −14.0 |
| t2 veiled (31) | 41.9% | 71.0% | −29.1 |
| t3 boundary pairs (55) | **69.1%** | 67.3% | **+1.8** |
| **overall (150)** | **62.7%** (94/150) | **74.0%** | **−11.3** |

Starvation is not the explanation: 5-fold CV *within* v2, with 120 in-distribution
training rows per fold, gives 62.0% (random rows) and 60.0% (boundary-pair grouped
— each t3 twin family held out whole). Three protocols, 60–63%.

The shape is worth keeping: the linear model **wins t3**, the deliberately
confusable boundary pairs where a 3B is supposed to be the instrument and a
keyword heuristic is supposed to die, and it **collapses on t2**, where intent is
stated without any op vocabulary. A lexical classifier is a boundary-pair
specialist and an intent-veiling failure. `web_search` is its worst class at
44.4%.

## 6. Two things that disqualify it as a drop-in default

**The probabilities never arm the gate.** Raw margin softmax at T=1.0 is far too
flat: p_max median is **0.171** on the 18-way ballot and **0.193** on the 8-way,
its maximum is **0.392** and **0.330**, and only 3–4% of rows clear 0.30 at all.
Against the shipped `policy.gate_route = 0.70` its **coverage is 0.0% on both
sets** — behind the real gate the backend would commit to nothing and every hop
would escalate. That is a temperature-calibration problem, not an accuracy problem,
and it cannot be fixed
without a held-out calibration set this repo does not have. Note d1-3B reports
confidences around 0.9 by construction, which is why it clears the same gate.

**It misses 40% of the abstain cases.** none-detection is 0.60 against d1-3B's
**1.00** on the same 382 rows. Ten of 25 off-scaffold utterances get a tool call,
and for a `choice` behind a state-mutating hop that is the expensive direction of
error. d1-3B captured all 25.

## 7. What the classical model *is* good for — measured, not argued

Used as a front gate in front of d1-3B on the same 150 v2 rows (rows above the
gate never reach the 3B; the rest take d1's recorded per-row answer from
`bench/results/dell/v2-d1-3B-Q4-systemone-props-asserted*.json`, pooled over both
runs):

| gate (p_max) | cascade accuracy | gate coverage | mean ms | rows fixed / broken |
|---|---:|---:|---:|---:|
| off (d1 alone) | 74.0% | 0% | 169 | — |
| ≥ 0.15 | 65.3% | 96% | 8 | 26 / 39 |
| ≥ 0.18 | 74.0% | 67% | 57 | 21 / 21 |
| **≥ 0.20** | **79.3%** | **44%** | **95** | **16 / 8** |
| ≥ 0.25 | 76.0% | 15% | 143 | 4 / 1 |
| ≥ 0.30 | 74.7% | 4% | 163 | — |

"Fixed / broken" counts rows on which the cascade is right where d1 was wrong,
and vice versa. At ≥ 0.20 that is 16 against 8: exact McNemar two-sided
**p = 0.152** at n = 150, identical in both d1 runs. From 0.20 upward the cascade
is never worse than d1 and 2–5 points better, while mean latency falls to roughly
half.

But **do not read +5.3 points as established on 150 rows**, and the threshold was
chosen by looking at this set — which is why the whole sweep is printed rather
than one row. At 0.15 the cascade is clearly *worse* than d1, so the sign of the
answer depends on a threshold nobody has fitted. The honest framing: the
classical model is a plausible *reject option* for the front of a cascade at
around 40% of traffic, worth a properly sized eval (±8 points at n=150 is not
enough to settle it; ~1,960 paired rows would be, per the arithmetic already in
`docs/assets-inventory.md`).

## 8. Verdict on the shipped default

**Do not change `service/ti/conf/defaults.json`.** `backends_in_use.choice`
stays `pair` (stage 1 = `d1_systemone`). The case for swapping it rests on the
382-row table, and that table measures a different task:

- the 382 set is **18-way with a learned `__none__` class**, generated from
  templates, single-sentence home-automation commands. The linear model wins
  there on both accuracy (93.19 vs 91.36) and latency (0.04 vs 242.7 ms);
- v2 is **8-way with no `none` class**, and its t1/t2/t3 bands were authored to
  defeat exactly the surface-keyword strategy a TF-IDF classifier is. The linear
  model loses there by 11.3 points.

Those two facts are not in tension; they are two different questions, and only
one of them is this repo's. What the 382-row result *does* establish is that the
premise "a neural decision model is required behind `choice`" is **overstated for
a well-specified closed label set with in-vocabulary phrasing** — the 3B is not
earning its 242 ms on that task. It also establishes something sharper and
uncomfortable: on the *hardest* band of this repo's own set (t3, boundary pairs)
a 0.04 ms linear model is 1.8 points **better** than the 3B. That is the result
worth chasing, not a default swap.

Recommended next step, in order: (1) a temperature fitted for `classical` so its
p_max is on the gate's axis at all, plus a threshold fitted on held-out rows,
(2) the same measurement at n ≥ 500 paired, (3) only then a default decision.

## 9. Caveats, stated plainly

1. **Different tasks.** The 382-row set and `utterances_v2.json` have different
   class counts (18 vs 8), different label spaces (`__none__` exists only in the
   first) and different authoring intent (template expansion vs hand-written
   adversarial bands). Every cross-set comparison in this document is labelled;
   none is a like-for-like ranking.
2. **93.19% is template transfer, not novel phrasing.** The same protocol is
   66.2% with whole templates held out. Anyone quoting 93.19% must quote that.
3. **The original number is held-out by row**, and this document shows why
   (zero overlap; in-sample would be 100%). The suspicion that it was in-sample
   did not survive the check.
4. **n is small.** 382 rows → ±2.6 points at 95% for 93.19%; 150 rows → ±8 points,
   so the whole d1-vs-classical v2 gap and the cascade gain are single-sample
   estimates.
5. **The cascade threshold was selected on the evaluated set.** Section 7's table
   is a sweep for shape, not an unbiased estimate. 0.20 is where this set peaks;
   a fitted threshold would not be the argmax of its own test data.
6. **The temperature is not calibrated.** T=1.0 is the raw margin softmax. That
   is why p_max tops out near 0.39 and the shipped 0.70 gate commits to nothing.
7. **The shipped seed is one of a family.** Determinism is per (config, seed);
   18 fits spanning 6 budgets x 3 seeds land on 355–356 of 382, so the reported
   number is representative but not the only value the config can produce.
8. **The `__none__` class is atomic in the template-grouped CV** (its 25 rows have
   no templates), so that fold structure tests the none class only once.
9. **Latency here is not comparable to the recorded one across implementations.**
   Reported as measured; the recorded 0.61 ms includes sklearn's per-call
   overhead, and neither number is a throughput claim.
10. **d1-3B was not re-run.** Its numbers are the recorded ones
    (`docs/model-sweep-consolidated.md`, `bench/results/dell/`), used unchanged.

## 10. Reproduce

```
python3 bench/classical_baseline.py --out-dir /tmp/cls   # every number above
python3 -m pytest tests/test_classical_backend.py -q     # the contract
```

`--out-dir` is not decoration: without it the script rewrites the two artifacts
under `bench/results/` that hold the recorded numbers (see below).

**At `ca7ffec` — the HEAD this section was written against, 2026-11-03 15:41 EST —
the first command did not run at all.** It exited with

```
json.decoder.JSONDecodeError: Invalid \escape: line 8 column 16 (char 193)
  File "bench/classical_baseline.py", line 332, in main
    d1_runs = [json.loads(p.read_text()) for p in sorted(
```

because 37 of the 40 result files under `bench/results/dell/` are **invalid JSON** — every
one that carries a `model` block; the three `heuristic_micro` files, which carry none, still
parse. Commit `e523336`
redacted the personal prefix to `C:\Users\<user>` by string substitution and left the
remaining backslashes unescaped, so every one of them fails `json.loads` at its
`model.path` line. The 382-row half is written before line 332, so `classical_382.json`
is produced; the v2 half and the cascade sweep are not, and `classical_v2.json` is never
written. That is a defect in the recorder→loader chain, not in the numbers — see the
verification below, which drives this file's own functions.

Re-verified 2026-11-03 15:41 EST from worktree `p2-numbers` @ `ca7ffec`, with the escaped
backslashes repaired **in memory** (`bench/results/repro2-2026-11-03/classical_382.json`,
`classical_v2.json`):

| figure | published | re-derived |
|---|---:|---:|
| 382-row accuracy | 93.19% | **93.19%** (356/382) |
| none-detection | 0.60 | **0.60** (15/25) |
| false-none | 0.00 | **0.00** (0/357) |
| template-grouped CV | 66.23% | **66.23%** (253/382) |
| random-row CV | 92.41% | **92.41%** (353/382) |
| in-sample on 382 | 100.0% | **100.0%** (382/382) |
| model fingerprint | `2eff58ac…` | `2eff58ac9806fd8b25bd4d9e32582569fa518811dc8dba09c52a0a4fa37f09a3` |
| v2 transfer, v1-trained | 62.7% | **62.7%** (94/150) |
| v2 bands t1/t2/t3 | 67.2 / 41.9 / 69.1 | **67.2 / 41.9 / 69.1** |
| v2 CV grouped / random | 60.0 / 62.0 | **60.0 / 62.0** |
| d1-3B recorded, both runs | 74.0% | **74.0%** (111/150 × 2) |
| cascade at gate ≥ 0.20 | 79.3% at 44%, 95 ms | **79.3%** at 44%, **93.9 ms**, 16 fixed / 8 broken |
| p50 latency, 382 rows | 0.04–0.14 ms | **0.027 ms** (p95 0.033) |

The 382-row latency row is the only published figure that moved: 0.027 ms sits below the
recorded 0.04–0.14 ms band, which is the same single-shot timing noise this document
already warns about in §2, in the same direction. Nothing else moved by more than the
0.1 ms rounding on the cascade's mean.

Artifacts: `bench/results/classical_382.json` (reproduction, in-sample contrast,
both CVs, reference-distance buckets, coverage curve, config, model fingerprint)
and `bench/results/classical_v2.json` (v1→v2 transfer, per-band, both CVs,
coverage curve, the d1 cascade sweep).

### The crash is closed — status 2026-11-03 16:20 EST, at `a105d50`

**The `JSONDecodeError` above no longer reproduces, and the paragraph describing
it is a record of one afternoon rather than a property of the tree.** Commit
`7113ef8` (2026-11-03 16:01:17, "chore: close the path leak the guard could not
see") rewrote every recorded `model.path` — the half-redacted Windows form
`e523336` left behind — to `<workspace>\models\<model>.gguf`, which is portable
*and* valid JSON. Measured on this worktree:

- all **61** result files under `bench/results/` parse with `json.loads`. That
includes the 37 of the 40 named above, the three `props/*.json` beside them (also
invalid at `ca7ffec`, and outside the glob the count was taken over), and the three
`heuristic_micro` files that always parsed because they carry no `model` block;
- `python3 bench/classical_baseline.py --out-dir /tmp/cls-verify` exits **0 in 52 s**
and writes both artifacts, so the v2 half and the cascade sweep run;
- every figure in the table above is reproduced by that run: accuracy 0.9319
(356/382), none-detection 0.60 (15/25), false-none 0.00 (0/357), template-grouped
253/382 = 0.6623, random-row 353/382 = 0.9241, in-sample 382/382, v2 transfer
94/150 = 62.7% with bands 67.2 / 41.9 / 69.1, v2 CV 60.0 / 62.0, cascade 79.3% at
44% coverage and gate ≥ 0.20, model fingerprint `2eff58ac…`.

A second, independent defence exists on an unmerged branch (`p2-regress` @
`ae9961e`, 16:07:17, "fix(bench): the probes survive auth and a bad artifact"): a
`load_runs` that names each unreadable artifact and its reason, checks that every
file's `per_row` count matches the rows it is paired with, and continues on the
files that parse — so a future damaged recording degrades this run instead of
aborting it. It is not on `main`; the edit that makes the files parse is.

### The documented command overwrites the evidence it reproduces — 2026-11-03 16:20 EST

This is the defect this document did not have. `--out-dir` defaults to
`bench/results` (`bench/classical_baseline.py:247`) and both writes are
unconditional `write_text` calls (`:319`, `:354`). So the reproduction command
above does not reproduce the recorded numbers — it **replaces the files that
record them**, `bench/results/classical_382.json` and `classical_v2.json`, with
whatever the box produced this run.

Observed when this baseline was first re-run by hand on 2026-11-03: the accuracy
was identical and the p50 latency moved **0.062 → 0.024 ms** — the same 3×
single-shot timing noise §2 warns about — and both files had to be `git checkout`-ed
back. A third run, in this pass and into `/tmp`, gave **0.17 ms** p50 for the same
356/382. Three values for a figure that is not changing; two of those runs wrote
over the recording, which is why it had to be restored by hand.

Practical rule, and the reason §10 now shows the flag:

```
cp -r bench/results /tmp/results-before          # keep the recording
python3 bench/classical_baseline.py --out-dir /tmp/cls-verify
```

Without `--out-dir` the run is not read-only, and a reader who "just reproduces
the numbers" deletes the numbers being reproduced. The same hazard applies to any
documented command whose script writes into `bench/results/` by default.
