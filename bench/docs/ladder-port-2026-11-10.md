# The localAlexa ladder, ported — all four arms reproduce to the row

2026-11-10, branch `ladder-port/2026-11-10`. The four winners of the localAlexa classifier
ladder now run as harness arms on the harness's own CLI and CV machinery, scored on the
*ladder's* folds and the *ladder's* training pool. Every arm reproduces its recorded number
exactly — not within a row, to the row, on both protocols:

| arm (harness spec) | grouped (5 folds) | ladder | rowwise | ladder |
|---|---:|---:|---:|---:|
| `sklearn_text|vectorizer=tfidf_word|clf=linsvc|C=1` | 343/382 | 343 | 356/382 | 356 |
| `sklearn_text|vectorizer=tfidf_word|clf=logreg|C=100` | 347/382 | 347 | 357/382 | 357 |
| `torch_text` + `trainer=localalexa.harness_adapters:train_setfit` | 359/382 | 359 | 360/382 | 360 |
| `ensemble|mode=mean|standardise=true` over the two above | 363/382 | 363 | 370/382 | 370 |

`ladder_port.py --verify` exits non-zero if any arm drifts more than one row
(`results/ladder-port-2026-11-10/summary.json`). The run also carries the fold map, the
per-fold fit-pool sizes, and the count of pool rows each fold had to drop.

## What the harness could express without changes

Both scikit-learn arms. `sklearn_text` already had the word TF-IDF vectoriser and both
classifiers with `C=` as a spec key, so the two linear arms were spec strings and nothing
else. That is the harness working as designed.

## What it could not, and what was added

Five additions, each small and each a real gap rather than a workaround. Defaults are
unchanged, so no existing arm moves.

### 1. `localalexa_intents|group_by=phrasing`

The existing adapter sets `group = row["family"]`, the seven-value *domain* column. Five
folds over seven domains withholds a whole domain per fold, so every fold contains intents
the training side has never seen: measured **0/382 out of fold**, which is the grouping
failing rather than the arm. `group_by=phrasing` uses the 110 generating templates — the
grouping the ladder measured with, and the one that leaves a router something to learn from.
The domain column stays in `meta` under both.

### 2. `--fit-pool` and the `localalexa_refs` dataset

The harness's CV fits on the other folds of the rows being scored. The ladder fits on the
200 **reference utterances** the router is trained on, minus any reference that is a literal
slot fill of a held-out phrasing family. Those are different experiments and on this corpus
they differ by several points, because the scored rows are generated from the same templates
the references paraphrase.

`localalexa_refs` serves the reference pool as rows with `meta.leaks_into` naming the
phrasing families each reference is a slot fill of (29 of 200 are), and `--fit-pool` makes
`grouped_cv` train on that pool, dropping a row when its own group is held out or when it
leaks into a held-out one. The per-fold training sizes come out at 194/199/195/195/194 —
identical to `localalexa.ladder.pools.fold_pool`, which is the check that the contamination
filter is the same filter.

### 3. `--fold-map`

"Same grouped folds" cannot mean "folds that look similar" if the point is to reproduce a
number. `--fold-map` takes the group→fold assignment the ladder wrote into its artifact and
uses it instead of assigning new ones. `grouped_cv` now reports `fold_map: "explicit"` beside
the map itself, so an artifact says which of the two it was.

### 4. `torch_text`: `normalize`, and a checkpoint that does not exist yet

Two changes, both needed for a faithful port rather than a convenient one.

**`normalize`.** `sentence-transformers/all-MiniLM-L6-v2` has a Normalize module after its
mean-pooling layer, so SetFit's head is trained on unit-norm embeddings — verified:
`encode(..., normalize_embeddings=False)` still returns vectors of norm 1.0. The harness
pooled without normalising, so a SetFit checkpoint scored through it would have been a model
that never existed. The checkpoint contract now accepts `"normalize": true`, and `_probabilities`
L2-normalises the pooled vector before the head when it is set. This is the difference
between "the port runs" and "the port is the same model", and it is worth 0 rows only because
it is right.

**A checkpoint that does not exist yet.** `_checkpoint_file` refused a path that was not on
disk, which makes training-from-scratch inexpressible: grouped CV builds a fresh predictor
per fold and the `trainer=` hook writes that fold's weights. With a trainer the path is now
allowed to be absent at build time; without one the guard is unchanged.

The training itself stays in the owning project, which is what the harness's `trainer` hook is
for: `localalexa.harness_adapters.train_setfit` fits SetFit on the fold's rows and writes the
checkpoint in the harness's contract — encoder state under `encoder.*`, the fitted logistic
head as `head.weight`/`head.bias`, labels from the head's own `classes_` so the row order
cannot disagree with the weights. Training settings are imported from the ladder rung rather
than copied, so the arm and the rung cannot drift.

### 5. `ensemble|mode=mean|standardise=true`

The winning ensemble averages *row-standardised* member distributions. Without that, an
average hands the whole vote to whichever member is sharpest: the linear model's probabilities
run 0.1–0.9 while the encoder head's sit around 0.2–0.4, so the linear arm decides every row
and the ensemble is that arm plus a cost. `standardise=true` row-standardises each member
before averaging.

**This one cost 8 rows before it was right.** The first version standardised each member and
then shifted-and-renormalised it back to a distribution before averaging. That rescales a
member's vote by its own spread, which is a different aggregation: the port scored 355/382
against the ladder's 363. The fix is to keep z-scores through the average and softmax once at
the end — the same thing the ladder's `StandardisedEnsemble` does — which brought it to 363/382
and 370/382 exactly. The lesson is in the docstring so the next reader does not repeat it.

## The torch question, answered directly

The brief allowed "if the harness cannot express a torch predictor, say so rather than forcing
it". It can, and this port is that answer: `torch_text` loads a checkpoint, runs the encoder
and a linear head on CPU, and a `trainer=module:function` hook hands training back to the
project that owns the corpus. Two gaps had to be closed (`normalize`, a not-yet-existing
checkpoint) and both are additive. Nothing here needed a new predictor kind.

What it cannot do is train SetFit itself, and that is by design: the corpus, the schedule and
the checkpoint contract belong to the owning project.

## What was deliberately not ported

* **Rung 6 (augmentation).** It is a data policy, not a predictor: the rung is defined by
  extra rows (`augment(base, held_out)` per fold), and porting it would mean teaching the
  harness a fold-aware data pipeline. It was also refuted (+1.6 ± 1.9 grouped, and the
  uncapped variant that looks like +3.15 rowwise is the protocol being gamed).
* **Rung 4 (cross-encoder).** Refuted relative to SetFit: 92.15% grouped at 11.8 ms against
  93.98% at 7.75 ms. It would be expressible as a `torch_text` head over a *pair* encoder,
  which is a different checkpoint shape; not worth a contract revision for a losing arm.
* **Rung 5 linear-only ensemble.** Refuted: exactly the baseline's accuracy at 8× the cost.

## Caveats

* **Latency is not comparable in this artifact.** The ensemble's p50 reads 56.97 ms here
  against the ladder's 7.85 ms: the box was at load ~23 from other work while this ran, and
  `torch_text` re-encodes through plain `transformers` per row where SetFit's own path batches.
  Accuracy reproduces; the latency column in this harness run should not be quoted.
* **The fold map is read from a path** (`/home/andrew/localAlexa/results/ladder/rung5-ensemble.json`).
  That is deliberate — the point is to use the assignment that produced the numbers — but it
  means the port is not self-contained without that artifact.
* **Both protocols are checked.** Rowwise (fit on all 200 references, score 382) reproduces
  too, which is what makes the check worth having: a port that only matched the grouped
  number could still be scoring a different row set.

## Reproduce

```bash
cd ~/toolInterfacer/bench
python3 ladder_port.py --skip-torch          # the two sklearn arms, ~15 s
/home/andrew/localAlexa/.venv-embed/bin/python ladder_port.py   # all four, ~10 min of SetFit
/home/andrew/localAlexa/.venv-embed/bin/python ladder_port.py --arms r5-ensemble-setfit-z
```

Equivalent through the harness CLI, for one arm:

```bash
python3 -m harness.run \
  --dataset "localalexa_intents|group_by=phrasing" \
  --fit-pool localalexa_refs \
  --fold-map /tmp/ladder-fold-map.json \
  --cv 5 \
  --arm "r2=sklearn_text|vectorizer=tfidf_word|clf=logreg|C=100" \
  --reference r2
```

Artifacts: `results/ladder-port-2026-11-10/summary.json` (all four arms) and
`results/ladder-port-ensemble-2026-11-10/summary.json` (the ensemble after the aggregation
fix). Tests: `harness/tests/test_fold_map_and_fit_pool.py` — 15 cases over the fold map, the
fit pool and its leak filter, the dataset grouping, ensemble standardisation, and the two
`torch_text` changes.
