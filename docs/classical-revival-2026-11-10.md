# The classical tier was dead for its whole life

2026-11-10. Revival and measurement. Every number here is from the eval split, with the corpus
built from the training split only.

## The mistake

`decide.classical.corpus_path` pointed at `bench/registry_382.json` — a registry of **17
voice-assistant tools** (`timer_create`, `shoppinglist_add`, `calendar_today`, …). The backend is
asked to choose among the **eight code/file ops** (`filesys_grep`, `web_search`, …). The label sets
share **no names**, so the fitted model could not name a single candidate and returned
`ChoiceAnswer(None, {uniform}, 0.125)` on every request it has ever received.

It was invisible because **an abstaining tier looks exactly like a cautious one**: it says "no
opinion", the expensive model answers, the answer is correct, and every accuracy number stays where
it was. Nothing logged coverage. One person did look at it and set `temperature: 1.0` — the fix for
*bad confidence*, not for a *wrong vocabulary* — so the investigation ended with the misdiagnosis
intact and the tier still dead.

## What changed

1. `bench/build_classical_corpus.py` builds `service/ti/data/classical_v3_train.json` from
   `bench/data/heldout_v3.json` — **training rows only**, one flat `{"utterance", "label"}` row
   each, 208 rows over all eight ops (25–27 per label). Training on the eval rows would have
   produced a better-looking and meaningless number. The generator refuses to write a corpus whose
   labels do not intersect the offered option set, which is the check that was missing.
2. `decide.classical.corpus_path` points at it. One line, and it was the easy part.

Re-fit is automatic: the backend caches on a `(config, corpus)` fingerprint, so the new file is a
new model. `--check` on the generator fails if the file goes stale.

## Coverage, which is the measurement that matters

`bench/harness` gained a `classical` predictor that builds **the shipped backend from the shipped
config**, so a run measures the deployed tier rather than a copy of it. Every arm now reports
`n_answered`, `n_abstained` and `n_errors` next to its accuracy, and an abstention is recorded as an
abstention (`choice=""`, `meta["abstained"]`) instead of as a wrong answer. A tier that declines
everything reads as `answered 0/207` in the table, the summary and the row records.

The rot, reproduced on purpose against the old corpus, now prints:

```
arm               n correct  err  abst  accuracy (95% Wilson)
dead             40       0    0    40   0.0% [0.0,8.8]
           abst: dead declined 40/40 rows. A tier that declines everything is dead, and its
           accuracy says nothing.
```

`tests/test_classical_front.py` asserts exactly that, so it cannot regress quietly.

## What it is worth, measured on the 207 eval rows

| | |
|---|---|
| the tier, answering everything | **148/207 = 71.5%**, 0 abstentions, p50 **0.13 ms** |
| the 3B, replayed | 165/207 = 79.7%, p50 2668 ms |
| paired, tier vs 3B | fixed 20, broke 37, **−17 rows, p = 0.033** |

So as a *replacement* it is worse, and significantly so. As a **front door** it is worth having:

| threshold | coverage | precision on covered | blended accuracy | 3B alone | model time saved / 1,000 req |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 66.7% | 87.0% | 83.1% | 79.7% | 1 779 s |
| **0.22** | **58.0%** | **90.8%** | **83.1%** | 79.7% | **1 547 s** |
| 0.25 | 42.5% | 97.7% | **84.1%** | 79.7% | 1 134 s |
| 0.30 | 28.0% | 100% | 83.1% | 79.7% | 748 s |

At the 90%-precision point it takes **58% of requests** at 90.8% precision, lifts blended accuracy
by **+3.4 points**, and removes **1 547 seconds of model time per 1,000 requests** — 26 minutes of
a 3B for every thousand decisions.

## The part that is not proven, and why

**That operating point was read off the eval rows, so it is not a result yet.** Choosing it there is
the same error as the cascade that "won" earlier in this repo by having its threshold picked on the
rows it was scored on.

The obvious repair — choose the threshold on the training rows — **does not work, and the reason is
measurable**:

| p_max | training rows (in-sample fit) | eval rows (out of sample) |
|---|---:|---:|
| median | **0.456** | **0.235** |
| p90 | 0.480 | 0.426 |

Confidence on rows the model memorised is nearly twice what it is on rows it has not seen, so the
90%-precision point on the training rows sits at `t = 0.504`, which covers **1%** of them and
**0%** of the eval rows. A threshold picked from an in-sample fit is not conservative, it is
arithmetically wrong.

The honest fix is out-of-fold probabilities on the training split: cross-fit the tier over the 208
training rows, pick the threshold on those held-out folds, then apply it to the eval split once. The
harness already does this (`--cv` with `stratify`), but the shipped backend is not a `fit()`-taking
predictor — it fits itself from its config — so that arm needs a small predictor that fits per fold.
**That is the next unit of work, and it is the only thing standing between this and a shipping
decision.**

Second caveat: the gate in the policy (`policy.gate_slot`, 0.70) is on a different axis. The
classical tier's p_max tops out at 0.500 and has a median of 0.235; a 0.70 gate would let it fire on
nothing at all. Two tiers with different confidence scales need two thresholds, and the config has
one.

## Files

- `bench/build_classical_corpus.py`, `service/ti/data/classical_v3_train.json` — the corpus and its
  generator
- `bench/harness/predictors/classical.py` — the shipped tier as a harness arm
- `tests/test_classical_front.py` — the test that would have caught it, plus the rot reproduced
- `tests/test_classical_backend.py` — the corpus test no longer asserts the broken vocabulary
- `service/ti/conf/defaults.json` — one line: `corpus_path`
