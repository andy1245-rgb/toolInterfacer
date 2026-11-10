# The shared harness — make it modular, and make it score classifiers

Decided 2026-11-10 by the operator: **this repository's bench becomes the shared one** for all
three projects (toolInterfacer, localAlexa, agentFleet). It already has the parts that matter —
arms, an A/A floor, pre-registered pass conditions, a per-option ledger, and (as of this morning)
an order control. What it does not have is seams: every experiment is a bespoke script, and there
is no way to score a scikit-learn model at all.

Goal: **any dataset, any predictor, one set of metrics and controls.** Adding a classifier must be
a class, not a new script.

## The four seams

### 1. Dataset adapters

One interface, several sources. A row is:

```python
@dataclass(frozen=True)
class Row:
    id: str            # stable, unique within the dataset
    text: str          # the input the predictor sees
    label: str         # the correct answer, one of the option names
    group: str         # for grouped CV and for per-group reporting (family, intent kind, session)
    options: tuple[str, ...]   # the closed option set, in the order the predictor gets it
    meta: dict         # anything else (stratum, split, band) carried through unchanged
```

Adapters to ship:

| adapter | source | rows | notes |
|---|---|---:|---|
| `tool_routing` | `bench/data/heldout_v3.json` | 415 | the eight ops; must reproduce today's arms exactly |
| `localalexa_intents` | `localAlexa/data/routing_382.json` + `results/historical/*.json` | 382 | 17 intents, `group` = family; the corpus for the classifier work |
| `generic_csv` | `--csv path --text-col --label-col` | any | so a new corpus is never a code change |

`options` comes from the dataset, not from a global — that is what lets a 17-way intent task live
next to the 8-way routing task.

### 2. Predictor protocol

```python
class Predictor(Protocol):
    name: str
    def predict(self, text: str, options: tuple[str, ...]) -> Prediction: ...

@dataclass
class Prediction:
    choice: str
    probabilities: dict[str, float] | None   # None when the predictor cannot produce them
    latency_ms: float
```

Implementations that must exist:

- **`llama_systemone`** — the existing `Server.ask` path, unchanged behaviour.
- **`logit_read`** — the existing letter read.
- **`constant`** — always-one-op. The baseline every model must beat. This is not a joke: on the
  current agentFleet corpus always-nudge scores better than the local engine.
- **`sklearn_text`** — a `sklearn.pipeline.Pipeline` built from a small spec dict, e.g.
  `{"vectorizer": "tfidf_word", "clf": "linsvc", "calibrate": "sigmoid"}`. Must support the
  vectoriser union `FeatureUnion(word, char_wb)` because that is the first cheap win on short
  utterances. In-process, so `latency_ms` is real.
- **`torch_text`** — a fine-tuned classifier head over a frozen (or partially unfrozen) encoder,
  CPU-capable. This is the seam that makes "fine-tuning" a first-class arm rather than a side
  project. Keep it to inference plus a `fit` hook; training scripts stay in the owning project.
- **`ensemble`** — comboses other predictors by vote, mean of probabilities, or a stacked
  meta-learner fitted on out-of-fold predictions. Must report the **correlation of the members'
  errors**, because an ensemble of predictors that fail on the same rows buys nothing.

Order matters and must be explicit: the harness passes `options` in the order given, and records
that order plus a hash of it in every artifact. This morning's result is why — reversing the eight
descriptions with no text change cost **9 rows and flipped 34 argmaxes**, against an A/A floor of
one flip.

### 3. Metrics

Per arm, per group, and overall:

- accuracy, and the **per-label ledger** (`n`, base, treat, fixed, broke, net) that already exists
- exact **McNemar** against a named reference arm
- **calibration**: ECE, Brier, NLL — required whenever `probabilities` is not None
- **the reject curve**: accuracy and coverage as the confidence threshold sweeps, plus
  `coverage@precision=0.9` (one number, daisy-chained off the sweep)
- **latency** p50/p95, and the same measured per row (already recorded)
- for a fitted predictor, **grouped CV** by `group`, with the fold assignment recorded, because a
  random split on utterances from the same family is how a classifier reports 97% and ships 88%

### 4. Controls, and pre-registration

The controls are what make a number believable, and they must be impossible to forget:

- **`--aa`**: runs the reference arm twice; reports the flip count as the floor. A delta smaller
  than 3× the floor is not a result.
- **`--order rev|shuffle:<seed>`**: re-runs an arm with the option order permuted, text unchanged.
  This is now mandatory for any claim about wording.
- **`--paraphrase`**: a second wording of the same edit, so "the clarification worked" is
  distinguishable from "any rewording worked".
- **`--preregister <path>`**: a YAML/JSON file declaring arms, the metric, and the pass conditions
  **before** the run. The run writes `passed: true|false` per condition and refuses to print a
  verdict without it.

## Deliverables

1. `bench/harness/` — `rows.py`, `predictors/`, `metrics.py`, `controls.py`, `run.py` (one CLI:
   `--dataset --predictor --arms --reference --aa --order --preregister --out`).
2. `bench/harness/tests/` — a deterministic `FakePredictor` so the whole thing is testable with no
   network and no model. Every metric gets a hand-computed case; the McNemar and ECE
   implementations get a known-answer test.
3. `docs/harness.md` — the seams, and a worked "add a predictor in 20 lines" example.
4. **`bench/rewrite_transfer.py` keeps working unchanged.** Either keep it as-is or make it a thin
   shim; do not break the reproduction command in `docs/grep-arm-2026-11-10.md`.
5. Re-run **one existing arm through the new harness** and assert it reproduces today's numbers
   (`floor_a` = 325/415). A harness that cannot reproduce a record it was built from is not
   finished.

## Hard limits

Branch commits only, never push. No hosted model calls — the operator's $1 is nearly spent and this
job needs none. Do not edit `service/ti/conf/defaults.json`. Do not delete `rows.jsonl` or any
existing arm; the historical records are the regression suite.

---

## Open questions from the first build (2026-11-10, night)

Written down instead of asked. Each one has the default I chose so the build is not blocked; any
of them can be reversed without touching the seams.

1. **Do the harness tests run in CI?** `ci.yml:31-32` runs `python3 -m pytest tests/ -q`, so a
   suite in `bench/harness/tests/` is invisible to it. I put the detailed tests there (as the spec
   asked) and added `tests/test_harness_contract.py` as a bridge so CI covers the reproduction and
   the metric cross-check. *Default taken:* bridge only. *The alternative:* add the path to
   `ci.yml` and to `pyproject.toml`'s `testpaths`, which makes CI slower and makes one command run
   two suites — your call, because it changes a shared config.

2. **Is `--limit` allowed to be a sample?** It truncates in dataset order, which is neither random
   nor stratified, so it is a smoke-test switch and not a cheap experiment. *Default taken:* smoke
   only, documented as such. *The alternative:* `--limit` with a seed and a stratum quota, which is
   a real design (and would make "run the 52-row class" a one-liner).

3. **How many order permutations is a claim worth?** The control implements `rev` and
   `shuffle:<seed>`; one reversal moved 8.2% of argmaxes and cost 9 rows, which is an upper bound
   for a single permutation, not an estimate. *Default taken:* one run, hash recorded, no policy.
   *The alternative:* require three seeded shuffles for any wording claim, as the plan's E3 says.

4. **What does `llama_systemone` present for the 382-intent task?** It presents
   `row.meta["descriptions"]` when present, else the option name. localAlexa's registry has
   `id`, `family` and `patterns` (regexes), and no description field, so an 18-way llama arm today
   reads 18 bare ids. *Default taken:* bare ids. *The alternative:* present the patterns, which is
   closer to what the tool actually matches and changes the task.

5. **Is `group` the family or the intent for localAlexa?** The spec says family, and the adapter
   uses it — but there are only 7 families for 382 rows, so a 5-fold split is lumpy and two folds
   can hold one family each. *Default taken:* family. *The alternative:* group by intent id (18
   groups, tighter folds) accepting that same-family paraphrases then straddle the split.

6. **Should the legacy artifacts carry their order hash?** `recorded` refuses to replay under a
   permuted order only when `order_sha=` is passed, because the historical `rows.jsonl` rows
   predate order recording. *Default taken:* the guard is opt-in. *The alternative:* annotate the
   artifact (a data edit to 2,905 rows of the historical records — I did not do it).

7. **Is nested cross-validation wanted?** `--cv` fits one predictor per fold; for an `ensemble` the
   members are fitted inside the same factory, and `stacked` runs its own inner out-of-fold loop.
   That is nested CV with no separate accounting, so a stacked ensemble's `--cv` number is
   honest but not comparable to a single model's. *Default taken:* documented, not changed.

8. **Are run artifacts committed?** `--out` writes `summary.json` and `rows.jsonl` wherever you
   point it; tonight's went to `/tmp`. The historical record lives in git, so the convention is
   probably "a run that gets quoted gets committed". *Default taken:* nothing new in git.

9. **`coverage@precision=0.90` is new and unanchored.** `grep_only` reaches 78.3% coverage at
   0.90 precision against `floor_a`'s 73.7%, and nothing in the repo says whether 0.90 is the
   right target for any project. *Default taken:* 0.90 because the spec named it.
