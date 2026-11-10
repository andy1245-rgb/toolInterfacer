# The shared harness

One dataset seam, one predictor seam, one set of metrics, one set of controls. A scikit-learn
pipeline, a fine-tuned encoder and a llama-server are the same kind of arm, so "score this model
on this corpus" stops being a new script.

Spec: [`docs/harness-spec-2026-11-10.md`](harness-spec-2026-11-10.md) (the open questions from
the first build are appended there).

## Why it exists

Three projects need the same measurement, and this repo already had the parts that make a number
believable: arms, an A/A floor, pre-registered pass conditions, a per-option ledger, and an order
control. What it did not have was a way to point any of that at anything but the frozen 8-op
probe. Two measured facts drove the seams:

* **Order is not a detail.** Reversing this repo's eight descriptions with no text change moved
  **34 of 415 argmaxes** and cost **9 rows** (`p = 0.0784`) against an A/A floor of one flip. So
  the option order is an input, it is recorded in every artifact, and it has its own control.
* **A dataset whose labels and options disagree cannot be scored.** An 18-class corpus behind an
  8-op router is how the classical backend came to answer `None` on every row. `Row` refuses
  that at load.

## Quickstart

```bash
# reproduce the record the harness was built from, with no server and no model
python3 -m bench.harness.run \
    --dataset tool_routing \
    --arm floor_a='recorded|arm=floor_a' \
    --arm grep_only='recorded|arm=grep_only' \
    --reference floor_a --aa --out /tmp/repro
```

```
dataset   tool_routing  rows 415  options 8  groups 133  order_sha 6b70f586ec061715

arm               n correct  accuracy (95% Wilson)     p50 ms  p95 ms  ECE/Brier/NLL
*floor_a        415     325   78.3% [74.1,82.0]          2714    3044  0.046/0.324/0.799
 grep_only      415     328   79.0% [74.9,82.7]          2909    3188  0.041/0.314/0.797

pair: floor_a -> grep_only
  fixed 10  broke 7  delta +3 rows  exact McNemar p = 0.6291
```

That is `docs/grep-arm-2026-11-10.md`, recomputed from the artifact through the new code path:
325/415, 10 fixed, 7 broke, `p = 0.6291`, and the per-label ledger down to `filesys_grep`
`+10 / 0` and `web_search` `−3`. `--aa` on a replayed arm is a determinism check of the harness,
not a serving floor, and the CLI says so.

The historical records are the regression suite: every arm in
`bench/results/p3-transfer/rows.jsonl` is replayable as a predictor.

## The four seams

| seam | module | what crosses it |
|---|---|---|
| dataset | `bench/harness/rows.py` | `Row(id, text, label, group, options, meta)`; `options` is per-row, so a 17-way and an 8-way task coexist |
| predictor | `bench/harness/protocol.py` | `predict(text, options) -> Prediction(choice, probabilities, latency_ms)` |
| metrics | `bench/harness/metrics.py` | accuracy, per-label ledger, exact McNemar, ECE/Brier/NLL, reject curve, latency, grouped CV |
| controls | `bench/harness/controls.py` | `--aa`, `--order`, `--paraphrase`, pre-registered conditions |

### Datasets

| adapter | source | rows | options |
|---|---|---:|---|
| `tool_routing` | `bench/data/heldout_v3.json` | 415 | the 8 ops, in the frozen instrument's order; `group` = authored family |
| `localalexa_intents` | `<localAlexa>/data/routing_382.json` + `registry.json` | 382 | 17 intents + the `__none__` abstain route; `group` = family |
| `generic_csv` | `--path` | any | derived from the label column, or fixed with `options=a,b,c` |

The instrument's *text* travels in `row.meta["descriptions"]`, because an option name is a label
and the description is what the model actually reads. `llama_systemone` uses the descriptions;
`sklearn_text` uses the text and the labels.

### Predictors

| kind | needs | notes |
|---|---|---|
| `llama_systemone` | a llama-server on `:8080` | the existing `/v1/systemone` path, unchanged behaviour |
| `logit_read` | a llama-server | the existing letter read |
| `constant` | nothing | always one op; the baseline every model must beat |
| `sklearn_text` | `scikit-learn` | TF-IDF word, word+char_wb union, LinearSVC / logistic / SGD, optional sigmoid or isotonic calibration |
| `torch_text` | `torch`, `transformers` | a fine-tuned classification head over a frozen encoder, CPU; `fit` delegates to a caller-supplied trainer |
| `ensemble` | nothing (vote/mean), `scikit-learn` (stacked) | reports the members' error agreement, because an ensemble that fails on the same rows buys nothing |
| `fake` | nothing | deterministic, needs no model: the fixture the whole suite runs on |
| `recorded` | a `rows.jsonl` | replay any published arm; this is what makes the records a regression suite |

Spec strings are `kind|key=value|key=value`, so the CLI needs no imports:
`recorded|arm=floor_a`, `constant|choice=web_search`,
`sklearn_text|vectorizer=tfidf_word_char|clf=linsvc|calibrate=sigmoid`.

`scikit-learn` and `torch` are **not** installed on this box. Their modules import cleanly anyway
(everything is imported inside the factory), and building one without the dependency raises a
message naming the install command. The harness core, its tests and every arm above the dotted
line run on the standard library plus pytest.

## The CLI

```
python3 -m bench.harness.run \
    --dataset 'tool_routing'                 # or 'generic_csv|path=x.csv|text_col=a|label_col=b'
    --arm floor_a='recorded|arm=floor_a'     # repeatable; or --predictor SPEC for a single arm
    --arms 'a=SPEC;b=SPEC'                   # the same thing on one line
    --reference floor_a                      # everything is paired against this
    --aa                                     # score the reference twice: the floor
    --order rev                              # or shuffle:<seed>; text unchanged
    --paraphrase words.json                  # {option: alternate wording}
    --preregister conditions.json            # read BEFORE the run
    --cv 5                                   # grouped CV for predictors with fit()
    --out /tmp/run                           # writes summary.json and rows.jsonl
```

Artifacts carry the option order, its sha256, the per-row probabilities, and the group of every
row, so a run can be reproduced from its own output.

### Controls

* **`--aa`** — the reference arm twice. The delta must exceed **3× the flip count** to count
  (`controls.gate_verdict`), and the A/A figure is printed whether or not it flatters the result.
* **`--order rev|shuffle:<seed>`** — permutes the option tuple, never the strings. Mandatory for
  any claim about wording; a `recorded` arm refuses to replay under a permuted order, because
  that replays a fixed order and measures nothing.
* **`--paraphrase`** — rewrites *descriptions*, never option names. A label that moves is a
  different class; a description that changes is a different instrument.
* **`--preregister`** — conditions are `PATH OP VALUE`, resolved against the run's own numbers:
  `mcnemar_p < 0.05`, `delta <= -3`, `ledger.web_search.broke <= 3`,
  `ledger_untouched_net >= 0`. No `eval`; an unresolvable path is an error, not a False. Without
  the file the CLI prints numbers and **no verdict**.

## Add a predictor in 20 lines

```python
# bench/harness/predictors/my_thing.py
from ..protocol import Prediction, register


class MyThing:
    def __init__(self, model_dir: str = ""):
        self.model_dir = model_dir
        self.name = f"my_thing:{model_dir or 'default'}"

    def predict(self, text: str, options: tuple[str, ...]) -> Prediction:
        import time
        t0 = time.perf_counter()
        scores = self._score(text, options)          # your model, returning one score per option
        choice = max(options, key=lambda o: scores[o])
        total = sum(max(s, 0.0) for s in scores.values()) or 1.0
        probs = {o: max(scores[o], 0.0) / total for o in options}
        return Prediction(choice=choice, probabilities=probs,
                          latency_ms=(time.perf_counter() - t0) * 1000)


@register("my_thing")
def build(model_dir: str = "", **_kw) -> MyThing:
    return MyThing(model_dir=model_dir)
```

Then `python3 -m bench.harness.run --predictor 'my_thing|model_dir=/tmp/m'`. Two rules matter:
**only return options you were given** (a fitted model's classes may be a superset of a call's
options, so restrict and renormalise), and **import heavy dependencies inside `build`/`fit`**, so
the harness stays importable on a box that only wants to replay a record.

If your predictor can be fitted in-process, add `fit(rows)`. `--cv 5` will then build a fresh one
per fold, fit it on the other folds, and write the fold assignment into the artifact — which is
the difference between a 97% in-sample number and an 88% shipped one. Keep training loops in the
owning project: the seam is fit-then-predict, not a trainer.

## Tests

```bash
python3 -m pytest bench/harness/tests -q          # 78 tests, no network, no model
python3 -m pytest tests/ -q                       # the repo suite, unchanged
```

* `test_metrics_known_answer.py` — every metric against arithmetic on paper: McNemar against six
  known values **and** against `bench/ti_bench/stats.mcnemar_exact` on a 21×21 grid, Wilson
  against the published `[74.1, 82.0]`, ECE and Brier and NLL hand-computed, the reject curve's
  prefix, fold stability.
* `test_rows_contract.py` — the `Row` invariants, the three adapters, the domain split, and the
  label-outside-options refusal.
* `test_pipeline_fake.py` — `FakePredictor` end to end, the controls, the condition grammar, and
  the CLI writing artifacts and refusing a verdict without a pre-registration. The fake's hit
  rate is recomputed independently from `hashlib` in the test, so it cannot silently agree with
  itself.
* `test_replay_reproduces_records.py` — the published arms, replayed: `floor_a` 325/415,
  `grep_only` +10/−0 on `filesys_grep` with `p = 0.6291`, `order_rev` 316/415 with 6/15,
  `floor_b` one flip from `floor_a`, and no gate rows in any of them.

`bench/rewrite_transfer.py` is untouched: the reproduction command in
`docs/grep-arm-2026-11-10.md` still works exactly as published, which is why the harness replays
the same artifact rather than replacing the script.

## What this does not do yet

* **No live arm has been run through the new path.** The reproduction above is a replay. The live
  path is the same seam (`llama_systemone`), but it needs the server up, and nothing on this box
  has been served since the order arm finished.
* **`sklearn_text`, `torch_text` and `ensemble` are unexecuted** for the same reason: neither
  `scikit-learn` nor `torch` is installed here, so their factories have been exercised only to
  the point of their import error. Their logic is a seam, not a validated model.
* **Grouped CV has no published number attached.** `--cv` works and is tested with a fitted fake,
  but the 382 corpus has 7 groups, so 5-fold splits are lumpy and the per-fold accuracies will be
  noisy until the corpus is grouped more finely.
* **The reject curve is new and unflattering to be compared with**: `coverage@precision=0.90` is
  73.7% for `floor_a` and 78.3% for `grep_only`, which is the first time this repo has had that
  number and nothing to read it against.
