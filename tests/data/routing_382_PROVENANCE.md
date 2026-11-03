# routing_382 — provenance

Imported 2026-11-03 from the operator's sibling research project. **Nothing in
this file or the two data files has been edited**: rows, labels and registry
entries are byte-identical to the source, which is why the digests below are
recorded.

## Origin

| | |
|---|---|
| source directory | `~/mirror/PersonalCodingProjects/personal/projects/localAlexa/jevResearchCompiled/experiments/` |
| rows | `data/test.json` (39,265 B) |
| registry | `registry.json` (21,872 B) |
| generator | `gen_data.py`, `SEED = registry["seed"]` = **42**, `PER_TOOL = 25` |
| classical runner | `exp_classic.py` — the run that produced 93.19% |
| recorded results | `results/classic_results.json`, `results/d1_results.json`, `results/router_results.json` |

## What landed in this repo

| file | sha256 | source |
|---|---|---|
| `tests/data/routing_382.json` | `7c91fa0644f96378e9d529d877218f864361bf18cc9442d19592f63f6f9ebcf6` | `data/test.json` |
| `bench/registry_382.json` | `7408ef5afaeefaeac725414e71b57cb30b6f15b4e864ccc6d313f6ded8a7859f` | `registry.json` |

`tests/data/routing_382.json` is the frozen evaluation set; `bench/registry_382.json`
is the **training** corpus the recorded baseline fitted on — its 200 `references`
strings, 17 tools plus a `none_route`. Keeping the training corpus is what makes
the recorded protocol reproducible: fitting on the 382 rows themselves would be a
different (and much easier) experiment.

## The 382 rows

18 classes, 382 rows, `{"text", "tool", "family"}` per row.

| class | rows | | class | rows |
|---|---:|---|---|---:|
| `__none__` | 25 | | `thermostat_set` | 25 |
| `docs_query` | 25 | | `thermostat_offset` | 23 |
| `light_query` | 25 | | `calendar_today` | 21 |
| `light_set` | 25 | | `timer_cancel` | 19 |
| `reminder_cancel` | 25 | | `projects_query` | 18 |
| `reminder_create` | 25 | | `timer_status` | 13 |
| `research_deep` | 25 | | `email_unread` | 7 |
| `shoppinglist_add` | 25 | | `shoppinglist_view` | 6 |
| `shoppinglist_remove` | 25 | | | |
| `timer_create` | 25 | | **total** | **382** |

Families: `knowledge` 96, `timers` 57, `shopping` 56, `reminders` 50, `lights` 50,
`climate` 48, `none` 25. The four short classes are the four whose template x slot
expansion could not yield 25 unique utterances after the references were removed
(`gen_data.py` prints a warning per tool when that happens).

## How the rows were made — the thing that matters for reading the score

`gen_data.py` expands each tool's `registry["templates"]` across its slot
vocabularies, **drops any utterance that is verbatim a reference**, shuffles with
seed 42 and keeps 25 per tool; `__none__` is 25 hand-written off-scaffold
questions from a literal list in the generator.

So `test.json` is not hand-labelled and it is not sampled from a corpus: it is a
systematic instantiation of the same template set the references were written
against. Every row is out-of-sample **by row**, and no row is a phrasing family
the reference author never saw. `bench/classical_baseline.py` measures the
consequence: training on the references scores 93.19%, but holding out whole
templates from a fit on the 382 rows drops to 66.2%.

## The recorded command

Run from the source directory, no arguments, no environment:

```
python exp_classic.py
```

which prints three models and writes `results/classic_results.json`. The 93.19%
row is the third:

```python
run("tfidf-word+linearsvc",
    TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
    LinearSVC(C=1.0))
```

Training input: `registry["tools"][*]["references"]` plus
`registry["none_route"]["references"]` — 200 strings, 18 labels. Test input: all
382 rows of `data/test.json`. The run was made with scikit-learn on the
operator's own box; scikit-learn is **not** installed in this repo, so
`service/ti/decide/linear_text.py` re-implements the vectoriser and the solver in
stdlib Python and `bench/classical_baseline.py` re-runs this protocol in-repo.

## Recorded numbers, and the arithmetic that checks them

| metric | recorded | exact fraction |
|---|---:|---|
| accuracy | 93.19% | 356/382 = 0.93194 |
| none-detection | 0.60 | 15/25 |
| false-none rate | 0.00 | 0/357 |
| p50 latency | 0.61 ms | — |

Both rate metrics land on an integer row count, which is the cheapest available
check that they came from this set: 0.9319 x 382 = 356.0 and 0.60 x 25 = 15.0.

## Caveat carried forward

The 382-row set is **not** this repo's task. It is 18-way including a learned
`__none__` class, generated from templates, and its utterances are
single-sentence home-assistant commands. `tests/data/utterances_v2.json` is
8-way, has no `none` class, and its t2/t3 bands were written specifically to
defeat surface-keyword routing. Numbers from one do not transfer to the other;
`docs/classical-baseline-2026-11-03.md` reports both and says so.
