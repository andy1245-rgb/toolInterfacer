# routing_382 — provenance

Imported 2026-11-03 from the operator's sibling research project. The rows,
labels and registry entries are the source's, with **one documented exception**:
on 2026-11-04 the four private project names in the `projects_query` slot
vocabulary were replaced by neutral names of the same shape, because this repo
is public and that vocabulary was one person's own project list. Nothing else
changed — not a row, not a label, not a byte of formatting. Digests for the
source, for what was first imported and for what ships now are all recorded
below, so the exception is checkable rather than asserted.

## Origin

| | |
|---|---|
| source | a private sibling benchmark tree — the six-backend tool-router benchmark this set was generated for. Read 2026-11-03; the directory is not published and does not resolve for a reader, so the digests below are what make the import checkable |
| rows | `data/test.json` (39,265 B) |
| registry | `registry.json` (21,872 B) |
| generator | `gen_data.py`, `SEED = registry["seed"]` = **42**, `PER_TOOL = 25` |
| classical runner | `exp_classic.py` — the run that produced 93.19% |
| recorded results | `results/classic_results.json`, `results/d1_results.json`, `results/router_results.json` |

## What landed in this repo

| file | source sha256 | imported 2026-11-03 | ships as |
|---|---|---|---|
| `tests/data/routing_382.json` | same as imported | `7c91fa0644f96378e9d529d877218f864361bf18cc9442d19592f63f6f9ebcf6` | `dc63af16426ef573d795279987a858c8dbc55c649956177109a5cd35466ecf9d` |
| `bench/registry_382.json` | same as imported | `7408ef5afaeefaeac725414e71b57cb30b6f15b4e864ccc6d313f6ded8a7859f` | `ec27371aa4b42e9b6089e6b6077ad1f2bc53489046e26474bb2eddd2ea0eb2c2` |

Both were byte-identical to the source when imported, so "imported" and "source"
are the same digest — the source tree is not published, which is why the import
digest is the only thing a reader could ever check.

`tests/data/routing_382.json` is the frozen evaluation set; `bench/registry_382.json`
is the **training** corpus the recorded baseline fitted on — its 200 `references`
strings, 17 tools plus a `none_route`. Keeping the training corpus is what makes
the recorded protocol reproducible: fitting on the 382 rows themselves would be a
different (and much easier) experiment.

## The one edit: the `projects_query` slot vocabulary (2026-11-04)

The source registry's `projects_query` tool carried a five-entry `slots.project`
list of project names, and `gen_data.py` expanded it into 15 of the 18
`projects_query` rows. Four of those five were the importing operator's own
private projects; one is a generic app category and stays. The four were
replaced, and **the originals are not republished here** — quoting them to
record their removal would defeat the removal. What is recorded instead is a
fingerprint for each: `sha256` of the UTF-8 bytes, which anyone holding the
source can check against, and the neutral name that replaced it.

| removed — `sha256(bytes)[:12]` | length | ships as | length |
|---|---:|---|---:|
| `55c87f261c39` | 11 | `nimbusTune` | 10 |
| `d4b284efc56a` | 19 | `workshopProjects` | 16 |
| `e3eb975543c1` | 10 | `localSpeaker` | 12 |
| `b5555df826d6` | 12 | `liveSubtitles` | 13 |

Each replacement keeps the grammatical slot (a one-word project name, camelCase
where the original was), the token count, and the length class within a couple of
characters. This was a disclosure decision, not a data-quality one: two of the
four appear in no audit document anywhere, so the vocabulary was simply one
person's project list sitting in a published dataset.

Where the four names occurred: `registry.tools[13].slots.project` (4 of its 5
entries) and two of the same tool's `references` (each rewritten in place with
one substituted token); and in `tests/data/routing_382.json` the rows those slots
expanded into — three phrasings per name (`what's happening in <project>`,
`how is <project> doing`, `give me a summary of the <project> project`) × five
names = the 15 `projects_query` slot rows, of which 12 carry a renamed project
and the 3 generic ones are unchanged.

The same four tokens were replaced in the three recorded replays that quote the
382 rows (`bench/results/classical_382.json`,
`bench/results/repro2-2026-11-03/classical_382.json`,
`bench/results/d1-382-2026-11-04.json`), so a recorded `per_row` entry still
quotes the row that ships. Their scores are untouched; the `sha256` before and
after is in the digest record below.

### Evidence that nothing else moved

- **Byte-reversible.** The substitution is a token-for-token replacement, so
  reversing it on each of the five files reproduces the 2026-11-03 bytes exactly
  (`git show HEAD:<path>` compared byte for byte; verified for all five).
  The byte delta is arithmetic: `bench/registry_382.json` ±0 B (21872 →
  21872), the other four −3 B. No line was reflowed, no key reordered.
- **Structure.** 382 rows, 18 tools, 25 `__none__`, key order
  `["text", "tool", "family"]`, 18 `projects_query` rows, 110 template groups —
  all unchanged, and pinned by
  `tests/test_classical_backend.py::ImportedDataTest` and
  `tests/test_eval_harness.py::test_r382_template_groups_match_classical_baseline`.
- **Numbers.** `bench/classical_baseline.py --out-dir <tmp>` run against the
  imported files and re-run after the substitution produced **identical**
  results: template groups 110, in-sample 382/382, `cv_random_rows` 0.9241,
  `cv_template_grouped` 0.6623, and every field of
  `bench/results/classical_382.json` other than latency, fingerprint and
  timestamp. So the recorded numbers further down still describe the shipped
  files.
- **The contamination audit.** `data/distill/manifest.json`'s `routing_382`
  block, re-run with `tools/distill/certify.py`'s own `audit_contamination` over
  the 412 committed label rows: every number identical (`max_ratio` 0.7216, same
  five `closest` pairs, `exact` 0, no leak-rule or subsumption hits). Only its
  recorded `sha` moved, `7c91fa0644f9` → `dc63af16426e`.

### Digest record

| file | before (2026-11-03) | after (2026-11-04) |
|---|---|---|
| `tests/data/routing_382.json` | `7c91fa0644f96378e9d529d877218f864361bf18cc9442d19592f63f6f9ebcf6` | `dc63af16426ef573d795279987a858c8dbc55c649956177109a5cd35466ecf9d` |
| `bench/registry_382.json` | `7408ef5afaeefaeac725414e71b57cb30b6f15b4e864ccc6d313f6ded8a7859f` | `ec27371aa4b42e9b6089e6b6077ad1f2bc53489046e26474bb2eddd2ea0eb2c2` |
| `bench/results/classical_382.json` | `63040e265ee6d540beb99661affda19199713765334d3e8c046d296fef8da41b` | `fcd563bac112ae72ccf76cc050bd0cd7987f2912b131df190d979394ba4bdbe7` |
| `bench/results/repro2-2026-11-03/classical_382.json` | `20d880918e87a190c028fff1cceff45631fea6eb7af0b323426eb7aa2992335e` | `5bafa109254fb701d70952ffdf84141cdff9526a5a9acec4cf595054df7356b1` |
| `bench/results/d1-382-2026-11-04.json` | `bc39939ecbd504a2914ba6b20ac9a8a1bb542955597833018ff0ff293d216ea3` | `ce8de9e052799bcdd0b996f90539edd9b7f540382a9692f29c83cd1d8f492e7a` |

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
