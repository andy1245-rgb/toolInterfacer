# The distilled corpus, certified as training data

2026-11-03. Answers the two questions that decide whether the teacher-labelled
corpus under (git-ignored) `var/distill/` can be trained on: **is it contaminated
by the rows everything is evaluated on**, and **are its labels good enough**. Both
are answered from the bytes on disk — no teacher call, no spend.

The artefact is `data/distill/` (committed), produced by
`tools/distill/certify.py`. This document is the record of what it contains and
what was removed to produce it.

**Verdict: clean, and usable — with two rows removed and a hard limit on what may
be claimed from it.** 410 of 412 labelled rows survive: 329 train, 81 val. Zero
rows collide with either frozen eval set under the repo's own leakage rule. The
labels are the frames' authored answers, not the teacher's; the teacher is
demonstrably noisy and is excluded from the training data.

## 1. Contamination — clean

Every distilled row was compared against every row of both frozen eval sets, with
the repo's own guard (`tools/distill/similarity.py`: normalised equality, **or**
token-Jaccard ≥ 0.70, **or** `SequenceMatcher.ratio` ≥ 0.80).

| eval set | rows | exact | leak-rule hits | max ratio | max jaccard |
|---|---:|---:|---:|---:|---:|
| `tests/data/utterances_v2.json` | 150 | 0 | **0** | 0.793 | 0.667 |
| `tests/data/routing_382.json` | 382 | 0 | **0** | 0.722 | 0.500 |

These are all-pairs maxima, not per-row nearest-neighbour scores. The distinction
matters: a nearest-neighbour sweep reports the closest reference *for each row*,
while a contamination claim needs the closest pair *anywhere*, and those are
different numbers. The sweep prunes a pair only when `2·min(len)/len_a+len_b` —
the hard ceiling on `SequenceMatcher.ratio` — already proves it cannot reach the
threshold, so no pair is skipped that could have tripped the rule.

The guard was already working, and the evidence is in the drop log: `dropped.jsonl`
records four rows refused at generation time for colliding with the frozen 150,
one of them an exact duplicate ("what search terms fit this ask?") and three
paraphrases of `docs.python.org` fetch rows at ratio 0.808–0.848. Those rows never
reached `candidates.jsonl`. That is why the survivors are clean.

### Two rows removed anyway

The leak guard's 0.80 ratio bar is calibrated for *paraphrase*, which makes an
utterance longer. **Deleting** a clause from an eval row makes it shorter, which
lowers the ratio while leaving the request identical. So a stricter rule was
applied on top — a strict token-subset of an eval row at ratio ≥ 0.70 — and it
caught two rows the guard let through:

| id | utterance | nearest eval row | ratio |
|---|---|---|---:|
| `rt-0027` | "list every file ending in .toml" | "list every file ending in .toml anywhere in the repo" | 0.74 |
| `rt-0036` | "any config files lying around?" | "any config files lying around that look relevant?" | 0.75 |

Both are `t1/filesys_glob`. Both are in `data/distill/excluded.jsonl` with the
rule, the colliding eval row and the ratio.

To be clear about what this does and does not accomplish: the corpus's highest
ratios are **not** removed, and that is correct. The worst surviving pair is
`rt-0047` "where is guard_state defined?" against the frozen "where is
handle_stream defined?" at 0.793, followed by `rt-0049` "which files mention the
cascade thresholds?" against "which files mention the scheduler hooks?" at 0.790.
Both are the *same frame with a different object* — a different request with a
different correct answer, which is the domain overlap a router is supposed to
generalise across. The two excluded rows are a distinct failure mode: a literal
**deletion** of an eval row, where the request is the same one. Sharing a frame is
fine; being the eval row minus a clause is not.

A third subsumption — "which one should we ship?" against a longer `decide_pick`
row — scores 0.56 and is a generic short phrase, so it was reviewed and **kept**;
the rule sits at 0.70 precisely so that case stays visible rather than being
silently swept up.

### `routing_382` is not the eval set for this corpus

`routing_382` is an **imported** benchmark (`tests/data/routing_382_PROVENANCE.md`)
with 18 classes — `timer_create`, `light_set`, `thermostat_set`, `__none__` … —
and its own training corpus at `bench/registry_382.json`. The intersection with
this corpus's 8 ops is **empty**. It cannot be contaminated by these labels, and,
more importantly, **no classifier trained on this corpus can be scored on it.**
The 93.19% / 66.23% row-wise-vs-family-held-out lesson in
`docs/classical-baseline-2026-11-03.md` is real and worth heeding, but it is about
`routing_382`'s own registry, not about this corpus. The eval set for this corpus
is `utterances_v2.json`, and it is the one the guard covers.

## 2. The count — 412, not 267

The repo quotes "267 usable labelled rows". That is a stale snapshot of a label
run that was still going. The arithmetic reconciles exactly:

| quantity | count | what it is |
|---|---:|---|
| `candidates.jsonl` | 414 | the templates stage's surviving rows |
| `labels.jsonl` | **412** | rows with a teacher answer; `label_method: measured` on every one |
| unrecovered rejects | 2 | `rt-0326`, `rt-0400` — never got a label |
| `rejects.jsonl` | 25 | failed *attempts*, not rows: 21 provider errors, 4 schema-invalid replies |
| `dropped.jsonl` | 40 | generation-time drops that never became candidates: 19 dedup, 18 band, 3 frozen-collision |

`414 − 2 = 412`, and `412` is the usable count: every row carries a non-empty
`intended`, `label_method` is `measured` on all 412, and `intended` is present in
its own `candidates` ballot on all 412 (verified — a label not on its own ballot
would be a labelling impossibility). The 25 rejects are retries: 23 of those ids
were relabelled successfully on a later attempt, leaving the same 2.

The cost confirms which snapshot "267" came from. Mean label cost is \$1.16e-5, so
\$0.0028 buys **265 rows** — the 08:00, 09:00 and 10:00 batches (196 + 39 + 30 =
265). The run resumed at 19:00 and added 147 more, finishing at **\$0.0043 for
412**. `TODO.md` already flags the 267 as "an earlier draft"; this is the count
behind it.

## 3. Labels — the teacher is not the label

`intended` is the label these frames were authored to carry, and
`tests/data/utterances_v2_NOTES.md` is explicit about what it means: for a
boundary row "the label is the semantically correct one, `tier_target` is the
confusable rival". The hosted teacher's answer (`target.tool`) is a *measurement
of the teacher*. It is not shipped in the frozen rows.

| band | rows | teacher agrees | mean conf | rows at conf ≥ 0.95 |
|---|---:|---:|---:|---:|
| t1 literal | 134 | **100.0%** | 0.934 | 95 |
| t2 veiled | 106 | **76.4%** | 0.777 | 12 |
| t3 boundary | 172 | 93.6% | 0.919 | 114 |

### The teacher is measurably noisy, not just low-confidence

36 of 412 rows have the teacher contradicting the label (25 in t2, 11 in t3). The
concerning part is not the count, it is that the contradictions are not
concentrated where confidence is low. On 9 of the 117 multi-row frames the teacher
gives **different answers to two rows that differ only in a slot value**:

| family | id | teacher | conf | utterance |
|---|---|---|---:|---|
| `t3:decide_extract->filesys_read:01` | `rt-0265` | `decide_extract` | 0.80 | "pull the values out of bench/lanes.json, don't paste the file" |
| | `rt-0266` | `filesys_read` | **0.90** | "pull the values out of pyproject.toml, don't paste the file" |
| `t3:decide_extract->filesys_read:05` | `rt-0271` | `filesys_read` | 0.90 | "read through bench/lanes.json and pull out the literal values" |
| | `rt-0272` | `decide_extract` | 0.90 | "read through pyproject.toml and pull out the literal values" |
| `t2:filesys_grep:06` | `rt-0181` | `reddit_digest` | 0.60 | "pack keeps turning up — how widespread is it?" |
| | `rt-0182` | `filesys_grep` | 0.85 | "handle_stream keeps turning up — how widespread is it?" |

The filename cannot change the correct operation. Whichever of the pair is right,
the teacher is wrong on the other, at 0.90 confidence. All 18 affected ids and the
full 36-row contradiction set are in `data/distill/diagnostics.jsonl`.

This is the concrete form of the risk `TODO.md` flags ("a confidence target taken
from this without scaling trains a model that is confidently wrong"). The
consequence for this corpus is specific: **train on `intended`, never on
`target.tool`, and do not fit a temperature/confidence target on this
distribution until the audit CSV is annotated** — it is still blank, and
`stats.py` is right to report calibration as UNKNOWN.

For t3 the teacher's behaviour is otherwise reassuring: it picks the authored label
on 161 of 172 rows and falls for the decoy `tier_target` on only 9.

## 4. The corpus's own split does not leak families

295 families, 117 of them with more than one row, and **zero** crossing
train/val. The split unit is `family` (the frame), hashed by
`corpus.assign_split`, so paraphrases of one frame cannot land on both sides. The
guarantee is asserted in `manifest.json` and re-asserted by a test rather than
assumed.

This means `val.jsonl` is a fair test of generalisation **across frames of the
same (band, op) cells** — it is not a test of generalisation to an unseen op,
because every op appears in both sides.

## 5. Fitness for purpose

**It can train a classifier for the `useTool` 8-op decision layer.** 410 rows with
authored labels, family-disjoint validation, and a clean relationship to the only
eval set that shares its taxonomy.

**It cannot train anything for `routing_382`** — different label space, different
task. If a plan expected a corpus trained here to be scored on those 382 rows,
that plan is a category error rather than a leakage risk.

Where the corpus is thin, stated so a trainer can act on it:

- **t1 teaches keyword matching and will inflate every headline number.** 134 rows
  (33% of the corpus), teacher agreement 100%, 95 of 134 at confidence ≥ 0.95, and
  the requests name the operation outright. Report t1 and t2/t3 separately, always.
- **t2 is both the most valuable and the least confirmed band**: 106 rows, 76.4%
  teacher agreement, and every one of the 8 t2 cells is under 15 rows
  (`decide_pick` 11, `web_fetch` 12).
- **`decide_pick` is the thinnest class overall** — 39 rows, 11 of them t2.
- **The corpus is entirely synthetic** (`source: template` on all 414). It provides
  boundary-structure coverage, not the distribution of real user phrasing.
- **Object vocabulary is shared with the eval set.** 33 of 150 `utterances_v2` rows
  use a slot object that also appears in this corpus; 68 of 69 slot values appear
  here. Rows do not duplicate, but nouns recur — so a gain measured on
  `utterances_v2` is partly "same objects", and is optimistic against genuinely
  new ones.

### What a fair evaluation looks like

1. Fit on `train.jsonl` only; select on `val.jsonl` (family-disjoint).
2. Score **once** on `utterances_v2.json`, reporting per band, per op and
   macro-F1 — never only the headline.
3. Put the baselines on the same rows or the number is meaningless: the recorded
   figures are d1-3B at **74.0%** and the stdlib TF-IDF + linear SVM at **62.7%**
   (`docs/classical-baseline-2026-11-03.md`).
4. Do not tune on `utterances_v2`. It has already been the eval set for every arm
   in this repo; it is a reported set now, and each further look costs it standing.

## 6. What a reader can check

```
python3 tools/distill/certify.py --data-dir var/distill --check   # re-run the audit
python3 -m pytest tests/test_distill_certified_corpus.py -q       # the frozen artefact
```

`certify.py` re-derives every number in this document from the label store and
exits non-zero on a leak. `data/distill/manifest.json` carries the counts, the
per-file hashes, the thresholds, and each exclusion with its rule. The test pins
the committed artefact — row counts against the manifest, every label on its own
ballot, no family crossing the split, and no collision with the frozen 150 — so a
future regeneration that reintroduces eval content fails CI instead of quietly
inflating a score.

One accounting gap is recorded rather than fixed: `manifest.json`'s own generator
tally reports 90 template drops (dedup 55, band 29, frozen 6) while `dropped.jsonl`
logs 40 rows (19 / 18 / 3). The log is the more conservative record and is what
this document uses; the 50-row difference is unexplained and would need a
regeneration run to resolve. The frozen set does not depend on it — it is built
from the 412 rows that actually carry labels.
