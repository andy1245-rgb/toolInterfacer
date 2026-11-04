# data/distill — the frozen training set

**Use `label`. Do not use the teacher's answer.** That is the whole contract; the
rest of this file is why.

This directory holds the distilled routing corpus, frozen and audited by
`tools/distill/certify.py`. The live label store it came from lives in
git-ignored `var/distill/`, so this is the only copy a reader or a trainer can
actually depend on.

```
python3 tools/distill/certify.py --data-dir var/distill --out data/distill
python3 tools/distill/certify.py --check     # audit and print, write nothing
```

## The label

`label` is the frame's **authored** answer — the op the request semantically
wants, per `tests/data/utterances_v2_NOTES.md` ("the label is the semantically
correct one, `tier_target` is the confusable rival").

The hosted teacher's answer is **not** in these files, deliberately. It disagrees
with `label` on 36 of 412 rows, and on 9 frames it gives *different answers to two
rows that differ only in a slot value*, at high confidence both times:

| id | teacher | conf | utterance |
|---|---|---|---|
| `rt-0265` | `decide_extract` | 0.80 | "pull the values out of bench/lanes.json, don't paste the file" |
| `rt-0266` | `filesys_read` | 0.90 | "pull the values out of pyproject.toml, don't paste the file" |

Same frame, same phrasing, one filename apart, opposite answers. That is teacher
noise, not signal. Training on the teacher's choice would fit it. The contradicting
rows are kept in `diagnostics.jsonl` for inspection and for an ablation, not as a
label source.

For `t3` rows, `tier_target` is the **decoy** — the defensible-but-wrong rival op
the row was authored to tempt. It is never the label. It is `null` on t1/t2.

## Contamination: clean

Both frozen eval sets were swept against every row, all pairs, not nearest
neighbour only.

| eval set | rows | exact | leak-rule hits | max char-ratio | max jaccard |
|---|---:|---:|---:|---:|---:|
| `tests/data/utterances_v2.json` | 150 | 0 | **0** | 0.793 | 0.667 |
| `tests/data/routing_382.json` | 382 | 0 | **0** | 0.722 | 0.500 |

The leak rule is the repo's own (`tools/distill/similarity.py`): normalised
equality, **or** token-Jaccard ≥ 0.70, **or** `SequenceMatcher.ratio` ≥ 0.80. No
row trips any branch against either set.

Two rows were nonetheless removed as near-misses, under a stricter rule that
catches *deletion* rather than paraphrase: a strict token-subset of an eval row at
ratio ≥ 0.70. The leak guard's 0.80 bar is calibrated for paraphrases, which get
longer; deleting a clause from an eval row shortens it and lands under that bar
while still being the same request.

| id | utterance | nearest eval row | ratio |
|---|---|---|---:|
| `rt-0027` | "list every file ending in .toml" | "list every file ending in .toml anywhere in the repo" | 0.74 |
| `rt-0036` | "any config files lying around?" | "any config files lying around that look relevant?" | 0.75 |

Both are in `excluded.jsonl` with the rule and the numbers.

This is **not** a claim that the corpus's highest ratios are gone — they are not,
and they should not be. The worst remaining pair is `rt-0047` "where is
guard_state defined?" against "where is handle_stream defined?" at ratio 0.793,
and next to it `rt-0049` "which files mention the cascade thresholds?" against
"which files mention the scheduler hooks?" at 0.790. Those are the *same frame
with a different object* — a different question with a different correct answer,
which is exactly the domain overlap a generalising classifier should learn from.
The two excluded rows are a different failure mode: a literal **deletion** of an
eval row, where the request is the same one. Different objects are fine; the same
request with words removed is not.

`routing_382` deserves its own note: it is an **imported** benchmark with a
disjoint 18-class taxonomy (`timer_create`, `light_set`, `thermostat_set`, …) and
its own training corpus at `bench/registry_382.json`. There is zero overlap with
this corpus's 8 ops. It cannot be contaminated by these labels in any meaningful
sense, and **no classifier trained here can be scored on it.**

## Files

| file | rows | what |
|---|---:|---|
| `train.jsonl` | 329 | training rows |
| `val.jsonl` | 81 | held-out rows, **family-disjoint** from train |
| `excluded.jsonl` | 2 | every removal, with its rule, the colliding eval row, and the ratio |
| `diagnostics.jsonl` | 44 | the 36 teacher contradictions and the 9 unstable frames — diagnostics, not labels |
| `ballot.json` | 8 | the op-key → description map; every row's `candidates` draws from these keys |
| `manifest.json` | — | all counts, hashes, thresholds, and the verdicts above |

Row shape:

```json
{"id": "rt-0296", "utterance": "...", "label": "web_search", "band": "t3",
 "split": "train", "family": "t3:web_search->decide_extract:01",
 "tier_target": "decide_extract",
 "candidates": ["decide_extract", "...", "web_search"]}
```

`candidates` is the row's own ballot (4–8 of the 8 ops), so a row never asks the
model to choose an op that was not offered. `label` is always in `candidates` —
asserted in `tests/test_distill_certified_corpus.py`.

## Split integrity

295 families, 117 with more than one row, and **zero** families crossing
train/val. The split unit is `family` (the frame), hashed by
`corpus.assign_split`, not the row — paraphrases of one frame stay on one side.

## Where this corpus is weak — read before trusting a number

- **t1 is saturated and teaches keyword matching.** 134 rows, teacher agrees on
  100% of them, mean confidence 0.934, and 95 of 134 sit at ≥ 0.95. t1 requests
  name the operation outright ("grep the tree for …", "search the web for …").
  A model trained on this corpus will look excellent on t1-like rows and that
  will mean almost nothing. **Report t1 and t2/t3 separately, always.**
- **t2 is where the value is and where the labels are least confirmed.** 106 rows,
  teacher agreement 76.4%, mean confidence 0.777 with a proper spread. It is also
  the band with the least data and 9 thin cells (11–14 rows per op).
- **The whole corpus is synthetic** (`source: template` on all 414). Frames ×
  slots, three registers. It carries no organic user phrasing. Its value is
  coverage of the boundary structure, not of the surface distribution.
- **Object vocabulary is shared with the eval set.** 33 of 150 `utterances_v2`
  rows use a slot object that also appears in this corpus (68 of 69 slot values
  appear here). Rows do not duplicate, but objects recur, so an accuracy gain on
  `utterances_v2` is partly "same nouns". Anything measured on it is optimistic
  relative to genuinely new objects.
- **`decide_pick` is the thinnest class**: 39 rows total, 11 in t2.

## A fair evaluation protocol

1. Fit on `train.jsonl` only. Select hyper-parameters/checkpoints on `val.jsonl`
   (family-disjoint, so this measures generalisation across frames rather than
   memorisation).
2. Score once on `tests/data/utterances_v2.json` — the frozen 150 — and report
   **per band, per op, and macro-F1**, never only the headline.
3. Report the baseline on the same rows or the number means nothing. The recorded
   figures are d1-3B at **74.0%** and the stdlib TF-IDF+SVM at **62.7%**
   (`docs/classical-baseline-2026-11-03.md`).
4. Do not tune on `utterances_v2`. It has already been used as the eval set for
   every arm in this repo; it is now a reported-set, not a development set, and
   each extra look costs it standing.
5. This corpus is for the `useTool` 8-op decision layer only. `routing_382` is a
   different task with a different label space and is out of scope here.
