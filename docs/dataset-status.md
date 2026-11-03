# Dataset status — what the runtime evidence actually contains

Measured, not estimated. Snapshot: `service/logs/hops.jsonl` at 1398 lines
(mtime 02:39:16) and `service/logs/live.jsonl` at 13 lines. **The hop log is live
output** — it was 1202 lines when this work started and 49 lines in this
worktree's own copy after one test run — so re-running `ingest` against a later
snapshot will report different line counts. Every count below is stated per
source so a mismatch is visible rather than confusing.

## Headline

**`hops.jsonl` yields 14 trainable choice records, and all 14 carry the same
label.** (15 on the current log — the re-check immediately below is the current
number and the 14 is the snapshot this section describes.) It is not a decision
log. It is an execution trace with an intent string attached to some hops, and the
three things a router would need — a state, a candidate set, and the answer the
model picked — are not in it. What it has is which op *ran*.

**Re-checked 2026-11-03 15:48 EST against the live log, which has since grown.**
The same rule, run over the current `service/logs/hops.jsonl` (**5,243 lines**, up
from 1,398; `live.jsonl` unchanged at 13) yields **15** records instead of 14, and
**all 19 observed records — 15 from `hops.jsonl`, 4 from `live.jsonl` — are still
`filesys_glob`**. So the count is a snapshot that moves with the log and the
finding does not: entropy is still zero and the file is still not a decision log.
The per-op table below describes the 1,398-line snapshot; on the current 5,243
lines the three kinds are `filesys_glob` 2,400, `decide_pick` 1,501,
`filesys_read` 1,342, with 2,985 hops rejected as contradictory and 1,484 for no
intent. `TI_DATASET_LOGS=<live log dir> TI_DATASET_OUT=/tmp/<dir> python3 -m
tools.dataset ingest` reproduces both. *(Checked 2026-11-03 16:2x EST: the rule in
`tools/dataset/ingest.py` — `soft.intent_context` non-empty, op in the 8 routable
set, op consistent with the intent across the whole log — is unchanged since
`fb19e16`, so both counts come from one rule and neither is a rule change wearing a
snapshot change. The per-op figures above add up to the line count
(2,400 + 1,501 + 1,342 = 5,243). They could not be re-measured from this worktree:
`service/logs/` is git-ignored and absent here, so the 5,243-line snapshot is taken
as recorded rather than re-counted.)*

Worse for training: because every surviving record is `filesys_glob`, a model
trained on this file learns "answer `filesys_glob`" and nothing else. The label
distribution is 14/14, entropy zero.

## hops.jsonl, per op kind

| op | hop lines | with `intent_context` | records kept | why the rest are dropped |
|---|---|---|---|---|
| `filesys_glob` | 622 | 622 | **14** | 608 lines are repeats of an intent already counted, or contradictions |
| `decide_pick` | 416 | 17 | **0** | 399 lines are pure execution trace (no intent); the 17 that have one share intents with `filesys_glob` |
| `filesys_read` | 360 | 360 | **0** | every one of its intents was also executed as `filesys_glob` — no single correct answer |
| **total** | **1398** | **999** | **14** | 399 dropped for no intent, 815 for contradiction |

The 14 records, with how many hop lines each one summarises:

| # | observations | label | intent |
|---|---|---|---|
| 1 | 1 | filesys_glob | web_search: Sam Cooke "Bring It On Home to Me" … |
| 2 | 75 | filesys_glob | list the files |
| 3 | 18 | filesys_glob | what files are in the ghost directory |
| 4 | 18 | filesys_glob | list the files in the bench directory |
| 5 | 18 | filesys_glob | list all the files in the repo |
| 6 | 18 | filesys_glob | ls |
| 7 | 15 | filesys_glob | list the .txt files |
| 8 | 15 | filesys_glob | list the .sh files in the bench directory |
| 9 | 1 | filesys_glob | web search for Clickteam Fusion 2.5 asset extraction tools |
| 10 | 1 | filesys_glob | Find published measurements of ultra-wideband ranging accuracy |
| 11 | 1 | filesys_glob | search the web for the Vasarhelyi 2018 Science Robotics paper |
| 12 | 1 | filesys_glob | Find the arXiv ID and title for the paper "MCPTox" |
| 13 | 1 | filesys_glob | Search the web for measured ESP-NOW latency and throughput |
| 14 | 1 | filesys_glob | Find Canadian court judgments describing controlled delivery |

Rows 1 and 9–14 are **observational mislabels**, not training data: seven
web-research intents that the router executed as `filesys_glob`. The consistency
rule cannot catch them — each appears once, so no contradiction is visible — and
they are only identifiable by reading them. They are why `hops` records are
`label_source="observed"` and are excluded from the SFT export by default. The
seven sound rows (2–8, 177 of the 184 observations) are real: a listing intent
whose routing decision really was `filesys_glob`.

Note what is missing from the log entirely: `filesys_grep`, `decide_extract`,
`web_search`, `web_fetch`, `reddit_digest` never appear as hops. Six of the eight
routable ops have no runtime evidence at all.

## live.jsonl

13 sessions, 4 of which routed before erroring. The other 9 died on timeouts or an
`unknown op '__decide_extract__'` error and produced no hop.

All 4 records are `filesys_glob`, and two of them are visibly not glob decisions:
"read the service logs live.jsonl and summarize the last few records" ran
`filesys_glob → decide_pick → filesys_read` (the glob is candidate discovery, the
answer is read), and "Find Canadian court judgments…" is web research. Treat the
live contribution as 4 candidate rows of which 2 are sound.

## The frozen files are the corpus

The only gold labels in the repo are the vetted utterance→label pairs:

| source | rows | task | labels |
|---|---|---|---|
| `tests/data/utterances.json` | 50 | routing | 8 ops |
| `tests/data/utterances_v2.json` | 150 | routing | 8 ops, bands t1/t2/t3 |
| `tests/data/shape_expectations.json` | 150 | shape | singular / enumerative |
| `tests/data/heldout_phase3.json` | 76 | shape | singular / enumerative |
| `tests/data/heldout_phase4.json` | 70 | shape | singular / enumerative |

The five files share no utterance with each other (checked), so every gold record
is a distinct row. 496 gold records, 18 observed, 514 total — the observed half
being 14 from `hops.jsonl` plus 4 from `live.jsonl` on this snapshot; on the
5,243-line log measured above it is 19 observed and 515 total — the same
arithmetic, with one more row from a longer trace.

## What the pipeline emits

`TI_DATASET_LOGS` and `TI_DATASET_OUT` select the inputs and the derived-data
directory (`service/logs` and `var/dataset` by default, both repo-relative;
`var/dataset/` is git-ignored).

```
python3 -m tools.dataset ingest     # 514 records -> corpus.jsonl + ingest_report.json
python3 -m tools.dataset validate   # schema gate, exit 1 with every error listed
python3 -m tools.dataset split      # 415 / 51 / 48
python3 -m tools.dataset export     # sft.jsonl (200), manifest.jsonl, labels.json, eval_*.json
python3 -m tools.dataset stats      # counts, label distribution, band coverage
```

- `sft.jsonl` — 200 rows, `{"messages": [system, user, assistant]}`, assistant is
  a bare letter. Only `routing.ops_v1` gold records: 50 from v1 plus 150 from v2.
  The 314 skipped are 296 shape-task rows (a different question) and 18 observed
  rows (an unverified answer).
- `labels.json` — the letter legend, A–H → op, plus the option text sent to the
  model. A test asserts this text still matches `bench/live_probe.py`.
- `eval_val.json` / `eval_test.json` — the shape `bench/live_probe.py --utterances`
  already reads: a JSON array of `{"utterance", "label"}`, plus `band` and
  `tier_target` where the source declares them (the harness ignores extra keys).
  20 and 18 rows.

## The rule for a trainable record

A hop becomes a `choice` record only if (1) `soft.intent_context` is non-empty —
that is the only field that can serve as a state; (2) the op is one of the 8
routable ops — otherwise the label is not an option of the question; and (3) the
op is a consistent function of the intent across the whole log. An intent executed
as both `filesys_glob` and `filesys_read` has no single correct answer, so it is
dropped rather than resolved by majority vote — a vote would invent a label the
log never asserts. Surviving hops dedupe to one record per intent: 75 repetitions
of `list the files` are one decision observed 75 times.

The rejected alternative: segment hops into sessions by timestamp gap and take the
first hop as the routing decision. That is closer to how the runtime behaves — a
plan really is glob, then pick, then read — but the log carries no session id and
the gap distribution has no valley (272 gaps above 10 ms, 190 above 50 ms, 63
above 5 s), so any threshold is a tuning knob. Measured anyway: at a 1 s cutoff it
gives 12 intents instead of 14, every one still `filesys_glob`. The conclusion
does not depend on the threshold; the threshold-free rule is the one implemented.

## Split

415 train / 51 val / 48 test (80.7 / 9.9 / 9.3 — the drift from 80/10/10 is
group granularity: the v2 and shape rows for one utterance are one group of two
rows and cannot be halved). Every op and every band appears in every split.

Leakage key: `u:<lowercased, whitespace-collapsed utterance>` for text-derived
rows, `s:<session ts>` for live sessions. A whole group goes to one split. This
also keeps a v2 routing row with its `shape_expectations` annotation, since both
carry the same utterance.

**Residual leakage it does not prevent:** paraphrase families. `heldout_phase3`
declares a `family` such as `listing-extension`; two members of one family can
land in different splits with different utterances and the same structure.
Grouping on family instead would pack 146 held-out rows into 24 families
("listing-extension" alone covers dozens of rows) and make the split coarser than
the measurement it exists to serve, so utterance identity is the deliberate
trade. Anyone quoting a held-out number off this split should know that.

## Reproducing

`tests/test_dataset_pipeline.py` builds its own fixtures in temp dirs and touches
no real log, so the suite passes on a machine with no `service/logs/` at all.
To reproduce the numbers above, point `TI_DATASET_LOGS` at a directory containing
the live `hops.jsonl` and `live.jsonl`.
