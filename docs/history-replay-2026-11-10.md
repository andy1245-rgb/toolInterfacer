# History replay: real requests from the pi corpus, and what they can and cannot answer

2026-11-10. Collection only. `bench/history_replay.py` reads pi session transcripts, pairs every
user request with the tool the assistant actually called and how that call ended, and asks the
shadow's own logic what it would have picked. It executes nothing, calls no model, and writes no
request, argument or path — the corpus is real traffic, so only counts and rates leave it
(`docs/harness-spec-2026-11-10.md`, the data rule).

```bash
python3 bench/history_replay.py --max-sessions 400 --max-pairs 20000 --json /tmp/replay.json
```

## What the corpus yields

| | |
|---|---|
| transcript files scanned | 203 |
| **usable request/outcome pairs** | **20 000** (capped; the walk stopped there) |
| sessions with at least one pair | **182** |
| distinct tools | 28 |
| tool-call error rate | **4.4%** |
| agent-management tool share | 16.7% |

A pair is a user message, the tool call in the assistant turn that followed it, and the matching
tool result's own `isError` flag. Pairs without a result, or without a preceding user message, are
dropped rather than guessed.

| tool | pairs | share | error rate | p50 |
|---|---:|---:|---:|---:|
| `bash` | 14 137 | **70.7%** | 5.7% | 59 ms |
| `forge_result` | 1 393 | 7.0% | 0.2% | 97 ms |
| `forge_run` | 1 327 | 6.6% | 0.1% | 19 ms |
| `edit` | 765 | 3.8% | 7.1% | 8 ms |
| `read` | 732 | 3.7% | 0.3% | 13 ms |
| `write` | 776 | 3.9% | 0.1% | 5 ms |

## What the shadow would have chosen — the two answers

**1. The shipped member cannot speak this vocabulary at all.** Its eight ops
(`filesys_read`, `web_search`, …) share **zero** names with the assistant's tools
(`bash`, `read`, `edit`, …):

```
pairs where the eight ops overlap the session's tools at all: 0 / 20000
answered 0 (0.0% of pairs)
```

That is the same `vocab_match: 0.0` the live shadow log records on `decide_pick`, at corpus scale.
The measurement is not wrong; the member is pointed at the wrong vocabulary, exactly as the
classical tier was for its whole dead life.

**2. A member fitted on this corpus still loses to a constant.** Fitting a word+char TF-IDF
logistic model on the 24 tools with ≥5 examples, held out **by session** (182 sessions, 5 folds):

```
19 990 pairs over 182 sessions, accuracy 68.8% [0.681, 0.694]
per fold [0.704, 0.663, 0.691, 0.688, 0.694]
coverage at 90% precision: 0.4%
the constant answer 'bash' scores 70.7% on the same pairs — worse than always answering 'bash'
```

**The classifier is 1.9 points below the majority class, and no threshold reaches 90% precision.**
So on this corpus, the user's request text does not predict which tool the agent will reach for, at
least not above "it is almost always `bash`".

## Why, and what it means for shadow mode

The agent's tool choice is not a function of the request text. It depends on what the session has
already done — the file it just read, the command that failed, the artifact it is holding. 71% of
calls are `bash` because most work is shell work, and the interesting 29% are decided by context
that never appears in the user's message.

Three consequences, in order of how much they should change the plan:

1. **This corpus cannot answer the fusion question, and no amount of replay will make it.** It is a
   record of an agent's choices, not of a router's options. The shadow's meaningful traffic is the
   *router's own* decision points — the ones that produce `decision` records — and those only exist
   when the service serves a request.
2. **The shipped member is mute on this traffic** (0 of 20 000). Any shadow collection aimed at the
   assistant's tools needs a member fitted on the assistant's tool vocabulary; the eight-op router
   is the wrong instrument, and `vocab_match` is the field that says so per record.
3. **Even with the right vocabulary, the ceiling here is low.** A text-only member under a constant
   baseline is not a front door on this traffic; it is a way to lose. The 4.4% error rate and the
   per-tool p50s are the useful numbers this corpus does give: they say what a wrong choice costs in
   retries and seconds, and 5.7% of `bash` calls error.

## The honest summary of the two jobs

* **History replay works and yields 20 000 pairs**, and it is safe by construction: it reads JSONL
  and prints counts.
* **It does not give the operator a shadow comparison yet.** The shipped member cannot answer, and a
  purpose-fitted member does not beat a constant — so there is nothing to compare, and saying
  "the shadow agreed X% of the time" would be a statement about a model that is worse than always
  answering `bash`.
* **What would make it informative:** a member for this vocabulary (which the 68.8%-vs-70.7%
  result says is hard), or a corpus of *router* decisions rather than agent decisions — which is
  what the shadow log becomes the moment the service is restarted with it enabled.
