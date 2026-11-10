# After the rewrite study: what the damage is, and how to move on

2026-11-09, at `cb3d708`. **Phase 0: research and a plan. No code was changed, nothing was
pushed, `service/ti/conf/defaults.json` was not edited, no server was started or stopped.**

Four read-only units ran against the tree: the instrument (margin arithmetic on the recorded
vectors and a trace of how the choice is scored), the capacity question (the repo's own
sensitivity record), the three fix candidates, and the four claims the backlog asserts. Their
charts are `/tmp/ti-phase0/w{1,2,3,4}-*.md`; the numbers below are the ones that survived a
second pass, and the ones I re-derived myself are marked **[verified]**.

---

## 0. The answer in one page

* **The brief's numbers are the archived reading, and the archive is not the model.**
  80.5% → 77.3% with 27 broken rows is the 2026-11-04 pairing of a live arm against an archived
  one; the 2026-11-05 correction replaced it with a live-vs-live 78.3% → 77.4%, 16 fixed /
  20 broken, p = 0.618, and 18 broken (not 27) on the six untouched classes. **[verified]**
* **The 2026-11-05 correction is right about the result and wrong about the cause.**
  74 of the archive's 415 rows (17.8%) were never answered by the model. They are
  `heuristic_micro` pre-gate hits: `ms` 0.017–0.089 s, vector `{1.0, seven × 1e-6}`,
  `conf` 0.99 exactly, 74/74 correct, and all 74 are `web_fetch` (38) or `reddit_digest` (36).
  Ten of the eleven rows the correction calls "session drift" are those gate rows. **[verified]**
  Recorded live arms contain zero such rows. **[verified]** So the two arms were recorded by
  two different backends, and the honest reproducibility floor on this probe is **1 changed row
  in 341** (0.29%), not 9 in 415. **[verified]**
* **The damage is option entanglement in transport, semantic in selection, and not a capacity
  effect.** The eight descriptions are rendered into one prompt and read out of one next-token
  distribution through one softmax at T = 1 with a fixed order, so the vector sums to 1 and
  editing one string renormalises the other seven. The two edited options gain +0.127 of the
  vector in total, and the donor is each row's own winner. Which rows break is not random and
  not "nearest in meaning": 17 of the 18 untouched-class breaks land on an edited option (14 on
  `decide_extract`, 3 on `filesys_grep`, 1 on `filesys_glob`), and the same edit costs nothing on
  the 254 rows whose baseline margin was ≥ 0.70. Capacity is a modifier, not the cause: Q4 → Q8 moves 4 of 150
  argmaxes and one row, and an A/A repeat moves none. **[verified]**
* **The only clean number in the whole thread is the live-vs-live one, and it is a wash.**
  Nothing in this phase turns −4 rows into a direction.
* **Two of the six "unbuilt" items do not mean what the brief says they mean.**
  The `pytest-subtests` premise is **false** — pytest 9.2.0 has subtests builtin, and the suite
  is `852 passed, 586 subtests passed`, exit 0, exactly as `README.md:125` claims. The classical
  backend at temperature 1.0 is not the first defect: its corpus points at `registry_382.json`,
  whose 17 tool names share **zero** names with the 8 ops, so it answers `None` on 415/415 rows
  at any temperature. **[verified]**

---

## 1. Three corrections to the numbers this task was written on

### 1.1 The brief cites the superseded reading

| reading | how the arms were recorded | result | status |
|---|---|---|---|
| 2026-11-04 (doc body) | live targeted vs **archive** baseline | 80.5% → 77.3%, 16 fixed / **29** broken, p = 0.0725 | superseded |
| 2026-11-05 (correction) | live targeted vs **live** baseline | 78.3% → 77.4%, 16 fixed / **20** broken, p = 0.618 | current |
| correction, six untouched classes | live vs live | 86.2% → 80.4%, 0 fixed / **18** broken | current |
| correction, test side | live vs live | 165 → 165, 7 fixed / 7 broken, p = 1.000 | current |

`docs/rewrite-transfer-2026-11-04.md:329-512`. The `+17.3` on the grep half and the "extract half
was a fit" both survive unchanged; what changes is that the whole-set number is a null, not a
failure, and that the damage is 18 rows, not 27.

### 1.2 New: the archived baseline is 18% heuristic gate, and that is what "drift" was

`bench/results/p2-heldout/v3-d1-3B-Q4.json` carries 415 `per_row` records. 74 of them have:

| field | value |
|---|---|
| `ms` | 0.017 – 0.089 s (the other 341: p50 8 225 ms) |
| `probabilities` | one label at exactly 1.0, the other seven at exactly 1e-06 |
| `conf` | exactly 0.99 |
| `correct` | 74 of 74 |
| `label` | `web_fetch` 38, `reddit_digest` 36 |

That is the fingerprint of `service/ti/decide/backends/heuristic_micro.py:28-30`, which returns
0.99 for a web-fetch or reddit-digest keyword hit. The archive was recorded through
`bench/live_probe.py`, which builds its backend through the service's resolver, so
`decide.routing = "heuristic_micro_then_d1"` (`defaults.json:3`) short-circuits confident rows
and the recording is not a recording of the model. `bench/rewrite_transfer.py` talks to
`/v1/systemone` through `routing_diagnose.Server` (`:54`, `:141`), so its arms never contain a
gate row. **[verified: 830 rows in `bench/results/p3-transfer/rows.jsonl`, zero gate rows in
either arm]**

Where else it bites **[verified]**:

| recording | rows | gate rows | reported accuracy | model-only accuracy |
|---|---:|---:|---:|---:|
| `p2-heldout/v3-d1-3B-Q4.json` (v3 baseline) | 415 | 74 | 80.5% | **76.2%** (260/341) |
| `p1-diagnose/baseline_live.json` (v2 baseline) | 150 | 11 | 74.0% | **71.9%** (100/139) |
| `p3-transfer/rows.jsonl`, `arm: baseline` | 415 | 0 | 78.3% | 78.3% |
| `p3-transfer/rows.jsonl`, `arm: targeted` | 415 | 0 | 77.4% | 77.4% |

Consequences, in the order they matter:

1. **The 74.0% → 82.0% in-sample claim compared a gated baseline with a raw rewrite.** It was
   already retired as in-sample; it was also measured against the wrong baseline.
2. **The "9 rows of drift" was not drift.** Ten of the eleven changed rows are in the two classes
   the gate covers, and the gate answers them right where the model answers them wrong. The
   live baseline replaced those rows with model answers, which is why the archive looked 9 rows
   too good. `docs/rewrite-transfer-2026-11-04.md:481-489` diagnoses this as session drift and
   prescribes latency-matching; latency was not the variable.
3. **The three-point protocol in that section is still correct about what to *do*** (both arms
   live, one session, p50 recorded, never leave a control arm unread) but its stated cause is
   wrong, and one item needs adding: assert per row that the recording is model-only.
4. **Anything quoting 80.5% or 74.0% is quoting an instrument that answered 18% of the rows
   itself.** `bench/data/heldout_v3_NOTES.md:107` ("the endpoint is deterministic, 0/150 argmax
   disagreement across two builds") is a bound on the endpoint, measured on rows the gate was
   answering.

This does **not** change the wash verdict, because the verdict rests on the two live arms, and
both are model-only and in one session.

### 1.3 The reproducibility floor is 1 in 341, not 9 in 415

Join the archive to the live baseline on `utterance` (all 415 join), drop the 74 gate rows, and
compare correctness: **341 rows, 1 changed** (0.29%). The v2 pair agrees too: re-running
d1-3B-Q4 on the same 150 rows moved **0 of 150** argmaxes
(`bench/results/dell/v2-d1-3B-Q4.json` vs `-run2.json`, `bench/results/dell/d1-3B-Q4-dell*.json`
50/50 too). **[verified]**

So the edit's 36 flips are about 30× the floor, and I agree with unit W1 that the instrument is
not the cause. It is still a real second-order term: a single order reversal on 46 rows moves the
argmax on **8 of 46 (17.4%)**, and swapping the adapter from `systemone` to `logit_read` moves
**28 of 150** rows for +1 net row — both the same size as the edit's own 16.4% (68/415). Nothing
in this repo has ever permuted the option order more than once.

---

## 2. Why a 3B loses rows it was never asked about

### 2.1 The instrument is one softmax, and the eight options are one distribution

From unit W1's trace (cite it, do not re-derive): one `/v1/systemone` request carries the
utterance as `state` and the eight `OP_DESCRIPTIONS` as `criteria` on a `choice` question
(`bench/routing_diagnose.py:112-134`); the server renders state and options into a single prompt,
reads the eight label logits out of **one** next-token distribution at the last prompt position,
runs a joint softmax at T = 1.0 with no temperature metadata, and takes the argmax. There is no
per-option scoring, no order averaging for this template, and the order is fixed.

That single fact does most of the work: **Σp = 1 is not a modelling choice here, it is the
readout**, so any change to any string re-normalises all eight. "Two lines changed, six classes
insulated" was never physically available.

### 2.2 The transport is entanglement; the selection is semantic

The recorded vectors say both things at once **[unit W2, recomputed from `rows.jsonl`]**:

* the two edited options take +0.127 of the vector in total, supplied by the six others;
* `p(decide_extract)` rises on 98–100% of rows in **every** class, including the untouched ones,
  and still rises on the 347 rows whose argmax never moved — median +0.054;
* the donor is each row's own winner, not a proportional slice (pick-truth rows lose pick
  −0.108; web_search-truth rows lose web_search −0.083).

And the recorded breaks are not a random sample of that pressure:

| fact | value | what it rules out |
|---|---|---|
| untouched-class breaks that landed on an edited option | **17 of 18** (14 extract, 3 grep) | "the renormalisation alone decides" |
| broken rows' median baseline margin | 0.20 (kept rows: p50 0.885) | "the broken rows were coin-flips" |
| rows with baseline margin ≥ 0.70 (top1 − top2) that broke | **0 of 254** | "margin does not matter" |
| broken rows where baseline rank 2 was extract | 10 of 20 | "damage is spread evenly" |
| correlation of break with the margin the option already held | **r = +0.97** (grep) | — |
| correlation of break with lexical similarity to the edited text | **r = +0.16** (extract), **−0.46** (grep, own class out) | "nearest in meaning wins the row" |

So: the mechanism by which a row can move without being mentioned is entanglement; the reason
*these* rows moved is that they were already thin and already leaning toward the option whose
text was widened. The widened `decide_extract` text ("queries, keywords, paths, names … words
from the request") is a description of the *object* of a request, not of an operation, which is
exactly what a noun-routing model keys on. That is the diagnosis `p1-diagnose` made, reproduced
one level deeper.

### 2.3 Capacity is a modifier, not the cause

The discriminating measurement is *the same edit at two capacities or two quantisations on the
same rows in the same session*, and **this repo has never run it.** What it has:

| measurement | model / quant | n | effect | source |
|---|---|---:|---|---|
| Q4 vs Q8, same rows, same session | d1-3B Q4_K_M vs Q8_0 | 150 | **4 argmax moves, 111 → 110 correct**, mean per-label \|Δp\| 0.0104 | `bench/results/dell/v2-d1-3B-Q{4,8}.json` **[verified]** |
| A/A repeat, same session | d1-3B Q4 | 150 | 0 argmax moves | `...-run2.json` **[verified]** |
| option count 8 → 6 | d1-3B Q4 | 46 | **+1 row**, 30.4% → 32.6% | `p1-diagnose/FINDINGS.md:125-132` |
| option order reversed | d1-3B Q4 | 46 | 8/46 argmax moves, 30.4% → 28.3% | `p1-diagnose/FINDINGS.md:134-138` |
| global rewrite (8/8) | d1-3B Q4 | 150 | 111 → 106, `decide_pick` 18/18 → 5/18 | `p1-diagnose/FINDINGS.md:206-210` |
| a 35-rule cue table | no model | 150 | **78.7%**, beats the 3B's 74.0% (p = 0.040) | `docs/local-classifier-cascade-2026-11-03.md:111` |
| distilled 3B-class classifier | trained | 150 | 68.7%, loses to the 3B | `docs/distill-classifier-2026-11-04.md` |

Four times the weight precision buys one row. Six options instead of eight buys one row. A
hand-written rule table beats the model. The information is in the surface form and the model is
not short of bits, it is short of *separation*: the failure is thin margins, and a description
edit redistributes exactly the thin ones. Capacity is neither the cause nor the cure — but that
is an inference from a model *class* comparison, not from the experiment that would settle it,
and I am not going to present it as the latter.

### 2.4 The three hypotheses, scored

| hypothesis | verdict | evidence | what would still change it |
|---|---|---|---|
| **instrument** (probe noise, order, drift) | **ruled out as cause**, kept as a second-order term | 36 flips vs a 1/341 floor; broken rows were not coin-flips; 17/18 land on an edited option | a same-session A/A at 415×2 showing a floor of the same size |
| **option entanglement** | **the mechanism**, confirmed | Σp = 1 by readout; +0.127 to the edited options from each row's own winner; mass moves on 100% of rows and on 347 rows whose argmax did not | an option set scored independently per option would break the argument; this server does not offer that |
| **capacity** | **modifier, not cause**; the decisive test has never been run | Q4 → Q8 = 4/150 argmax; 8 → 6 options = +1 row; a 35-rule table beats the model | the same edit on Q8, or on a second 3B, same rows, same session |

---

## 3. The cheapest experiments that separate them

Costs use this repo's own recorded latency: 3.959 s/row on the live baseline arm
(`docs/rewrite-transfer-2026-11-04.md:342-346`), 1.055 s per decide hop. No money is involved in
any row below; the one paid lane is named at the end.

| # | experiment | what it separates | calls | wall clock | needs |
|---:|---|---|---:|---|---|
| E0 | **gate audit of every recording** (Appendix A) | instrument identity | **0** | 5 s | nothing — **done in this phase** |
| E1 | **A/A interleaved at 415 × 2**, same session | instrument floor at the reporting unit | 830 | ~55 min | llama-server on :8080 |
| E2 | **semantically null one-string edit** (same meaning, new words), 415 rows | "this rewording" vs "any rewording" — the entanglement test that costs one arm | 415 | ~27 min | :8080 |
| E3 | **3 seeded order permutations** of the unchanged eight, 1 245 rows | pure instrument: position vs text | 1 245 | ~82 min | :8080 |
| E3, half-run 2026-11-10 | **one full reversal**, 415 rows | position alone, 8.2% of argmaxes moved, 9 rows net, p = 0.078 — see `docs/grep-arm-2026-11-10.md` §E3 | 415 | 21 min | done |
| E4 | **grep-only arm**, both arms live in one session | the shipping decision (see (a)) | 830 | ~55 min | :8080 |
| E5 | **grep-only arm on Q8_0** | capacity, the one test never run | 830 | ~55 min + a serve | a second gguf, **not on this box** |
| — | **lexical/embedding distance vs the mass ledger** | "nearest in meaning" | **0** | minutes | nothing — **done by W2; r = +0.16 / −0.46** |

Already answered with zero calls, so do not spend a server on them: E0 and the similarity test.
E1–E4 can run as one session of about 3 320 calls (~3.7 h) with the A/A rows interleaved into the
others so the floor is measured on the same server at the same time; that is what makes the
protocol in §4 checkable rather than aspirational.

What no experiment on this list settles: whether real user phrasing behaves like v3's frames
(every row here is a slot-filled authored sentence), and whether an independent-per-option
readout would remove the coupling (it would be a different probe, not a different arm).

---

## 4. The re-baseline protocol

Six rules. They are what the record needed and did not have; the first three would have caught
the 74 rows.

**P1 — One session, both arms, one probe.** The baseline and the treatment are recorded in the
same session, through the same code path, against the same served model, and `/props` is asserted
before the first call and after the last. Record `ts` and `ms` per row. Refuse to report a delta
when the two arms' p50 differ by more than 20%.

**P2 — Assert the recording is model-only.** Every row must have `ms` above a floor (1 s on this
box) **and** no `{1.0, seven × 1e-6}` vector. A row failing either is a gate answer and is
excluded and counted, never silently kept. `bench/live_probe.py` goes through the service resolver
and `bench/rewrite_transfer.py` does not; until that is unified, cross-probe comparisons are not
comparisons. *(This is the rule the 2026-11-05 correction did not have, and its absence is the
entire "drift" episode.)*

**P3 — An archive needs a gate audit and a per-class drift bound.** Re-run a hash-chosen sample
of *every* class, sized to the smallest per-class cell the report will quote, and publish the
drift beside the delta. A bound measured on one class does not bound another.

**P4 — Measure the floor in the same session.** Re-ask a fixed 20-row hash sample at the start
and the end of the run and report the A/A flip count. A delta smaller than 3× the floor is not a
result. On this probe the floor is expected to be 0–1 rows.

**P5 — Report three units, not one.** Rows (with the exact McNemar), families (sign test), and
**the per-option ledger** — the unit that was actually edited. The grep half is why: 9 rows
bought, 3 broken, net +6 on the option, and those 9 rows are 3 authored families (p = 0.25).

**P6 — Pre-register the pair and the prediction, and name the arm.** The option dictionary's
identity (a hash of the eight strings) goes into the artifact, because the option set *is* the
instrument. `report()` in `bench/rewrite_transfer.py:368-375` currently drops every record whose
arm is not `targeted`, and reads the archive for the baseline at `:358`; until both are fixed, its
output is the archive comparison, not the result.

---

## 5. How to move on

### (a) Ship the grep wording alone — prepared, not wired

**The text**, verbatim from `bench/routing_diagnose.py:289-290` **[verified]**:

```
filesys_grep: "Search inside local file contents for a pattern or string; the answer is which
              code or text contains it."
```

against the frozen instrument's `bench/ti_bench/prompts.py:16`
`"Search inside file contents for a regular expression."`

**The change, prepared and not applied** — one line and one dict key, no string retyped:

```python
# bench/routing_diagnose.py, after line 293
TARGETED_GREP = dict(OP_DESCRIPTIONS)
TARGETED_GREP.update({"filesys_grep": TARGETED["filesys_grep"]})

# bench/rewrite_transfer.py:64
ARMS = {"baseline": OP_DESCRIPTIONS, "targeted": TARGETED, "grep": TARGETED_GREP}
```

`--arm` is a closed enum (`rewrite_transfer.py:456`), so this is the whole plumbing: `run()` keys
off `ARMS[arm]` at `:174` and already writes `arm` per row at `:204`, so the rows land in
`rows.jsonl` beside the others and the A/A logic keys on `(arm, id)` at `:109-117`. Trap:
`v2_spot_check` chooses its archive with `V2_TARGETED if arm == "targeted" else V2_RUN`
(`:138`), so a grep arm would be spot-checked against the wrong archive; the spot check is not
part of this arm.

**Cost.** 415 × 3.959 s ≈ 27 min per arm; the protocol pair is ~55 min. `models/d1-3B-Q4_K_M.gguf`
is on disk, `extern/llama.cpp/build/bin/llama-server` is on disk, and nothing is listening on
8080 today. `--expect-model d1-3B-Q4_K_M.gguf` (byte compared, `model_guard.py:40-42`).

**The pre-registered prediction, so the arm can lose.** grep +9 / −3 (69.2% → 86.5%, the cell is
identical under both baselines); `decide_extract` unchanged at 20/51 (39.2%); `web_search` −3,
not −8, because 5 of its 8 were the extract text's captures. Whole-set delta between −1 and +5.

**Ship it if and only if** the pre-specified pair clears p < 0.05 in the same session, the
`web_search` loss is ≤ 3 rows, and the untouched-class ledger is not net-negative. Anything less
and it stays a bench arm. Note what the record already implies and I will not oversell: the 9
grep rows are **3 authored families**, so the row-level p = 0.004 is a sentence-level effect
worth p = 0.25. It is real and it is small, and it is the only option string in this repo with a
positive out-of-sample ledger.

### (b) Whether another backend removes the damage — the run is not there

**The backend research run is not in flight and left no artifact.** I looked: `runs.jsonl` in
`~/.pi/agent/subforge` has nothing for this repo since `i1-verdict` on 2026-11-05; no `pi`
process is alive on the box; the fleet's "Research and test toolInterfacer" pane (`w49:p1`) is a
dead registry row that `read` refuses; no file under `bench/results/`, `docs/`, or `~/reports`
postdates 2026-11-05 in this repo. **[verified]** So (b) is answered from the tree, not from its
result, and if it lands later, read it against §4 before believing it.

What the tree says:

| backend | needs | enabled now | recorded |
|---|---|---|---|
| `d1_systemone` | llama-server | yes (stage 1 of `pair`) | 74.0% @163 ms (v2, gated baseline) |
| `logit_read` | llama-server, any gguf | no | 74.7% @204 ms (v2) |
| `classical` | nothing (in-process) | no | see (c) |
| `heuristic_micro` | nothing | yes (noul) | 7.3%, 139/150 abstain |
| `fixture` | a fixture file | yes (noul) | replay only |
| `openrouter` | `TI_OPENROUTER_API_KEY` | no | 96.0% @2 575 ms, **paid** |
| `pair` | both | yes (choice) | 90.0% @3 118 ms, **paid** |
| `cascade` | llama-server, x2 | off | 79.3% @95 ms on v2, p = 0.152, n = 150 |

**No backend can score the 415 v3 rows through the harness today**: `bench/ti_bench/sets.py:26-29`
has `v1` and `v2` only, `bench/arms.json` has no classical arm, and the ids it mints (`v3:NNN`)
do not join `rows.jsonl`'s (`v3-0001`). Offline, in-process, the classical and heuristic backends
score 415 rows in under a second — but that is a bench call, not an arm.

**The "another backend" that could plausibly remove the damage is another model, not another
transport**, because the coupling is in this prompt's readout and every transport here feeds the
same `/v1/systemone`. The cheap version is Q8 of the same gguf, and **the Q8 gguf is not on this
box** — only `models/d1-3B-Q4_K_M.gguf` and a vision projector. The `v2-d1-3B-Q8*` pair that
exists is a *recording* under `bench/results/dell/`, not a file. Getting it is a download or a
copy from dell321, not a purchase; the hosted lane is ~$0.0022 per 415-row arm at the recorded
per-token rate and is your call, not mine to spend.

### (c) The classical backend: temperature is the second defect, not the first

Temperature enters at `service/ti/decide/linear_text.py:142` as a **divisor** on the margin
vector, so T = 1.0 is the raw softmax and T < 1 sharpens; confidence is `p_max`
(`classical.py:122-123`) and the gate is 0.70 (`defaults.json:82`). Eight close margins put
`p_max` near 1/8, median 0.171 — 1.37× uniform, max 0.371 — so nothing clears 0.70 and coverage
is 0.0%.

**But it cannot answer the 8-op task at all.** `decide.classical.corpus_path` is
`bench/registry_382.json` (`defaults.json:20`), whose 17 tool names (`timer_create`,
`thermostat_set`, …) share **zero** names with the eight ops. `classical.py:103-108` builds
`fitted` from the offer set intersected with the fitted margins, gets nothing, and returns
`choice=None`. **[verified]** The 8-op lane's corpus is `tests/data/utterances.json`
(50 rows), which is what `bench/classical_baseline.py:377-379` passes when it reports 62.67%.

With the corpus corrected, the fit is free and offline (0.16 s in-process, no server, no network),
and unit-measured on 415 rows:

| objective | optimum | what it gives |
|---|---|---|
| ECE | **T ≈ 0.17** (ECE 0.079) | coverage **44.6%** at the 0.70 gate, precision 0.865 |
| NLL | T ≈ 0.20 (0.506) | same operating point |
| accuracy | **invariant, 0.607 at every T** | temperature buys coverage, not correctness |

Three things follow. Accuracy is invariant, so the fit is a coverage calibration and must be
stated as one. 44.6% at 0.865 precision is the same operating point the repo already chose
(`classical-baseline-2026-11-03.md:206`), so this reproduces rather than improves it. And 0.607
against d1's 0.783 on the same 415 rows is a real gap: the classical-front cascade's 79.3% at
95 ms is an n=150 v2 number at p = 0.152, and nothing here shows it holds on v3.

Order of operations, both edits in `defaults.json` and both waiting on you: **(1)** point
`corpus_path` at the 8-op corpus, **(2)** then set the temperature to the fitted value, fitted on
rows held out from the ones it is scored on. The bench form, which needs no conf edit, is
`"override": {"decide.classical.temperature": 0.17}` in a new `bench/arms.json` entry.

### (d) The three unconfigured ops — true, and the fix is smaller than "vendor it"

`web_search.script`, `web_search_provider.script`, `web_fetch.script` ship `""`, `reddit.cwd` and
`cwd_candidates` ship empty, and `docs/ops.md:116-124` says that is deliberate. Each raises
`CliMissingBinary` naming the key to set, before any subprocess runs
(`ops/web/search.py:114-119`, `ops/web/fetch.py:47-52`, `ops/reddit/digest.py:98-107`).

What is not true is that nothing works here: the two web CLIs **exist on this box**
(`~/.pi/profiles/registry/web-search/scripts/free-search`, 49 415 bytes; `.../web-fetch`, 13 970
bytes) — conf just does not point at them. The reddit project genuinely is not on disk
(`~/redditScrapingSearching` and `~/tools/reddit-research` both absent), so `reddit_digest` stays
unrunnable until it is vendored or cloned.

The smallest honest action is a test, not a vendored CLI: **no test anywhere asserts the empty-script
message** (`grep "is not set" tests/` → 0 hits), and the one existing test that looks like it
(`tests/test_ops_web.py:155-165`) sets a *nonexistent* path and exercises a different branch.
One test per op pinning the literal message makes the promise checkable and costs nothing. For
`reddit_digest`, the backlog's "vendor it or delete it" is still the right fork, and it is a
product decision.

### (e) `pytest-subtests` — the premise is false

**`852 passed, 586 subtests passed in 42.73s`, exit 0**, under Python 3.10.12 / pytest 9.2.0 with
`pytest_subtests` **not installed**. Subtest support is a pytest 9 builtin
(`_pytest/subtests.py`, "Builtin plugin that adds subtests support"; `terminal.py:74` lists
`"subtests passed"` as a known type). The standalone PyPI package is not required, and
`README.md:125`'s "852 tests, 586 subtests" is exactly right.

The real finding is smaller and actionable: `pyproject.toml:38` declares
`dev = ["pytest", "numpy"]` with **no floor**, so the 586 figure is a property of whatever pytest
the runner resolves, not of the project. `ci.yml:27-32` installs the same unpinned set. One line,
`dev = ["pytest>=9", "numpy"]`, makes the published count stable. (`tests/` has 66 `subTest` call
sites across 16 files; under a pre-9 pytest the code still passes but the summary loses the
figure.)

### (f) L3 and L4 — everything is honestly marked unbuilt, which makes deletion cheap

`service/ti/planner/free.py` is one line: `"""free — L3 placeholder: model-planned DAG (v2)."""`.
L4 has **zero** hits for browser/playwright/mcp under `service/` and `tests/`. Both are labelled
"not built" everywhere they appear: `spec/1.0-spec.md:36,123,202` and `:38`; `README.md:132-137,162`
and `:163-164`; `ARCHITECTURE.md:82-83,184-186`; `TODO.md:129-133,151-155`. Two corrections to the
backlog's own reasoning: `docs/roadmap-2026-11.md` makes **no** L3 promise (0 hits), and
`README.md`'s capability table (`:117-119`) has **no** L3 or L4 row — only `ARCHITECTURE.md:78-83`
has a table with both.

| | delete the promise | build it |
|---|---|---|
| **L3** | ~10 doc lines, delete one file, remove one import-list entry | a DAG scheduler: `core/runner.py` walks a linear step list with an `over=` fan-out and has no `depends`/graph machinery (0 hits); plus a planner, an inspector branch, and new tests |
| **L4** | 4 doc lines, no code (nothing to delete) | a new op package, a new conf block, an inspector route, tests, a browser dependency that contradicts `pyproject.toml:35` and fails `ci.yml:132-135`, plus the hard-coded op count at `ci.yml:170` |

Deletion is an order of magnitude cheaper for both, and cannot create a contradiction because
every line it touches already says "not built". For L3 there is a third option worth naming
because it is the cheapest of all and is honest: keep the level in the spec, drop it from the
README's gap list and the TODO item, and let the spec's own "(v2)" tag carry it as future work.

---

## 6. What I did not verify

* **I did not run a live probe.** No llama-server was started, nothing was sent to :8080, and the
  3 320 calls in §3 are estimates from the recorded 3.959 s/row, not measurements. `127.0.0.1:1113`
  is a live service that is not mine and was not touched.
* **The instrument trace in §2.1 is unit W1's reading of the vendored llama.cpp source**, not mine.
  I verified its output (the fingerprint, the arithmetic, the arm composition) but not its
  `server-decision.cpp` / `server-context.cpp` line citations.
* **I did not re-run the classical fit.** T ≈ 0.17 / coverage 44.6% / accuracy 0.607 are unit
  W3's in-process numbers; the recipe is reproducible in under a second and should be re-run
  before it is quoted, because it is currently in-sample on v3.
* **I did not confirm that the archive's 74 gate rows were produced by
  `heuristic_micro_then_d1`.** That is inference from the fingerprint, the conf value, the two
  classes, and the routing key. The git history of the recording script was not opened, and I did
  not reproduce a gate row.
* **The Q4/Q8 comparison is n = 150 on v2, not on v3**, and it is a *different* question from the
  one §3's E5 asks (it perturbs the quant, not the option set).
* **The 1/341 floor is one archive/live pair**, and the archive ran at 2.1× the live p50, so it
  bounds that pair only. The clean floor is the 0/150 A/A repeat.
* **No p-value in this document is mine.** 0.618, 0.004, 0.008, 0.0725, 0.25, 0.040 are from the
  cited documents or from unit W2's recomputation over `rows.jsonl`.
* **`git status` is clean and nothing was pushed.** This document is the only file written.

---

## 7. Decisions for you

1. **(a)** Apply `TARGETED_GREP` (one line, one dict key) and run E1–E4 as one ~3.7 h session on
   the local model — no money, but the box is busy for that long. Alternatively run only E4
   (~55 min) to settle the shipping question first and take the rest later.
2. **(c)** The order: point `classical.corpus_path` at the 8-op corpus first, then set the
   temperature. Both are `defaults.json` edits and I have not made either.
3. **(f)** L3 and L4: delete, or (L3 only) keep the spec tag and drop the README/TODO promises.
4. **(b)** If the backend research run exists somewhere I could not read, point me at it. If it
   does not, say whether Q8 belongs in the next session — that is the only capacity test worth
   running and it needs a gguf that is not on this box.

---

## Appendix A — the gate audit, reproducible

Twelve lines, no server, five seconds, and it is the check that would have caught the 74 rows:

```python
import json, glob
def gate(r):
    if not isinstance(r, dict):          # p1-diagnose/orders.json stores ids, not rows
        return False
    v = sorted((r.get("probabilities") or {}).values(), reverse=True)
    return (r.get("ms") or 99) < 1.0 and v and abs(v[0] - 1.0) < 1e-9 \
           and all(abs(x - 1e-6) < 1e-15 for x in v[1:])
for p in sorted(glob.glob("bench/results/**/*.json", recursive=True)):
    d = json.load(open(p))
    rows = d.get("per_row") if isinstance(d, dict) else None
    if rows and (g := [r for r in rows if gate(r)]):
        print(f"{len(rows):5d} rows  {len(g):4d} gate  {d.get('accuracy')}  {p}")
```

Its output on this tree is the table in §1.2 plus the `heuristic_micro` recordings (11, 11 and 6
gate rows, which are the heuristic scoring itself, correctly) and the iGPU live probe (6).

## Appendix B — the four unit reports

| unit | file | what it holds |
|---|---|---|
| W1 instrument | `/tmp/ti-phase0/w1-instrument.md` | probe trace, margin quartiles per split, the 20 broken rows, the floor recomputation, the gate fingerprint |
| W2 capacity | `/tmp/ti-phase0/w2-capacity.md` | the 31-row evidence table, per-class mass ledger, the never-measured gaps, the ranked experiment list |
| W3 fixes | `/tmp/ti-phase0/w3-fixes.md` | the arm plumbing, the backend table, the classical fit |
| W4 unbuilt | `/tmp/ti-phase0/w4-unbuilt.md` | the four claims, the suite run, the L3/L4 touches |

They live in `/tmp` and will not survive a reboot. Everything load-bearing is restated above with
its source; if a unit file is gone, its numbers are still checkable from the committed artifacts.
