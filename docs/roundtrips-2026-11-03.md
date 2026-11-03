# Does `useTool` actually save round trips?

Measured 2026-11-03 on the CPU-only laptop (`i5-1235U`, 12 threads), nine fixed tasks, two
arms, the real HTTP service with the shipped config. Raw: `bench/results/roundtrips-2026-11-03.json`.
Reproduce: `python3 bench/roundtrips.py --serve --json bench/results/roundtrips-<date>.json`.

## Verdict

**The README's "one round trip for a whole pipeline" holds, but it is not the same claim as
"cheaper".** On this task list the layer removes **2 of 14 round trips (14.3%)** and is
**7.5 s slower** in wall time, because the decisions it makes cost more than the round trips
they replace. The saving only turns positive when a model pass costs more than **~5.8 s** —
equivalently, when the decision costs less than **~1.06 s** *at that pass cost*. The two
thresholds are one point in (pass, decision) space, not two independent ones: at the
headline 2 s pass the break-even is **~303 ms per decision**.

That is a real result about where the layer belongs, and it is narrower than the spec
implies. Two facts produce it:

1. **Half the task list is single-call shapes** (`l0_read`, `l0_grep`, `l1_list`,
   `l2_reddit`): one tool call either way, so the layer can only add overhead. A round-trip
   count cannot improve on one.
2. **The decision is the expensive term on this box.** 10 decide hops cost **10,555 ms**
   (≈1,055 ms each, the CPU llama-server) against **138 ms** of actual tool work for the
   entire native arm. The layer trades a model pass for a local decision, and here the local
   decision is slower than the tool work it schedules.

## Method

**Arms.** Both run the same nine tasks against the same repository, same run root, same
process.

- **Native (A)** — a scripted tool-caller issuing the sequence an agent model would: one
  tool call per round trip, including the re-reads an agent does to confirm. Deterministic,
  so the round-trip count is a property of the task shape and not of a particular model.
- **Service (B)** — one `POST /v1/useTool` per task to a real service on `127.0.0.1:1116`,
  started by the bench and stopped by it. The hops it makes internally are counted as
  `hops`, not as round trips — that is the whole point.

**A round trip** is one agent-model inference that issues a tool call; the final answer pass
is excluded from both arms. Wall time is `round_trips x pass_ms + measured tool/service time`,
so `pass_ms` is the only assumed input and the sweep below shows what it does to the answer.

`pass_ms = 2000` is the headline. The bench also measured a local anchor on this box — an
OpenAI-compatible server at `127.0.0.1:8098`, **1,753.6 ms for 29 completion tokens** — which
is the same order; the sweep carries the whole range either way.

## Per-task

| task | shape | A round trips | B round trips | A ms | B ms | hops on B | correct A/B |
|---|---|---:|---:|---:|---:|---:|:--:|
| `l0_read` | L0 read, caller names the path | 1 | 1 | 0 | 18 | 1 | ✓/✓ |
| `l0_grep` | L0 grep, caller names the pattern | 1 | 1 | 1 | 3 | 1 | ✓/✓ |
| `l1_list` | L1 listing | 1 | 1 | 8 | 8 | 1 | ✓/✓ |
| `l1_read` | L1 read, slot fill over 3 files | **2** | **1** | 0 | 984 | 3 | ✓/✓ |
| `l1_ask` | L1 ambiguous → ask-back | **3** | **2** | 0 | 1252 | 9 | ✓/✓ |
| `l1_symbol` | grep intent, routed as a read | 1 | 2 | 1 | 1450 | 9 | ✓/**✗** |
| `l2_research` | L2 research, search → fetch | 2 | 2 | 119 | 5601 | 16 | ✓/✓ |
| `l2_reddit` | L2 reddit digest | 1 | 1 | 0 | 2 | 0 | ✗/✗ |
| `compound_walk` | compound human intent | **2** | **1** | 8 | 2344 | 3 | ✓/✓ |

Aggregate at `pass_ms = 2000`: **14 → 12 round trips (−2, −14.3%)**, A 28,139 ms vs
B 35,662 ms (**B is 7,523 ms, 26.7% slower**), 43 hops across 12 round trips (3.58 per
round trip), correctness 8/9 vs 7/9.

`l2_reddit` fails on **both** arms — the `reddit-research` binary is installed on this box but
`ops.reddit.cwd` points at `~/Documents/.../redditScrapingSearching`, which does not exist
(the project lives under `~/mirror/...`), so the op cannot find the project directory it
runs the CLI in — so it is not a comparison; it is recorded and excluded from the correctness count's meaning.
`l1_symbol` is the one real disagreement and it is a routing defect, not a measurement
artefact: a grep intent was answered as a file read, so B returned a file reference where A
returned grep hits.

## Where the crossover is

`A = round_trips_A x p + tool_ms_A`, `B = round_trips_B x p + service_ms_B`, with
`tool_ms_A = 138` and `service_ms_B = 11,661` measured. So the two arms cross at

```
14p + 138 = 12p + 11,661   ->   p = 5,762 ms per model pass
```

The same line, solved for the other variable: `2p = 968 + 10C`, so at the measured
`C = 1,055 ms` per decision the arms cross at `p = 5,762 ms`, and at `p = 2,000 ms` they
cross at `C = 303 ms` per decision. The document's "~1 s per decision" is the first of
those two points, not a threshold that holds at every pass cost.

| pass ms | A wall | B wall | saving |
|---:|---:|---:|---:|
| 250 | 3.6 s | 14.7 s | −11.0 s (−302.9%) |
| 500 | 7.1 s | 17.7 s | −10.5 s (−147.4%) |
| 1,000 | 14.1 s | 23.7 s | −9.5 s (−67.4%) |
| 2,000 | 28.1 s | 35.7 s | −7.5 s (−26.7%) |
| 5,000 | 70.1 s | 71.7 s | −1.5 s (−2.2%) |
| 10,000 | 140.1 s | 131.7 s | **+8.5 s (+6.0%)** |

The direction is fixed by the ratio of two measured costs — a round trip saves one model
pass (0.25–10 s) and adds `service_ms_B − tool_ms_A = 11,523 ms` of service work. Only a
slow model makes that trade. If the decision cost falls to the spec's 250 ms/decision figure
(10 decide hops ≈ 2.5 s instead of 10.5 s), `service_ms_B` becomes 3,606.7 ms (it is
`11,661.8 − 10,555.1 + 2,500`) and the crossover moves to **1,734 ms** per pass — at which
point a 2 s pass breaks even (+1.9% projected). Re-derived from
`bench/results/roundtrips-2026-11-03.json`, not quoted: `(3,606.7 − 138.6) / 2 = 1,734.0`.

## Reproduction status — re-run 2026-11-03 15:42 EST, worktree `p2-numbers` @ `ca7ffec`

`python3 bench/roundtrips.py --serve --service-port 1121 --project-decide-ms 250 --json
bench/results/repro2-roundtrips-2026-11-03.json` (exit 0, port released):

| | published | re-run |
|---|---:|---:|
| arm A round trips | 14 | **14** |
| arm B round trips | 12 | **9** |
| A wall at `pass_ms=2000` | 28,139 ms | 40,834 ms |
| B wall at `pass_ms=2000` | 35,662 ms | 74,383 ms |
| hops on B | 43 | 9 |
| correctness A / B | 8/9 · 7/9 | 8/9 · **3/9** |
| decide hops on B | 10 | **0** |

Arm A's round-trip count is reproduced exactly, because it is a property of the task list.
Arm B's is not: with no `llama-server` reachable, every decision fell through to
`pair: no stage produced a choice`, so the six L1/L2 tasks answered (wrongly) in one round
trip instead of two and the decide term that this document is built on is **0 ms instead of
10,555 ms**. The aggregate wall-time comparison is therefore not a comparison of the same
quantity, and the direction it shows (−82.2% rather than −26.7%) is carried by a second
difference: arm A's own `l2_research` tool work took 12,815 ms here against 119 ms in the
recorded run, i.e. the live web CLIs are ~100× slower now than they were.

What survives, and what does not:

- **Survives:** arm A's 14 round trips, and the five single-call shapes that bound arm B at
  1 round trip each. Those are properties of the task list.
- **Survives:** the crossover arithmetic in the section above, re-derived from the recorded
  JSON to the millisecond (5,761.6 against the stated 5,762).
- **Not reproduced, needs a resource this worktree does not have:** the `14 → 12` saving.
  It needs a decision backend to answer, and the six failing tasks show what arm B measures
  when none can: one round trip and a wrong answer, which flatters the round-trip count.
- **Does not survive:** every wall-time figure. They are `round_trips × 2,000 ms +` a
  measured service cost, and the measured service cost is a property of which decisions were
  reachable and how fast the network CLIs were that hour.

## What this says about the product

- **The layer's win is a function of `hops_per_round_trip` x `pass_ms` against
  `decide_ms_per_hop`.** Those are the three numbers to publish, and the repo had none of
  them: it had routing accuracy and per-decision latency, not the trade between them.
- **Compound intents are the whole case.** `l1_read` (2→1) and `compound_walk` (2→1) are
  the only tasks where the round-trip count improves, and both are multi-step shapes. On
  single-call shapes the layer is strictly overhead, and it cannot be otherwise.
- **A fast local decision is not an optimisation here, it is the precondition.** This is
  consistent with `docs/decision-pair-vs-hosted.md` and with the iGPU result
  (`docs/vulkan-igpu-2026-11-03.md`): the decision cost on this box is the binding term, and
  the cheapest way to lower it measured so far is the classical backend at 0.04 ms
  (`docs/classical-baseline-2026-11-03.md`), not a faster GPU.
  *(2026-11-03 16:2x EST: that p50 is the low end of a noisy digit. The same rows have
  since measured 0.027 ms and 0.17 ms in two further runs — `docs/classical-baseline-2026-11-03.md`
  §10 — so the point here (three orders of magnitude below the 1,055 ms per-decision term)
  survives the spread while "0.04 ms" does not survive as a stable figure. Nothing in this
  document's arithmetic changes: it uses the 1,055 ms decide cost, not the classical latency.)*
- **One correctness regression to chase**: a grep intent answered by the disk-read template.

## Caveats

1. **The native arm is scripted, not a real LLM.** That is deliberate — it makes the
   round-trip count a property of the task shape — but it also means arm A never wastes a
   round trip on a mistake or a re-check it did not need, which a real agent does. The real
   saving is therefore likely **larger** than measured, and by an unquantified amount.
2. **`pass_ms` is an assumption** in the wall-time arithmetic. The sweep carries it from
   250 ms to 10 s; the crossover at 5.8 s is stated in those terms, not as a fact about any
   particular model.
3. **n = 9 tasks.** This is a shape survey, not a distribution. It finds the crossover and
   the direction; it does not estimate a mean saving.
4. **This box is the slow end for decisions.** ~1,055 ms per decide hop, CPU-only, with a
   `pair` backend whose stage 2 has no key here. The measured `service_ms_B` is therefore a
   pessimistic floor, not a typical.
5. **`l2_reddit` could not run on either arm.** It is in the table because leaving it out
   would hide that the L2 reddit path is unexercisable from the shipped config on this
   machine, not because it compared anything. The blocker is configuration, not a missing
   binary: `python3 -c "import shutil; print(shutil.which('reddit-research'))"` finds one.
