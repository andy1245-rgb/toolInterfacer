# Roadmap to first publication (2026-11-03)

Working plan for an overnight run. Each phase ends with a commit and a measured
result. Nothing here is a claim; a phase is done when its acceptance criterion is
executed, not when its code exists.

**The run is over.** The plan below is the plan as it was written, kept as the record.
What is new is a closing verdict under each phase heading and one corrected deliverable
in the Phase 0 table. The **Outcome — what was executed, and what it measured** section
at the end is the account of what the run actually produced, including the parts it did
not reach.

## Where the project actually stands

*(As of 2026-11-03, the start of this run. Every figure below is that night's and is
superseded: the README's Test suite row carries the current test count and
`docs/dataset-status.md` the current hop log. The rest of this file is the record of that
night's plan and is not rewritten.)*

- Service, planner, four decide backends (d1 `/v1/systemone`, Jev-style
  `logit_read`, `pair`, `openrouter`) and the ops wall exist and are covered:
  `python3 -m pytest tests/ -q` → 326 passed, 159 subtests *(that night's count; the
  README's Test suite row carries the current one)*.
- The incumbent decision backend is **d1-3B Q4_K_M**: 74.0% overall on the frozen
  150-row v2 set at p50 163 ms (`docs/model-sweep-consolidated.md`). Qwen3.5-4B
  Q4 scores 58.0% on the same set — the "4B is better" assumption does not hold.
- The pair-vs-hosted experiment is measured (`docs/decision-pair-vs-hosted.md`,
  n=30, pair 90.0% / 3118 ms p50, openrouter 93.3% / 827 ms p50, McNemar p=1.000).
  The shipped default stays `pair` for availability, not for latency.
- `service/logs/hops.jsonl` holds 1202 hop records; how many are *trainable
  decision records* rather than execution trace is unknown and is being counted.

## Phases

### Phase 0 — Recon (parallel, 6 tracks)

**Closed, all six committed.** One deliverable is corrected in place: the recall track's
answer is the assets survey, not a document of its own — see the row below.

| Track | Question | Deliverable |
|---|---|---|
| recall | What has already been decided, measured and abandoned across this project, its private sibling trees, cactus/needle, and decision-model tuning? | [`docs/assets-inventory.md`](assets-inventory.md) — the survey answers it for every one of those projects. No separate recall document was ever written; this row pointed at `docs/session-recall.md`, which is in no commit on any branch |
| audit | Which modules are real, which are stubs, and what blocks a stranger's first clone? | `docs/audit-2026-11-03.md` |
| assets | What can be imported instead of rebuilt from the sibling projects? | `docs/assets-inventory.md` |
| publish | LICENSE, packaging, CI, a README that states what is true. | repo root + `.github/` |
| dataset | A validated ingest → split → export pipeline over the hop logs. | `tools/dataset/` |
| bench | One command that scores and compares any decide arm, McNemar included. | `bench/run.py` |

Acceptance: all six committed, full suite green on each branch.

### Phase 1 — Repair and integration

Driven by the audit. Two streams, disjoint files:

- **Repair** the blockers the audit ranks highest, each with a regression test.
- **Integration**: drive a real `useTool` call through the real service with a
  real llama-server behind it, end to end, and assert the returned hops. The
  suite today is fixture-driven; nothing proves the live path works.

Acceptance: a live end-to-end probe script that fails loudly when the stack is
down, plus green tests.

**Closed.** Repair is the blocker table in the Outcome. Integration is
`bench/live_e2e.py`, which drives ingress → plan → decide → ops against a live
llama-server and now sends the service token; the run recorded in
[`docs/live-e2e-2026-11-03.md`](live-e2e-2026-11-03.md) predates auth and reads 0/4
for it.

### Phase 2 — Data and the fine-tune question

The decision-model line only pays off if there is a dataset. The hop logs may not
be enough, so the plan is teacher distillation into a controlled set:

- Build utterances beyond the frozen 150 from the operator's real session corpus,
  band-labelled (t1 literal / t2 veiled / t3 compound) to match `utterances_v2`.
- Label with the hosted teacher (`openrouter` backend), then human-check a sample.
- LoRA SFT rank 32–64 on an instruct 3B per `docs/assets-inventory.md` §1, 10–20%
  general-instruction mix.

Acceptance: a labelled corpus with a stated size, and a fine-tuned checkpoint
scored on the same frozen set by the same harness as every other arm.

Budget: **$1.50 of rented GPU**, cheapest card that fits a 3B QLoRA. If the
number is not achievable inside that, the honest deliverable is the measured
throughput plus a costed plan, not a half-run.

**Half closed.** The distillation pipeline exists and ran — `tools/distill/`, 414
rows, 199 teacher-labelled, $0.002 — and the LoRA run did not. That is the *Still
open* item at the end of this document.

### Phase 3 — Applications

A tool layer that is not applied is a library with no users. Pick avenues the
operator already runs, wire `useTool` to each, and measure the round trips saved
against the LLM doing the tool calls natively:

1. OSCAR Plus postings through the local mcm-portal checkout — read-only, structured.
2. The web/reddit research stack the ops already wrap.
3. One market-scrape corpus read path.

Acceptance: per avenue, a before/after count of model round trips and wall time
on a fixed task list.

**Partly closed.** Avenue 2 is measured in
[`docs/roundtrips-2026-11-03.md`](roundtrips-2026-11-03.md): nine fixed tasks, two
arms, 2 of 14 round trips removed and 7.5 s slower at the headline pass cost.
Avenues 1 and 3 were not wired — no op, template or bench in the tree reads an
mcm-portal checkout or a market corpus.

### Phase 4 — Review and publish

- Independent code review of everything merged tonight, findings fixed or filed.
- README/docs reconciled with what the code does.
- Tag `v0.1.0`.

**Partly closed.** Both reviews ran ([`docs/review-2026-11-03.md`](review-2026-11-03.md),
[`docs/review2-2026-11-03.md`](review2-2026-11-03.md)) and their findings are fixed or
filed; the reconciliation they asked for is still running
([`docs/doc-review-2026-11-03.md`](doc-review-2026-11-03.md)). No tag was cut:
publication waits on the history decision in
[`docs/pre-publish-audit.md`](pre-publish-audit.md), and [`RELEASE.md`](../RELEASE.md) is
the runbook for the day it is taken.

## Rules for every track

- Isolated worktree under `~/ti-worktrees/`, branch named for the track.
- Full suite green before commit: `python3 -m pytest tests/ -q`.
- No hardcoded personal absolute paths (`tests/test_no_personal_paths.py`).
- No pushes, no tags without an explicit go-ahead.
- A number is quoted with its n and its source file, or it is not quoted.

---

## Findings that changed the plan

Recorded as the recon landed, because two of them move the goalposts.

**1. The classical baseline may be the whole answer.** The sibling-project survey
(`docs/assets-inventory.md`) found a six-backend head-to-head on 382 rows / 18
classes: TF-IDF + LinearSVC **93.19% at 0.61 ms**, d1-3B **91.36% at 242.7 ms**,
MiniLM + logistic 86.65% at 57.8 ms, needle3 59.42%. That is a 400x latency gap
at equal-or-better accuracy. Either the linear number is in-sample (the original
script fitted and scored on the same rows, which is the first thing to check) or
the project's premise -- a neural decision model behind the `choice` primitive --
is the wrong instrument. `ov-classical` exists to settle it.

**2. Every accuracy claim in `docs/` is underpowered.** Per
`docs/assets-inventory.md` §19, detecting two
points unpaired needs ~5,000 rows per arm; paired with McNemar at 10%
discordance, ~1,960. At n=300 the 95% interval is +/-4.0 points. This repo reports
74.0% vs 58.0% on n=150 and 90.0% vs 93.3% on n=30 -- both are directional,
neither is a result. Any public claim needs the interval printed beside it, and
the frozen set needs to grow before it can support a comparison. That is
`ov-distill`'s job and it reorders Phase 2 ahead of Phase 1.

**3. Needle is a measured no for routing and an unmeasured maybe for `extract`.**
1.07 GB of local evidence says needle3 scores 59.4% node-fixed and a local LoRA
fine-tune dropped it from 26% to 6% against an 81% majority baseline. But
`resolve("extract")` raises `KeyError` today -- the `extract` primitive has **no
backend at all** -- and needle's contract (one call-acceptance scalar, no
distribution over tools) makes it structurally unsuitable for `choice` while
being the right shape for `extract`. Worth one spike, not an adoption.

**4. The bar to beat is higher than this repo thinks.** The incumbent d1-3B
number in `docs/` is 74.0% on the 150-row v2 set, but the same model scores
**91.4% with none-detection 1.00** on the 382-row set. The v2 set's bands (t1
literal / t2 veiled / t3 compound) are harder than the 382 set's task. Any
comparison across the two is invalid; the harness must name which set it scored
or the number is meaningless.

---

## Outcome — what was executed, and what it measured

Written at the end of the run. The plan above was the intent; this is what
happened, including the parts that changed it.

### Publish blockers: closed and verified, not asserted

Two independent reviews converged on the same defects. All are fixed, and the
security ones were re-probed live against a running service:

| was | now |
|---|---|
| an unauthenticated POST returned `/etc/hostname` and listed `/etc` | `401` without the token, `403 path_refused` for the path |
| `filesys_glob` walked a caller-chosen root with no cap; an `os.walk` above the sshfs mount hangs uninterruptibly | run root confined to `ops.filesys.roots`, deny-list pruned before the first `lstat`, depth/hit/deadline caps |
| `filesys_grep` compiled a caller-supplied regex; `(a+)+$` wedged a worker | four shape refusals at compile, matching against bounded windows |
| `confidence: 1.0` on an error run with `hops: []` | the field is absent when nothing was measured; a vague intent now reports 0.215 and asks back |
| a one-entry probability vector read as `p_max 1.0`, so `tau` could never fire | K<2 raises instead; a 0.02 answer no longer reports 0.87 |
| the L2 research template named ops that do not exist — every "research" intent errored at step 0 | the template runs: extract → search → pick → fetch (gated, per-page `noul`) → assemble |
| `pip install .` shipped 2 of 53 modules | 53 modules, console script, CI wheel check |
| `tsc` reported 66 errors, the extension could not load | `tsc` exit 0, contract check green |
| seven web-research intents routed to `filesys_glob` in production | an explicit research verb outranks `r/`; real-log regression table |

**588 → 626 tests** *(the count at the end of that night; the README's Test suite row
carries the current one)*. The suite grew by the regressions that would have caught
each item above.

### What the measurements changed

**The classical baseline was not the answer, but it found the real question.**
The 93.19% reproduces exactly and is genuinely held out — but holding out whole
*phrasing families* instead of rows drops it to **66.23%**, and on this repo's own
set it scores 62.7% against d1-3B's 74.0%. It *wins* the hardest band (t3
boundary pairs, 69.1% vs 67.3% at 0.04 ms) and collapses on veiled intent (41.9%
vs 71.0%). So the premise "a neural decision model is required" is overstated
for a closed label set with in-vocabulary phrasing, and the boundary-pair band is
where a lexical model belongs — worth chasing, not a default swap.

**The iGPU is not compute.** On the routing workload the Iris Xe is 2.72× slower
per decision and 6.9× slower at prompt processing, losing 50 of 50 paired rows.
Routing is prefill-dominated, which is exactly an iGPU's worst case. It also
surfaced a long-lived-server state bug on the Vulkan backend.

**The hop log is not a decision log.** Of 1398 hop records, 14 are trainable
choice records and all 14 carry the same label. Fine-tuning on runtime evidence
is not available; the corpus has to be generated and teacher-labelled, which is
what `tools/distill/` now does (414 rows, 199 teacher-labelled, $0.002).

**The teacher's confidence is saturated exactly where the literature predicts.**
t1 literal: mean 0.940, 50 of 72 rows in the 0.95–1.00 bin. t2 veiled: mean
0.776 and genuinely spread. t3: 0.923. A confidence target from this needs
temperature scaling per band, not trust — and t2's 78.5% agreement with its own
frame is the band whose frames need a human audit first (`var/distill/audit.csv`).

### Still open

The fine-tune itself. The corpus is feedstock, not yet a trainable set — 199 rows
against the 300-row floor in `docs/assets-inventory.md` §9, and the labelling stall
was provider congestion, not policy. Before training: finish to 414 rows, fit the
per-band temperature, and annotate `audit.csv` so calibration stops reading
UNKNOWN.
