# Roadmap to first publication (2026-11-03)

Working plan for an overnight run. Each phase ends with a commit and a measured
result. Nothing here is a claim; a phase is done when its acceptance criterion is
executed, not when its code exists.

## Where the project actually stands

- Service, planner, four decide backends (d1 `/v1/systemone`, Jev-style
  `logit_read`, `pair`, `openrouter`) and the ops wall exist and are covered:
  `python3 -m pytest tests/ -q` → 326 passed, 159 subtests.
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

| Track | Question | Deliverable |
|---|---|---|
| recall | What has already been decided, measured and abandoned across toolInterfacer, jevResearch, localAlexa, cactus/needle, decision-model tuning? | `docs/session-recall.md` |
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

### Phase 2 — Data and the fine-tune question

The decision-model line only pays off if there is a dataset. The hop logs may not
be enough, so the plan is teacher distillation into a controlled set:

- Build utterances beyond the frozen 150 from the operator's real session corpus,
  band-labelled (t1 literal / t2 veiled / t3 compound) to match `utterances_v2`.
- Label with the hosted teacher (`openrouter` backend), then human-check a sample.
- LoRA SFT rank 32–64 on an instruct 3B per `decisionModelTuning/02`, 10–20%
  general-instruction mix.

Acceptance: a labelled corpus with a stated size, and a fine-tuned checkpoint
scored on the same frozen set by the same harness as every other arm.

Budget: **$1.50 of rented GPU**, cheapest card that fits a 3B QLoRA. If the
number is not achievable inside that, the honest deliverable is the measured
throughput plus a costed plan, not a half-run.

### Phase 3 — Applications

A tool layer that is not applied is a library with no users. Pick avenues the
operator already runs, wire `useTool` to each, and measure the round trips saved
against the LLM doing the tool calls natively:

1. OSCAR Plus postings through the local mcm-portal checkout — read-only, structured.
2. The web/reddit research stack the ops already wrap.
3. One market-scrape corpus read path.

Acceptance: per avenue, a before/after count of model round trips and wall time
on a fixed task list.

### Phase 4 — Review and publish

- Independent code review of everything merged tonight, findings fixed or filed.
- README/docs reconciled with what the code does.
- Tag `v0.1.0`.

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
`decisionModelTuning/03-evaluation-and-calibration.md` section 3, detecting two
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
