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
