# TODO

The backlog, ordered by what the measurements say matters. Every item names the number or
the failure that put it here, so it can be argued with.

Status key: **next** · **queued** · **blocked** · **won't**

---

## 1. Make the decision cheap — the whole game

The layer's economics reduce to one inequality:

```
hops_per_round_trip × pass_ms   >   decide_ms_per_hop × decide_hops
```

Measured on this box: 10 decisions cost **10,556 ms** against **138 ms** of actual tool work,
so the right side wins and the layer is a **net loss on 7.5 s** over nine tasks
(`docs/roundtrips-2026-11-03.md`). Routing accuracy is worth nothing until this is fixed.

- [ ] **Fit a temperature for the `classical` backend.** Its margin softmax puts p_max at a
      median of 0.171, so coverage at the shipped 0.70 gate is **0.0%** — every hop would
      escalate, which makes the 0.04 ms backend useless in production.
      `docs/classical-baseline-2026-11-03.md` §6. **next**
- [ ] **The classical-front cascade, as the default.** Measured **79.3% at 95 ms** against
      d1 alone at **74.0% at 169 ms** — 16 rows fixed, 8 broken, McNemar p = 0.152 at n=150.
      Best configuration anyone has measured here, and it is not wired. Needs the fitted
      threshold above, chosen on held-out rows rather than on the set it is scored on.
      **next**
- [ ] **Re-measure at n ≥ 500 paired.** At n=150 the 95% interval is ±8 points, so the
      +5.3 point cascade gain is not yet a result. Paired detection of 2 points needs ~1,960
      rows; n=500 buys ±2 points on the arms themselves. **next**
- [ ] **Q8 instead of Q4 under 2B.** `decisionModelTuning/04` §3 records Q4 changing the
      *failure mode*, not just the scale (xLAM-2-3B 58.00% → 51.75%, Qwen3-1.7B 14.50% →
      8.75%). Every lane in `bench/results/` is Q4. **queued**
- [ ] **A d1 server that survives being resident.** The iGPU result (`docs/vulkan-igpu-2026-11-03.md`)
      caught the Vulkan backend collapsing to `/b/b/b/b` after a generation call in the same
      process. Not isolated. **queued**

## 2. Grow the eval before believing anything

- [ ] **Finish the distilled corpus.** 267 of 414 rows labelled at **$0.0028** total; the
      stall was provider congestion (0.7 s/row → 20–60 s/call), not budget. Below the
      300-row floor in `decisionModelTuning/01` §5. **next**
- [ ] **Annotate `var/distill/audit.csv`.** 18 rows, 6 per band, blank `verdict` column.
      Until it is filled in, `tools/distill/stats.py` reports calibration as **UNKNOWN** and
      it is right to. **next**
- [ ] **Fit a per-band temperature on the teacher's confidence.** t1 literal means 0.940 with
      50 of 72 rows in the 0.95–1.00 bin; t2 veiled spreads properly at 0.776. A confidence
      target taken from this without scaling trains a model that is confidently wrong.
      **blocked on the annotation above**
- [ ] **Audit the t2 frames.** Teacher agreement with its own frame is **78.0% at n=82** and
      **78.5% at n=65** — stable across two samples, so it is a property, not noise. Either
      ~22% of the veiled frames are genuinely ambiguous or the teacher errs on them; the
      audit CSV is how you tell which. **next**

## 3. Close the P2 gap — let the model choose the plan

Today `planner/inspector.py` is an ordered keyword list, which the runtime proved wrong:
seven web-research intents in `service/logs/hops.jsonl` were executed as `filesys_glob`,
globbing the filesystem for a song title. Those seven are fixed and pinned by
`tests/test_route_real_traffic.py`, but the *class* of bug survives because the layer is
still keywords.

- [ ] **Route the plan choice through the decision model** (spec §2, P2). The rules become
      the fallback, not the mechanism. **next**
- [ ] **A grep intent is answered as a file read.** Found in the round-trip run
      (`l1_symbol`): the disk template returns a file reference where grep hits were asked
      for. One task, one defect, still open. **next**

## 4. Things that are promised and do not work

- [ ] **`reddit_digest` has never run against a real CLI.** The op is written against a
      contract inferred from documentation, and the project it drives is not vendored.
      Vendor it, or delete the op and the template. **queued**
- [ ] **`web_search` / `web_fetch` need configured CLIs.** They default to skill scripts that
      a clone does not have, so an L2 research intent errors out of the box. Ship a
      vendored fallback or make the error tell the reader what to set. **queued**
- [ ] **L3 (`planner/free.py`) is a docstring.** Build it, or remove L3 from the spec and
      the README table so the roadmap stops promising it. **queued**
- [ ] **L4 (browser) is an unwired adapter slot.** Same choice: build or delete. **queued**

## 5. Deliberately out of scope for 0.1.0

- **Destructive tools stay native.** `bash`/`write`/`edit` are not routed through `useTool`.
  Routing a mutation is a trust decision, and no evidence yet earns it.
- **No multi-agent orchestration.** This schedules one call.
- **No new dependencies.** The Python core is stdlib-only by policy; `pyproject.toml` has
  `dependencies = []` and the suite enforces the spirit of it.

## Known limits, not bugs

These will not be fixed, and should be quoted whenever the numbers are:

| limit | number |
|---|---|
| Frozen eval set is small | n=150 → ±8 points at 95% |
| The classical baseline's headline is template transfer | 93.19% holds out rows, **66.23%** holds out phrasing families |
| The round-trip saving is conditional | 14.3% of round trips, **7.5 s slower**, crossover at ~5.8 s per model pass |
| Decision latency on the CPU-only box | ~1,055 ms per decide hop |
| Teacher confidence is saturated on the easy band | t1 mean 0.940 |
| The iGPU is not faster | 2.72× slower per decision, 6.9× at prefill, 50/50 rows |
