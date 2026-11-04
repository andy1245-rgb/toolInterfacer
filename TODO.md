# TODO

The backlog, ordered by what the measurements say matters. Every item names the number or
the failure that put it here, so it can be argued with.

Status key: **next** · **queued** · **blocked** · **won't**

---

## 1. Make the decision cheap — the whole game

The layer's economics reduce to one inequality: a round trip saved is one model pass, and it
costs whatever the service adds on top of the tool work it replaces.

```
round_trips_saved × pass_ms   >   service_ms − tool_ms
      2 × pass_ms             >   11,661.8 − 138.6 = 11,523.2
                              →   pass_ms > 5,762 ms per model pass
```

Measured on this box over nine fixed tasks (`n=9` tasks, `n=10` decide hops, 43 hops total;
raw `bench/results/roundtrips-2026-11-03.json`): 10 decisions cost **10,555 ms** of the
**11,661.8 ms** service time, against **138.6 ms** of actual tool work in the native arm.
At the 2,000 ms pass the run assumed, the left side is 2 × 2,000 = 4,000 ms against
11,523 ms, so the layer is a **net loss on 7.5 s** (`docs/roundtrips-2026-11-03.md`).
Routing accuracy is worth nothing until this is fixed.

- [ ] **Fit a temperature for the `classical` backend.** Its margin softmax puts p_max at a
      median of 0.171, so coverage at the shipped 0.70 gate is **0.0%** — every hop would
      escalate, which makes the 0.04 ms backend useless in production.
      `docs/classical-baseline-2026-11-03.md` §6, n=150 rows. Still open as of this
      writing: `decide.classical.temperature` is `1.0` in `service/ti/conf/defaults.json`,
      the uncalibrated default. **next**
- [ ] **The classical-front cascade, as the default.** Measured **79.3% at 95 ms** against
      d1 alone at **74.0% at 169 ms** — 16 rows fixed, 8 broken, McNemar p = 0.152 at n=150.
      Best configuration anyone has measured here, and it is not wired. Needs the fitted
      threshold above, chosen on held-out rows rather than on the set it is scored on.
      **next**
- [ ] **Re-measure at n ≥ 500 paired.** At n=150 the 95% interval is ±8 points (worst case,
      `1.96 × 0.5/√n`), so the +5.3 point cascade gain is not yet a result. Paired detection
      of 2 points needs ~1,960 rows (`docs/assets-inventory.md` §19); n=500 buys ±4.4 points
      on the arms themselves at the same convention, which is not enough either. **next**
- [ ] **Q8 instead of Q4 under 2B.** A private decision-model research tree (`04` §3, summarised
      in `docs/assets-inventory.md` §25) records Q4 changing the *failure mode*, not just the
      scale (xLAM-2-3B 58.00% → 51.75%, Qwen3-1.7B 14.50% → 8.75%). The recorded lanes are
      Q4_K_M, with two exceptions: d1-3B also has a Q8_0 lane pair
      (`bench/results/dell/v2-d1-3B-Q8*`, 73% against Q4's 74% in
      `docs/model-sweep-consolidated.md`), and qwen3.5-0.8b was only ever run at Q8_0. No
      model under 2B has a Q4/Q8 pair, which is where the failure-mode change bites.
      **queued**
- [ ] **A d1 server that survives being resident.** The iGPU result (`docs/vulkan-igpu-2026-11-03.md`)
      caught the Vulkan backend collapsing to `/b/b/b/b` after a generation call in the same
      process. Not isolated. **queued**

## 2. Grow the eval before believing anything

- [x] **Finish the distilled corpus.** Done: **412 of 414 rows labelled**, $0.0043, and
      frozen at `data/distill/` — 410 usable (329 train / 81 val) after two near-duplicate
      exclusions. Both earlier counts were snapshots of a run still in progress: 199 was the
      first hour, and 267 is the ~$0.0028 mark (at $1.16e-5/row that buys 265). The label
      store stays git-ignored; the certified artefact does not. Verdict, per-band label
      audit and the fair-evaluation protocol are in
      `docs/distill-corpus-certification-2026-11-03.md`. **next** is training on it.
- [ ] **Annotate `var/distill/audit.csv`.** `tools/distill/audit.py` samples 6 rows per
      band (`--per-band`, default 6), so a full run is 18 rows with a blank `verdict`
      column. Until it is filled in, `tools/distill/stats.py` reports calibration as
      **UNKNOWN** and it is right to. **next**
- [ ] **Fit a per-band temperature on the teacher's confidence.** t1 literal means 0.940 with
      50 of 72 rows in the 0.95–1.00 bin; t2 veiled spreads properly at 0.776, t3 sits at
      0.923 (`docs/roadmap-2026-11.md`, `tools/distill/stats.py`). A confidence target taken
      from this without scaling trains a model that is confidently wrong.
      **blocked on the annotation above**
- [ ] **Audit the t2 frames.** Teacher agreement with its own frame is **78.5%**
      (`docs/roadmap-2026-11.md`). Either ~22% of the veiled frames are genuinely ambiguous or
      the teacher errs on them; the audit CSV is how you tell which. (An earlier draft of
      this item paired that with a second sample, 78.0% at n=82, which is not in the repo —
      re-print it from `tools/distill/stats.py` before quoting it.) **next**
- [x] **Train on it, and measure whether it beats the 3B.** Done, negative:
      **68.7% [59.6–77.0]** against d1-3B's 74.0% [66.5–81.3] on the frozen 150 — not
      distinguishable from the 3B (exact McNemar p = 0.38) and **significantly worse than
      `local_clf`'s 35-rule cue table at 78.7%** (p = 0.040), which is fitted on nothing. It
      does beat the 3B on the two classes `p1-diagnose` broke it on (`decide_extract` 79.2%
      vs 25.0%, `filesys_grep` 63.6% vs 36.4%) and loses `web_search` 38.9% vs 100%. Not
      shipped. Recipe at `data/distill/clf_recipe.json`, report at
      `docs/distill-classifier-2026-11-04.md`.
- [ ] **Give the corpus a split that holds slot values out.** The reason the arm above is
      negative is not only the label bands: family-held-out CV ranks seven candidates
      **backwards** against transfer (Spearman ρ = −0.71, `docs/distill-classifier-2026-11-04.md`
      §6), because a frame's slot values recur across frames, so holding a *family* out leaves
      every noun in the training fold and rewards the model that memorised them. Object-held-out
      CV exists as a diagnostic (`tools/distill/train.py --object-cv`) and scores *higher* than
      the family split, so it is not the answer either. This blocks using `val.jsonl` to choose
      between two models on this corpus, which is what the certification's protocol §5 step 1
      currently recommends. **next**

## 3. Close the P2 gap — let the model choose the plan

Today `planner/inspector.py` is an ordered keyword list, which the runtime proved wrong:
seven web-research intents in `service/logs/hops.jsonl` were executed as `filesys_glob`,
globbing the filesystem for a song title. Those seven are fixed and pinned by
`tests/test_route_real_traffic.py`, but the *class* of bug survives because the layer is
still keywords.

- [ ] **Route the plan choice through the decision model** — spec §2's **L3**, not the
      spec's P2; which label is which is resolved in
      [`spec/1.0-spec.md`](spec/1.0-spec.md#divergences-from-this-spec)'s §8 row. The rules
      become the fallback, not the mechanism. **next**
- [ ] **A grep intent is answered as a file read.** Found in the round-trip run
      (`l1_symbol`): the disk template returns a file reference where grep hits were asked
      for. One task, one defect, still open — reproduced while this file was being audited
      with the README's own recipe (a `TI_CONF_OVERLAY` that sets
      `decide.backends_in_use.choice` to `fixture`): `python3 bench/replay.py --intent "grep
      for the token budget constant"` plans glob → `decide_pick` → `filesys_read`, three
      hops, no `filesys_grep`. `planner/inspector.py` sends it to `disk_lookup` and
      `planner/shape.py` calls it SINGULAR, so the read shape is the only one reachable.
      **next**

## 4. Things that are promised and do not work

- [ ] **`reddit_digest` has never run against a real CLI.** The op is written against a
      contract inferred from documentation, and the project it drives is not vendored.
      Vendor it, or delete the op and the template. **queued**
- [ ] **`web_search` / `web_fetch` need configured CLIs.** They default to skill scripts that
      a clone does not have, so an L2 research intent errors out of the box. Ship a
      vendored fallback or make the error tell the reader what to set. **queued**
- [ ] **L3 (`planner/free.py`) is a docstring.** Build it, or remove L3 from the spec and
      the README table so the roadmap stops promising it. **queued**
- [ ] **L4 (browser) is a row in the spec's level table and nothing else.** No adapter, no
      code path, and no reference to one anywhere in `service/`. Same choice: build or
      delete. **queued**

## 5. Deliberately out of scope for 0.1.0

- **Destructive tools stay native.** `bash`/`write`/`edit` are not routed through `useTool`.
  Routing a mutation is a trust decision, and no evidence yet earns it.
- **No multi-agent orchestration.** This schedules one call.
- **No new dependencies.** The Python core is stdlib-only by policy; `pyproject.toml` has
  `dependencies = []`, and a third-party import would fail
  `tests/test_skeleton.py::test_every_module_imports` in CI, where nothing but this package
  and pytest is installed.

## Known limits, not bugs

These will not be fixed, and should be quoted whenever the numbers are:

| limit | number |
|---|---|
| Frozen eval set is small | n=150 → ±8 points at 95% |
| **The frozen eval set cannot measure its own optimism** *(added 2026-11-04)* | 63.3% atomic rows (95/150); family-held-out moves accuracy **−0.4 to +3.7 pts, both directions**; the 23.5-point penalty this model class shows on a set that *can* be family-split is the order of what v2 cannot see (`docs/eval-harness-2026-11-04.md` §4) |
| **`routing_382` is not this repo's task** *(added 2026-11-04)* | 18 smart-home classes (`timer_create`, `thermostat_set`, …), own registry, **empty** intersection with the 8 ops — only the 62.7% on `utterances_v2` is comparable |
| **`nested_selection_acc` is an oracle** *(added 2026-11-04)* | 84.7% applies the per-class map on the row's true label (`bench/local_clf.py:1085`); the union ceiling above it is 92.8% [88.5–96.6] (`docs/eval-harness-2026-11-04.md` §5(b)) |
| The classical baseline's headline is template transfer | 93.19% holds out rows, **66.23%** holds out phrasing families (n=382 — a different task, see above) |
| The round-trip saving is conditional | 14.3% of round trips, **7.5 s slower**, crossover at ~5.8 s per model pass (n=9 tasks) |
| Decision latency on the CPU-only box | ~1,055 ms per decide hop (n=10 hops) |
| Teacher confidence is saturated on the easy band | t1 mean 0.940 (72 rows) |
| The iGPU is not faster | 2.72× slower per decision, 6.9× at prefill, 50/50 paired rows (n=50) |
