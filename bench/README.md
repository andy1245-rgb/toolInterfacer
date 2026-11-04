# bench/

- `run.py` — **the one command**: score any arm on any frozen set and compare two
  reports. `bench/arms.json` is the arm spec (`list` reads it; nothing is
  hardcoded), `ti_bench/` is the implementation.

  ```bash
  python3 bench/run.py list
  python3 bench/run.py run --arm d1 --set v2 --out bench/results/d1-v2.json
  python3 bench/run.py compare bench/results/d1-v2.json bench/results/pair-v2.json
  python3 bench/run.py selftest          # offline, no model, CI-able
  ```

  Arms are built with `make_backend` directly, so neither the decision cache nor
  the `heuristic_micro` pre-gate can answer a row and put a 0 ms sample into the
  latency distribution (docs/decision-pair-vs-hosted.md). A `llama-server` arm
  asserts `/props` through `model_guard` **before** any probe traffic and aborts
  non-zero on mismatch, and an arm that cannot say which model it expects is
  refused unless `--expect-model` names one — the guard exists for exactly the
  failure the CORRECTION in `docs/model-sweep-consolidated.md` records, a lane
  labelled for one model that a server was answering with another.
  `warmup.calls_discarded` states how many throwaway calls were dropped; error
  rows are excluded from the latency distribution and counted.

  A report (`schema: ti-bench/1`) carries per-row `id`/`band`/`expected`/`chosen`/
  `confidence`/`latency_ms`/`real_call`, per-band accuracy, overall accuracy with
  a Wilson 95% interval and n, p50/p90/p99 latency, and a confidence-band
  calibration table. An existing output path is refused unless `--force`.
  Exit codes: 0 ok · 1 refused/malformed · 2 usage or I/O · 3 guard refused.
- `ti_bench/` — the harness modules: `stats` (percentile, Wilson, McNemar exact,
  calibration), `prompts` (the frozen probe, shared with `live_probe.py`), `sets`
  (frozen sets + stable row ids), `arms` (spec reader), `guard` (`/props` + key),
  `scoring`, `report` (no-clobber write, shape check), `compare`, `selftest`.
- `arms.json` — the arm spec: backend, what it `requires` (`[]` offline ·
  `llama-server` · `hosted`), the `expect_model` gguf basename the server must be
  serving, the `endpoint` conf path `--base-url` repoints, and `override` (dotted
  conf paths) so a variant is a named arm instead of an unrecorded flag.
- `replay.py` (P1) — runs one plan through the real engine and prints its hops,
  confidences and wall time. It uses the **configured** backend, which defaults
  to `pair`, so "no decision model" holds only once `TI_CONF_OVERLAY` points the
  choice decision at `fixture` (README Quickstart); and *zero network* holds
  only for a plan with no web op — an L2 `research` plan still runs the real
  `web_search` / `web_fetch` CLIs and fetches live pages, which takes minutes,
  not milliseconds. The fixture overlay replaces decision-making, not the ops.
- `live_probe.py` (P2) — routing probe over held-out pi-tool utterances
  against a live d1 server; writes per-row timing + conf to stdout/JSONL.
  `--expect-model <gguf-basename>` asserts the served model via `/props` and
  aborts before any probe traffic on a mismatch (warns loudly if omitted).
- `ab_run.py` — three-arm A/B (`pair` / `openrouter` / `d1_systemone`) over a
  stratified sample of `tests/data/utterances_v2.json`. All three arms run on each
  utterance back to back with the order rotating per row, because between-arm
  comparison across minutes is invalid on a box whose local latency swings
  ~25%: a spike inside a row hits every arm, a spike between arms would be
  read as a difference. Reports percentiles (never a mean), the pair's
  bimodality split and additivity residual, McNemar on paired correctness, and
  the four degradation paths over real dead ports and a stub 429.
  `--summarize <run>.json` re-derives the report from a recorded run with no
  traffic and no spend, so a reader can check the arithmetic without the model.

  This is **not** the harness behind `docs/decision-pair-vs-hosted.md`: that run
  is `results/ab-interleaved-30-run2.json`, built with `make_backend` directly,
  and no committed script generates it. `ab_run.py` is a second interleaved run
  of the same three arms on a different 30-row sample, driven through
  `resolve()`, recorded at `results/ab-three-arm-30rows.json`.
- `live_e2e.py` — the live proof that `POST /v1/useTool` runs end to end: it
  asserts the served model via `/props`, reads `/health` for the op registry *and
  the decide backend map* (a `done` from the replay fixture proves nothing), posts
  a fixed task list (L0 passthrough, L1 slot-fill, L2 research, an ambiguous
  intent that must ask, and the answer that resolves it), checks every response
  field by field against spec §3, and prints a PASS/FAIL table. Attaches to a
  running stack, or `--serve` starts and tears down its own on unused ports.
  Exit 0 pass, 1 contract/case failure, 2 stack unreachable. Evidence:
  `docs/live-e2e-2026-11-03.md`, `bench/results/live_e2e_summary.json`.
- `model_guard.py` — the `/props` served-model assertion (shared by
  `live_probe.py`, `live_e2e.py`, `run.py` and `lane.sh`); pure check plus a
  small CLI.
- `ab_probe.py` — one process per library root: imports the `ti` package named by
  `--service-root`, runs every `utterances_v2` row through `use_tool()` in-process,
  and records plan shape (hop ops, status, confidence) plus a listing-quality
  diagnostic for `filesys_glob` rows. Two lanes then compare with
  `--report-only --compare <other.json>`. It is the only bench that A/Bs two
  *checkouts* rather than two backends or two models, so it is the one that can
  show a plan-shape regression between commits: `run.py` scores decisions, this
  scores what the plan did. It writes results the way the section below
  specifies — every home path in the record is `$HOME/...`.
- `serve_lane.py` + `lanes.json` — argv-safe llama-server launcher for a lane;
  builds the serve argv in-process so a `--chat-template-kwargs` JSON value
  never crosses a shell command line. `--print-argv` emits the argv as JSON.
- `lane.sh` — one full lane on bench-host: serve via `serve_lane.py`, `/props`
  assertion, probe, teardown.
- `roundtrips.py` + `roundtrip_tasks.py` — the scripted LLM-driver baseline the P4
  acceptance criterion asks for: two arms over one fixed task list, the agent's own
  tool calls (arm A, no LLM: the round-trip count is a property of the shape, the
  per-pass cost is a stated number you can replace) against one `useTool` call (arm
  B), in round trips and wall time. Recorded: `docs/roundtrips-2026-11-03.md`.
- `classical_baseline.py` — the non-model baseline: re-runs the sibling
  benchmark's TF-IDF + linear-SVM protocol in-repo with the provenance, overlap,
  near/far and template-grouped CV checks it omitted, then scores the same backend
  on this repo's 150-row set. Every number in
  `docs/classical-baseline-2026-11-03.md` comes from it.
- `eval_harness.py` — **the measurement instrument**: group-aware splits (whole
  phrasing families held out), repeated splits across seeds, cluster-bootstrap
  intervals, and one `Predictor` interface so any signal lands in the same table
  under the same split. It refuses to emit a number without a split, because the
  repo already published one figure (93.19%) that reads as generalisation and is
  template reuse. `eval_signals.py` registers this repo's signals through it;
  `docs/eval-harness-2026-11-04.md` is the report, and its headline is that the
  150-row set has no families to hold out (63% of rows are atomic) while the
  382-row set does (−23.5 points when its templates are held out).
- `gen_heldout_v3.py` + `data/heldout_v3.json` — **the evaluation set that can be
  split by construction**: 415 rows over the 8 ops, generated from the authored
  frames in `tools/distill/frames.py`, with a `family` on every row and a train/test
  side that the *generator* assigned from the frame inventory before any utterance
  existed. A family is one authored frame (all slot-filled variants of one sentence
  skeleton) and never appears on both sides; 86% of rows sit in a multi-row family,
  against 37% on `utterances_v2.json`. `data/heldout_v3_families.json` records the
  seed and the whole assignment. `python3 bench/gen_heldout_v3.py --check` reports
  drift against the committed file; `report_v3.py` recomputes the proxy table
  (this set vs v2 on d1-3B) from two recorded runs, offline. Read
  `data/heldout_v3_NOTES.md` before quoting a number from it: it is synthetic, it
  cannot speak for real user phrasing beyond the frames, and its two decision-op
  classes are largely single-form families.
- `or_bench.py` — the hosted sweep: one chat completion per utterance per model
  against OpenRouter, an accuracy pass at bounded concurrency plus a strictly
  sequential latency pass (the concurrent pass's timings are never reported as
  latency). Prompt material is the frozen probe, imported from `ti_bench.prompts`,
  so accuracy is comparable to `live_probe.py`'s local runs. It is **not a local
  lane**: it needs an `OPENROUTER_API_KEY` (read only from the environment, never
  logged, never in argv) and it spends real money, which is why the two spend caps
  exist. A run is one labelled *configuration* and the sweep dimensions multiply:
  `--providers` (`Relace`, `Wafer`, …, `sort:price`, `order:Relace+Wafer`),
  `--variants` (`full`, `compact`), `--answers` (`json`, `json-noconf`, `label`,
  `letter`), `--reasoning` (`default`, `off`, `minimal`, `exclude`). Each row
  records the endpoint that actually answered (`provider`), a `fallback` flag when
  a requested endpoint list was not honoured, and the `cached_tokens` /
  `reasoning_tokens` / `cost_details` split, so cost is attributed to the endpoint
  that served it rather than assumed. A pin with `only` fails closed by default
  (`--allow-fallbacks` to relax it) — a silent fallback would otherwise measure a
  different endpoint than the one under test. `--sample N` takes a deterministic
  stride sample across the whole set; prefer it to `--limit N`, which is the first
  N rows and is band-skewed (the first 50 are all t1). `--model-max-tokens MODEL=N`
  raises the cap for a reasoning model whose JSON answer would otherwise truncate,
  and `--model-budget-usd` / `--budget-usd` stop a config or the whole run on
  provider-reported spend — **both default to 0 (off)**, because rows dropped by a
  cap leave the accuracy denominator and a capped run must not read as a bad one;
  when a cap does fire the entry records `budget_stop`, `skipped` and `n_attempted`.
  Per-endpoint prices for `z-ai/glm-5.3-flash` live in `PROVIDERS`; they are what is
  billed, unlike the catalogue block, which is a fallback only.
- `prefilter.py` — the deterministic pre-filter in front of a hosted routing call:
  15 regex rules over the utterance, tiered by how much of the surface form has to
  line up, first match wins. No model, no network, no state. It asserts no accuracy
  of its own; the rules are ranked by measured precision elsewhere.
- `cost_curve.py` — call-reduction cost curve from a recorded `or_bench.py` run:
  applies `prefilter.py` to every row, ranks rules by measured precision, and
  reports coverage / pre-filter precision / blended accuracy / $ per 1k decisions
  for every prefix of the ranking, plus the 20/40/60% points. Cost per point is
  measured per-decision cost × the calls the cascade still makes. Reads only a
  committed result file, so it runs offline.

See `docs/cost-reduction-glm-flash.md` for what those levers actually bought, and
for which raw result files were merged with it and which were not.

## The hosted sweep: three branches, one audit, one port

`docs/model-sweep-consolidated.md` is about the **local** lane sweep
(`live_probe.py` + `band_summary.py`, both here). The **hosted** sweep
(`or_bench.py`, a different experiment on the same 150 frozen rows) was authored on
three unmerged branches cut from the same base commit, and only the parts below
were ported. Recorded so the next reader does not have to re-decide it.

| branch | what it was | verdict |
|---|---|---|
| `bench/qwen-family` | the base commit alone: `or_bench.py` at 508 lines (22,486 B), a 12-line README bullet, two raw runs (`or-bench-2026-10-26.json` 482,469 B at 512 tokens, `-run2` 509,114 B at 2048). 33,245 insertions. | **discarded — superseded.** Its script is the ancestor of both continuations and the ported `or_bench.py` reproduces its default prompt byte for byte, so it carries nothing the port lacks. Its two runs are the truncated-at-512 baseline that the 2048 run was made to replace. |
| `bench/or-models` | base + `da16e7f` "provider pinning, prompt compaction and minimal reasoning, measured": the script grown to 38,915 B, `prefilter.py` (4,560 B), `cost_curve.py` (7,391 B), 15 raw runs totalling 4,535,614 B, and `docs/cost-reduction-glm-flash.md` (15,362 B). 20 files, 147,451 insertions. | **ported: script + both modules + the doc + one result file.** The 147k insertions are 4,535,614 B of raw model output; committing all of it would have added 76% to a 5,788,005-byte repository. Its headline numbers reproduce against the merged files (see the doc's corrections section for the two that do not). |
| `bench/or-models-2` | base + `095f0ef` "record the serving provider, cap tokens per model, and stop on spend": the script at 28,865 B, two raw runs (457,240 B and 114,609 B). 6 files, 52,070 insertions. | **complementary, partly ported.** A divergent continuation of the same base, not a stack: it has the spend ledger and per-model token cap that `or-models` lacks, and lacks the pinning / variant / answer / reasoning axes that `or-models` has. Both features were hand-merged into the ported script, with the run-wide cap hoisted out of the per-model loop (upstream it was constructed inside the loop, so `--budget-usd` reset for every model and never bounded the run) and **both caps defaulted to 0 instead of $0.50/$2.00** so the default run is unchanged. Its provider recording is redundant with `or-models`' `providers_served`. Its two result files were not merged: `run3`'s `meta/muse-wonder` headline accuracy of 98.9% is measured on **90 of 150 rows with zero `t3` rows**, because a $0.50 per-model cap stopped the pass — the JSON discloses this in `skipped` and `budget_stop`, but the accuracy field alone does not. |

## `results/` — recorded evidence, with the machine redacted

Every file under `results/` is a run that happened, kept as proof rather than
decoration: `docs/model-sweep-consolidated.md` exists because a lane was
mislabelled, and the `/props` capture under `results/dell/props/` is what proved
it. Do not edit a number in them.

The **machine-identifying path prefixes are redacted down to the model basename**, and
nothing else:

| recorded as | was | why it is safe to change |
|---|---|---|
| `$HOME/…` | an operator's home path | the gguf *basename* is what proved the mislabel, and that is untouched |
| `<workspace>\models\<model>.gguf` | a Windows profile path under `Documents\…`, with the sibling-project tree in front of it | same — the basename is the evidence, and the directory tree around it says nothing about the run |
| `bench-host` | a hostname | names the box for a reader; the date and the build id pin the run |

The numbers, timings, build ids, model basenames and `/props` bodies are
byte-identical to what was recorded. `tests/test_no_personal_paths.py` covers
`results/` now, so a new run cannot reintroduce a personal path — or a captured
Windows directory tree — into the repo.

## Unmerged branch: `bench/ab-probe`

`bench/ab-probe` (one commit, `5022904`, 2026-10-26) carries `ab_probe.py` and two
150-row result files. **The script was merged here; both result files were not.**

What the two files measured, because nothing else in the repo records it. `old` is
`6943650`, before `c5396c7` ("a listing covers the directory it names"); `new` is
`98efee2`, which contains it. Both are in `main`'s history now. 150 rows,
`d1-3B-Q4_K_M` on `:8080`, in-process `use_tool()`, 21 of the rows
`filesys_glob`-labelled.

- **`filesys_glob` rows: the fix returned listings and respected the constraint on
  none of them.** Old returned a path list on **0** of the 21 (13 of them stated a
  constraint and got no list at all). New returned one on **7**, and of the 5 the
  diagnostic could judge, **0 respected** the constraint the utterance named and
  **5 violated** it — each of the five was a **159-path** listing of the whole
  tree, so "every file ending in `.toml` anywhere in the repo" was answered with
  every file in the repo.
- All 150 rows: hops 3.68 → 3.46 (`filesys_glob` 3.762 → 2.952), confidence mean
  0.1202 → 0.4408, degraded rate 87.3% → 81.3%, wall mean 4559.8 → 3989.8 ms. Wall
  is indicative only: one box, two lanes, no interleaving and no discarded warm-up,
  which is the flaw `docs/decision-pair-vs-hosted.md` avoids by rotating arms.

That is a negative result about a fix, and it was never written up: no figure in
`docs/`, in `CHANGELOG.md`'s Measured section, or in `main`'s `bench/results/`
traces to either file.

**Why the result files stayed out.** They are 215,963 of the branch's 245,097
bytes (107,150 + 108,813 against 27,762 for the script), they carry 47 occurrences
of the operator's home path in the bare form `tests/test_no_personal_paths.py`
fails on, and the defect they size is gone. Re-checked 2026-11-03 at `4951262`,
same 21 rows, against this worktree rather than the `abharness` one the branch used:
5 of the 6 judged rows respect the named constraint, no row returns the whole tree,
and the remaining miss is row `57` (`bench_` prefix matched by substring instead of
by prefix). The glob path is one hop with no decide in it, so the re-check's
stand-in decision model cannot have influenced it; the list sizes are not comparable
across the two frames (159 then, 286 now, for rows that name no constraint).
Superseded rows published as if current is the failure `docs/model-sweep-consolidated.md`
is a CORRECTION about.

`ab_probe.py` itself needed no API change to run against today's `service/ti/`
(`use_tool`, `conf.load` and `model_guard.preflight` are unchanged in signature);
it was merged with one addition, the `$HOME/...` recording above.
