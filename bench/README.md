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
