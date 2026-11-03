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
  refused unless `--expect-model` names one — the guard
  docs/model-sweep-consolidated.md (CORRECTION) exists for. `warmup.calls_discarded`
  states how many throwaway calls were dropped; error rows are excluded from the
  latency distribution and counted.

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
- `replay.py` (P1) — runs a plan end-to-end through the fixture-player backend
  with zero network; prints hops, confidences, wall time.
- `live_probe.py` (P2) — routing probe over held-out pi-tool utterances
  against a live d1 server; writes per-row timing + conf to stdout/JSONL.
  `--expect-model <gguf-basename>` asserts the served model via `/props` and
  aborts before any probe traffic on a mismatch (warns loudly if omitted).
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
- `serve_lane.py` + `lanes.json` — argv-safe llama-server launcher for a lane;
  builds the serve argv in-process so a `--chat-template-kwargs` JSON value
  never crosses a shell command line. `--print-argv` emits the argv as JSON.
- `lane.sh` — one full lane on bench-host: serve via `serve_lane.py`, `/props`
  assertion, probe, teardown.
- `baseline.py` (P4) — scripted LLM-driver baseline for the research template
  comparison.

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
