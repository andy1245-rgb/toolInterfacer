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
- `model_guard.py` — the `/props` served-model assertion (shared by
  `live_probe.py`, `run.py` and `lane.sh`); pure check plus a small CLI.
- `serve_lane.py` + `lanes.json` — argv-safe llama-server launcher for a lane;
  builds the serve argv in-process so a `--chat-template-kwargs` JSON value
  never crosses a shell command line. `--print-argv` emits the argv as JSON.
- `lane.sh` — one full lane on dell321: serve via `serve_lane.py`, `/props`
  assertion, probe, teardown.
- `baseline.py` (P4) — scripted LLM-driver baseline for the research template
  comparison.
