# bench/

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
  `live_probe.py`, `live_e2e.py`, and `lane.sh`); pure check plus a small CLI.
- `serve_lane.py` + `lanes.json` — argv-safe llama-server launcher for a lane;
  builds the serve argv in-process so a `--chat-template-kwargs` JSON value
  never crosses a shell command line. `--print-argv` emits the argv as JSON.
- `lane.sh` — one full lane on dell321: serve via `serve_lane.py`, `/props`
  assertion, probe, teardown.
- `baseline.py` (P4) — scripted LLM-driver baseline for the research template
  comparison.
