# bench/

- `replay.py` (P1) — runs a plan end-to-end through the fixture-player backend
  with zero network; prints hops, confidences, wall time.
- `live_probe.py` (P2) — routing probe over held-out pi-tool utterances
  against a live d1 server; writes per-row timing + conf to stdout/JSONL.
  `--expect-model <gguf-basename>` asserts the served model via `/props` and
  aborts before any probe traffic on a mismatch (warns loudly if omitted).
- `model_guard.py` — the `/props` served-model assertion (shared by
  `live_probe.py` and `lane.sh`); pure check plus a small CLI.
- `ab_run.py` — three-arm A/B (`pair` / `openrouter` / `d1_systemone`) over a
  stratified sample of `utterances_v2.json`. All three arms run on each
  utterance back to back with the order rotating per row, because between-arm
  comparison across minutes is invalid on a box whose local latency swings
  ~25%: a spike inside a row hits every arm, a spike between arms would be
  read as a difference. Reports percentiles (never a mean), the pair's
  bimodality split and additivity residual, McNemar on paired correctness, and
  the four degradation paths over real dead ports and a stub 429.
  `--summarize <run>.json` re-derives the report from a recorded run with no
  traffic and no spend.
- `serve_lane.py` + `lanes.json` — argv-safe llama-server launcher for a lane;
  builds the serve argv in-process so a `--chat-template-kwargs` JSON value
  never crosses a shell command line. `--print-argv` emits the argv as JSON.
- `lane.sh` — one full lane on dell321: serve via `serve_lane.py`, `/props`
  assertion, probe, teardown.
- `baseline.py` (P4) — scripted LLM-driver baseline for the research template
  comparison.
