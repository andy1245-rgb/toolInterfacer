# bench/

- `replay.py` (P1) — runs a plan end-to-end through the fixture-player backend
  with zero network; prints hops, confidences, wall time.
- `live_probe.py` (P2) — routing probe over held-out pi-tool utterances
  against a live d1 server; writes per-row timing + conf to stdout/JSONL.
  `--expect-model <gguf-basename>` asserts the served model via `/props` and
  aborts before any probe traffic on a mismatch (warns loudly if omitted).
- `model_guard.py` — the `/props` served-model assertion (shared by
  `live_probe.py` and `lane.sh`); pure check plus a small CLI.
- `serve_lane.py` + `lanes.json` — argv-safe llama-server launcher for a lane;
  builds the serve argv in-process so a `--chat-template-kwargs` JSON value
  never crosses a shell command line. `--print-argv` emits the argv as JSON.
- `lane.sh` — one full lane on dell321: serve via `serve_lane.py`, `/props`
  assertion, probe, teardown.
- `baseline.py` (P4) — scripted LLM-driver baseline for the research template
  comparison.
- `or_bench.py` — hosted-model sweep over the same 8-way routing task: one
  chat completion per utterance per model, accuracy pass at bounded
  concurrency plus a strictly sequential latency pass (the concurrent pass's
  timings are never reported as latency). Prompt material is imported
  verbatim from `live_probe.py`, so accuracy is comparable to the local probe
  runs. Cost comes from the provider's own `usage.cost` when OpenRouter
  returns it, else from token counts at the catalog prices. The key is read
  only from `OPENROUTER_API_KEY` — export it from your funBot `.env`; the
  harness never reads that file, logs the key, or puts it in argv. Results:
  `bench/results/or-bench-<date>.json` (`or-bench-2026-10-26.json` is the
  `--max-tokens 512` run, whose truncated replies understate the
  reasoning-heavy models; `-run2` is the 2048-token run, the primary one).
