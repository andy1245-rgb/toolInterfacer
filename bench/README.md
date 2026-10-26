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

  A run is one labelled *configuration*, and the sweep dimensions multiply:
  `--providers` (`Relace`, `Wafer`, …, `sort:price`, `order:Relace+Wafer`),
  `--variants` (`full`, `compact`), `--answers` (`json`, `json-noconf`,
  `label`, `letter`), `--reasoning` (`default`, `off`, `minimal`, `exclude`).
  Each row records the endpoint that actually answered (`provider`), a
  `fallback` flag when a requested endpoint list was not honoured, and
  `cached_tokens` / `reasoning_tokens` / the `cost_details` split, so cost can
  be attributed to the endpoint that served it rather than assumed. A pin with
  `only` fails closed by default (`--allow-fallbacks` to relax it) — a silent
  fallback would otherwise measure a different endpoint than the one under
  test. `--sample N` takes a deterministic stride sample across the whole set;
  prefer it to `--limit N`, which is the first N rows and is band-skewed (the
  first 50 rows are all t1). Per-endpoint prices for `z-ai/glm-5.3-flash` live
  in `PROVIDERS`; they are what is billed, unlike the catalogue block.
- `prefilter.py` — the deterministic pre-filter in front of a routing call: 15
  regex rules over the utterance, tiered by how much of the surface form has to
  line up, first match wins. It asserts no accuracy of its own; the rules are
  ranked by measured precision elsewhere.
- `cost_curve.py` — call-reduction cost curve from a recorded run: applies
  `prefilter.py` to every row, ranks rules by measured precision, and reports
  coverage / pre-filter precision / blended accuracy / $ per 1k decisions for
  every prefix of the ranking, plus the 20/40/60% points. Cost per point is
  measured per-decision cost × the calls the cascade still makes.

See `docs/cost-reduction-glm-flash.md` for what those levers actually bought.
