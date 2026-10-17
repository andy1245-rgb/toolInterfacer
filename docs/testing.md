# Testing & benchmarking notes

## Where tests run

Benchmark/serve work goes to the **dell321** box over ssh — the local Lenovo
is compile-only territory (5–8W/hybrid CPU, powersave governor, heavy ambient
load makes latency numbers noisy).

```
ssh dell321 "powershell -Command ..."   # dell321 is Windows — schedule/serve there
```

## d1-3B launch recipe (measured best on this box so far)

```
llama-server -m models/d1-3B-Q4_K_M.gguf --mmproj models/mmproj-d1-3B-F16.gguf \
  --host 127.0.0.1 --port 8080 -c 8192 -t 8
```

- llama.cpp build lives in `extern/llama.cpp/` (gitignored), master @ b11706
- thread counter-intuitive on hybrid CPUs: t8 best here; t12 catastrophic
  (2P+8E spanning). Re-derive per box.
- `--cache-ram` tested, no effect (prefill is not the dominant term at these sizes)
- calibration observations (P2): conf ≥ 0.9 → 100% acc; the 0.7 route gate is valid

## Result files

`bench/results/*.json` — each run is one file with `timestamp`, config,
per-row data, calibration bands. Canonical summary: `live_probe_summary.json`.
Never overwrite; write new files (variants get suffixes like `-t8-v2`).
