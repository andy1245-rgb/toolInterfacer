# Testing & benchmarking notes

## Running the stack locally

This is the recipe to use. Two processes, both lightweight enough to leave resident:

```
bash bench/serve_local.sh            # decision model on 127.0.0.1:8080 (text-only, no mmproj)
PYTHONPATH=service python3 -m ti.http.api   # service on 127.0.0.1:1113 (extension spawns it if down)
```

`bench/serve_local.sh` needs a built `extern/llama.cpp` and `models/d1-3B-Q4_K_M.gguf`
(`TI_SERVER` / `TI_MODEL` override both) and exits 1 naming whichever is missing. Nothing
else needs the model: `python3 -m pytest tests/ -q`, `bench/demo.py` and the service on its
own all run without one.

Text-only serving drops ~850 MB and a second of init; routing never needs the
vision projector. Measured here: model ready in ~3 s, ~2.9 GB RSS.

Latency on this box (d1-3B Q4, CPU): **~1.7 s per routing decision**, ~50 ms on a
decision-cache hit. Prefill is ~19 ms/token, so *state size is the latency knob* —
which is why `decide.max_candidates` defaults to 6 and the candidate list is sent
once (criteria), never duplicated into the state. Raising it to 64 costs ~31 s.

## Where benchmark numbers come from

Latency and serve work goes to the **bench-host** box over ssh — the local Lenovo
is compile-only territory (5–8W/hybrid CPU, powersave governor, heavy ambient
load makes latency numbers noisy). Reach for it when you are *measuring* something;
everything else in this file applies to whatever box you are on.

```
ssh bench-host "powershell -Command ..."   # bench-host is Windows — schedule/serve there
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

## Portability rule (enforced)

Do not hardcode machine-specific absolute paths in source, config, tests, or docs
(`/home/<user>/...`, `C:\Users\<user>\...`). Use an environment variable, a
`~`-relative default, a repo-relative path, or `<placeholder>`. `tests/test_no_personal_paths.py`
fails the suite if one appears. Machine-specific bench paths belong in
`bench/lane.env` (gitignored) — see `bench/lane.env.example`. Recorded evidence under
`bench/results/` is **guarded too**: the recorded model paths were redacted to `$HOME/` and
`C:\Users\<user>`, keeping the basenames that prove what ran where, so that tree is no
longer exempt from the check.

## Proving the wire, not just the engine

`tests/` drives the runner with the fixture backend, so it cannot fail on a break
in HTTP ingress, the decide wall, or the response contract. `bench/live_e2e.py`
can, over a real socket against a live model:

```
python3 bench/live_e2e.py --serve        # start llama-server + service, probe, tear down
python3 bench/live_e2e.py --skip-l2      # attach to what is running (the L2 case drives the web CLIs)
```

It needs a llama.cpp server answering `/props` — `--expect-model <gguf-basename>`
asserts what it is serving *before* any probe traffic and aborts on a mismatch;
without the flag the run prints a warning and proceeds unguarded, which is how a
stale server gets measured silently. It refuses to report success when
`/health` says the choice backend is a replay fixture, and exits non-zero when the
stack is unreachable. Drop the L2 case with `--skip-l2` unless the `web_search` /
`web_fetch` CLIs are configured: that case really searches and really fetches, and
takes minutes.

Every POST carries the service token, resolved in the same order the service uses
(`service/ti/http/auth.py`: the variable named by `http.token_env`, then the 0600
runtime file); `--serve` hands the child it spawns that same value, and `--token`
overrides both when attaching to a service someone else started. A 401 aborts the
run with exit 2 instead of printing a table of failures. `tests/test_probe_guard.py`
pins the header and the resolution order.

The one recorded run predates that: `docs/live-e2e-2026-11-03.md` shows `0/4` with
`HTTP 401` on every case, because the probe sent no token at all — so it measured
nothing, and its PASS/FAIL table is not evidence about the wire. Until a run with
the token in place is recorded, the evidence for the HTTP route (including the
ask-back) is `tests/test_live_e2e_regressions.py::HttpRouteTest`, which serves the
service's own handler on a loopback port with the fixture backend and posts real
requests through it.
