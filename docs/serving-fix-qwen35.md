# Serving fix — qwen35 in-process GPU offload on dell321 (2026-10-17)

Task: root-cause why qwen35-arch lanes served in-process (pyhost recipe:
`e3\python.exe` hardlinked into the llama.cpp bin dir + `LoadLibrary
llama-server-impl.dll` → `llama_server(argc, argv)`) ran CPU-only while
`llama-bench` on the same build offloads the same GGUFs.

**Verdict: unresolved at driver-flag level. Strongest hypothesis below.**
Every documented experiment is first-hand measured on dell321 today; nothing
inherited from q4-scout except where cited.

## Reproduced baseline (all today, build b11512/a11f57ba9 unless noted)

| # | Path | Build | Context shape | qwen35 prefill | Source |
|---|------|-------|---------------|----------------|--------|
| 1 | `llama-bench.exe -p 146 -n 0 -ngl 99` | b11512 | real exe, from e3 dir | ~1.6 ms/tok (585–676 t/s) | E-bench-A |
| 2 | `llama-bench -p 256/2048, -ub 512 -b 2048 -n 1, --no-warmup, -fa off, --no-op-offload, --no-host, --repack 0, --poll 100` | b11512 | same as 1, all knobs | 585–919 t/s in all variants | E-bench-B |
| 3 | `llama-bench -ngl 0` | b11512 | CPU-only | 83.5 t/s (12 ms/tok) | E-bench-C |
| 4 | pyhost server, exact `lane.sh` baseline (argc=13: `-m <gguf> --host 127.0.0.1 --port 8080 -c 8192 -t 8 -ngl 99`) | b11512 | llama-server in-process | 97 t/s cold (13.4 ms/tok), cached-4-tok 261 ms/tok | E-qdiag-1 |
| 5 | llama-completion (single-shot CLI context) `-p` 22 tok, `-c 8192 -n 1 -lv 5` | b11512 | llama_context via common | 121 t/s (8.27 ms/tok) | E-cli-1 |
| 6 | pyhost server, d1-3B control (`-c 8192 -ngl 99`, same invocation shape) | b11512 or b11510 | same as 4 | **GPU-fast** (2.57 ms/tok at 64 tok — scout's d1control log; re-run today 547 ms for 145 tok ≈ 3.8 ms/tok) | E-d1-1 |
| 7 | server via recreated `d1srv_fix.exe` (real llama-server exe, AV-safe carve, same args as 4) | b11512 | llama-server exe | 18.8 t/s cold / 263 ms/token cached | E-exe-1 |
| 8 | pyhost against engine dir (`b11510/c35b66744`) impl.dll | b11510 | llama-server in-process | same CPU-class (78 MiB GPU compute buffer) | E-b10-1 |
| 9 | server `--rpc 127.0.0.1:50052` (local ggml-rpc-server held the Vulkan0 device) | b11512 | RPC device | still CPU-class throughput | E-rpc-1 |

Intern-Decision-4B reproduced the scout's broken state first-hand as well
(E-qdiag-0): argc=13 baseline, RSS 3.9 GB, prefill 13.4–20.8 ms/token, no
capacity for the 300 ms bar; matches their q4-Inter-*.json p50 ~2454 ms.

## What is NOT the cause (each ruled out first-hand)

Server-side flags — none of these change the outcome (all still ~97 t/s
class, "cached 4-token prompt eval" tens→260 ms per token):

- `--parallel 1` (→ `n_slots=1, kv_unified=false`), and the default
  `n_slots=4, kv_unified=true` (`n_parallel` auto->4 on these builds)
- `-c 256/512/1024/4096/8192`, `-b 256`, `-ub 64/128/256/512`
- `-fa on|off|auto`
- `--no-host` (bypass host buffers), `--op-offload true|false`
- `--fit off`, `--no-repack`, `-lzm on`, `-sm none|tensor`, `--no-warmup`
- `--jinja --chat-template-kwargs {"enable_thinking": false}` (scout's known
  fast-lane template config)
- `--rpc <local rpc server>` (compute in a separate process)
- exchanged build: b11510 vs b11512 (`llama-b11510-win-vulkan-x64.zip` vs
  `llama-b11512-win-vulkan-x64.zip` in Research/jevResearch/lab/d1bench)
- recreated-exe serving (tibench-style `llama-server.exe` carve, E-exe-1)
- token-array `/completion` prompt (bypasses the chat pipeline entirely):
  still 73 t/s cold, 281 ms/token cached

Model/things — ruled out:

- the three qwen35 GGUFs (Qwen3.5-4B / Intern-Decision-4B / Kev-4B) are the
  same arch `qwen35`, 32 blocks, same `ssm.*`/attention params; Kev carries
  extra `qwen35.decision.*` keys that make the server switch to embedding
  mode (its own quirk), but Intern and Qwen3.5 are structurally identical
  and behave identically slow — so it is not a per-file quant artifact
- backend discovery: in-process at `-lv 5` the server prints
  `Vulkan0 : Intel(R) Arc(TM) 140V GPU (18425 MiB, 17657 MiB free)` and
  `llama_prepare_model_devices: using device Vulkan0` — Vulkan registers
  fine under the pyhost. `GGML_BACKEND_PATH` (E-exe's `.cmd`-level test) had
  no effect on the compute path
- model tensor placement is identical between fast and slow paths:
  `load_tensors: Vulkan0 model buffer size = 2603.50 MiB / Vulkan_Host model
  buffer size = 497.31 MiB` in **both** llama-bench and the server

## The one measurable difference (and where it leads)

For the same graph (`nodes = 1929, splits = 2`, same FLASH_ATTN_EXT node ids,
same `input objects = 4, input tensors = 10`, same worst-case reserve
`n_tokens=256, n_seqs=1, n_outputs=1`), the compute-buffer split differs:

| Context | Vulkan0 compute buffer | Vulkan_Host compute buffer | prefill |
|---|---|---|---|
| llama-bench (fast) | 252.64 MiB @ub256 / 513 MiB @ub512 (completion path) | 10.14 / 28 MiB | 585–919 t/s |
| llama-server (slow) | 37.13 MiB @ub256 / 82.02 MiB @ub512 | 10.14 / 28 MiB | 73–97 t/s |

i.e. the server's scheduler keeps ~6–7× as much of the hybrid graph's
executed working set off the device compute buffer (host-attached
`Vulkan_Host`/CPU chunk) — which on Arc-140V UMA runs at CPU pace. The CVE,
if you will, is in how this build's llama-server (and llama-completion,
513 MiB GPU chunk but 121 t/s — a partial mismatch, see caveat) constructs/
reserves its context for the qwen35 hybrid batch path vs llama-bench.
LFM2 (d1) in-process is GPU-fast through the very same server, so the defect
is specific to the qwen35 hybrid prefill scheduling in the common/server
context.

Caveat: llama-completion got a 513 MiB GPU compute chunk (like bench) yet
still ran at 121 t/s, so compute-buffer size tracks the split but is not
itself the causal variable; no context-level flag we could reach changes the
execution class. The scout's fast lane (their q4-Qwen3.5-4B json, p50
201/91 ms, "GPU engaged") is currently not reproducible by any invocation
recombination we tested; its serve args are lost (not in `logs/`).

Strongest remaining hypotheses (in order):

1. A build-level scheduler/graph-composition regression for the qwen35
   hybrid prefill inside llama-server/llama-completion contexts on Windows/
   Vulkan (llama-bench bypasses it because it builds its context directly
   through `llama_context_params` without the common init path).
2. A UMA-specific (`uma: 1`) buffer-policy difference (`Vulkan_Host`
   buffer "not the default buffer type for device Vulkan0 for async
   uploads" appears in bench logs too) — the host side silently eats the
   heavy ops.
3. Something in the environment of the scout's 16:21 run (descriptor state,
   driver clock state, or an arg we cannot reconstruct).

Follow-up worth an hour: patch `serve_pyhost.py`-equivalent to construct the
context via llama-bench's `to_llama_cparams()` path, or run the llama.cpp
equivalent of `llama-sweep` upstream (b11513+ had qwen3.5/vulkan fixes
(`theneuralfeed` reported gated-deltaNet-vulkan merge ~b1151x)) and test
whether a newer build moves the qwen35 server path off the host chunk.

## Reproduce today

Dell-side driver (new files, this run): `toolInterfacerLab/qdiag.py`
(config-in-json pyhost; fd-redirect gives full load/offload lines at `-lv 5`),
`qdiag_b10.py` (same, pointed at the b11510 engine dir), plus
`qdiag_config.json` / `qdiag.log` state and `mkrexe.py` (AV-safe carve of
exes from the zips into toolInterfacerLab; used for E-exe-1 and the rpc
server test). Local repo copy: `bench/pyhost_qwen35.py`.

Acceptance bar not met: best qwen3.5-4B lane p50 measured ≈1100 ms (probe
post-cache), far above the 300 ms target; **no `fix-*.json` lane files are
written** because neither run qualifies as a pass. q4-scout's `q4-*.json`
untouched.

Teardown verified: dell:8080, dell:50052 free; local 18080 free; no python/
rpcsrv/llama processes running. AV consumed the treated exes
(`d1srv_fix.exe`, `rpcsrv_fix.exe`, `serve_fix.cmd`, `rpc_launch.cmd`) per
the known-§7.2 pattern; the retained diagnostics (`qdiag.py`, `qdiag_b10.py`,
`mkrexe.py`, `mkrpc.cmd`, `qdiag_config.json`, `logs/qdiag.log`) sit in
`toolInterfacerLab/`.
