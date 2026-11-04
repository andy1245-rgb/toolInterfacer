# d1-3B on the Iris Xe iGPU (Vulkan) vs the CPU path — 2026-11-03

Question: this laptop's idle iGPU is free compute. Does ggml's Vulkan backend run
the d1-3B routing decision faster than the CPU build, and is it worth the shared
memory it takes?

**No, and no.** On this box the iGPU is **2.72× slower per routing decision**
(p50 8943.6 ms vs 3283.7 ms, n=50) and **6.9× slower at prompt processing**
(9.3 vs 64.3 tok/s). It lost **all 50 paired rows** — there is no subset of the
workload where it wins. Decode is roughly a tie (10.8 vs 13.1 tok/s), which is
the interesting part: the routing workload is a single forward pass with zero
generated tokens, so it is *prefill*-dominated, and prefill is precisely where a
power-limited iGPU with no cooperative-matrix support and no separate memory
does worst.

Evidence: `bench/results/igpu-vulkan-v1-2026-11-03.json` (primary interleaved
run), `igpu-vulkan-throughput-2026-11-03.json`, `igpu-vulkan-liveprobe-cpu-2026-11-03.json`
(canonical probe, CPU arm), `igpu-vulkan-freshserver-retest10-2026-11-03.json`
(the server-state check in "One thing that went wrong").

**Edit — 2026-11-04: one platform name abstracted.** The box table, the memory finding and the
first caveat named the platform the two resident VMs were running on; they say "two VMs" now.
The platform belongs to an unrelated private workload and is not what any of those three
sentences needs. The load confound they rest on — two guests were resident and holding shared
memory — is unchanged, as are every measurement, delta and conclusion below.

## Box

| | |
|---|---|
| CPU | i5-1235U, 2P + 8E, 12 threads |
| RAM | 15.7 GB, shared with everything else (two VMs were resident) |
| GPU | Intel Graphics (ADL GT2), 0x8086:0x46a8, Xe-LP, 96 EU |
| Driver | Mesa 23.2.1 ANV, Vulkan 1.3.255 device / 1.3.204 loader |
| Model | d1-3B-Q4_K_M.gguf, 1.67 GB, Q4_K - Medium |
| llama.cpp | master `f74e2fa` (the fork with `/v1/systemone`), build `b1-f74e2fa` |
| Arms | CPU build (`extern/llama.cpp/build`), Vulkan build (out-of-tree, `build-vk/`) |

Both arms served the *same* gguf, one process each, `-c 4096 -t 8 --parallel 1`;
the Vulkan arm added `-ngl 99 --device Vulkan0`. `/props` was asserted on both
arms before any probe traffic (`--expect-model d1-3B-Q4_K_M.gguf`): both reported
`d1-3B-Q4_K_M.gguf`, `Q4_K - Medium`, `b1-f74e2fa`.

## Build: what it actually took

Vulkan is *not* a flag-flip on this distro (Ubuntu 22.04 / jammy). Four separate
things were missing; none of them is the driver.

1. **`glslc` does not exist here.** `libvulkan-dev`, `vulkan-tools` and
   `glslang-tools` are installed, but ggml-vulkan's CMake does
   `find_package(Vulkan COMPONENTS glslc REQUIRED)` and `glslc` ships in the
   `glslc` (shaderc) package, which is not in jammy's index and whose only pool
   builds need glibc 2.38 (this box has 2.35 — `GLIBC_2.38 not found`).
   Fix: `bench/vulkan/glslc-shim.py`, a small translator over the installed
   `glslangValidator` covering the exact argument shape ggml uses
   (`-fshader-stage= --target-env= -o -O -MD -MF -D`). It also injects
   `#extension GL_GOOGLE_include_directive : require` after `#version` (glslc
   enables includes implicitly, glslangValidator does not) and writes the
   depfile CMake expects.
2. **`SPIRV-Headers` missing.** Extracted the jammy `spirv-headers` deb
   (headers only) and pointed `-DSPIRV-Headers_DIR` at it.
3. **Vulkan headers too old.** The 1.3.204 headers have no `VK_EXT_device_fault`,
   and `ggml-vulkan-types.h` references `PFN_vkGetDeviceFaultInfoEXT`
   unconditionally:
   `error: 'PFN_vkGetDeviceFaultInfoEXT' does not name a type`. Fix: Khronos
   `Vulkan-Headers` v1.3.275 (headers only) via `-DVulkan_INCLUDE_DIR`.
4. **`spv` namespace undeclared.** `ggml-vulkan-types.h` reaches
   `spirv/unified1/spirv.hpp` through `__has_include`, and ggml finds but never
   *links* the `SPIRV-Headers` target, so the include directory never reaches the
   compiler. Fix: `-DCMAKE_CXX_FLAGS=-I<vendor>/root/usr/include`.

Working recipe (all paths machine-local, nothing installed system-wide):

```bash
ROOT=$(pwd)                                  # the worktree
SRC=${TI_LLAMA_SRC:-$HOME/toolInterfacer/extern/llama.cpp}
BUILD=$ROOT/build-vk                         # gitignored
V=$BUILD/vendor

mkdir -p $V/bin $V/root
# 1. shim over glslangValidator (glslc is absent / ABI-incompatible here)
install -m755 bench/vulkan/glslc-shim.py $V/bin/glslc
# 2. SPIRV-Headers, headers only, from the jammy pool
curl -sSO http://ca.archive.ubuntu.com/ubuntu/pool/universe/s/spirv-headers/spirv-headers_1.6.1+1.3.216.0-1~ubuntu0.22.04.2_all.deb
dpkg-deb -x spirv-headers_*.deb $V/root
# 3. Vulkan-Headers >= 1.3.206 (1.3.204 lacks VK_EXT_device_fault)
git clone --depth 1 --branch v1.3.275 https://github.com/KhronosGroup/Vulkan-Headers $V/Vulkan-Headers

cmake -S $SRC -B $BUILD \
  -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
  -DVulkan_GLSLC_EXECUTABLE=$V/bin/glslc \
  -DSPIRV-Headers_DIR=$V/root/usr/share/cmake/SPIRV-Headers \
  -DVulkan_INCLUDE_DIR=$V/Vulkan-Headers/include \
  -DCMAKE_CXX_FLAGS="-I$V/root/usr/include" \
  -DCMAKE_C_FLAGS="-I$V/root/usr/include"
cmake --build $BUILD -j12
```

**Build outcome: success.** Configure ~1 s per attempt (four attempts), build
**630 s wall (10 m 30 s) with `-j12`**, exit 0. The source tree was not modified.

One consequence of the shim is worth stating: shader feature probing at configure
time reports `GL_KHR_cooperative_matrix`, `GL_NV_cooperative_matrix2`,
`GL_EXT_integer_dot_product`, `GL_EXT_bfloat16`, `GL_EXT_float_e2m1/e4m3` as
"not supported by glslc" (glslang 11.8 predates them), so those kernel variants
are compiled out. On Xe-LP that costs nothing for coopmat (the hardware has no
`VK_KHR_cooperative_matrix`), and nothing for this model (Q4_K, not bf16/fp4).
Integer-dot-product is the one that could plausibly matter for quantized matmul
and is the first thing to restore — by building shaderc from source — before
concluding the hardware cannot do better.

### The Mesa 23.2 driver was never the blocker

The iGPU loads and runs the model correctly:

```
Vulkan0: Intel(R) Graphics (ADL GT2) (11787 MiB, 3508 MiB free)
load_tensors: offloading output layer to GPU
load_tensors: offloading 29 repeating layers to GPU
load_tensors: offloaded 31/31 layers to GPU
load_tensors:      Vulkan0 model buffer size =  1589.03 MiB
load_tensors:  Vulkan_Host model buffer size =   205.08 MiB
load_tensors:    Vulkan0 KV buffer size =     8.00 MiB
ggml_backend_vk: Vulkan0 compute buffer size =    79.51 MiB
```

That is **~1676 MiB device-side (≈1.64 GiB)** — about double the ~800 MB this
experiment budgeted — plus 205 MiB of host buffers. Fresh-server output is
numerically identical to the CPU arm (`"The capital of France is" -> " Paris."`).

## Protocol

Servers: `bench/serve_local.sh` (CPU, :8080) and `bench/serve_vulkan.sh`
(Vulkan, :8081), both resident simultaneously so no arm is measured on a
different machine state.

1. `/health` then `/props` on both arms, asserted against
   `--expect-model d1-3B-Q4_K_M.gguf` (`bench/model_guard.py`). A mismatch aborts
   before any probe traffic.
2. **2 warm-up calls per arm discarded** (`warmup_discarded_ms` in the JSON):
   cpu 1910.5 / 3536.7 ms, vulkan 6683.5 / 10093.8 ms.
3. 50 rows of `tests/data/utterances.json`, in file order, in **blocks of 5**.
   Every row is asked of *both* arms back-to-back inside its block, and the lead
   arm **rotates per block** (cpu, vulkan, cpu, vulkan, …), so neither arm is
   systematically first or second. 215 load-average samples were taken every 3 s
   across the run: min 1.86, median 2.93, p90 3.64, max 4.14.
4. `ti.decide.resolve` is bypassed (`make_backend("d1_systemone")` per arm): the
   `choice` path wraps the backend in a regex pre-gate and a decision cache, and
   under interleaving a cache hit in the second arm would report cache latency as
   model latency. Same choice as `bench/results/ab-interleaved-30-run2.json`.
   The prompt and criteria are imported verbatim from `bench/live_probe.py`.
5. The result used is the **paired** per-row delta (same row, both arms, seconds
   apart). The per-arm percentiles are reported alongside it, but the pairing is
   what makes them trustworthy on a box that is not otherwise quiet.

Reproduce:

```bash
bash bench/serve_local.sh &                      # :8080  (TI_SERVER=<cpu build>)
bash bench/serve_vulkan.sh &                     # :8081
python3 bench/probe_interleaved.py \
  --arm cpu=http://127.0.0.1:8080 --arm vulkan=http://127.0.0.1:8081 \
  --expect-model d1-3B-Q4_K_M.gguf --block 5 --warmup 2 \
  --out bench/results/igpu-vulkan-v1-<date>.json
```

## Results — routing decisions, frozen v1 set (n=50, interleaved)

| arm | n | p50 | p90 | p99 | accuracy |
|---|---|---|---|---|---|
| cpu (8 threads) | 50 | **3283.7 ms** | 3543.5 ms | 3702.4 ms | **84.0%** |
| vulkan (`-ngl 99`, Vulkan0) | 50 | 8943.6 ms | 9644.5 ms | 11544.6 ms | 80.0% |

- **p50 ratio: 2.72× slower on the iGPU.** Mean 3328.6 vs 9136.1 ms.
- Paired: median delta **+5649.6 ms** (vulkan − cpu); **50/50 rows slower on the
  iGPU**, 0 rows faster. Spread is tight on both arms (cpu 3058–3706 ms,
  vulkan 8483–12410 ms), so this is not a tail effect.
- Accuracy 84.0% vs 80.0% is *two rows* at n=50, so it is not an accuracy
  regression measurement. The arms agreed on the choice in 45/50 rows; on those
  45 the confidence differs by a median of 0.021 (max 0.43). Of the 5
  disagreements, the CPU was right in 3 — including idx 0, where the CPU said
  `filesys_read` at conf 0.941 and the iGPU said `decide_extract` at conf 0.364 —
  the iGPU was right in 1, both wrong in 1.
- On a *freshly started* GPU server the two arms track much more tightly: in the
  10-row re-test the confidences matched to ~0.001 per row (0.949/0.951,
  0.966/0.967, 0.887/0.886, …). The 0.021 median gap in the 50-row run is one
  more symptom of the long-run drift described under "One thing that went
  wrong", not a separate numerics difference.
- For parity, the canonical probe (`bench/live_probe.py --expect-model`, which
  *does* go through `resolve`) on the CPU arm gave p50 4190.5 ms / p95 5295.5 ms /
  accuracy 84.0% at n=50 — but that run happened while a peer session was
  hammering the box, so it is 28% above the interleaved CPU number and is quoted
  only as parity evidence, not as the baseline.

Recorded history for context: `docs/testing.md` claims ~1.7 s per decision
(that matches the recorded *min*, 1816 ms), and `bench/results/live_probe_summary.json`
from 2026-10-17 has p50 3496.5 ms at 84.0% accuracy. Today's clean CPU arm
(3283.7 ms) reproduces that run; the "1.7 s" figure is a floor, not a p50.

## Prefill vs decode — where the 2.7× comes from

The routing endpoint is prefill-dominated by construction: one POST, one forward
pass, zero generated tokens, ~131 prompt tokens. Measured two ways:

**llama-server's own per-request log, routing requests (131-token prompts):**

| arm | prompt processing | per 131-token prompt | n |
|---|---|---|---|
| cpu | **35.4 tok/s** (28.3 ms/tok) | 3.77 s | 2 |
| vulkan | **11.8 tok/s** (84.7 ms/tok) | 11.12 s | 5 |

**`/completion` timings block**, 135-token prompt, `cache_prompt=false`, median of
3 reps (full samples in `igpu-vulkan-throughput-2026-11-03.json`):

| arm | prompt processing | generation | prefill reps | decode reps |
|---|---|---|---|---|
| cpu | **64.3 tok/s** (15.6 ms/tok) | 13.1 tok/s | 65.1 / 64.0 / 64.3 | 13.0 / 13.1 / 13.5 |
| vulkan | **9.3 tok/s** (107 ms/tok) | 10.8 tok/s | 9.3 / 9.5 / 8.5 | 10.8 / 10.3 / 10.9 |

So: **prefill 6.9× slower on the iGPU, decode 1.2× slower.** The CPU's 15.6 ms/token
prompt processing reproduces the ~19 ms/token in `docs/testing.md` (that figure
was measured with the mmproj loaded, so slightly heavier). The two prefill numbers
for the GPU (9.3 tok/s for a clean 135-token batch, 11.8 tok/s inside the routing
call) are the same order; the difference is batch shape and graph reuse.

Read together with the routing result: a decision costs ~3.3 s on the CPU, of
which roughly 2 s is prompt processing; on the iGPU the prompt processing alone
is ~9–11 s. There is nothing left for the GPU to win back.

## Verdict

**The iGPU loses, and it is not close.** 2.72× slower at p50 over 50 paired rows,
0/50 wins, 6.9× slower at the prefill that *is* the workload. It also costs
**1.64 GiB** of shared memory (not ~800 MB), which on this box is 1.64 GiB the
VMs and the CPU arm cannot use. Do not switch the routing lane to Vulkan,
and do not spend the memory.

Why, in order of how confident the evidence makes me:

1. **Decode being a near-tie while prefill is 6.9× worse is the signature of a
   shared-memory-bandwidth machine.** Both engines read the same 1.67 GB of
   weights through the same LPDDR5; batch-1 decode is bandwidth-bound, so they
   land within 20% of each other (13.1 vs 10.8 tok/s). Prefill is compute-bound,
   and that is where the iGPU has to justify itself — and at 9 tok/s it is doing
   ~2 GFLOP/s effective on a 96-EU part with a ~2 TFLOP/s fp32 peak. It is not
   winning on math, so it is losing on overhead.
2. **No cooperative matrix, no integer dot product in this build.** Quantized
   matmul on the Vulkan path dequantizes to fp16 and runs plain fp16 matmul
   shaders; the CPU path uses AVX2/VNNI on the quantized data directly. Xe-LP
   genuinely has no `VK_KHR_cooperative_matrix` (so that costs nothing), but
   `GL_EXT_integer_dot_product` was unavailable to my shim because glslang 11.8
   can't compile it — that is a *build-tool* limitation, not a hardware one, and
   it is the one plausible chunk of headroom I could not test.
3. **Driver maturity is a real but secondary factor.** Mesa 23.2.1 is old; the
   per-node dispatch cost of a ~500-node graph at batch ~131 is on the critical
   path in a way it would not be for a big batched prefill. I have not separated
   that from (2), and doing so needs a shaderc-built glslc plus, ideally, a newer
   Mesa — neither of which I installed.

### One thing that went wrong (and why the numbers above are from fresh servers)

My first 50-row pass reported **6.0% accuracy** on the Vulkan arm — identical
choices and identical confidences for runs of consecutive rows, and `/completion`
producing `/b/b/b/b…`. The CPU arm, which had been through the same sequence,
stayed at 84%.

The trigger was state in the long-lived server, not broken math: a **freshly
started** Vulkan server was correct, and reproduced the CPU arm row for row
(80% on the first 10 rows, confidences matching to ~0.001 —
`igpu-vulkan-freshserver-retest10-2026-11-03.json`). `test-backend-ops -b Vulkan0
-o MUL_MAT` passes on this driver, which is consistent with that.

I did not isolate the exact trigger. The observed history before the collapse
included a 128-token generation request (`/completion`, `n_predict=128`) on the
same slot, so either slot/KV reuse after a generation, or Vulkan graph reuse
across prompts with different states, is the likely culprit — a state-handling
bug on the Vulkan backend, not a shader numerics bug. It is a **separate reason
not to serve this workload on Vulkan**, and it is why the numbers in this doc were
taken with a freshly started server and no generation probe in the same process.

## Caveats

- **Absolute latencies are on a busy laptop.** The clean pass ran at load-average
  1.86–4.14 with two VMs resident; the CPU arm's p50 moved ~28% between a
  loaded pass (4190 ms) and the clean pass (3284 ms). The *paired* deltas are the
  headline for that reason, and they are all one-directional (50/50), so noise
  cannot be manufacturing the result — only its exact size.
- **n=50.** The accuracy difference (84% vs 80%) is two rows and is not a
  measured accuracy regression. The 150-row `utterances_v2.json` set would settle
  it; it was not needed to answer the latency question.
- **One model, one quantization.** Q4_K_M at 3B on Xe-LP. A larger model, a
  smaller one, or an fp16 weight format would shift the balance between bandwidth
  and compute and could change the answer.
- **The shim is mine, not shaderc.** Everything above is "this ggml Vulkan
  backend, compiled with glslang 11.8, on Mesa 23.2.1 ANV". A modern shaderc
  and/or newer Mesa might recover the integer-dot-product path; that is the
  experiment I would run next, and it is the only plausible route to a different
  verdict.
- Both arms used `-t 8` (the recorded best on this hybrid CPU; `t12` spanning
  2P+8E is catastrophic here). The Vulkan arm's `-t` only affects its CPU-side
  ops, so it was not swept.
