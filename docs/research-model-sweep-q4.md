# Model sweep Q4 (2026-10-17) — scout for a decision-routing backend ~4B-class

Gate conditions honored: datasets frozen (`tests/data/utterances.json` 50 v1,
`tests/data/utterances_v2.json` 150 v2), prompt/criteria untouched, lanes run
through the Jev-style `logit_read` adapter (`/v1/chat/completions`,
`max_tokens=1`, top_logprobs=20, softmax over A–K letters, conf = 1−H/lnK).
Bar to beat: d1-3B Q4_K_M on dell = 84% @ p50 ~155ms (Q8: 86%).

## Selection requirements

- llama.cpp-servable GGUF, `Q4_K_M`/`Q4_0`, file ≤ 3.5 GB, params ≤ ~6.5B
- one-forward-pass logit read eligible: base model, non-thinking instruct, or
  think-mode disablable via `chat_template_kwargs` (the thinking trap cost
  hours before — Qwen3.5 instructs think by default; llama-server applies
  `--chat-template-kwargs '{"enable_thinking": false}'` correctly, see
  llama.cpp issue #20409: llama-server confirmed applying the kwargs; the
  broken shell quoting there was llama-cli's, not the server's)
- license OK (Apache-2.0 / permissive)
- structured-decision / classification evidence from the internet

## Findings from the web

1. **The "decision model" class is now real and llama.cpp-served.** Liquid's
   d1 line (our incumbent, `/v1/systemone` via llama.cpp PR #29818) reported
   Decision Index 0.2.1 = 48.57 (d1-3B), "ahead of every model under 10B and
   on par with Decider 35B-A3B (47.11)"; independent Winnow-12B sits higher
   (12x class, out of size budget). Sources:
   - https://rits.shanghai.nyu.edu/ai/liquid-ai-open-d1-decision-models/
   - https://www.laura-martel.com/blog/liquid-ai-open-d1-decision-models/
   - https://huggingface.co/LiquidAI/d1-3B
2. **Qwen3.5-4B (post-trained instruct)** is a serious agentic router at this
   size: model card reports **BFCL-V4 50.3** (Qwen3.5-9B 66.1; Qwen3-30BA3B-
   Thinking-2507 only 42.4) and **TAU2-Bench 79.9** (beats Qwen3.5-9B 79.1),
   MMLU-Pro 79.1, IFEval 89.8. **BFCL-V4 50.3 at 4B vs 42.4 for Qwen3-30B-A3B
   is the headline routing-strength evidence.** Thinks by default;
   `enable_thinking: false` is supported via chat-template kwargs. Apache-2.0.
   Sources:
   - https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/README.md
   - https://unsloth.ai/docs/models/qwen3.5 (small series 0.8B/2B/4B/9B, hybrid
     reasoning, Qwen3.5 base mounts shared with Qwen3-Next arch)
   - https://github.com/ggml-org/llama.cpp/issues/20409 (llama-server applies
     `--chat-template-kwargs`; also has `--reasoning off` switch)
   - llama.cpp supports the `qwen3_5` hybrid arch (Gated DeltaNet merged for
     Vulkan: https://theneuralfeed.com/article/gated-delta-net-for-vulkan-merged-in-llama-cpp/fmLjqU22);
     internal (pre-existing) evidence: dell-sweep served qwen3.5-0.8b/2b fine
     on b11512 vulkan.
3. **Intern-Decision-4B** (InternLM, 2026-09-26) is a Jev-schema decision
   fine-tune **of exactly Qwen3.5-4B** — reads a state + typed questions and
   scores option symbols A,B,…,Z — single-token logits read at one position each;
   strongest reported card at this size class: avg 90.02 across 7 suites
   (Jevbench-Easy 100 / Hard 73.87 / Typed-Decision 80.55 / ToolACE 96.45 /
   AG News 90.82), Brier 0.347, ECE 0.065; ~44 ms/query on RTX 4090. Apache-2.0
   (Qwen license preserved upstream). Decision-fidelity-checked GGUF quant: community
   **yeaay/Intern-Decision-4B-GGUF** with a **decision-level fidelity eval vs
   BF16**: Q4_K_M avg **89.42%** (BF16 89.88; Q5 89.96; Q8 89.88) — Q4_K_M
   costs only ~0.5pp avg. Caveat from that card: its native prompt format is
   the Jev question-schema, **not** the frozen Jev letter prompt of live_probe
   — via the frozen adapter it is being retested zero-shot, results may
   differ from card numbers (uncertain until lanes run).
   Sources:
   - https://huggingface.co/internlm/Intern-Decision-4B/raw/main/README.md
   - https://huggingface.co/yeaay/Intern-Decision-4B-GGUF/raw/main/README.md
     (bf16-source-verified conversion, SHA-256 list, `--no-nextn` handled)
   - https://thegtmdirectory.com/models/internlm-intern-decision/intern-decision-4b (release/license meta; URL is thegtmdirectory.com)
   - https://github.com/InternLM/Intern-Decision
4. **Kev-4B** (Jared Palmer, Kev 1.0, 2026-09-24) is the dedicated System One
   decision LoRA on Qwen3.5-4B-**Base** + pointer head + T=2.41 calibration:
   out-of-domain transfer-v4 **locked test 83.8 acc / Brier 0.224 / ECE 0.017
   / coverage 0.701** at ≤5% error; trained breadth of public classification
   data; `POST /v1/systemone` API. Official ggml-org GGUF conversion exists.
   Caveat: pointer-head exists only in the Kev serve path; a plain
   `llama-server` GGUF lane exercises the LoRA'd backbone through the frozen
   letter prompt, not the pointer head (uncertain until lanes run).
   Sources:
   - https://huggingface.co/jaredpalmer/kev-4b/raw/main/README.md
   - https://huggingface.co/ggml-org/Kev-4B-GGUF (30.9k DLs, apache-2.0 tags)
   - ecosystem context: https://www.explainx.ai/blog/llama-cpp-decision-model-support-2026
     lists llama.cpp-servable open decision GGUFs incl. Kev-4B, Laya, OpenJev,
     Julia-1.
5. **qwen3.5-4b-Base zero-shot lane — excluded.** No reliable Q4 GGUF exists
   directly (only safetensors; community conversions scattered), and the
   internal evidence (qwen3.5-0.8b base at 16% on logit_read, near-uniform
   letters) plus this class's lack of letter-routing training make base
   lanes weak. Its decision-tuned counterpart (Kev-4B) is the useful lane.

## Shortlist (top 3, in research-ranking order)

| # | Model | Q4 file | Size | Key evidence (source) | License | Thinking status |
|---|---|---|---|---|---|---|
| 1 | **Intern-Decision-4B** (decision-tuned Qwen3.5-4B) | `Intern-Decision-4B-Q4_K_M.gguf` (yeaay) | 2.71 GB | avg 90.02 over 7 decision suites; ToolACE 96.45; Brier 0.347; Q4 keeps 89.42 (card + yeaay fidelity eval) | Apache-2.0 | non-thinking by design (single forward pass; no `generate()`) |
| 2 | **Kev-4B** (System One decision LoRA on Qwen3.5-4B-Base) | `Kev-4B-Q4_K_M.gguf` (ggml-org) | 3.03 GB | transfer-v4 locked OOD 83.8/Brier 0.224/ECE 0.017; breadth 0.690; 0.8B sibling exists (model card) | Apache-2.0 | base backbone + decision LoRA; no thinking template |
| 3 | **Qwen3.5-4B (instruct)** | `Qwen3.5-4B-Q4_K_M.gguf` (unsloth) | 2.74 GB | BFCL-V4 50.3 & TAU2 79.9 at 4B (card) | Apache-2.0 | thinking by default, disablable server-side (`--chat-template-kwargs` / `--reasoning off`) |

All three are Qwen3.5-4B-derived, which matches the spec's intended fine-tune
target, and all dirs ship official/community-verified conversion docs.

## Downloads (Q4_K_M GGUFs; first bytes "GGUF" verified at acquisition)

All fetched fresh with `curl -L` (no resume), placed directly on the dell via
the Samba mount (mount write ~10.5 MB/s == HF direct ~12.9 MB/s; single copy
wins):

| File (exact) | Dell path (Windows) | Bytes | SHA-256 (head) | Verified |
|---|---|---:|---|---|
| `Intern-Decision-4B-Q4_K_M.gguf` | `<dell lab>\models\Intern-Decision-4B-Q4_K_M.gguf` | 2,708,804,416 | `3c3be9bd…` (full: 3c3be9bd66008743fc24a763f28e2de51fb2417d2e4406ae59a3015ec7eaeaab) | GGUF magic; **SHA-256 matches yeaay card's published hash** |
| `Kev-4B-Q4_K_M.gguf` | `…\toolInterfacerLab\models\Kev-4B-Q4_K_M.gguf` | 3,033,489,824 | `33ae6b18…` | GGUF magic; size == HF tree (3.03 GB); no upstream SHA published |
| `Qwen3.5-4B-Q4_K_M.gguf` | `…\toolInterfacerLab\models\Qwen3.5-4B-Q4_K_M.gguf` | 2,740,937,888 | `00fe7986…` (full: 00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4) | GGUF magic; size == HF tree (2.74 GB); no upstream SHA published |

Out: no mmproj downloaded (no vision lanes; adapter runs text-only).

## Phase B results (dell321, 2026-10-17)

Lock state when acquired: dell 8080 free for the entire 45-min window + 10-min
idle grace; v2 lanes (dell-sweep-v2 outputs) had not appeared -> proceeded under
the documented idle-grace clause (0 v2-* files at start, 4 landed mid-run,
dell-sweep-v2 serving again after my teardown finished).

Serve recipe: e3\python.exe serve_pyhost.py (LoadLibrary llama-server-impl.dll
from jevResearch/lab/d1bench/e3, built b11512 / a11f57ba9), foreground python,
`-c 8192 -t 8 -ngl 99`; ssh -f -N -L 18080 tunnel; per lane: serve -> health ->
RSS -> probe x4 (v1+run1/run2 x v2+run1/run2) -> teardown (server killed, port
free verified, tunnel killed).

### Per-lane benchmark table (logit_read backend, frozen prompt)

| Model | Dataset | Acc | p50 (ms) | p95 (ms) | mean conf | conf≥.9 -> acc (cov) | conf≥.7 -> acc (cov) | malformed | serving_stable |
|---|---|---:|---:|---:|---:|---|---|---:|---|
| Intern-Decision-4B | v1 (50) | 50.0% | 2454.5 / 2159.5 | 2991.4 / 2335.3 | 0.696 | 60% (10%) | 57% (42%) | 0 | true |
| Intern-Decision-4B | v2 (150) | 36.0% | 2379.7 / 2433.4 | 2469.4 / 2663.2 | 0.696 | 45% (13%) | 44% (41%) | 0 | true |
| Kev-4B | v1 (50) | 68.0% | 2441.1 / 2136.8 | 3051.8 / 2582.6 | 0.645 | 80% (10%) | 86% (28%) | 0 | true |
| Kev-4B | v2 (150) | 41.3% | 2432.8 / 2425.1 | 2941.1 / 2983.7 | 0.691 | 33% (12%) | 55% (35%) | 0 | true |
| Qwen3.5-4B (instruct, think=off) | v1 (50) | **80.0%** | **201.6 / 90.9** | 212.0 / 477.0 | 0.601 | **100%** (4%) | **100%** (44%) | 0 | true |
| Qwen3.5-4B (instruct, think=off) | v2 (150) | **74.7%** | 405.6 / 236.2 | 472.3 / 273.7 | 0.479 | **100%** (1%) | **100%** (19%) | 0 | true |

Backdrops (committed): d1-3B Q4_K_M (systemone path) 84% @ p50 134-155ms; d1
Q8 86%; jev-style-2b logit_read 46%; qwen3.5-0.8b base logit_read 16%.

### What this says

1. **Qwen3.5-4B (instruct) is the pick for the toolInterfacer router.** 80/75%
   through the frozen letter protocol — within 4-10pp of the d1-3B fine-tuned
   bar — at logit_read-fast latency (91-406ms p50 vs ~1.05s+ for the other two
   4B qwen35-family lanes; GPU offload engaged for this lane). Calibration is
   usable: every row above conf .7 was correct in both datasets (ideal 0.7
   route gate); low global conf mean (.48-.60) means wide ask-back coverage,
   but tolerable at ~200ms/hop.
2. **Intern-Decision-4B failed as a zero-shot logit_read backend** (50/36%).
   Its decision tuning is tied to its own Jev-schema prompt/placeholder
   format; through the frozen "Answer:" letter prompt its outputs drift and
   conf-bunch at 0.696 regardless of input (moderate calibration mismatch).
   It would need a dedicated adapter speaking its format (future work, not a
   vanilla llama.cpp drop-in).
3. **Kev-4B is decent zero-shot on v1 (68%) but collapses on v2 (41%)**;
   its pointer-head + systemone semantics are not exercised by plain chat
   completions, and its base-backbone LoRA reads weakly on pi-tool routing.
   Not a drop-in either.
4. Research-vs-run deltas confirmed the "uncertain until lanes run" marks:
   the two decision-tuned models do NOT port through a plain letter-prompt
   adapter; the generic instruct model ports and delivers the cards strength
   (BFCL-V4/TAU2 trend).

### Platform notes for future agents

- In-process (LoadLibrary) serving of qwen35-family GGUFs stays CPU-side for
  the other two lanes' runs and ran without vulkan offload lines; the Qwen3.5
  instruct lane came up fast (partial GPU), the mechanism controlling which
  qwen35 lane engages vulkan in-process is **unresolved** (llama-bench on the
  same build offloads the same GGUF at 377 t/s). Latency comparisons across
  qwen35 lanes should not yet be treated as a stable serving-recipe section.
- The P1/P2 "Answer:" protocol needs no adapter change on any lane; adapter
  normalization tok.strip()==letter already covers bare and spaced letters.
- Do not kill dell port-8080 processes unless they are yours; after the sweep
  dell-sweep-v2 was serving again (v2 files appearing).
