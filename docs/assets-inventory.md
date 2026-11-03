# Assets inventory — what the sibling projects already have

Surveyed 2026-11-03 for toolInterfacer's three open needs: **(a)** route decisions
better with a small model, **(b)** build a fine-tuning dataset, **(c)** serve a
decision model cheaply.

Method: every claim below comes from a file I opened in this session. Numbers are
quoted with the file they came from. Where a directory was too large to read
usefully I say so rather than guessing at its contents. Scopes were addressed with
non-recursive `ls` plus `grep` scoped to a single named directory; nothing
recursive was run over `/home/andrew` or the sshfs mount.

**Headline:** the operator has already run most of this. Two complete measured
pipelines exist — a needle (Cactus) fine-tune with a LoRA adapter, a `.cact`
archive and an eval harness in `~/agentFleet`, and a six-backend tool-router
benchmark with a shared 382-row test set in `~/mirror/.../localAlexa/jevResearchCompiled`.
The needle result is **negative and documented as such**. The router benchmark is
**positive and includes the classical baseline toolInterfacer has never run**.

---

## Table

Asset vocabulary: `dataset` / `finetune` / `adapter` / `serving recipe` /
`eval harness` / `prompt design` / `measured result` / `none`.

| path | what it is | reusable asset? | concrete file paths | import recommendation |
|---|---|---|---|---|
| `~/toolInterfacerExtras/devskills/` | The three CLI skills (web-search, web-fetch, reddit-research) plus `lib/` and `harness/`. 7.5 KB README. | serving recipe, eval harness | `README.md`, `web-search/`, `web-fetch/`, `reddit-research/`, `harness/`, `lib/` | Already absorbed — toolInterfacer has `tests/data/devskills/{web-fetch,web-search}/fixtures/`. Do not re-import; check fixture parity only. |
| `~/toolInterfacerExtras/research/PROGRAM.md` | The autonomous optimisation program: rounds 1–3, standing objective, verified infra facts, round-by-round findings digests. 17 KB. | prompt design, measured result | `PROGRAM.md` | **Cite, don't copy.** Its round-1 digest records the cascade wall and the agreement gate — the design rationale for `service/ti/decide/cascade.py` and `pair.py`. |
| `~/toolInterfacerExtras/research/round1/F-cactus-needle.md` | 35 KB dedicated Cactus/Needle research: Needle 1/2/3 version table, `.cact` container, the 5 serving shapes, I/O contract, confidence semantics, why there is no GGUF path. 24 cited sources. | prompt design, measured result, serving recipe | `F-cactus-needle.md` | **Top-5 import (docs).** It is the only document that answers "can Needle implement `choice`/`extract`" against toolInterfacer's own primitives. |
| `~/toolInterfacerExtras/research/round2/D-needle-decision.md` | 35 KB decision doc: **SPIKE-FIRST**. Router: do not adopt. `extract` sidecar: spike, adopt on pass — because `extract` has no backend at all. | measured result, prompt design | `D-needle-decision.md`, `analysis/fetch/` (raw fetched bytes) | **Top-5 import (docs).** Carries the vendor-vs-independent numbers (BFCL v4 50.2% vs Qwen3.5-0.8B 56.8%; SNIPS-7way 24.7%). |
| `~/toolInterfacerExtras/research/round2/A2-frontier-corrected.md` + `A-frontier.json` | The corrected coverage/precision frontier across lanes; 1.7 MB of raw frontier JSON. | measured result | `A2-frontier-corrected.md`, `A2-frontier-corrected.json`, `A-frontier.json` | Import the markdown; leave the 1.7 MB JSON where it is and cite the path. |
| `~/toolInterfacerExtras/research/round2/A-gate-cascade-design.md` | Gate-cascade design, 50 KB. | prompt design | `A-gate-cascade-design.md` | Read before rechanging gate arithmetic in `service/ti/policy/`. |
| `~/toolInterfacerExtras/research/round2/tscheck/`, `round3/{P1,P2,P3}-*.md`, `round3/verify/` | Round-3 swarm/needle plans, design docs, and ten small verification scripts (`v1_reach.py`, `v3_routing.py`, `gil_probe.py`, …). | measured result, eval harness (small) | `round3/P1-swarm-needle.md`, `round3/verify/*.py` | Skim `P1-swarm-needle.md` for needle-at-scale evidence; the verify scripts are probes, not a harness. |
| `~/toolInterfacerExtras/research/gpu-paper/` | 21 documents (1.4 MB total) on tiny-model theory: attention-only + thin-FFN architectures, tiny-scale calibration, quantisation-vs-calibration, edge hardware matrix, MCU weight streaming, plus five experiment arms. | measured result, prompt design | `04-attention-only-thin-ffn.md`, `09-tiny-scale-calibration.md`, `12-quantization-calibration.md`, `20-arm1-broken-confidence.md`, `EXPERIMENT-MENU.md` | **Selective citation only.** `20-arm1-broken-confidence.md` and `09-tiny-scale-calibration.md` are the ones relevant to a confidence gate. Do not import wholesale. |
| `~/decisionModelTuning/01…05-*.md` + `README.md` | The operator's own 5-document synthesis: task/data, training methods, evaluation/calibration, small-models/serving, diffusion. Every claim carries an arXiv ID; README says headline citations were spot-verified 2026-10-28. | prompt design, eval harness, measured result (cited literature) | `01-task-and-data.md`, `02-training-methods.md`, `03-evaluation-and-calibration.md`, `04-small-models-and-serving.md`, `05-diffusion-models-structured-output.md`, `README.md` | **Top-5 import (docs).** This is the design document for the dataset and the eval; see the synthesis section below. |
| `~/agentFleet/finetune/` | A working needle decision-classifier pipeline: 4-op tool grammar, datum converter, LoRA trainer driver, `.cact` build, eval with refusal accounting + per-class + ECE deciles. 25 KB README with a full verification log. | **finetune, dataset, eval harness, serving recipe, prompt design** | `tools.py`, `dataset.ts`, `eval.py`, `slate.py`, `slate.json`, `Makefile`, `measure_window.py`, `timesteps.py`, `README.md`, `fixtures/` | **Top-5 import.** The eval harness alone is worth it: it reports refusals first-class, which is the one thing `bench/band_summary.py` cannot do. |
| `~/agentFleet/mine/` | The corpus miner: `decision-points.ts` (20 KB) defines a "decision point" as a completed agent turn a human then answered; `out/decision-points.jsonl` now holds **527** rows (the docs say 502 — the miner kept writing after the README was taken). | **dataset** | `decision-points.ts`, `review-labels.ts`, `out/decision-points.jsonl` | **Top-5 import.** Labelled rows in the deployment's own distribution, already field-mapped to a training format. |
| `~/agentFleet/src/labeling/label-to-op.ts` | The single definition of label → op (`continue/nudge/redirect/stop/new_task` → `nudge/nudge/nudge/leave/flag`), shared by the miner, the converter and the benchmark. | prompt design | `label-to-op.ts` | Import as a **pattern** (one table, three consumers), not as code — its five labels are fleet-specific. |
| `~/agentFleet/checkpoints/needle3.safetensors` | The base Needle 3 checkpoint. **242,047,978 bytes**, 121,019,861 params at bf16. | adapter (base weights) | `checkpoints/needle3.safetensors` | Nothing to import — it is a HF-cache-derived base. Record the path and the param count; do not copy 242 MB. |
| `~/.cache/cactus-needle/v3/3.3.3/{needle3.cact,libneedle.so}` | The published deployable archive (**35,335,380 B**) and the engine (**1,590,584 B**). Auto-downloaded on first `Needle()` use. | adapter, serving recipe | `~/.cache/cactus-needle/v3/3.3.3/` | Reference the cache path. The `.cact` is what a needle backend would load — 35 MB, not 242 MB. |
| `~/agentFleet/finetune/out/{d4,d8,d20}/` | Trained LoRA exports: `.cact` (15.4 / 27.4 / 63.4 MB) + `eval.json` each. | **adapter, measured result** | `out/d4/{adapter.safetensors,d4.cact,eval.json}`, `out/d8/*`, `out/d20/*` | **Top-5 import (the measured result).** These are the numbers that say a local needle fine-tune on this corpus does not work. |
| `~/agentFleet/finetune/out/{control-d4,control-d8}/` | Untuned base sliced to 4/8 layers — the zero-shot baselines. | measured result | `out/control-d4/{control-d4.cact,eval.json}`, `out/control-d8/control-d8.cact` | Import with the tuned rows. Without the control, the tuned number means nothing. |
| `~/agentFleet/bench/` | The needle benchmark: runner, slate driver, result schema, committed records, and 32 KB of prose (`SLATE.md`, `NOTES-needle.md`, `NOTES-baselines.md`). | **eval harness, measured result** | `runners/run.ts`, `runners/slate.ts`, `schema/result.schema.json`, `results/*.json`, `SLATE.md`, `NOTES-needle.md`, `tasks/` | **Top-5 import.** `result.schema.json` is a reusable record shape; the NOTES files are the honest write-ups. |
| `~/agentFleet/src/decision/` | The needle serving path: Python sidecar (single long-lived engine handle), TS adapter, escalation ladder, prompt renderer, tool catalogue. | **serving recipe, prompt design** | `needle_sidecar.py`, `needle.ts`, `ladder.ts`, `prompt.ts`, `catalogue.py`, `types.ts` | **Top-5 import.** `needle_sidecar.py` is a 4.6 KB worked example of wrapping a non-HTTP engine behind a process boundary — exactly toolInterfacer's `decide/backends/` shape. |
| `~/agentFleet/docs/specs/tiny-model-landscape.md` | 807-line sub-200M base-model survey with **GGUF tensor-table** param counts (not card claims), embedding fractions, IFEval numbers, and the KD measurement. | measured result, prompt design | `tiny-model-landscape.md` | Import as a doc. §6.1 is the only controlled public KD-at-135M number the operator has. |
| `~/agentFleet/docs/specs/tiny-finetune-toolchain.md` | Toolchain spec: tool matrix, FLOPs arithmetic, measured `llama-finetune` run, full-FT-vs-LoRA argument, serving handoff, and a complete `train_sft.py` + `render_corpus.py` in the appendices. | **finetune, serving recipe** | `tiny-finetune-toolchain.md` (Appendix A, B) | Import. Appendix A is a runnable full-parameter SFT script for a 135M model on CPU or one T4. |
| `~/agentFleet/docs/METRICS.md` | 22 KB metric definitions. | eval harness (definitions) | `METRICS.md` | Read if the eval introduces a new metric; else skip. |
| `~/agentFleet/docs/specs/{role-grammars,train-serve-integrity,corpus-factory,prompt-trim,hi-orchestrator,plan-*,free-compute,runtime-scaffold,LIMITATIONS}.md` | The rest of the fleet's spec set — grammars, corpus factory, prompt trimming, free-compute survey, integrity rule. | prompt design, serving recipe | `docs/specs/` | `prompt-trim.md` and `corpus-factory.md` are the two worth reading. The rest is fleet-specific. |
| `~/swarmHarness/plans/hybrid-needle-swarm.md` | 46 KB plan, "every number measured on this box". Contains the **needle param breakdown** and the verdict that d8 is unusable as a router. | measured result, prompt design | `plans/hybrid-needle-swarm.md` | **Top-5 import (the §1.2 breakdown).** It is the only place the 121M is decomposed by component. |
| `~/swarmHarness/docs/design/{architecture,benchmark-design,experiment-contract-v3}.md` | Coordination-layer design + a benchmark design doc (60 KB) and an experiment contract. | eval harness (method), prompt design | `docs/design/benchmark-design.md`, `docs/design/experiment-contract-v3.md` | Import `experiment-contract-v3.md` only if toolInterfacer starts committing run records; otherwise cite. |
| `~/swarmHarness/{src,benchmarks}/` | Swarm coordination implementation and benchmark tasks/runners. | eval harness (unrelated domain) | `src/`, `benchmarks/` | **Skip** — multi-agent coordination, not routing decisions. |
| `~/mirror/.../projects/localAlexa/jevResearchCompiled/experiments/` | **The six-backend tool-router benchmark.** One 382-row test set, one 17-tool registry, six runners (needle, d1-3B, TF-IDF/LinearSVC, MiniLM, Qwen3.5-0.8B, cascade), one results dir. | **dataset, eval harness, measured result, prompt design** | `data/test.json`, `registry.json`, `exp_needle.py`, `exp_d1.py`, `exp_classic.py`, `exp_router.py`, `exp_qwen.py`, `exp_cascade.py`, `router_lib.py`, `gen_data.py`, `results/*.json` | **Top-5 import, ranked #1.** A node-fixed head-to-head including two classical baselines, on one box, with the row-level data on disk. |
| `~/mirror/.../localAlexa/jevResearchCompiled/10-d1-3b-benchmark.md` | The incumbent's own benchmark write-up: d1-3B 91.4% @ 243 ms on the 382-row set, plus calibration bands and the Arc-140V serving gotchas. | measured result, serving recipe | `10-d1-3b-benchmark.md` | **Top-5 import (docs).** It is the reference point every new backend must beat, and it explains why `/v1/systemone` needs build ≥ b11500. |
| `~/mirror/.../localAlexa/jevResearchCompiled/05-localjev-experiments.md` | Hardware/serving traps: Vulkan fine single-shot, unstable long-lived; thinking models return empty content and surface as 502; grammar crashes on large probability-array schemas. | serving recipe, measured result | `05-localjev-experiments.md` | **Import.** These are four bugs each worth a day; they are already cited by `agentFleet/AGENTS.md`. |
| `~/mirror/.../localAlexa/jevResearchCompiled/{02-benchmarks,08-decision-and-vision-models,09-vlm-benchmarks,12-real-screen-validation,13-validation-widening}.md` | Vision/screen-grounding benchmark docs and validation widening. | measured result (vision) | same dir | Skip for routing; relevant only if toolInterfacer grows a vision op. |
| `~/mirror/.../localAlexa/jevResearch/lab/` | The lab: `needle_bench.py`, `needle_extract_bench.py`, `needle_smoke.py`, `jevbench/bench_localjev.py` + per-model bench logs, `d1bench/tibench.py`, `localjev/` (TS bridge with `eval/` and `test/`), `typesafe-computer-use/`. | **eval harness, serving recipe** | `lab/needle_bench.py`, `lab/needle_extract_bench.py`, `lab/needle_smoke.py`, `lab/jevbench/bench_localjev.py`, `lab/jevbench/{bench_qwen3_0_6b.txt,bench_gemma3-1b_latest.txt,bench_lfm2_5-1_2b_latest.txt}`, `lab/d1bench/tibench.py`, `lab/localjev/` | Import `needle_extract_bench.py` — `extract` is the primitive with no backend, and this is the extract-shaped needle bench. |
| `~/mirror/.../localAlexa/minicpmTune/` | Docs only (no code, no weights): model choice, serving, inference-engine decision records, personas prompt set. | prompt design (thin) | `docs/decisions/model-choice.md`, `docs/decisions/inference-engine.md`, `docs/architecture/model-serving.md`, `docs/prompts/personas.md` | **None worth importing.** It is a plan that was not executed; cite only if MiniCPM is reconsidered. |
| `~/mirror/.../localAlexa/{overallStructure,autoSuggestProjects}/` | Structure notes; an electronics-project subtree. | none | — | Skip. |
| `~/mirror/.../Research/jevResearch/` | Docs 01–07 (a subset of `jevResearchCompiled`, which has 01–13) plus `bench/` (`run-bench.ps1`, `tasks/`, `results/`, sandbox dirs). | eval harness (thin) | `bench/run-bench.ps1`, `bench/tasks/`, `bench/results/` | **Superseded by `jevResearchCompiled/`** — same files 01–07, fewer docs, and `bench/` is PowerShell-only. Import nothing. |
| `~/mirror/.../Research/localLLMfunReseach/` | A domain-research project (grey-market compound price/consensus gathering) with 50+ scripts in `data/` and 8 in `scripts/`. Consensus/verification tooling, `errors.md`, `tool_issues.md`. | **none** (for decisions) | `docs/project_reference.md`, `scripts/*.py` | **Import nothing.** Nothing here trains, serves or evaluates a model. The consensus-merge pattern is the only transferable idea and it is not worth the read. |
| `~/mirror/.../Research/moneyLM/` | One file: a 10.5 KB money-making research sweep. | **none** | `RESEARCH_money_ideas_2026-10-09.md` | **Import nothing** — out of scope for this question. Noted so a later survey does not open it again. |

---

## The decisionModelTuning synthesis

The operator's own conclusions, 20 points, each with its doc and section. Source:
`~/decisionModelTuning/`.

1. **LoRA SFT at rank 32–64 on an instruct 3B, 2–10K examples, 10–20% general-instruction mix is the highest-expected-value first move.** Everything else is second. (`README.md`, headline #1)
2. **Constrained decoding raises schema validity to ~100% for free**, so structured-output reliability is an inference-time problem, not a fine-tuning problem: "train the content, constrain the syntax." (`README.md` #2; `02` §6; `05` §4)
3. **Calibrate and report cost, not accuracy.** With an asymmetric cost matrix, `Acc@1` alone is misleading. (`README.md` #3; `03` §1)
4. **Do not start with diffusion** — published code results put it 10–17 points below autoregressive at comparable size. (`README.md` #4; `05` §3)
5. **Default to routing-as-classification** (closed candidate set, label out) for a tens-of-tools catalog: cheapest to train, easiest to constrain, and confidence becomes a real distribution rather than a sampled token. (`01` §1)
6. **Train closed but hold tools out of training entirely** and evaluate on them — routing transfers across corpora and retrievers when trained for it. (`01` §1)
7. **Confidence should be a bucket, not a regressed float**: emit `high/med/low` or a discrete head over 0.05 bins, then temperature-scale. Buckets let you supervise from evidence you actually have. (`01` §2, "Confidence as a target")
8. **Keep `target` separate from `candidates`** in the row format so rows survive a re-serializer change; pin `schema_ver`, sort candidates by name so position carries no signal. (`01` §2)
9. **Volume is not the lever — verified quality is.** 300–800 verified rows for a usable 1–3B router, 2,000–5,000 for phrasing robustness; hand-authored confusable rows weighted ~10× teacher rows. NVIDIA's closest published analogue: Llama 3.2 1B LoRA on ~500 samples → ~93% function-name accuracy vs ~1.4% base. (`01` §5)
10. **Hard negatives are the asset.** Build choices so non-gold candidates *are* the negatives (2–5 confusable + 3–10 random); pairs above ~0.85 cosine on tool descriptions are confusable and deserve ten hand-authored requests each. (`01` §4)
11. **You need a "none of these" / `clarify` class at 5–10% of the set.** With a small catalog, abstention is a first-class behaviour and is what breaks in small models. (`01` §4; `04` §1)
12. **Preference optimisation mostly does not beat SFT on a classification-shaped task**, and the best evidence is a null result (Tülu 3: only the instruction-following persona set improved its own target). The mechanism is saturation: when the base already puts mass on the right tool, the DPO gradient vanishes. (`02` §2)
13. **RLVR/GRPO is real but conditional.** ToolRL reports +17% over base and +15% over SFT from 4K examples. Two cautions: GRPO with *random* rewards still lifted Qwen2.5-Math by 21.4 points, so a gain on a Qwen base is not proof the reward works; and RL sharpens, collapsing toward high-prior tools. (`02` §3)
14. **Distillation's dense signal is the teacher's distribution over tool names**, not one-hot labels — 40 floats per example, a few lines on top of the SFT loop. Hard-label imitation transfers style, not capability. (`02` §4)
15. **Curriculum reordering is a null result** except in the multi-hop staging form (train 1-hop, then 2-hop, then 3-hop), which is credit assignment rather than curriculum. (`02` §5)
16. **Constrained decoding helps classification and hurts reasoning** — JSON-mode "improves classification accuracy by constraining the answer space and eliminating answer-selection errors" — but greedy constrained decoding distorts the distribution, so parse the mask before reading a softmax as confidence. (`02` §6)
17. **Measure selection, binding and knowing-when-not-to-act separately**; log candidate list, chosen tool, arguments, confidence and decode config per hop or none of it is computable. (`03` opening)
18. **A softmax over tool names is not a probability.** Renormalising over tool-name logits redistributes whatever mass sat on prose, a clarifying question, or an invented name — destroying exactly the signal abstention needs. Treat it as a ranking signal until verified. (`03` §2)
19. **Size the eval for 2-point detectability, and pair it.** Unpaired 2-point resolution at p≈0.85 needs ~5,000 per arm; McNemar at 10% discordance needs ~1,960 — a 5× saving. At n=300 the 95% interval is ±4.0 points, so a 2-point win on 300 items is noise. Hold out whole phrasing families, not rows. (`03` §3)
20. **External benchmarks do not transfer** (catalog, grading, composition, fallible graders). Use BFCL/τ-bench as a smoke test that a checkpoint is not broken, nothing more. (`03` §4)
21. **Bucket errors by the first decisive error only, then size the next batch by count × cost.** "Over-abstention" and "under-abstention" are separate buckets; under-abstention is the dangerous one. (`03` §5)
22. **The minimum honest report is four numbers**: `Acc@1`, expected cost per 1,000 hops, destructive-misroute count, and the risk/coverage point. A single mutating misroute cost up to 92%/96% of task success (SABER). (`03` §6)
23. **The floor for a small router is ~125–230M, and "restraint" is what breaks below it** — `llama3.2:3b` and `granite4:3b` score 0.000 on judgment because they always emit a call. Parameter count is a weak predictor. (`04` §1)
24. **Cascade: hash → rules → MiniLM encoder + linear head → 1B LLM for the tail**, gated on *margin* as well as threshold (0.55/0.45 should escalate). RouteLLM's reference result is >2× cost reduction at ~90% of strong-model quality. (`04` §2)
25. **Quantisation changes the failure mode, not just the scale** — xLAM-2-3B 58.00% → 51.75% at Q4_K_M, Qwen3-1.7B 14.50% → 8.75% with wrong-argument rate 20.8% → 26.9%, producing "error-free-but-wrong trajectories that no runtime channel captures." Recommendation: **Q8_0 under 2B**, and re-run the eval at the quant you ship. (`04` §3)
26. **Latency is a policy problem, not a model problem**: batch 32 costs ~1.6× one request's latency for ~30× throughput, and prefix caching is the whole game — but only if the prefix is byte-identical. Local serving is 100–300× cheaper than hosted; the binding constraint is p99. (`04` §5)
27. **Semantic caching is disqualifying below ~0.97 similarity** for routing, because a false hit does not produce a slightly worse answer, it produces the wrong tool. Cache exact matches keyed on `(model, quant, adapter, system_prompt_hash, tool_schema_hash, query)`. (`04` §6)
28. **`temperature=0` is not reproducibility**, so log `(input_hash, output, logprobs, model, quant, seed)` and treat a >2% retry rate as a quality alarm. `04` §7's pick: `Qwen3-1.7B` at Q8_0, thinking off, batched, prefix-cached, FSM-constrained to a schema that includes `refuse`. (`04` §6–7)

Two caveats carried from the same directory: the docs contain **no measurements from
this machine** (`README.md`, "What these docs do and do not guarantee"), and
`04 §1` explicitly records that there are **no independent BFCL numbers for
`Needle`-class <50M tool-callers** — a gap `jevResearchCompiled/experiments/`
partly fills.

---

## Cactus / Needle

### What exists locally

| artefact | size | path |
|---|---|---|
| Base JAX checkpoint, Needle 3 | 242,047,978 B (121,019,861 params, bf16) | `~/agentFleet/checkpoints/needle3.safetensors` |
| Deployable archive | 35,335,380 B | `~/.cache/cactus-needle/v3/3.3.3/needle3.cact` |
| Engine shared object | 1,590,584 B | `~/.cache/cactus-needle/v3/3.3.3/libneedle.so` |
| Python package | `cactus_needle-3.2.8.dist-info` | `~/agentFleet/.venv/lib/python3.14/site-packages/` |
| Trained LoRA (rank 16, alpha 32) | 7,906,640 B | `~/agentFleet/finetune/out/slate/d4/adapter.safetensors` |
| Sliced exports | 15.4 / 27.4 / 63.4 MB | `~/agentFleet/finetune/out/{d4,d8,d20}/*.cact` |
| Research write-ups | 35 KB + 35 KB + 45 KB | `toolInterfacerExtras/research/round1/F-cactus-needle.md`, `round2/D-needle-decision.md`, `round3/P1-swarm-needle.md` |
| Bench drivers | 3.2 / 2.4 / 0.6 KB | `localAlexa/jevResearch/lab/{needle_bench,needle_extract_bench,needle_smoke}.py` |

So: **weights yes, adapter yes, engine yes, serving wrapper yes, benchmark yes.** This
is not a paper-only interest — the operator has run the whole pipeline end to end.

### What is measured

**Architecture** (`swarmHarness/plans/hybrid-needle-swarm.md` §1.1–1.2, param
breakdown from the safetensors header): 121,019,861 params, d_model 768, 20 layers,
12 heads / 2 KV heads, vocab 8192, sliding window 1024, global attention at layers
4/9/14/19. Decomposed: **36,488,980 (30.2%) transformer blocks, 70,778,880 (58.5%)
engram slot tables, 6,291,456 (5.2%) token embedding, 5,898,240 (4.9%) engram
key/value projections.** More than half the model is a lookup table — that is the
most useful single fact about what a "121M attention-only model" actually spends its
budget on.

**Routing accuracy, node-fixed** (`localAlexa/jevResearchCompiled/experiments/results/needle_results.json`,
382 rows, 18-way):

| backend | accuracy | none-detection | p50 latency | source |
|---|---:|---:|---:|---|
| TF-IDF + LinearSVC | **93.19%** | 0.60 | **0.61 ms** | `results/needle_results.json` → `classic_baseline_best` |
| d1-3B Q4_K_M (`/v1/systemone`) | **91.36%** | **1.00** | 242.7 ms | `results/d1_results.json` |
| MiniLM + logistic (embedding router) | 86.65% | 0.44 | 57.84 ms/row | `results/router_results.json` |
| Qwen3.5-0.8B zero-shot | 75.64% | 0.20 | 7,290 ms | `results/qwen_results.json` (n=78 subset) |
| **needle3 (pkg 3.0.4 / engine 3.0.1)** | **59.42%** | 0.32 | — | `results/needle_results.json` → `accuracy.overall` |

needle3 excluding the `__none__` rows: 61.34% over 357 rows. The
`10-d1-3b-benchmark.md` write-up quotes 59.4% and 91.4% for the same run, so the
documents and the raw JSON agree.

**After a local fine-tune, 4-op grammar, 50 held-out rows** (`agentFleet/finetune/out/*/eval.json`):

| variant | layers | accuracy | acc when answered | refused | p50 ms | confidence head |
|---|---:|---:|---:|---:|---:|---|
| `control-d4` (untuned base) | 4 | **26.0%** | 31.7% | 9/50 | 226 | present, ECE 0.1666 |
| `d4` (LoRA) | 4 | **28.0%** | 31.1% | 5/50 | 206 | **absent** |
| `d8` (LoRA) | 8 | **6.0%** | 7.3% | 9/50 | 320 | **absent** |
| `d20` (LoRA) | 20 | **10.0%** | 19.2% | 24/50 | 1100 | **absent** |

`d8` predicted the zero-row class `declare_done` 30 times out of 50. The majority
baseline is 0.81 (`swarmHarness/plans/hybrid-needle-swarm.md` §0). **The fine-tune
made the model worse, and the run is documented as a negative result** in
`bench/results/SLATE.md` and in the plan's verdict table.

**Throughput and cost, measured on this box:**

- Training: **262.4 s/step**, 29 steps/epoch, **126.8 min/epoch** (452 examples, batch 16, seq 1024, CPU JAX). `finetune/out/slate/slate-report.json` → `_probe.d4`. Verdict in that record: `OVER BUDGET` against a 600 s/epoch box.
- After prompt compaction: 57.9 s/step @ `--max-len 512`, 63.2 @ 640, 94.9 @ 768; 25.1 / 27.4 / 41.1 min per epoch. `bench/results/NOTES-needle.md`.
- The compaction itself: SYSTEM 709 → 515 chars (216 → 170 tokens), tool schemas 780 → 666 (174 → 150), static prefix **403 → 333 tokens**. Same file.
- Serving: warm ~1.4 s/decision for the full 20-layer archive; first call in a process 20–43 s (engine load + prefill). `finetune/README.md` §6.
- Zero-shot refusal, the reason the scaffold exists: `REFUSALS 1/1 (100.0%)`, `confidence 0.032`, one call withheld. `finetune/README.md` preamble.
- Sidecar, 22 rows: exact match 0.1364 both before and after prompt work, 22/22 refusals, coverage **0.0 at every threshold 0.3–0.9**. p50 3776 → 2023 ms. `bench/results/NOTES-needle.md`.
- Control at 8 layers over 40 rows: exact 15.0%, `flag` recall 85.7%, `nudge` 0%, `leave` 0%, 37/40 refusals, p50 994 / p95 1210 ms. `bench/results/2026-10-26T23-24-18-841Z__slate-control-d8.json`.

**Vendor vs independent numbers** (`round2/D-needle-decision.md` §1, citing round-1
§3.5 and the compiled 6-benchmark table): Needle 3 BFCL v4 **50.2%** against
Qwen3.5-0.8B's **56.8%**; SNIPS 7-way **24.7%**; and the vendor's own base-model
note says it fails 5 of 6 Cactus suites at the default gate.

### What is missing

- **No probability vector, ever.** Needle returns one call-acceptance scalar, `min(calibrated head, decode probability)`, floor 0.1, with no distribution over the declared tools (`F-cactus-needle.md` §2.4). toolInterfacer's `choice` primitive wants a distribution — so needle cannot be a `choice` backend without a lie in the interface.
- **No GGUF / llama.cpp path.** `.cact` is a proprietary container with Cactus Quants at 2.125 bit/weight and a nameless positional tensor directory (`F-cactus-needle.md` §3.1). Everything in `docs/model-sweep-consolidated.md` was measured through `llama-server`; needle cannot join that table.
- **The confidence head is lost on local fine-tunes.** `needle build --lora` drops it, so `confidence` is `None`, and `needle_sidecar.py` coerces `None → 0.0`, which makes `ladder.ts` escalate **100% of decisions** (`finetune/README.md` §5, open question 7). Any local needle adapter is therefore unroutable until that is fixed. Only `needle platform finetune` keeps the head.
- **Two class-balance defects in the corpus**: `declare_done` has 0 rows out of 502 and was still left in the grammar (the direct cause of d8's 30/50 over-prediction), and `leave` has 7 rows so its recall is a 7-example statistic (`finetune/README.md` §3, `docs/specs/LIMITATIONS.md` lineage).
- **No throughput number for the sliced rungs.** `d4` at 206 ms and `d20` at 1100 ms are eval-loop medians over 50 rows, not a latency benchmark; nobody has run `needle-bench`-style pp/tg on `.cact`.
- **No `extract` benchmark wired into toolInterfacer.** `jevResearch/lab/needle_extract_bench.py` exists and has never been pointed at `service/ti/decide/primitives.py::extract`, even though `round2/D-needle-decision.md` names `extract` as needle's correct seat and notes `resolve("extract")` raises `KeyError` today.

**Net:** the local evidence on Needle is a *measured no* for routing (59.4% node-fixed, a fine-tune that dropped 26% → 6%) and an *unmeasured maybe* for `extract`.

---

## Top 5 imports

Ranked by hours saved. Each names the exact source path and where it lands.

### 1. The six-backend router benchmark — data, registry, runners, results

**Copy from:** `~/mirror/PersonalCodingProjects/personal/projects/localAlexa/jevResearchCompiled/experiments/`
— specifically `data/test.json` (39 KB, **382 rows**, keys `text` / `tool` / `family`),
`registry.json` (21 KB, 17 tools + `__none__`), `exp_{needle,d1,classic,router,qwen,cascade}.py`,
`router_lib.py`, and `results/{needle,d1,classic,qwen,router,cascade}_results.json`.

**Lands at:** `tests/data/routing_382.json`, `bench/registry.json`, `bench/routers/`,
`bench/results/routing_382/`.

**Why it saves time:** it is the only head-to-head where six backends saw the same 382
utterances on the same box, and it includes the two baselines toolInterfacer has never
run: **TF-IDF + LinearSVC at 93.19% / 0.61 ms** and **MiniLM + logistic at 86.65% /
57.8 ms/row**. Until those are on the board, "74.7% for a 3B" reads as good; against a
0.61 ms classical model at 93.19% it reads as the wrong instrument. Rebuilding this
means re-generating 382 labelled utterances across 18 classes plus the tool registry —
`gen_data.py` exists, but the labelled set is the expensive half.

### 2. The decisionModelTuning five documents — as the dataset and eval spec

**Copy from:** `~/decisionModelTuning/{01-task-and-data,02-training-methods,03-evaluation-and-calibration,04-small-models-and-serving,05-diffusion-models-structured-output}.md`
plus `README.md`.

**Lands at:** `docs/decision-model-tuning/` (verbatim), with a one-page `docs/dataset-spec.md`
distilled from `01` §2 for the row format and `03` §1–3 for the eval.

**Why it saves time:** it is the written spec for needs (b) and (a) with citations
already check-verified — row schema, hard-negative rates, the abstention class, the
sample-size arithmetic (~1,960 paired vs ~5,000 unpaired for 2 points), the ECE and
threshold formula, and the four-number minimum report. Writing this from scratch is a
week of literature work, and the *reason* this repo's sweeps report accuracies at n=150
without confidence intervals is that this arithmetic was not on hand.

### 3. agentFleet's needle eval harness and sidecar

**Copy from:** `~/agentFleet/finetune/eval.py` (14.7 KB), `~/agentFleet/finetune/{tools.py,dataset.ts,slate.py,slate.json,Makefile,measure_window.py}`,
`~/agentFleet/src/decision/needle_sidecar.py` (4.6 KB), `~/agentFleet/bench/runners/run.ts`
and `~/agentFleet/bench/schema/result.schema.json`.

**Lands at:** `bench/needle/` (harness), `service/ti/decide/backends/needle_sidecar.py`
(backend), `bench/schema/result.schema.json` (record shape).

**Why it saves time:** `eval.py` already reports the four things a routing eval needs and
toolInterfacer's `band_summary.py` does not: **refusals as a first-class outcome**
(`REFUSALS n/N` plus what the engine withheld), accuracy split into overall vs
answered, a confusion matrix **with a `refused` column** so a never-called class is
visible as zeros, and calibration by confidence decile with ECE. `needle_sidecar.py` is
also the worked pattern for wrapping a non-HTTP, process-global engine behind
toolInterfacer's `decide` wall — including the single-long-lived-handle and
`reset()`-per-call discipline.

### 4. The 502-row mined corpus and its converter

**Copy from:** `~/agentFleet/finetune/out/real/{train.jsonl,val.jsonl}` — the newest
regeneration, **474 train / 53 val rows** (817 KB + 91 KB), 527 decision points total;
the older `~/agentFleet/finetune/out/slate/d4/{train.jsonl,val.jsonl}` (452/50) is the
set the recorded eval numbers were measured on, so keep both and label which is which;
plus `~/agentFleet/mine/{decision-points.ts,out/decision-points.jsonl}` and
`~/agentFleet/src/labeling/label-to-op.ts`.

**Lands at:** `tests/data/needle/{train,val}.jsonl`; the label→op table pattern into
`service/ti/planner/` or `ops/` as the single definition of what a correct decision is.

**Why it saves time:** ~500 rows in the deployment's own distribution, already
field-mapped (`humanText` → intent line, `assistantText` → tail, arguments made
verbatim spans of the query with a check that found 93 bad rows in an earlier cut),
already split by FNV-1a hash so re-running puts the same example on the same side, and
already carrying the **negative result** — d8 0.06 against a 0.81 majority baseline —
so nobody rebuilds this pipeline expecting a win. The valuable import is as much the
*post-mortem* as the data: zero-row classes must be filtered from both rows and the
tool catalogue, and a locally built `.cact` loses the confidence head.

### 5. d1-3B's benchmark and the `< 1 day` serving traps

**Copy from:** `~/mirror/.../localAlexa/jevResearchCompiled/10-d1-3b-benchmark.md`,
`05-localjev-experiments.md`, and `~/mirror/.../localAlexa/jevResearch/lab/jevbench/bench_localjev.py`
with the per-model logs beside it (`bench_qwen3_0_6b.txt`, `bench_gemma3-1b_latest.txt`,
`bench_lfm2_5-1_2b_latest.txt`).

**Lands at:** `docs/`, appended to `docs/model-sweep-consolidated.md` as a "reference
lane" section; the probe into `bench/`.

**Why it saves time:** this is the incumbent's own number — **91.4% @ 243 ms, with
none-detection 1.00** — which is the bar every candidate must beat, and it is 17 points
above the 74.7% this repo currently treats as the incumbent. Adopting it reframes the
whole live-probe table. The companion doc supplies four traps, each a lost day:
`/v1/systemone` needs llama.cpp build ≥ b11500 (the vendored b10964 has no
`systemone` symbols); the AV quarantines any binary named `llama-server.exe` so the
same bytes must be extracted as `d1srv.exe`; Vulkan is fine single-shot and dies as a
long-lived server; and grammar-constrained decoding crashes on large
probability-array schemas, which is why `LOCALJEV_GRAMMAR_MAX_OUTCOMES` defaults to 8.

---

### Two things worth not importing

- **`swarmHarness/src` and `benchmarks/`** — multi-agent coordination, a different
  question. Its `plans/hybrid-needle-swarm.md` is worth reading for §1.2's param
  breakdown, but the code is not.
- **`Research/localLLMfunReseach` and `Research/moneyLM`** — a domain-research pipeline
  and a business sweep. Neither contains a dataset, model, adapter or eval for
  decisions. Recorded here so a later survey does not spend a pass on them.
