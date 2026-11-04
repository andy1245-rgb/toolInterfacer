# Sibling assets — the findings, not the disk they came from

| | |
|---|---|
| surveyed | 2026-11-03 |
| question | for three open needs: **(a)** route decisions better with a small model, **(b)** build a fine-tuning dataset, **(c)** serve a decision model cheaply |
| status | findings. The directory listings, byte-for-byte inventory and survey method notes were cut on 2026-11-03; every measurement, `n`, quoted figure and negative result below survived the cut |

**The trees these figures come from are private.** The survey read a set of sibling trees that
belong to the operator: an agent-orchestration project, a multi-agent coordination project, a
decision-model literature tree, the working-notes tree kept alongside this repo, the benchmark
tree that produced the 382-row routing set, and a few others. None of them is published with
this repository and none of their paths appears here. The figures below were measured against
those trees on the dates given. A reader cannot re-run them, and this document does not pretend
otherwise — it is the record of a measurement, not a reproduction script.

What a reader *can* check is the half that was imported. `tests/data/routing_382.json`,
`bench/registry_382.json` and `bench/classical_baseline.py`, together with the recorded
`bench/results/classical_382.json` and `bench/results/classical_v2.json`, are in this repo and
are the reproducible half of everything claimed here. Where a claim rests on one of them it is
cited by exact in-repo path. Where a claim rests on a private tree it is attributed by
description and dated.

**How sources are named.** *The upstream benchmark* is the tree that produced the 382-row
routing set and the six runners that scored it. *The fleet project* is the agent-orchestration
tree that carries a working needle fine-tune pipeline, a session miner and a needle benchmark.
*The swarm project* is a multi-agent coordination tree whose needle plan is the only place the
121M parameter count is decomposed. *The decision-model research tree* is a five-document
literature synthesis, cited below by document number (`01`–`05` plus its `README`); it is not in
this repo. *The notes tree* is the working notes kept alongside this project — research rounds,
GPU-paper notes, and the three CLI skills. Every claim below comes from a file that was read;
where a tree was too large to read usefully that is said rather than guessed at.

---

## Headline

Two complete measured pipelines already existed. One is a needle (Cactus) fine-tune with a LoRA
adapter, a `.cact` archive and an eval harness; the other is a six-backend tool-router benchmark
with a shared **382-row** test set.

- **The needle fine-tune is a negative result and is documented as such.** A locally built
  adapter made routing worse than the untuned base (`d8`: **6.0%** against a **0.81** majority
  baseline).
- **The router benchmark is positive, and it includes the classical baseline this project had
  never run** — TF-IDF + LinearSVC at **93.19%** and **0.61 ms** on the same 382 rows.
- That baseline has since been re-run *in this repo*: `bench/classical_baseline.py`, written up
  in `docs/classical-baseline-2026-11-03.md`, artifacts in `bench/results/classical_382.json`
  and `bench/results/classical_v2.json`. See the status section at the end.

The rest of this document is the evidence behind those three sentences, then what was imported
and what was not.

---

## The 382-row router benchmark

One 382-row test set, one 17-tool registry (plus a learned `__none__` class), six runners
(needle, d1-3B, TF-IDF/LinearSVC, MiniLM, Qwen3.5-0.8B, cascade), one results directory, all on
one box. It is the only head-to-head in which six backends saw the same utterances.

### Recorded numbers, node-fixed, 382 rows, 18-way

| backend | accuracy | none-detection | p50 latency |
|---|---:|---:|---:|
| TF-IDF + LinearSVC | **93.19%** | 0.60 | **0.61 ms** |
| d1-3B Q4_K_M (llama.cpp `systemone` endpoint) | **91.36%** | **1.00** | 242.7 ms |
| MiniLM + logistic (embedding router) | 86.65% | 0.44 | 57.84 ms/row |
| Qwen3.5-0.8B zero-shot | 75.64% | 0.20 | 7,290 ms (n=78 subset) |
| **needle3** (package 3.0.4 / engine 3.0.1) | **59.42%** | 0.32 | — |

needle3 excluding the `__none__` rows: **61.34%** over **357 rows**. The incumbent's own
write-up quotes 59.4% and 91.4% for the same run, so the documents and the raw JSON agree.

### Why this reframes the project

A 3B model at **74.7%** reads as a decent small router until a classical model at **93.19%** is on
the same board, answering in **0.61 ms** where the 3B takes 242.7 ms — a 400× latency gap at
better accuracy. It makes the neural arm look like the wrong instrument rather than an
under-tuned one. The two baselines this project had never run were **TF-IDF + LinearSVC at
93.19% / 0.61 ms** and **MiniLM + logistic at 86.65% / 57.8 ms/row**.

Rebuilding this benchmark by hand means re-generating 382 labelled utterances across 18 classes
plus the tool registry. The generator exists, but the labelled set is the expensive half.

### What landed in this repo

`tests/data/routing_382.json` (the rows, 39 KB), `bench/registry_382.json` (the registry, 21 KB:
17 tools plus `__none__`, **200** reference strings — the training corpus, not the test set),
`bench/classical_baseline.py` (the classical half, stdlib Python), and the recorded
`bench/results/classical_382.json` / `bench/results/classical_v2.json`. Provenance, digests and
per-class row counts are in `tests/data/routing_382_PROVENANCE.md`.

The recorded protocol matters: fitting on the registry's references scores 93.19%, while fitting
on the 382 rows themselves and holding out whole templates drops to **66.2%**. Both numbers are
real; they measure different things, and `docs/classical-baseline-2026-11-03.md` reports both.

---

## The needle fine-tune — a measured negative

### The eval, 4-op grammar, 50 held-out rows

| variant | layers | accuracy | acc when answered | refused | p50 ms | confidence head |
|---|---:|---:|---:|---:|---:|---|
| untuned control | 4 | **26.0%** | 31.7% | 9/50 | 226 | present, ECE 0.1666 |
| LoRA `d4` | 4 | **28.0%** | 31.1% | 5/50 | 206 | **absent** |
| LoRA `d8` | 8 | **6.0%** | 7.3% | 9/50 | 320 | **absent** |
| LoRA `d20` | 20 | **10.0%** | 19.2% | 24/50 | 1100 | **absent** |

`d8` predicted the zero-row class `declare_done` **30 times out of 50**. The majority baseline is
**0.81**. The fine-tune made the model worse, and the run is recorded as a negative result in the
fleet project's slate write-up and in the swarm plan's verdict table.

The control rows cannot be dropped from this table: without the untuned base sliced to the same
depth, the tuned number means nothing.

An earlier control at 8 layers over **40 rows**: exact **15.0%**, `flag` recall **85.7%**,
`nudge` recall **0%**, `leave` recall **0%**, **37/40** refusals, p50 994 / p95 1210 ms.

### Throughput and cost, measured on this box

- Training: **262.4 s/step**, **29 steps/epoch**, **126.8 min/epoch** — 452 examples, batch 16,
  seq 1024, CPU JAX. The recorded verdict against a 600 s/epoch box is `OVER BUDGET`.
- After prompt compaction: **57.9 s/step** at `--max-len 512`, **63.2** at 640, **94.9** at 768;
  **25.1 / 27.4 / 41.1 min** per epoch.
- The compaction itself: SYSTEM **709 → 515** chars (**216 → 170** tokens), tool schemas
  **780 → 666** (**174 → 150**), static prefix **403 → 333** tokens.
- Serving: warm **~1.4 s/decision** for the full 20-layer archive; first call in a process
  **20–43 s** (engine load + prefill).
- Zero-shot refusal, the reason the scaffold exists: `REFUSALS 1/1 (100.0%)`, confidence
  **0.032**, one call withheld.
- Sidecar over **22 rows**: exact match **0.1364** both before and after the prompt work, **22/22**
  refusals, coverage **0.0 at every threshold 0.3–0.9**, p50 **3776 → 2023 ms**.

### What the pipeline established that is worth keeping

Four things a rebuilt pipeline needs and would otherwise rediscover:

1. **Refusals belong in the first column of the report**, not in a footnote. A 14.7 KB eval
   harness is enough to do it, and that harness reports `REFUSALS n/N` and what the engine
   withheld, splits accuracy into overall vs answered, prints a confusion matrix **with a
   `refused` column** so a never-called class shows as zeros, and calibrates by confidence decile
   with ECE. `bench/band_summary.py` in this repo cannot do any of that.
2. **Zero-row classes must be filtered from both the rows and the tool catalogue.** `declare_done`
   had **0 rows out of 502** and was left in the grammar; that is the direct cause of `d8`'s 30/50
   over-prediction. `leave` had **7 rows**, so its recall is a 7-example statistic.
3. **A locally built `.cact` loses the confidence head** — see the gaps below.
4. **Label → op must be one table, not three.** The miner, the converter and the benchmark each
   consumed the same mapping (`continue/nudge/redirect/stop/new_task` →
   `nudge/nudge/nudge/leave/flag`). Worth importing as a *pattern*, not as code: the five labels
   are fleet-specific.

### The mined corpus, in case it is ever reused

The miner defined a decision point as a completed agent turn a human then answered, and had
written **527** rows by the end (the README documents 502 — the miner kept writing after the
README was taken). The newest regeneration split **474 train / 53 val** (817 KB + 91 KB); the
older slate split was **452 / 50**, which is the set the recorded eval numbers above were
measured on, so the two must be labelled separately or the numbers become unattributable.

Those rows are in the deployment's own distribution and were already field-mapped, with an
arguments-made-verbatim-span check that found **93 bad rows** in an earlier cut, already split by
FNV-1a hash so a re-run puts the same example on the same side. They also carry the negative
result above — `d8` at **0.06** against the **0.81** majority baseline — so nobody rebuilds this
pipeline expecting a win. The valuable import is as much the post-mortem as the data.

---

## Needle as an interface: what it cannot do

### Architecture — the one decomposition worth keeping

**121,019,861** params: d_model **768**, **20** layers, **12** heads / **2** KV heads, vocab
**8192**, sliding window **1024**, global attention at layers **4/9/14/19**. Decomposed from the
safetensors header: **36,488,980 (30.2%)** transformer blocks, **70,778,880 (58.5%)** engram slot
tables, **6,291,456 (5.2%)** token embedding, **5,898,240 (4.9%)** engram key/value projections.

More than half the model is a lookup table. That is the most useful single fact about what a
"121M attention-only model" actually spends its budget on, and it is the reason the param count is
kept here while the checkpoint itself is not.

Artefact sizes, for anyone sizing a deployment: the bf16 base checkpoint is **242,047,978 B**
(**7,906,640 B** for the rank-16 / alpha-32 LoRA adapter), the published deployable archive is
**35,335,380 B**, and the engine shared object is **1,590,584 B**. Sliced exports come out at
**15.4 / 27.4 / 63.4 MB** for 4 / 8 / 20 layers. The packaging is 3.2.8, and the three research
write-ups that document the family run 35 KB + 35 KB + 45 KB against three bench drivers of
3.2 / 2.4 / 0.6 KB.

### Vendor vs independent numbers

Needle 3 scores **50.2%** on BFCL v4 against Qwen3.5-0.8B's **56.8%**; **24.7%** on SNIPS 7-way;
and the vendor's own base-model note says it fails **5 of 6** Cactus suites at the default gate.
Separately, the decision-model research tree records that there are **no independent BFCL numbers
for Needle-class <50M tool-callers** — a gap the upstream benchmark partly fills, and the reason
the 59.42% above is worth more than the vendor table.

### Five gaps, and why they matter

1. **No probability vector, ever.** Needle returns one call-acceptance scalar —
   `min(calibrated head, decode probability)`, floor **0.1** — with no distribution over the
   declared tools. This project's `choice` primitive wants a distribution, so needle cannot be a
   `choice` backend without a lie in the interface.
2. **No GGUF / llama.cpp path.** `.cact` is a proprietary container at **2.125 bit/weight** with a
   nameless positional tensor directory. Everything in `docs/model-sweep-consolidated.md` was
   measured through `llama-server`; needle cannot join that table.
3. **The confidence head is lost on local fine-tunes.** A local build drops it, so `confidence` is
   `None`, the sidecar coerces `None → 0.0`, and the escalation ladder therefore escalates
   **100% of decisions**. Any locally built needle adapter is unroutable until that is fixed. Only
   the vendor's own finetune path keeps the head.
4. **No throughput number for the sliced rungs.** The 206 ms and 1100 ms above are eval-loop
   medians over 50 rows, not a latency benchmark; nobody has run pp/tg on a `.cact`.
5. **No `extract` benchmark wired into this repo.** An extract-shaped needle bench exists in the
   private lab tree and has never been pointed at
   `service/ti/decide/primitives.py::extract`, even though the needle decision note names
   `extract` as needle's correct seat. *(Half of this closed 2026-11-03: the primitive has a
   backend — `rule_extract`, `service/ti/decide/backends/rule_extract.py`, shipped as
   `decide.backends_in_use.extract` — so the `KeyError` is gone and the seat is occupied, by a
   rule extractor rather than a model. The benchmark half stands: nothing under `bench/` drives
   the lab tree's extract bench.)*

**Net:** the local evidence on Needle is a *measured no* for routing — 59.4% node-fixed, and a
fine-tune that dropped 26% → 6% — and an *unmeasured maybe* for `extract`.

---

## The decision-model synthesis — 28 conclusions

The operator's own synthesis of a private decision-model research tree, in 28 points, each with
its document and section anchor inside that tree (`README` or `01`–`05`, with a section number).
The tree itself is not in this repo. (The summary line this section carried before the cut said
"20 points"; the list has always run to 28, and 28 is what the cross-references below count.)

> **Numbering is load-bearing.** Other documents in this repo cite these points by number —
> `TODO.md` cites §9, §19 and §25, and `docs/roadmap-2026-11.md` cites §1 and §19. The numbers
> have not been renumbered, reordered or merged in this revision. The section headings in this
> document are deliberately unnumbered so that a `§`-reference can only mean a point below.

1. **LoRA SFT at rank 32–64 on an instruct 3B, 2–10K examples, 10–20% general-instruction mix is
   the highest-expected-value first move.** Everything else is second. (`README`, headline #1)
2. **Constrained decoding raises schema validity to ~100% for free**, so structured-output
   reliability is an inference-time problem, not a fine-tuning problem: "train the content,
   constrain the syntax." (`README` #2; `02` §6; `05` §4)
3. **Calibrate and report cost, not accuracy.** With an asymmetric cost matrix, `Acc@1` alone is
   misleading. (`README` #3; `03` §1)
4. **Do not start with diffusion** — published code results put it 10–17 points below
   autoregressive at comparable size. (`README` #4; `05` §3)
5. **Default to routing-as-classification** (closed candidate set, label out) for a tens-of-tools
   catalog: cheapest to train, easiest to constrain, and confidence becomes a real distribution
   rather than a sampled token. (`01` §1)
6. **Train closed but hold tools out of training entirely** and evaluate on them — routing
   transfers across corpora and retrievers when trained for it. (`01` §1)
7. **Confidence should be a bucket, not a regressed float**: emit `high/med/low` or a discrete head
   over 0.05 bins, then temperature-scale. Buckets let you supervise from evidence you actually
   have. (`01` §2, "Confidence as a target")
8. **Keep `target` separate from `candidates`** in the row format so rows survive a re-serializer
   change; pin `schema_ver`, sort candidates by name so position carries no signal. (`01` §2)
9. **Volume is not the lever — verified quality is.** 300–800 verified rows for a usable 1–3B
   router, 2,000–5,000 for phrasing robustness; hand-authored confusable rows weighted ~10× teacher
   rows. NVIDIA's closest published analogue: Llama 3.2 1B LoRA on ~500 samples → ~93%
   function-name accuracy vs ~1.4% base. (`01` §5)
10. **Hard negatives are the asset.** Build choices so non-gold candidates *are* the negatives
    (2–5 confusable + 3–10 random); pairs above ~0.85 cosine on tool descriptions are confusable
    and deserve ten hand-authored requests each. (`01` §4)
11. **You need a "none of these" / `clarify` class at 5–10% of the set.** With a small catalog,
    abstention is a first-class behaviour and is what breaks in small models. (`01` §4; `04` §1)
12. **Preference optimisation mostly does not beat SFT on a classification-shaped task**, and the
    best evidence is a null result (Tülu 3: only the instruction-following persona set improved its
    own target). The mechanism is saturation: when the base already puts mass on the right tool, the
    DPO gradient vanishes. (`02` §2)
13. **RLVR/GRPO is real but conditional.** ToolRL reports +17% over base and +15% over SFT from 4K
    examples. Two cautions: GRPO with *random* rewards still lifted Qwen2.5-Math by 21.4 points, so
    a gain on a Qwen base is not proof the reward works; and RL sharpens, collapsing toward
    high-prior tools. (`02` §3)
14. **Distillation's dense signal is the teacher's distribution over tool names**, not one-hot
    labels — 40 floats per example, a few lines on top of the SFT loop. Hard-label imitation
    transfers style, not capability. (`02` §4)
15. **Curriculum reordering is a null result** except in the multi-hop staging form (train 1-hop,
    then 2-hop, then 3-hop), which is credit assignment rather than curriculum. (`02` §5)
16. **Constrained decoding helps classification and hurts reasoning** — JSON-mode "improves
    classification accuracy by constraining the answer space and eliminating answer-selection
    errors" — but greedy constrained decoding distorts the distribution, so parse the mask before
    reading a softmax as confidence. (`02` §6)
17. **Measure selection, binding and knowing-when-not-to-act separately**; log candidate list,
    chosen tool, arguments, confidence and decode config per hop or none of it is computable.
    (`03` opening)
18. **A softmax over tool names is not a probability.** Renormalising over tool-name logits
    redistributes whatever mass sat on prose, a clarifying question, or an invented name —
    destroying exactly the signal abstention needs. Treat it as a ranking signal until verified.
    (`03` §2)
19. **Size the eval for 2-point detectability, and pair it.** Unpaired 2-point resolution at
    p≈0.85 needs ~5,000 per arm; McNemar at 10% discordance needs ~1,960 — a 5× saving. At n=300
    the 95% interval is ±4.0 points, so a 2-point win on 300 items is noise. Hold out whole
    phrasing families, not rows. (`03` §3)
20. **External benchmarks do not transfer** (catalog, grading, composition, fallible graders). Use
    BFCL/τ-bench as a smoke test that a checkpoint is not broken, nothing more. (`03` §4)
21. **Bucket errors by the first decisive error only, then size the next batch by count × cost.**
    "Over-abstention" and "under-abstention" are separate buckets; under-abstention is the
    dangerous one. (`03` §5)
22. **The minimum honest report is four numbers**: `Acc@1`, expected cost per 1,000 hops,
    destructive-misroute count, and the risk/coverage point. A single mutating misroute cost up to
    92%/96% of task success (SABER). (`03` §6)
23. **The floor for a small router is ~125–230M, and "restraint" is what breaks below it** —
    `llama3.2:3b` and `granite4:3b` score 0.000 on judgment because they always emit a call.
    Parameter count is a weak predictor. (`04` §1)
24. **Cascade: hash → rules → MiniLM encoder + linear head → 1B LLM for the tail**, gated on
    *margin* as well as threshold (0.55/0.45 should escalate). RouteLLM's reference result is >2×
    cost reduction at ~90% of strong-model quality. (`04` §2)
25. **Quantisation changes the failure mode, not just the scale** — xLAM-2-3B 58.00% → 51.75% at
    Q4_K_M, Qwen3-1.7B 14.50% → 8.75% with wrong-argument rate 20.8% → 26.9%, producing
    "error-free-but-wrong trajectories that no runtime channel captures." Recommendation: **Q8_0
    under 2B**, and re-run the eval at the quant you ship. (`04` §3)
26. **Latency is a policy problem, not a model problem**: batch 32 costs ~1.6× one request's
    latency for ~30× throughput, and prefix caching is the whole game — but only if the prefix is
    byte-identical. Local serving is 100–300× cheaper than hosted; the binding constraint is p99.
    (`04` §5)
27. **Semantic caching is disqualifying below ~0.97 similarity** for routing, because a false hit
    does not produce a slightly worse answer, it produces the wrong tool. Cache exact matches keyed
    on `(model, quant, adapter, system_prompt_hash, tool_schema_hash, query)`. (`04` §6)
28. **`temperature=0` is not reproducibility**, so log
    `(input_hash, output, logprobs, model, quant, seed)` and treat a >2% retry rate as a quality
    alarm. `04` §7's pick: `Qwen3-1.7B` at Q8_0, thinking off, batched, prefix-cached, FSM-
    constrained to a schema that includes `refuse`. (`04` §6–7)

Two caveats carried from the same tree: the documents contain **no measurements from this
machine** (`README`, "What these docs do and do not guarantee"), and `04` §1 explicitly records
that there are **no independent BFCL numbers for `Needle`-class <50M tool-callers** — a gap the
upstream benchmark partly fills.

Why these 28 points are worth having in-repo even though their source is not: they are the written
spec for needs (b) and (a), with citations already spot-verified (2026-10-28). Row schema,
hard-negative rates, the abstention class, the sample-size arithmetic (~1,960 paired vs ~5,000
unpaired for 2 points), the ECE and threshold formula, and the four-number minimum report are all
in them. Writing this from scratch is a week of literature work, and the reason this repo's
sweeps report accuracies at n=150 without confidence intervals is that this arithmetic was not on
hand.

The distillation corpus this project is growing sits under the 300-row floor in §9: the last
recorded snapshots were 199 of 414 rows at $0.002, then 267 of 414 at $0.0028. The stall was
provider congestion (0.7 s/row → 20–60 s/call), not budget.

---

## The documents that were read, and what each one settled

Sizes and verdicts only. No path, and no list of what else sat in the same folder.

| what it is | size | what it settled |
|---|---|---|
| An autonomous-optimisation program note covering rounds 1–3, with a round-by-round findings digest | 17 KB | The cascade wall and the agreement gate — the design rationale behind `service/ti/decide/cascade.py` and `service/ti/decide/pair.py`. Cite it, do not copy it. |
| A dedicated Cactus/Needle research note: version table, `.cact` container, the five serving shapes, the I/O contract, confidence semantics, why there is no GGUF path | 35 KB, 24 cited sources | The only document answering "can Needle implement this project's own primitives" — it is also the source of the `choice` gap above. |
| A needle decision note: spike-first, router do-not-adopt, extract-sidecar adopt-on-pass | 35 KB, plus raw fetched bytes | The split between the two needle verdicts. It is why `extract` was pursued and the router was not. |
| A corrected coverage/precision frontier across lanes, plus the raw frontier JSON | markdown + 1.7 MB | The per-lane coverage/precision trade-off the cascade gate is chosen against. The markdown was the transferable half; the 1.7 MB JSON stayed where it was. |
| A gate-cascade design note | 50 KB | Gate arithmetic. Read before changing it in `service/ti/policy/`. |
| Round-3 plans and ten small verification scripts (reach, routing, GIL probes) | plans + 10 scripts | Needle-at-scale evidence. The scripts are probes, not a harness — nothing was imported from them. |
| Tiny-model theory notes: attention-only + thin-FFN architectures, tiny-scale calibration, quantisation-vs-calibration, edge hardware matrix, MCU weight streaming, five experiment arms | 21 documents, ~1.4 MB | Selective citation only. The broken-confidence arm and the tiny-scale-calibration note are the two relevant to a confidence gate; the broken-confidence arm is where the "confidence head" gap above comes from. |
| A sub-200M base-model survey with GGUF tensor-table param counts (not card claims), embedding fractions, IFEval numbers, and a KD measurement | 807 lines | §6.1 is the only controlled public KD-at-135M number measured here, which is why it is worth importing as a document rather than citing. |
| A tiny-finetune toolchain spec: tool matrix, FLOPs arithmetic, a measured `llama-finetune` run, the full-FT-vs-LoRA argument, serving handoff, and a full-parameter SFT script + corpus renderer in the appendices | — | The appendices are a runnable full-parameter SFT for a 135M model on CPU or one T4. |
| Metric definitions | 22 KB | Read only if the eval introduces a new metric. |
| A spec set: role grammars, a corpus factory, prompt trimming, a free-compute survey, an integrity rule, orchestration plans | — | Prompt trimming is where the 403 → 333 token prefix number comes from. The rest is project-specific. |
| A hybrid needle/swarm plan, "every number measured on this box" | 46 KB | The only place the 121,019,861 is decomposed by component, and the verdict that `d8` is unusable as a router. |
| A benchmark design note and an experiment contract | 60 KB | Method only. Worth importing if this repo starts committing run records as contracts rather than results. |
| The incumbent's benchmark write-up: accuracy, calibration bands, and the serving gotchas | — | The bar every candidate must beat. See the serving-trap list below. |
| Hardware/serving trap notes, vision and screen-grounding benchmark notes, validation widening | — | Four bugs worth a day each, all carried into `docs/serving-fix-qwen35.md`. The vision half is relevant only if this project grows a vision op. |

---

## What was imported

Everything that came across, with where it landed. Every path below is in this repo and
resolves; run `python3 -m pytest tests/ -q` and the ones with tests are exercised.

### The three CLI skills — fixtures only

A 7.5 KB README plus three skill directories (web-search, web-fetch, reddit-research) and a
shared `lib/` and `harness/` came from the notes tree. This is a *serving recipe and eval harness*
asset, and it was already absorbed before this survey ran.

**Landed at** `tests/data/devskills/` (`web-fetch/` and `web-search/` fixture trees), replayed by
`tests/test_ops_web.py`. Do not re-import; check fixture parity only.

### The 382-row benchmark — data, registry, runner, results

**Landed at** `tests/data/routing_382.json`, `bench/registry_382.json`,
`bench/classical_baseline.py`, `bench/results/classical_382.json`,
`bench/results/classical_v2.json`, with provenance in `tests/data/routing_382_PROVENANCE.md`.

This is the single highest-value import: it is the only head-to-head where six backends saw the
same 382 utterances on one box, and it carries the two baselines this project had never run. Until
those are on the board, "74.7% for a 3B" reads as good; against a 0.61 ms classical model at
93.19% it reads as the wrong instrument. *As landed, there is no `bench/routers/` and no
`bench/results/routing_382/` — the six runners collapsed to one script,
`bench/classical_baseline.py`, whose artifacts are the two `classical_*.json` files.*

### The decision-model synthesis — as the dataset and eval spec

**Landed at** `docs/assets-inventory.md` §1–§28 above. The five source documents did **not** land
verbatim; the conclusions did. A verbatim copy of the five documents, and a distilled dataset-spec
page beside them, were both proposed and both were dropped: the conclusions are what a reader
needs, and `docs/assets-inventory.md` §8 already fixes the row format, `§19` the sample-size
arithmetic and `docs/dataset-status.md` the current corpus state.

### The needle eval harness and sidecar pattern

**Landed at** nothing imported verbatim — but the *shape* is recorded above and the four
capabilities it has that `bench/band_summary.py` lacks are listed in the negative-result section.
The sidecar is a 4.6 KB worked example of wrapping a non-HTTP, process-global engine behind this
project's `decide` wall, including the single-long-lived-handle and `reset()`-per-call discipline;
`service/ti/decide/backends/` is where such a backend would go if a needle seat is ever wanted.

### The mined corpus and its converter — as a post-mortem

**Landed at** nothing. The rows stay in the private tree. What was taken is the specification of
what a corpus must look like (474/53 split, 527 points, FNV-1a splitting, verbatim-span argument
checks) and the negative result that stops anyone rebuilding it expecting a win.

### The incumbent's benchmark and the sub-day serving traps

**Landed at** `docs/serving-fix-qwen35.md`, `docs/research-model-sweep-q4.md` and
`docs/model-sweep-consolidated.md`, as the reference lane those sweeps are read against; the probe
landed in `bench/`.

This is the bar every candidate must beat: **91.4% @ 243 ms with none-detection 1.00**, which is
**17 points** above the 74.7% this repo currently treats as the incumbent. It reframes the whole
live-probe table. The companion notes supply four traps, each a lost day:

- the `systemone` endpoint needs llama.cpp build **≥ b11500** (the vendored **b10964** has no
  `systemone` symbols);
- the AV quarantines any binary named `llama-server.exe`, so the same bytes must be extracted under
  another name;
- Vulkan is fine single-shot and dies as a long-lived server;
- grammar-constrained decoding crashes on large probability-array schemas, which is why the outcome
  cap defaults to **8**.

---

## What was not imported

Recorded so that a later pass does not spend itself on these again.

| asset | why not |
|---|---|
| A multi-agent coordination implementation and its benchmark tasks | A different question — coordination, not routing decisions. Only its needle plan's param decomposition was worth reading, and that is in the architecture section above. |
| A second, earlier copy of the same benchmark research (docs 01–07 of 13, plus a bench directory that is PowerShell-only) | Superseded by the upstream benchmark: the same files, fewer documents, and a bench half that only runs on one shell. Nothing imported. |
| A docs-only fine-tuning plan for a fourth model family (model choice, serving, inference-engine decision records, prompt personas) | A plan that was not executed — no code and no weights. Cited only if that family is reconsidered. |
| A decision-point miner's implementation | The pattern, not the code — see the mined-corpus section. Its five labels are project-specific. |
| Two further trees outside this project's subject | Neither contains a dataset, model, adapter or eval for decisions. Nothing importable, and nothing about them is recorded here beyond this line. |

Everything else in the documents table above was read and cited rather than copied. The size
column is the guide to which ones reward a read: 807 lines of base-model survey with tensor-table
param counts and 22 KB of metric definitions are different investments, and only the first is
repeatedly useful.

Two are worth *not* importing even though they are close to the subject: the swarm project's
implementation (coordination, a different question — its plan is worth reading for the param
breakdown only) and the out-of-subject trees (nothing to import between them).

---

## Status of this survey — 2026-11-03 16:20 EST

*(Appended. The survey above is the record of what existed when it was written; three of its
"has never been run" claims have since stopped being true, and they are marked inline as well as
here.)*

| claim in this document | status now |
|---|---|
| "the classical baseline this project had never run" (headline, and the benchmark section's two never-run baselines) | **done, in-repo.** `bench/classical_baseline.py` reproduces the upstream benchmark's 93.19% / 0.60 / 0.61 ms protocol in stdlib Python, plus the checks that number needed: **66.23%** with whole templates held out, **0.0%** coverage behind the shipped 0.70 gate, and **62.7%** against d1-3B's **74.0%** on this repo's own frozen set. Written up in `docs/classical-baseline-2026-11-03.md`, artifacts in `bench/results/classical_382.json` and `classical_v2.json`. The MiniLM baseline still has not been run. |
| the benchmark section's landing paths | the rows landed at `tests/data/routing_382.json` and the registry at `bench/registry_382.json`; there is no `bench/routers/` (the runners became one script, `bench/classical_baseline.py`) and no `bench/results/routing_382/`. |
| "`resolve("extract")` raises `KeyError` today" | **closed.** `extract` has a backend (`rule_extract`) and ships as `decide.backends_in_use.extract`. What is still missing is the *benchmark*: nothing under `bench/` drives the private lab tree's extract bench. |

Everything else here is a reading of a private tree's files and is unaffected by this repo's work.
The one number a later document qualifies is in `docs/classical-baseline-2026-11-03.md` §5–6,
which measures the same classical model on a different, harder task and reaches the opposite
verdict for it; that is a scope difference, not a contradiction, and that document says so.

---

## Corrections — 2026-11-04

Appended, not edited. The survey above is the record of what the private trees held when it was
read; the numbers in it are quoted correctly from those files. What does not survive Phase 1 is
the **framing** that lets a reader carry them into this repo.

### C1. The 382-row set is a smart-home router; the intersection is empty

`routing_382` is a smart-home assistant router: `timer_create`, `thermostat_set`,
`shoppinglist_add`, `calendar_today` — **18 classes, its own 17-tool registry, its own
200-reference training corpus**. **Its intersection with this repo's 8 ops is empty.** Re-verified
2026-11-04 against `tests/data/routing_382.json` and `tests/data/utterances_v2.json`: 18 labels,
8 labels, intersection empty.

So every figure in "The 382-row router benchmark" (93.19%, 91.36%, 86.65%, 75.64%, 59.42%) is a
measurement **of a task this repo does not perform**, and none of them is comparable to an
accuracy on `useTool`'s 8 ops. Only the `62.7%` figure — `docs/classical-baseline-2026-11-03.md`
§5, the same linear model scored on this repo's own frozen 150 rows — is. The `Status of this
survey` row that runs 93.19 / 66.23 / 62.7 / 74.0 together in one sentence should be read as two
sentences: two figures from the smart-home router, two from this repo's 8 ops.

### C2. "74.7% against 93.19%" — the two numbers are from different sets

Two places make the rhetorical comparison this survey is built around, and both mix a
150-row figure into a 382-row board:

- **line 77**: "A 3B model at **74.7%** reads as a decent small router until a classical model at
  **93.19%** is on the same board, answering in **0.61 ms** where the 3B takes 242.7 ms — a 400×
  latency gap at better accuracy."
- **line 390**: "Until those are on the board, '74.7% for a 3B' reads as good; against a 0.61 ms
  classical model at 93.19% it reads as the wrong instrument."

74.7% is **not** a 382-row number and is not on that board. In this repo's recorded sweep it is
d1-3B-Q4's **`logit_read`** lane on the **150-row, 8-op** set
(`docs/model-sweep-consolidated.md`; 81.2 / 67.7 / 70.9). On the 382-row board the same model's
recorded figure is **91.36% `systemone`** — and it is the row directly above 93.19% in this
document's own table.

What that does to the claim:

- the **400× latency gap** is real and is measured on the same rows (0.61 ms against 242.7 ms on
the 382) — unaffected;
- "**better accuracy**" is true **on the 382 board**: 93.19% against 91.36%, a difference of
  **1.83 points** — not the 18-point impression the 74.7%/93.19% pairing creates;
- and the inference "**it reframes the whole live-probe table**" (line 424, "**17 points** above
  the 74.7% this repo currently treats as the incumbent") does **not** follow. The 17 points span
  two datasets, two adapters and two label spaces. There is no 17-point gap on any single board in
  this repo, and the incumbent's own 382-row score is 91.36%.

The correct reading: on a closed, in-vocabulary smart-home label set, a linear model matches a 3B
at ~400× lower latency. That is a real and useful result about *that* task. It says nothing about
how a linear model performs on this repo's ops — `docs/classical-baseline-2026-11-03.md` §5
measures that, and finds it 11.3 points *behind* the 3B.

**Evidence.** `tests/data/routing_382_PROVENANCE.md` and
`docs/distill-corpus-certification-2026-11-03.md` §1 (the 18 classes, the own registry, the empty
intersection); `docs/model-sweep-consolidated.md` (74.7% = d1 `logit_read` on the 150-row set);
`docs/classical-baseline-2026-11-03.md` §2 (91.36% on the 382, recorded there and not re-run). Of
the 382-row figures listed in C1, `bench/classical_baseline.py` reproduces 93.19%, 92.41% and
66.23%. The other four — d1-3B 91.36%, MiniLM 86.65%, Qwen3.5-0.8B 75.64%, needle3 59.42% — are
recorded from the sibling project and are not re-run here.
