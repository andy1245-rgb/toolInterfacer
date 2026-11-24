# docs/

Every document in this directory, grouped by what you are trying to do — dated files record one tree
on one day and are not rewritten, undated files are current reference. Start from the front page,
[`../README.md`](../README.md); the module contracts are [`../ARCHITECTURE.md`](../ARCHITECTURE.md) and
[`../spec/1.0-spec.md`](../spec/1.0-spec.md).

## Design and contracts

| file | what it covers |
|---|---|
| [`architecture.md`](architecture.md) | the per-module contract: what each module owns, and explicitly does not own |
| [`ops.md`](ops.md) | the CLI contract the subprocess ops rely on, the mismatches found against live CLIs, and §9's HTTP status contract |
| [`confidence-sources.md`](confidence-sources.md) | every place a confidence is produced, each classified measured / derived / asserted, and the rule that came out of it |
| [`consequence-model-2026-11-10.md`](consequence-model-2026-11-10.md) | why a misroute costs one call and a retry: no op mutates, no subprocess has a shell, paths are confined before the first syscall |
| [`consequence-and-fusion-brief-2026-11-10.md`](consequence-and-fusion-brief-2026-11-10.md) | the brief that produced the consequence model, plus the options considered for the fusion's C3 failure and what was left open |
| [`logging-2026-11-04.md`](logging-2026-11-04.md) | what the tool records by default, asked of the consumer first — hence opt-in capture, a reduced record, and `0600`/`0700` modes |
| [`security-surface-2026-11-04.md`](security-surface-2026-11-04.md) | what a token holder can reach with the shipped config, enumerated rather than argued, and what a deny list would and would not close |
| [`testing.md`](testing.md) | where the suite and benchmarks run, the d1-3B launch recipe, the placeholders-not-paths rule, and how to prove the wire rather than the engine |

## The measurement instrument

| file | what it covers |
|---|---|
| [`harness.md`](harness.md) | the shared harness: one dataset seam, one predictor seam, one set of metrics, one set of controls — with the reproduce-the-record quickstart |
| [`harness-spec.md`](harness-spec.md) | the specification the harness was built to, followed by the open questions the first build left |
| [`eval-harness-2026-11-04.md`](eval-harness-2026-11-04.md) | group-aware splits, repeated seeds, cluster-bootstrap intervals — and every local signal re-measured through them, including the 150-row set that cannot be split by family |

## Routing measurements and their verdicts

| file | what it covers |
|---|---|
| [`model-sweep-consolidated.md`](model-sweep-consolidated.md) | the frozen 150-row sweep: d1-3B-Q4_K_M at 74.0% per band against the 4B challengers, plus the correction where a lane turned out never to have served the model it named |
| [`research-model-sweep-q4.md`](research-model-sweep-q4.md) | the 4B-class scout behind those lanes: candidates sourced, downloaded, benched through `logit_read`, scored against the d1-3B bar |
| [`serving-fix-qwen35.md`](serving-fix-qwen35.md) | why the qwen35 lanes served CPU-class in-process — **unresolved**, with the flags, builds, RPC, driver and GGUFs each ruled out by measurement |
| [`vulkan-igpu-2026-11-03.md`](vulkan-igpu-2026-11-03.md) | the iGPU against the CPU build on this workload: 2.72× slower per decision and 0 of 50 paired rows won, because routing is prefill-dominated |
| [`classical-baseline-2026-11-03.md`](classical-baseline-2026-11-03.md) | the TF-IDF + linear-SVM baseline rebuilt in stdlib: the recorded figure reproduces but is a different task, and on this repo's own rows it is 62.7% against the 3B's 74.0% |
| [`classical-revival-2026-11-10.md`](classical-revival-2026-11-10.md) | the classical tier was pointed at a 17-label registry for its whole life; this is the revival measured on the eval rows, coverage first, with the part that is not proven |
| [`local-classifier-cascade-2026-11-03.md`](local-classifier-cascade-2026-11-03.md) | how much a free local model can take off a hosted fallback: every coverage above 2.7% lowers blended accuracy, and the per-class alternative was an oracle |
| [`router-policy-2026-11-04.md`](router-policy-2026-11-04.md) | the per-class routing figure measured without the answer: 79.3% [73.6–84.8] at n=150, family-grouped, with the accuracy/cost curve and why a top-2 margin gate is no better than top-1 probability |
| [`distill-classifier-2026-11-04.md`](distill-classifier-2026-11-04.md) | the 8-way classifier trained on the distilled corpus and scored on the frozen set: 68.7%, losing to the cue table, **not shipped**, with the two classes where it wins |
| [`distill-corpus-certification-2026-11-03.md`](distill-corpus-certification-2026-11-03.md) | the distilled corpus certified as training data: 412 rows labelled, 0 collisions with either frozen eval set, 410 frozen after two near-duplicate exclusions |
| [`dataset-status.md`](dataset-status.md) | the hop log measured as training data: 14 of 1398 records trainable and all one label — it is an execution trace, which is why the corpus is distilled rather than harvested |
| [`cost-reduction-glm-flash.md`](cost-reduction-glm-flash.md) | the hosted routing bill cut by measurement: $0.0202 → $0.0053 per 1k decisions (−73.8%) at 145/150, by pinning the cheapest endpoint and three prompt/answer changes |
| [`grep-arm-2026-11-10.md`](grep-arm-2026-11-10.md) | the grep-description arm measured properly: one served model with `/props` asserted, five arms, 2075 rows — **it does not ship**, and the per-option ledger is why |
| [`fusion-2026-11-10.md`](fusion-2026-11-10.md) | the fusion arm, pre-registered then measured: the four conditions, the one it failed first, the sensitivity that moves a fitted arm by 35 rows, and what is and is not established |
| [`order-and-variants-2026-11-10.md`](order-and-variants-2026-11-10.md) | shadow mode, the seven-arm variant sweep with every loss on the table, and order selection on one half validated on the other |
| [`rewrite-transfer-2026-11-04.md`](rewrite-transfer-2026-11-04.md) | the two op-description rewrites run out of sample: no net effect, with two significant opposite components — 9 rows bought on `filesys_grep`, 8 paid on `web_search` |
| [`post-rewrite-diagnosis.md`](post-rewrite-diagnosis.md) | where the 3B loses rows it was never asked about, the cheapest experiments that separate the explanations, and the re-baseline protocol that follows |
| [`roundtrips-2026-11-03.md`](roundtrips-2026-11-03.md) | whether `useTool` saves round trips: 2 of 14 removed and 7.5 s slower at a 2 s pass, with the crossover, and the re-run that does not reproduce the saving |
| [`decision-pair-vs-hosted.md`](decision-pair-vs-hosted.md) | `pair` vs `openrouter` vs `d1_systemone` over 30 interleaved utterances, and why `pair` ships anyway: hosted-only hard-fails without a key, the pair degrades to local |
| [`direction-adjudication.md`](direction-adjudication.md) | four candidate directions for the decision layer, one section each, with the measurement that decides it and the condition that would reverse it |
| [`shadow-mode-2026-11-10.md`](shadow-mode-2026-11-10.md) | how to read the shadow log: `vocab_match` first, which question the log actually answers today, and the routing change that would make it answer the other |
| [`history-replay-2026-11-10.md`](history-replay-2026-11-10.md) | the assistant's own tool calls replayed as routing traffic: 20 000 pairs, 0 answered by the eight-op member, and a fitted member that loses to a constant |
| [`closed-experiments-2026-11-10.md`](closed-experiments-2026-11-10.md) | the register: five experiments shut with the number that closed them, what would reopen each, and one that met its local objective and still failed its global condition |

## Audits, reviews and corrections

| file | what it covers |
|---|---|
| [`audit-2026-11-03.md`](audit-2026-11-03.md) | the spec audited module by module: four blockers (a wheel shipping 2 of 53 modules, 66 `tsc` errors, a template naming ops that do not exist, a default needing a model and a key) and what became of each |
| [`review-2026-11-03.md`](review-2026-11-03.md) | five blockers with the proof for each — `confidence: 1.0` on a run that measured nothing, a one-entry probability vector read as certainty, an unauthenticated POST |
| [`review2-2026-11-03.md`](review2-2026-11-03.md) | the same five re-checked after twelve repair merges, plus four new blockers, among them a ReDoS guard a bounded quantifier walks through |
| [`doc-review-2026-11-03.md`](doc-review-2026-11-03.md) | the docs read as a stranger: the suite count wrong in seven places, a bound the README calls structural although `sanitize` does not apply it, and the claims that held up |
| [`review-apparatus-2026-11-04.md`](review-apparatus-2026-11-04.md) | the measuring apparatus itself reviewed: what the harness could not see, the fold rule that moves a fitted arm by tens of rows, and the controls the review added |
| [`adversarial-review-2026-11-04.md`](adversarial-review-2026-11-04.md) | the adversarial pass over the same claims, one at a time, with the verdict and the constraint each one leaves behind |
| [`consistency-review-2026-11-04.md`](consistency-review-2026-11-04.md) | a contradiction matrix for every retired number, which conflicts are a dated body doing its job and which mislead a reader, and the reachability ledger for corrections |
| [`number-sweep-2026-11-04.md`](number-sweep-2026-11-04.md) | every place in the tree a retired figure appears, each marked fixed / clean / coincidence with the reason, plus the eleven places the same patterns appear without those digits |
| [`pre-publish-audit.md`](pre-publish-audit.md) | the gate before a public push: what a stranger can see, download and infer from the clone — history, secrets, licence, the tracked tree — with a resolution appendix |
| [`quickstart-stranger-2026-11-03.md`](quickstart-stranger-2026-11-03.md) | the README followed literally from a fresh clone: five wrong turns to a first working call, and the two documentation defects behind them |
| [`rot-audit-2026-11-10.md`](rot-audit-2026-11-10.md) | the rot register: every backend and composition asked whether it can degrade into "no opinion" indistinguishably from a component with nothing to add |
| [`rot-detection.md`](rot-detection.md) | the generalisable lesson from that audit as a method: one symptom per row and the check that exposes it, plus the protocol for reading a revived tier's measurement |
| [`assets-inventory.md`](assets-inventory.md) | what the sibling projects already hold, surveyed against this repo's needs: ranked import candidates with a recommendation for each, and two things worth not importing |

## Plans and end-to-end probes

| file | what it covers |
|---|---|
| [`roadmap-2026-11.md`](roadmap-2026-11.md) | a work plan with a closing outcome: which publish blockers were closed and re-probed rather than asserted, and the three measurements that changed the plan |
| [`live-e2e-2026-11-03.md`](live-e2e-2026-11-03.md) | the shipped path probed over a real socket through the whole stack, 5/5 cases, and the ten defects it found and fixed |
