# Decision-backend sweep — consolidated report

Session handoff. Consolidates three workstreams run on dell321 on 2026-10-17:

1. **Stratified v2 backend sweep** — the primary comparison, every lane scored on
   the same frozen 150-row set (`tests/data/utterances_v2.json`).
2. **Q4 model scout** — sourced shortlist + benchmark of 4B-class decision
   candidates (`docs/research-model-sweep-q4.md`); its three lanes are folded into
   the same per-band table here.
3. **Serving RCA** — why qwen35-family lanes don't reach GPU speed in server
   contexts (`docs/serving-fix-qwen35.md`).

Bar to beat (incumbent): **d1-3B Q4_K_M on the native `/v1/systemone` path**.

---

## CORRECTION (2026-10-18) — the "Qwen3.5-4B" lanes never served Qwen3.5-4B

> **Superseded:** the original Qwen3.5-4B numbers in this document (table row and
> findings 2, 3, and next-session recommendation 1) are wrong and are kept only
> for the record. Read the corrected numbers below.

The four result files `bench/results/dell/q4-Qwen3.5-4B-{v1,v2}{,-run2}.json`
actually contain **d1-3B-Q4_K_M** data — **no Qwen3.5-4B was ever loaded** for
those lanes. Three defects combined (full RCA: `toolInterfacerExtras/dell-rerun/c-recap.md`):

1. The Qwen lane's serve command carried `--chat-template-kwargs
   {"enable_thinking": false}`. An unquoted `{"enable_thinking": false}` in a
   PowerShell 5.1 command line is a **parse error**, so the serve never started
   and its `> log` redirect was never created — a silent remote failure.
2. A **stale d1-3B-Q4 server owned dell:8080** during the Qwen lane's probes
   (a concurrent session had left it running).
3. The lane loop gated only on **`/health`**, which *any* llama-server answers
   `ok`, so the stale model answered every probe.

**`/props` evidence.** All four phantom files record
`model.path = …\d1bench\models\d1-3B-Q4_K_M.gguf`, and the mislabelled lane's
logit_read accuracy (74.67%) is bit-identical to the corrected d1-3B-Q4
logit_read number below. Raw `/props` captures are kept at
`bench/results/dell/props/props-{d1,qwen35,intern}.json`.

**Corrected v2 lanes** (2 runs pooled, every lane gated on a `/props` assertion
before any probe traffic; raw JSONs in `bench/results/dell/` with the
`-props-asserted` suffix):

| lane (adapter) | served model (`/props` asserted) | t1 | t2 | t3 | overall | p50 ms |
|---|---|---:|---:|---:|---:|---:|
| d1-3B-Q4 (logit_read) | `d1-3B-Q4_K_M.gguf` ✔ | 81.2% | 67.7% | 70.9% | **74.7%** | 204 |
| d1-3B-Q4 (systemone) | `d1-3B-Q4_K_M.gguf` ✔ | 81.2% | 71.0% | 67.3% | **74.0%** | 163 |
| **Qwen3.5-4B-Q4_K_M (logit_read)** | `Qwen3.5-4B-Q4_K_M.gguf` ✔ | 67.2% | 41.9% | 56.4% | **58.0%** | 2516 |
| Intern-Decision-4B (logit_read) | `Intern-Decision-4B-Q4_K_M.gguf` ✔ | 43.8% | 25.8% | 32.7% | **36.0%** | 1866 |

Data: `bench/results/dell/v2-d1-3B-Q4-logit_read-props-asserted{,-run2}.json`,
`v2-d1-3B-Q4-systemone-props-asserted{,-run2}.json`,
`v2-qwen35-4B-logit_read-props-asserted{,-run2}.json`,
`v2-intern-decision-4B-logit_read-props-asserted{,-run2}.json`
(preflight `qwen35-4B-logit_read-props-asserted-preflight-limit3.json` carries no
`v2-` prefix so it is not swept into `bench/band_summary.py`).

**Consequences for the pick.** The genuine Qwen3.5-4B is **58.0% — 16.7pp below
the phantom 74.7%** and 16.7pp below the incumbent d1-3B-Q4 on the same adapter;
its t2 (veiled intent, 41.9%) is the worst band of any lane. The "Qwen is the
pick" narrative in findings 2/3 below does not survive a correct measurement.
The Intern re-run reproduced the scout's 36.0% exactly, so the harness is
apples-to-apples with the historical valid lanes — only the Qwen lane was ever
wrong.

**Guard now in place.** `bench/model_guard.py` reads `/props`, compares the
served model path basename (and optional sha/size) to the expected model, and
aborts non-zero before any probe traffic; `bench/live_probe.py --expect-model
<basename>` enforces it (and prints a prominent warning when omitted);
`bench/lane.sh` serves through `bench/serve_lane.py` (argv built in-process, so
the kwargs JSON never touches a shell) and asserts `/props` after health, after
settle, and through the tunnel. Unit tests: `tests/test_probe_guard.py`.

---

## The frozen set

`tests/data/utterances_v2.json` — 150 rows, 8 op classes, stratified by band
(design + risk notes in `tests/data/utterances_v2_NOTES.md`):

- **t1 unambiguous (64)** — direct intent, signature vocabulary may appear. Harness floor.
- **t2 veiled (31)** — intent stated with no op-signature keyword; phrasing leans on the object.
- **t3 boundary pairs (55)** — label sits between two defensible ops; stresses the confusable sibling.

Probe: `bench/live_probe.py`, Jev-style single-token letter read (`max_tokens=1`,
top_logprobs=20, softmax over A–K, conf = 1−H/lnK), `-c 8192 -t 8 -ngl 99`,
build b11512 / a11f57ba9. Each lane: serve → health → probe ×2 → teardown.

## Consolidated per-band table (pooled over 2 runs; acc = correct/rows)

| lane (file) | backend | t1 | t2 | t3 | overall | p50 ms |
|---|---|---:|---:|---:|---:|---:|
| **d1-3B-Q4** | d1_systemone | **81%** | **71%** | 67% | 74% | 153 |
| **d1-3B-Q8** | d1_systemone | 80% | 71% | 67% | 73% | 158 |
| **Qwen3.5-4B** (think=off) — *SUPERSEDED: measured d1-3B-Q4; see CORRECTION* | logit_read | 81% | 68% | 71% | 75% | 265 (runs 236/406) |
| Kev-4B | logit_read | 58% | 23% | 33% | 41% | 2430 |
| Intern-Decision-4B | logit_read | 44% | 26% | 33% | 36% | 2395 |
| jev-style-2b | logit_read | 41% | 39% | 36% | 39% | 630 |
| qwen3.5-0.8b (base) | logit_read | 12% | 13% | 9% | 11% | 643 |
| heuristic_micro | heuristic | 11% | 0% | 7% | 7% | 0 (local) |

Chance for an 8-way choice is 12.5%. `heuristic_micro` abstains on 139/150 every
run, so its headline 7% is really **100% precision on the 11 rows it commits to**
— a coverage gate, not a weak classifier (see findings).

## Findings

1. **The bands behave as designed — accuracy is monotone t1 ≥ t2 ≥ t3 on every
   lane that is above chance.** The stratified set is measuring difficulty, not noise.

2. **[SUPERSEDED — see CORRECTION] On the hard set the incumbent ties the scout's pick.** d1-3B-Q4 (3B,
   ~153 ms, systemone) **74%** vs Qwen3.5-4B-instruct (4B, 265 ms pooled, logit_read)
   **75%** — within 1pp. The scout's "Qwen3.5-4B is the pick" rested on the
   incumbent's **easy-set** number (84% on 50-row v1); once the 150-row stratified
   set is applied, the two are level and d1 is smaller and ~2.7× faster. This is
   the single most important reconciliation for next session.

3. **[SUPERSEDED — see CORRECTION] Their failure shapes differ and it matters for the gate.**
   - d1-3B: 81 → 71 → **67** — handles *veiled* intent better (t2 71%), leaks on
     *boundaries* (t3 67%).
   - Qwen3.5-4B: 81 → **68** → 71 — weak on *veiled* intent, recovers on
     explicit boundaries.
   - Conf-gate: d1 commits on **56%** of rows above 0.7 at 87% precision
     (98% on t1); Qwen3.5-4B commits on only **19%** but at **100%** precision.
     **More coverage with some error (d1) vs near-perfect precision with thin
     coverage (qwen).** The route gate and ask-back budget pick between them.

4. **The decision-tuned 4B lanes do not port through the frozen letter prompt.**
   Intern-Decision-4B (36%) and Kev-4B (41%) both collapse — their decision
   tuning is bound to their own schema/pointer-head (Jev-schema, `/v1/systemone`),
   and plain chat-completions letter reads drift. Confirmation of the scout's
   "uncertain until lanes run" caveat; neither is a vanilla drop-in.

5. **logit_read on the weak lanes is conf-dead, not just wrong.**
   qwen3.5-0.8b produces a near-uniform letter distribution (conf mean **0.000**)
   → chance. jev-style-2b never crosses conf 0.7 (mean 0.19). Neither can drive a
   route gate unaided.

6. **d1's t3 losses land exactly where the NOTES predicted** — pooled Q4 misses
   cluster on `decide_extract → filesys_glob` (10), `decide_extract →
   web_search` (8), `filesys_grep → web_fetch` (6): the extract↔glob,
   extract↔web_search, and grep↔web boundary families. The t2 misses are
   dispersed and low-confidence (conf mean 0.56), i.e. the model hedges rather
   than confidently errs.

7. **The probe is deterministic.** Every lane returns bit-identical accuracies
   across run1/run2 (same numerators, 8 op classes); only latency moves
   (thermal / GPU-offload variance, largest on the qwen35 lanes). Repeat runs
   buy latency confidence, not accuracy confidence.

## Conf-gate detail (contenders, pooled)

| lane | band | conf≥0.7 cov / acc | conf≥0.9 cov / acc |
|---|---|---|---|
| d1-3B-Q4 | t1 | 66% / 98% | 38% / 96% |
| d1-3B-Q4 | t2 | 39% / 75% | 6% / 100% |
| d1-3B-Q4 | t3 | 55% / 77% | 24% / 100% |
| Qwen3.5-4B | t1 | 27% / 100% | — |
| Qwen3.5-4B | t2 | 3% / 100% | — |
| Qwen3.5-4B | t3 | 18% / 100% | — |

All three `Qwen3.5-4B` rows here are **SUPERSEDED** — those runs were answered by
a stale d1-3B-Q4 server (see CORRECTION); the conf-gate numbers are d1's, not
Qwen's.

## Platform note (carried from the serving RCA)

- Every qwen35-family lane above (0.8b, 2b, 4B) runs **CPU-class in server
  contexts** — 73–121 t/s prefill vs 585–919 t/s from `llama-bench` on the same
  graph/build. Signature: the scheduler reserves only 37–82 MiB on Vulkan0 for
  the hybrid graph vs bench's 252–513 MiB. Not fixable at arg/env level on these
  builds; 30+ variants ruled out (`docs/serving-fix-qwen35.md`). **Latency
  comparisons across qwen35 lanes are not recipe-stable** until a newer upstream
  build (gated-deltaNet-Vulkan merge) is tested.
- d1-3B (non-hybrid) offloads normally → the ~150 ms figures are trustworthy.

## Next-session recommendations

1. **[SUPERSEDED — see CORRECTION] Re-frame the pick.** On the stratified set d1-3B-Q4 and Qwen3.5-4B-instruct
   are tied at ~74–75%. Decide the router on **coverage-vs-precision under the
   gate** (finding 3), not on headline accuracy. If the gate can absorb ask-backs,
   Qwen3.5-4B's 100%-above-0.7 precision is attractive; if hops are expensive,
   d1's 56% commit rate wins.
2. **Test a newer upstream build** (gated-deltaNet-Vulkan merge) before treating
   any qwen35 in-process latency as stable; re-run the 4B lane and see if it
   reaches GPU speed.
3. **Only pursue Intern-Decision-4B / Kev-4B with a native-schema adapter** —
   their letter-prompt numbers are not a fair test of the models.
4. Re-running the sweep is now cheap and reproducible: `bench/band_summary.py`.

## Artifacts

- Results: `bench/results/dell/v2-*.json` (5 lanes ×2), `bench/results/dell/q4-*-v2*.json` (3 scout lanes ×2)
- Corrected results + `/props` evidence (CORRECTION): `bench/results/dell/v2-*-props-asserted*.json`, `bench/results/dell/props/`
- Analyzer: `bench/band_summary.py` (per-band join + latency)
- Dataset: `tests/data/utterances_v2.json` + `tests/data/utterances_v2_NOTES.md`
- Prior reports: `docs/research-model-sweep-q4.md`, `docs/serving-fix-qwen35.md`, `docs/testing.md`

### Caveats

- Lane status files (`/tmp/sv2_*.status`) and probe logs were eaten by the AV
  (known §7.2 pattern); `serving_stable` couldn't be read from them post-hoc.
  Result integrity is the fallback evidence: every file has 150 rows, 0 malformed,
  0 abstains (except heuristic), and a populated `/props` model block.
- Cross-workstream numbers (d1/heuristic/jev/qwen0.8b swept in dell-sweep-v2;
  the three q4 lanes in the scout) ran in separate sessions on the same box,
  build, and dataset.
