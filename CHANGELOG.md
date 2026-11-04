# Changelog

All notable changes to this project are recorded here. The format is
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), the versioning is
[Semantic Versioning](https://semver.org/spec/v2.0.0.html), and every entry traces to a
commit in `git log` or to a document under `docs/`.

## [0.1.0] — 2026-11-03

*(Two claims in the `### Measured` section below were retired on 2026-11-04; read
[Corrections — 2026-11-04](#corrections--2026-11-04) with them.)*

First public release: 104 commits from 2026-10-17 to 2026-11-03, a suite of 700+ tests, 62
modules in 16 packages, no runtime dependencies.

*(Test counts in this file are floors, not readings. The current one is on the README's
Test suite row, which is the only place it is written down.)*

### Added

- **The decision service** (`service/ti/`) — `GET /health`, `POST /v1/useTool` and
  `POST /v1/plan`, on loopback, plus a `ti-serve` console script.
- **A planner** — `inspector` classifies an intent as L0 / L1 / L2, and three templates
  (`disk_lookup`, `research`, `reddit_digest`) emit a step graph the runner executes
  under a budget instead of a script.
- **Nine ops** — `filesys_glob`, `filesys_grep`, `filesys_read`, `web_search`,
  `web_fetch`, `reddit_digest`, `decide_pick`, `decide_extract`, `digest_assemble`.
- **Nine registered decision backends** — `pair` (local, escalating to hosted),
  `cascade`, `d1_systemone`, `logit_read`, `openrouter`, `heuristic_micro`,
  `rule_extract`, `classical` (TF-IDF + linear SVM, reimplemented in stdlib rather than
  adding scikit-learn), and `fixture` (recorded answers, which is what makes the suite
  and the demo offline).
- **Bounded decision state by construction**, a 256-entry decision cache, and a
  `heuristic_micro` pre-gate, all on by default.
- **A hops log** — every hop the service took, with the op, the latency and the
  confidence it reached, which is the receipt the product exists to produce.
- **The pi extension** (TypeScript) — `useTool`, a `/ti` command, service spawn on
  session start, shared-token resolution, and `contract-check` / `capture-check` scripts.
- **A benchmark harness** — `bench/run.py` (`list` / `run` / `compare` / `selftest`) over
  `bench/ti_bench`, plus drivers for round trips, the live end-to-end probe, model
  sweeps, the classical baseline, and cast-to-SVG rendering.
- **Dataset and distill pipelines** — ingest / validate / split / export of a decision
  corpus, and corpus generation with teacher labelling.
- **`bench/demo.py`** — three scenes (L0, L1, L2) that run offline, with the recorded
  asciinema cast and a renderer that survives a real terminal.
- **Documentation** — `ARCHITECTURE.md`, `CONTRIBUTING.md`, `TODO.md`, and `docs/`: an
  audit against the spec, two independent reviews, a first-contact quickstart, an ops
  note, and the measurement reports, including the ones that are bad news.
- **CI** — pytest on 3.10 and 3.12, extension typecheck on Node 22 and 24, the contract
  and capture checks, the offline benchmark harness and demo, and a wheel that is built,
  inspected, installed into a clean venv and driven to `/health`.

### Changed

- `choice` ships as **`pair`** — a local decision with a hosted fallback behind it —
  after measuring it against hosted-only.
- The service moved from `:8090` to **`:1113`**.
- **One confidence scale** (p_max) everywhere, and `decide.cascade.enabled` is
  authoritative for the choice kind rather than inferable from a neighbour.
- The candidate payload and the candidate descriptions that reach the model are
  **bounded**, and the pre-gate plus cache sit in front of every backend.
- Every backend registers **regardless of import order**.
- `ops.filesys.roots` ships as `["~", ".", "/tmp"]`, so a clone runs wherever it lives
  and a session started in a temp directory works at all.
- The repository is **portable**: machine-personal absolute paths were removed and
  `tests/test_no_personal_paths.py` fails the suite if one comes back.

### Fixed

- A **CRLF shebang** is detected and the script is run under the service interpreter,
  instead of the kernel looking for an interpreter named `python3\r` and exiting 127
  with no artifact — a failure that read as "every query failed" (`docs/ops.md` §7).
- A run that measured nothing no longer reports **`confidence: 1.0`**.
- `/health` names the **backend that answered**, so a probe cannot mistake a recorded
  fixture for the model — the guard that came out of finding a published lane which had
  been answered by a stale server.
- The HTTP surface **authenticates every POST, confines the run root, and bounds the
  body**; a nested-quantifier regex can no longer wedge a worker, including
  `(a+){10}b`, which the first version of that rule let through at 19.9 s on a
  36-character line.
- **`params.path` reached a `stat` before passing the file policy.** It is confined
  first, because a `stat` on a dead mount blocks in uninterruptible sleep and nothing in
  the process can stop it.
- A **picked relative path is read against the caller's cwd**, not the service's.
- A **gate block over HTTP degrades instead of dropping the request**, and a slot-gate
  block returns the best guess instead of an empty result.
- The **L2 research template names ops that exist**, its extractor sees the intent,
  `params.answer` closes the ask-back loop, the question no longer offers the rejected
  pick as its only option, a URL candidate's option id is a string rather than a dict
  repr, and an empty candidate list no longer blames the decide backend — ten defects
  in all, found by the live probe and listed in `docs/live-e2e-2026-11-03.md`.
- **Runner-level errors are forwarded** into the `useTool` response instead of being
  swallowed into a bare status.
- The **extension declares its SDK** and `npx tsc --noEmit` is clean.
- The demo's fixture CLI writes into the demo's temp dir instead of leaving an artifact
  in the repo, and `tests` skip a `node_modules` wherever it appears, not only at the
  root.

### Measured

- **Round trips — the headline, and its condition.** `useTool` removed 2 of 14 round
  trips (14.3%) across nine fixed tasks and was **7.5 s slower**, because 10 decisions
  cost 10.6 s against 0.14 s of tool work; the two arms cross at ~5.8 s per model pass
  (equivalently ~1 s per decision *at that pass cost* — at a 2 s pass the break-even is
  ~0.3 s). Caveat: the native arm is scripted rather than a real agent, n = 9 tasks is a
  shape survey and not a distribution, and `pass_ms` is an assumption
  (`docs/roundtrips-2026-11-03.md`).
- **Routing accuracy — 74.0%** on the frozen 150-row stratified set (t1 81.2%, t2 71.0%,
  t3 67.3%; 8-way, chance 12.5%) with p50 ≈ 0.15 s per decision. Caveat: two runs pooled,
  and one lane in that report was later found to have been answered by a stale server and
  is marked superseded rather than removed (`docs/model-sweep-consolidated.md`).
- **The classical baseline does not replace the 3B yet.** TF-IDF + linear SVM reproduces
  the sibling project's 93.19% to the row at 0.04–0.14 ms p50, but scores **62.7%** on
  this repo's own v2 set against d1-3B's 74.0%; as a front gate it measured **79.3% at
  95 ms** against 74.0% at 169 ms. Caveat: a different task, 16 rows fixed against 8
  broken — a direction worth chasing, not a result
  (`docs/classical-baseline-2026-11-03.md`).
- **The idle iGPU is not free compute.** The Vulkan path was **2.72× slower** per decision
  (p50 8943.6 ms vs 3283.7 ms, n = 50) and 6.9× slower at prefill, losing all 50 paired
  rows. Caveat: one power-limited iGPU with no cooperative-matrix support, on a
  prefill-dominated workload — the result is about this class of part, not about GPUs
  (`docs/vulkan-igpu-2026-11-03.md`).
- **The live hop log is not a training set.** 1,398 logged lines yield **14** trainable
  choice records, all 14 carrying the same label (`filesys_glob`), so a model trained on
  it learns one answer and nothing else (`docs/dataset-status.md`).
- **The shipped path works end to end:** 5 of 5 contract cases passed against the real
  service and a real d1-3B over `/v1/systemone`, having found 10 defects on the way.
  Caveat: `openrouter` had no API key, so every escalation was exercised through its
  degraded path rather than a live hosted call (`docs/live-e2e-2026-11-03.md`).
- **The suite needs nothing.** 700+ tests with no network, no model and no
  CLI: the choice decision is the `fixture` backend and the subprocess ops run against
  `tests/data/fake_cli.py`.

#### Reproduction status — re-run 2026-11-03 15:39–15:52 EST, branch `p2-numbers` @ `ca7ffec`

Every figure above was re-driven against the artifacts it cites. What still reproduces, and
what a fresh worktree cannot reproduce, is recorded per document: the round-trips report
carries a reproduction table, so does the classical baseline, the live probe, and the
dataset snapshot. In one place:

| claim | status | why |
|---|---|---|
| round trips 2/14 | **not reproducible without a decision backend** | arm B needs `d1_systemone`; with none reachable every L1/L2 task answers in one round trip and is wrong (B 3/9) |
| routing accuracy 74.0% | **reproduces from the recorded per-row data** (111/150 × 2, bands 81.2 / 71.0 / 67.3) | re-measuring needs d1-3B and a `llama-server`; the gitignored `models/` and `extern/` are outside a fresh worktree |
| classical 93.19% / 66.23% / 62.7% | **reproduces exactly**, fingerprint and all | but `python3 bench/classical_baseline.py` now *crashes*: 37 of the 40 files in `bench/results/dell/` are invalid JSON after the `C:\Users\<user>` redaction in `e523336`, so the documented command exits at its d1 load |
| iGPU 2.72× slower | **artifact only** | the gguf is on the box (`models/d1-3B-Q4_K_M.gguf`, 1.67 GB) and `vulkaninfo` still reports the ADL GT2, but no `build-vk/` exists, so re-measuring means rebuilding llama.cpp through the `glslc` shim first |
| hop log → 14 records | **moved with the log**: 5,243 lines now yield 15, all still `filesys_glob` | the snapshot is live, machine-local and gitignored; the conclusion holds |
| live 5/5 | **not reproducible**: `0/4`, HTTP 401 | `bench/live_e2e.py` sends no service token, and auth landed in `5dc537f` nine minutes after the recorded run |
| 679 tests / 349 subtests | **neither is current** | 679 was right at the release commit `1b4563b` and the README went on quoting it after the suite had moved; the count at HEAD is the README's Test suite row |

## Corrections — 2026-11-04

Four claims published in `0.1.0` were retired by the Phase 1 measurement pass. Nothing here is
silently edited: each affected document carries its own dated, appended correction, and the
released text above is left as it was published so a reader can see what was believed and when
it changed. Evidence for the first three is `docs/eval-harness-2026-11-04.md`; the fourth cites
`bench/results/p1-diagnose/FINDINGS.md` §3 and the same report for the split limit.

### Retired: an oracle quoted as a deployable accuracy

`docs/local-classifier-cascade-2026-11-03.md` reported **84.7%** for "per-class routing, decided
on training rows" against the 3B's 74.0%. It is an oracle: the per-class map is applied on each
row's ground-truth label (`bench/local_clf.py:1085`), so computing it requires already knowing
the answer being predicted. **84.7% was an oracle**, and no router can reach it. The map itself
is fitted honestly on training rows — the application is what consults the answer. The same
defect applies to the report's 84.0% static-table row (`bench/local_clf.py:1063`) and its 84.7%
pooled-OOF row (`:1097`), so that the two 84.7% figures were *identical* was arithmetic, not
replication.

The oracle's own bound — the union of `lr` and `d1` — is **92.8% [88.5–96.6] family-held-out**,
93.5% row-wise. The deployable numbers on that set are unchanged: `lr` 61.3%, `1-NN` 50.0%,
`d1-3B` 74.0%. **A deployable per-class figure is being measured on another track; no number is
quoted here for it and the report will carry a further correction when it lands.**

Corrected in: `docs/local-classifier-cascade-2026-11-03.md` (Corrections C1), `README.md`
("What was measured" row and the 2026-11-04 correction block).

### Retired: a different task presented as comparable

`routing_382` results — **93.19%** (references → 382 rows) and **66.23%** (template-grouped) —
were presented alongside this repo's numbers as if the same task at two split settings. They are
not. `routing_382` is a **smart-home assistant router**: 18 classes (`timer_create`,
`thermostat_set`, `shoppinglist_add`, `calendar_today`), its own 17-tool registry, its own
200-reference training corpus. **Its intersection with this repo's 8 ops is empty** (re-verified
2026-11-04). Only the classical baseline's **62.7%** on `utterances_v2` measures what `useTool`
does.

Corrected in: `docs/classical-baseline-2026-11-03.md` (Corrections C1),
`docs/assets-inventory.md` (Corrections C1–C2, including the "74.7% against 93.19%" comparison,
which pairs a 150-row figure with a 382-row one), `README.md`.

### Retired: a split that measured nothing

`utterances_v2` cannot support a generalisation claim. **63.3% of its rows (95 of 150) are
atomic** — they belong to no phrasing family — and the only grouping it declares is 55 t3 rows
in 11 boundary-pair families. Holding families out moves accuracy by **−0.4 to +3.7 points, in
both directions, inside the seed-to-seed spread**. The 8.0-point "leak" reported for this set
recomputes to **0.0 points** across 5 seeds. On `routing_382`, which *can* be family-split, the
same model class loses **23.5 points** (92.1% → 68.6%).

Corrected in: `docs/local-classifier-cascade-2026-11-03.md` (C2),
`docs/classical-baseline-2026-11-03.md` (C2), `docs/model-sweep-consolidated.md` (2026-11-04
correction), `README.md`.

### Retired: an in-sample improvement described as a finding about the model

`bench/results/p1-diagnose/FINDINGS.md` reports the 3B going **74.0% → 82.0%** by rewriting two
of eight op descriptions (18 fixed / 6 broken, exact McNemar p = 0.0227). The effect is real and
the mechanism is understood — the model routes on the object noun and not the operation — but
the two replacement strings were written by reading the errors of the same 150 rows they are
scored on. **It is in-sample and is an upper bound, not an estimate.** Another track is
validating it on a held-out set; this changelog does not pre-empt that result.

Corrected in: `bench/results/p1-diagnose/FINDINGS.md` (2026-11-04 correction).
