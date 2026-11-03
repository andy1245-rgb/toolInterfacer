# Changelog

All notable changes to this project are recorded here. The format is
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), the versioning is
[Semantic Versioning](https://semver.org/spec/v2.0.0.html), and every entry traces to a
commit in `git log` or to a document under `docs/`.

## [0.1.0] — 2026-11-03

First public release: 104 commits from 2026-10-17 to 2026-11-03, 715 tests and 349
subtests, 62 modules in 16 packages, no runtime dependencies.

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
  or ~1 s per decision. Caveat: the native arm is scripted rather than a real agent, n =
  9 tasks is a shape survey and not a distribution, and `pass_ms` is an assumption
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
- **The suite needs nothing.** 715 tests and 349 subtests with no network, no model and no
  CLI: the choice decision is the `fixture` backend and the subprocess ops run against
  `tests/data/fake_cli.py`.
