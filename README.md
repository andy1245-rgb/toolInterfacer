# toolInterfacer

[![CI](https://github.com/andy1245-rgb/toolInterfacer/actions/workflows/ci.yml/badge.svg)](https://github.com/andy1245-rgb/toolInterfacer/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Python 3.10 – 3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)

A decision-model tool-interfacing layer for the pi coding agent: the big LLM states an
**intent**, a small local model decides which tools to run and with what arguments, and the
service executes them.

```sh
git clone https://github.com/andy1245-rgb/toolInterfacer.git && cd toolInterfacer
python3 bench/demo.py            # offline: three scenes, no model, no network, no install
```

Three intents — one the caller fully specifies, one vague, one compound — each answered by a
single call, with the hop list printed as a receipt:

```
scene 2 — a vague intent, resolved against real files
  intent  'read the notes file'
  step            op                       ms   conf  artifact
  candidates      filesys_glob            0.4   1.00  path_list[3] e.g. server-out.log
  picked_file     decide_pick             7.5   0.93  picked_items[1] e.g. notes.md
  content         filesys_read            0.1   1.00  file_content :: # Project notes

  status done   confidence 0.93   3 hops in 13 ms wall
```

`--live` runs the same scenes against the configured decision model and the real CLIs.
[`ARCHITECTURE.md`](ARCHITECTURE.md) is how it works, [`TODO.md`](TODO.md) is what is next,
and the numbers are under [`docs/`](docs/) — including the ones that are bad news.

## The problem

An agent that reads, searches, fetches and re-checks spends most of its round trips moving
tokens rather than information. Every tool call is a full model pass — the model has to
remember the whole tool catalogue, emit a schema-valid argument object, and then read back
a result it did not need to see to decide the next step.

The work being done in those calls is mostly *decision*, not generation: which of these
eight operations, over which of these six candidate files, with which slot values. That is a
classification problem over a short state, and it does not need a frontier model.

toolInterfacer puts a decision model there instead. `useTool(intent, params)` is the only
thing the LLM calls; the service classifies the intent, resolves a step graph, fills slots
against real candidates, and runs the operations. The LLM pays one round trip for a whole
pipeline. A decision costs ~0.15 s on the hardware that served the sweep below, and zero
generated tokens.

**Whether that is cheaper is a trade, and it is measured.** Nine fixed tasks, two arms, the
real service: the layer removes **2 of 14 round trips (14.3%)** and is **7.5 s slower** on
the CPU-only box, because 10 decisions cost 10.6 s against 0.14 s of actual tool work. The
two arms cross at **~5.8 s per model pass** or **~1 s per decision** — so the layer pays for
compound intents on slow models with fast local decisions, and is pure overhead on
single-call shapes. Full table, method and caveats:
[`docs/roundtrips-2026-11-03.md`](docs/roundtrips-2026-11-03.md).

## Architecture

```
LLM (pi)
  │  useTool(intent, params?, mode?)
  ▼
pi extension (TypeScript)          pi-extension/
  │  POST /v1/useTool               one compact result: result, hops, assumptions, degraded
  ▼
decision service (Python, 127.0.0.1:1113)          service/ti/
  │
  ├─ planner.inspector   intent → level + template   (L0 / L1 / L2)
  ├─ planner.templates   the step graph              disk_lookup, research, reddit_digest
  ├─ core.runner         op → artifact → gate → next step, under a budget
  ├─ policy              gate arithmetic, ask-back, confidence
  │
  ├─ ops                 filesys_glob/read/grep, web_search, web_fetch,
  │                      reddit_digest, decide_pick, decide_extract
  └─ decide              the primitive wall: choice / noul / score / extract
                         ├─ pair           local, escalating to hosted under a confidence tau
                         ├─ cascade        the same idea, banded (implemented, off by default)
                         ├─ d1_systemone   llama.cpp POST /v1/systemone
                         ├─ logit_read     any local LLM, letter A–Z → top_logprobs
                         ├─ openrouter     a hosted model behind a provider pin
                         ├─ heuristic_micro a regex pre-gate that mostly abstains
                         ├─ rule_extract   deterministic, no model
                         └─ fixture        recorded answers, for tests and replay
```

Two rules hold the shape:

- **One primitive wall.** The runner and the templates address decisions as
  `choice` / `noul` / `score` / `extract` and never import a backend. Swapping the backend
  is one line in `service/ti/conf/defaults.json`.
- **Templates publish shape, config publishes degree.** No top-k, cap, gate threshold or
  timeout is hardcoded in a template; all of them are config keys.

Full notes: [`docs/architecture.md`](docs/architecture.md).

## What works today

Verified by running it, not by reading the spec:

| capability | state |
|---|---|
| HTTP service: `GET /health`, `POST /v1/useTool`, `POST /v1/plan` | works — see the quickstart below |
| L0 passthrough (caller fully specifies the op) | works — validates params, executes, returns in ~3 ms |
| L1 disk slot-fill (glob candidates → `decide_pick` → read) | works end to end model-free via `bench/replay.py` |
| L2 templates: `research` (queries → search → pick → fetch+verify loop → optional reddit cross-check → digest) and `reddit_digest` | `research` runs end to end model-free via `bench/replay.py` and returns a sourced digest; `reddit_digest` needs the reddit project (see gaps) |
| Nine registered ops | `decide_extract`, `decide_pick`, `digest_assemble`, `filesys_glob`, `filesys_grep`, `filesys_read`, `reddit_digest`, `web_fetch`, `web_search` |
| Decide backends and the pre-gate → cache → backend fast path | works — `heuristic_micro` pre-gate and the 256-entry decision cache are on by default |
| `pair` (local, escalating to hosted) and `openrouter` | implemented and measured by the author; running them needs a local llama.cpp server, and the hosted stage needs `TI_OPENROUTER_API_KEY` — [`docs/decision-pair-vs-hosted.md`](docs/decision-pair-vs-hosted.md) |
| Bounded decision state (structural, not conventional) | works — digest ≤ 8 turns × 240 chars, state ≤ `conf.context.state_max_tokens` |
| pi extension: `useTool`, `/ti` command, service spawn on session start | complete — `npm ci && npx tsc --noEmit` exits 0 and the contract check passes 11/11 against a stubbed transport; see [`pi-extension/README.md`](pi-extension/README.md) |
| Test suite | 662 tests, 349 subtests, no network, no model |

### Honest gaps

These are not wired, and the README would be lying if it did not say so:

- **Intent classification is keyword rules, not the decision model.**
  `planner/inspector.py` picks L0/L1/L2 from ordered keyword frames — an explicit op, a
  reddit route, a named web channel, a published source, a research verb gated on nothing
  local — which is wider than the two words (`"research"`, `"compare"`) it shipped with
  and is still a rule list. The decision model is used *inside* a plan — candidate
  selection, slot fill, verification — not to choose the plan. Spec §2 calls the routed
  version P2; `tests/test_route_real_traffic.py` is the table that holds the rules to the
  hop log.
- **Ask-back works over HTTP, but only as a second call.** The transport has no channel
  mid-request, so `use_tool(..., ask_via_response=True)` returns `status: "need_input"`
  with the candidate list as answerable options, and the caller resolves it by calling
  again with `params.answer`. A direct caller that passes `on_ask` gets the callback form
  instead; `tests/test_loop.py` and the live probe both exercise it.
- **`reddit_digest` cannot run out of the box.** The op is written against the
  `reddit-research` CLI contract but that project is not vendored, and its parser was
  inferred from the skill's docs rather than observed. `docs/ops.md` §6.
- **`web_search` / `web_fetch` shell out to CLIs you must configure.**
  `ops.web_search.script` and `ops.web_fetch.script` default to the author's
  `~/.pi/profiles/registry/...` skill scripts. Point them at any CLI that honours the
  contract in [`docs/ops.md`](docs/ops.md), or the ops raise `CliMissingBinary`.
- **L3 (model-planned DAG) is a placeholder** — `service/ti/planner/free.py` is a docstring.
  **L4 (browser)** is an unwired adapter slot.
- **Destructive tools stay native.** `bash` / `write` / `edit` are not routed through
  `useTool`; the v1 domain is read/glob/grep plus the web stack.

## Quickstart

Python 3.10+ is the only requirement. The service is stdlib-only — no dependencies.

```sh
git clone https://github.com/andy1245-rgb/toolInterfacer.git
cd toolInterfacer
python3 -m venv .venv && . .venv/bin/activate   # Debian/Ubuntu also need `python3-venv`
python3 -m pip install -e ".[dev]"
```

Run the service:

```sh
ti-serve            # or: PYTHONPATH=service python3 -m ti.http.api
```

It listens on `127.0.0.1:1113` (`conf.http.host` / `conf.http.port`). Check it:

```sh
curl -s http://127.0.0.1:1113/health
# {"status": "ok", "ops": ["decide_extract", ..., "web_search"], "conf": true}
```

### The service token

The service reads the operator's files on behalf of whoever asks, and anything on the box
can open a loopback socket — so every POST carries a shared token, checked before the body
is read. `/health` stays open: it is how the extension decides whether a service is already
running.

The token comes from `TI_HTTP_TOKEN` if it is set, else from the runtime file at
`http.token_file` (default `~/.local/state/ti/http.token`, mode 0600, created on first boot
if absent — and refused if group or other can read it). There is no "no token" mode: with
neither a usable environment token nor a usable file, the service refuses to start rather
than serving unauthenticated requests.

```sh
export TI_HTTP_TOKEN="$(cat ~/.local/state/ti/http.token)"
```

The extension resolves the same token in `pi-extension/src/token.ts` — environment first,
then the runtime file — and `bridge.ts` sends it as `X-TI-Token` on every POST. A service
the extension spawned is one it can authenticate to.

Run a real operation through it — L0 passthrough, no model involved:

```sh
curl -s -X POST http://127.0.0.1:1113/v1/useTool \
  -H "Content-Type: application/json" \
  -H "X-TI-Token: $TI_HTTP_TOKEN" \
  -d '{"intent":"list the pyproject file",
       "params":{"tool":"filesys_glob",
                 "args":{"pattern":"pyproject.toml","cwd":"."}}}'
# {"status": "done", "result": ["pyproject.toml"], "degraded": false,
#  "assumptions": [], "hops": [{"step": "L0", "op": "filesys_glob", "ms": 2.5, ...}]}
```

L0 may only name the ops in `http.allowed_ops` (the three `filesys_*` mirrors by default),
and every path they touch is confined to the run's cwd — the body's `digest.cwd`, which
must itself resolve inside `ops.filesys.roots`. `..`, an absolute path outside the run, a
symlink out of it, and a cwd outside the policy all come back `403`.

`ops.filesys.roots` defaults to `["~", "."]` — your home directory plus the directory
the service was started in — so the example above works from a clone wherever it lives. A
run root outside both answers `{"error": "scope_refused", ...}` and needs `roots` widened
through `TI_CONF_OVERLAY` (below).

Ask what a compound intent *would* do, without running it:

```sh
curl -s -X POST http://127.0.0.1:1113/v1/plan \
  -H "Content-Type: application/json" \
  -H "X-TI-Token: $TI_HTTP_TOKEN" \
  -d '{"intent":"research local inference latency"}'
# {"status": "plan", "meta": {"level": "L2", "template": "research"},
#  "steps": [{"id": "queries", "op": "decide_extract", ...}, ...]}
```

### A pipeline that actually decides something

The default backend is `pair`, which wants a local llama.cpp server. To watch the whole
L1 chain run with no model at all, point the choice decision at the fixture backend:

```sh
echo '{"decide": {"backends_in_use": {"choice": "fixture"}},
       "hops_log": {"enabled": false}}' > /tmp/ti-fixture.json
TI_CONF_OVERLAY=/tmp/ti-fixture.json python3 bench/replay.py \
  --intent "read the file we discussed"
```

It runs the real engine, the real ops and the real gates, and prints the chain:

```
== summary ==
status=done wall_ms=18.6 hops=3
  filesys_glob       2.5ms conf=1.00  path_list[179]  e.g. LICENSE
  decide_pick       15.9ms conf=0.95  picked_items[1]  e.g. README.md
  filesys_read       0.1ms conf=1.00  file_content :: # toolInterfacer
  decide-max conf=0.95 (gate=0.7)
```

Three hops, 18 ms, and the only decision — which of 179 candidate files the intent meant —
came back at 0.95 against a 0.7 gate. (The script truncates its JSON dump at 4000
characters, so on a long result body the block above the summary is cut mid-string and the
summary is the part that stays parseable.)

`TI_CONF_OVERLAY` deep-merges a partial config over the defaults (`TI_CONF` selects a whole
alternate file). It is how the tests and the bench repoint `ops.*` and `decide.*` without
copying `conf/defaults.json`.

### The pi extension

`pi-extension/` registers the `useTool` tool, bridges to the service over localhost HTTP,
and spawns the service on session start. pi executes the TypeScript directly via jiti, so
there is no build step. Discovery is a symlink into pi's extension directory:

```sh
ln -s "$(pwd)/pi-extension/src" ~/.pi/agent/extensions/toolinterfacer
```

| env var | effect |
|---|---|
| `TI_BASE_URL` | service base URL (default `http://127.0.0.1:1113`) |
| `TI_SERVICE_DIR` | cwd for the spawned service (default: the repo root) |
| `TI_HTTP_TOKEN` | the service token a POST must carry (inherited by the spawned service) |
| `TI_HTTP_TOKEN_FILE` | where the runtime token lives (default `~/.local/state/ti/http.token`) |
| `TI_PYTHON` | interpreter for the spawn (default `python3`) |
| `TI_DEV=1` | never spawn; print the exact command line instead, and trailer every result with the raw response |

Its own tooling (`tsc --noEmit`, `node scripts/contract-check.ts`) and the full contract
notes are in [`pi-extension/README.md`](pi-extension/README.md).

## Tests

```sh
python3 -m pytest tests/ -q
# 662 passed, 349 subtests passed
```

Runs from a fresh clone with nothing installed: each test file puts `service/` on
`sys.path` itself. No network, no model, no CLI — the choice decision is the `fixture`
backend (recorded lanes in `tests/data/lanes/` for the cascade tests) and the subprocess ops
run against `tests/data/fake_cli.py`, which reproduces all three CLIs' result conventions.
`python3 -m unittest discover -s tests -p 'test_*.py'` works too.

One test is a policy guard rather than a behaviour test:
`tests/test_no_personal_paths.py` fails the suite if a machine-personal absolute path
appears anywhere in source, config, tests or docs. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

CI runs the suite on Python 3.10 and 3.12 and asserts the built wheel contains every
subpackage.

## Benchmarks

`bench/` holds the drivers; recorded results are in `bench/results/`.

| script | what it does |
|---|---|
| `bench/demo.py` | three scenes, a hop list as the receipt; offline by default, `--live` for the real backend |
| `bench/run.py` | score any decide arm on a frozen set, and compare two reports (McNemar exact + a latency delta); `selftest` runs the whole path offline |
| `bench/replay.py` | runs one plan through the real engine — zero network once `TI_CONF_OVERLAY` points the choice decision at `fixture` (see the Quickstart); the script itself uses the configured backend |
| `bench/live_probe.py` | routing probe over the frozen utterance set against a live llama.cpp server; `--expect-model <gguf>` asserts `/props` before any probe traffic |
| `bench/model_guard.py` | the `/props` served-model assertion on its own |
| `bench/serve_lane.py`, `bench/lane.sh`, `bench/lanes.json` | argv-safe llama-server lane launch, serve → assert → probe → teardown |
| `bench/band_summary.py` | per-band accuracy join over `bench/results/` |

### Corpora (`tools/`)

Two stdlib-only pipelines that produce the training and evaluation data. Both are
derived-data producers: their output goes to `var/`, which is git-ignored.

| tool | what it does |
|---|---|
| `python3 -m tools.dataset {ingest,validate,split,export,stats}` | normalises the runtime hop logs plus the frozen utterance sets into one validated record schema, with a leakage-aware split and an SFT export |
| `python3 tools/distill/utterances.py` | generates band-labelled routing rows beyond the frozen 150, with a similarity check that refuses anything restating the eval set |
| `python3 tools/distill/label.py` | labels those rows with the hosted teacher, resumably, under a cost gate; refuses a row whose reply is not schema-valid |
| `python3 tools/distill/audit.py` / `stats.py` | the human-check sample, and per-band counts with the teacher's own calibration |

The hop log is *not* a decision log — see `docs/dataset-status.md` for the measured
count (14 trainable records out of 1398 hops, all one label), which is why the fine-tune
corpus is distilled rather than harvested.

### Measured, with their n

Routing accuracy on the frozen 150-row stratified set (`tests/data/utterances_v2.json`,
8 op classes, chance 12.5%), d1-3B-Q4_K_M served through `POST /v1/systemone` on a machine
with an Intel Arc 140V iGPU at `-ngl 99`, two runs pooled:

| band | accuracy |
|---|---|
| t1 unambiguous (64 rows) | 81.2% |
| t2 veiled (31 rows) | 71.0% |
| t3 boundary pairs (55 rows) | 67.3% |
| **overall (150 rows)** | **74.0%** |

p50 ≈ 0.15–0.16 s per decision across the two pooled runs. Source:
[`docs/model-sweep-consolidated.md`](docs/model-sweep-consolidated.md) — which also carries
the correction where one lane was found to have been answered by a stale server, and the
guard that came out of it.

Choice of decision layer, 30 utterances × 3 arms interleaved with a warm-up discarded per
arm (`n=30`, so these are bounds, not verdicts):

| arm | p50 | p90 | correct |
|---|---:|---:|---:|
| `pair` (τ 0.90) | 3118 ms | 4768 ms | 27/30 (90.0%) |
| `openrouter` | 827 ms | 2021 ms | 28/30 (93.3%) |
| `d1_systemone` | 2261 ms | 2901 ms | 22/30 (73.3%) |

McNemar pair vs hosted: p = 1.000 — no detectable accuracy difference, and 73% of the
pair's answers *are* the hosted answer. The shipped default stays `pair` anyway, because
hosted-only hard-fails when `TI_OPENROUTER_API_KEY` is absent while the pair degrades to
local. Reasoning and failure-path timings:
[`docs/decision-pair-vs-hosted.md`](docs/decision-pair-vs-hosted.md).

On a CPU-only box, d1-3B-Q4 costs ~1.7 s per routing decision and ~50 ms on a
decision-cache hit; prefill dominates, which is why `decide.max_candidates` defaults to 6.
[`docs/testing.md`](docs/testing.md).

**These numbers are from specific hardware and do not reproduce on CI.** The suite asserts
behaviour, never latency.

## Docs

| file | contents |
|---|---|
| [`spec/1.0-spec.md`](spec/1.0-spec.md) | the design of record: levels, tool contract, ask-back policy, phases |
| [`docs/architecture.md`](docs/architecture.md) | process view, module responsibilities, the two plugin walls, the bounded-state guarantee |
| [`docs/ops.md`](docs/ops.md) | the CLI contract the subprocess ops rely on, and the mismatches found against live CLIs |
| [`docs/testing.md`](docs/testing.md) | where tests run, serving recipes, the portability rule, latency knobs |
| [`docs/decision-pair-vs-hosted.md`](docs/decision-pair-vs-hosted.md) | pair vs hosted-only, measured |
| [`docs/model-sweep-consolidated.md`](docs/model-sweep-consolidated.md) | the backend sweep, per band, with its correction |
| [`docs/serving-fix-qwen35.md`](docs/serving-fix-qwen35.md) | why qwen35 lanes serve CPU-class in server contexts |
| [`docs/research-model-sweep-q4.md`](docs/research-model-sweep-q4.md) | the 4B-class model scout |

## Layout

```
service/ti/        Python decision service (stdlib only)
pi-extension/      TypeScript pi extension: useTool, /ti, service lifecycle
spec/              the written specification
docs/              architecture, ops contract, testing, measured sweeps
tests/             fixture-player tests: full engine, no model, no network
bench/             benchmark drivers, lane launch, recorded results
```

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) — tests, the portability rule, the commit style.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
