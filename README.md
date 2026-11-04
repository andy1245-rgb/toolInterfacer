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

<details>
<summary>The whole run, as one picture (a recording of <code>bench/demo.py</code>, offline)</summary>

<img src="demo/toolinterfacer.svg" width="760" alt="A recorded bench/demo.py session: a header naming the service port, nine registered ops, the fixture decision backend and a temp directory, then three scenes. Scene 1, intent 'read notes.md', is an L0 passthrough: one filesys_read hop at 0.1 ms and a status line reading 'confidence absent (nothing was measured)', 1 hop in 26 ms wall. Scene 2, intent 'read the notes file', is L1 slot fill: filesys_glob lists 3 candidates, decide_pick picks notes.md at confidence 0.93, filesys_read returns its text — 3 hops in 22 ms wall. Scene 3, intent 'research lora rank for a 3b router', is the L2 template: decide_extract makes 6 queries, web_search returns 7 hits, decide_pick keeps 4, four gated web_fetch hops each verify one page, digest_assemble assembles the digest — 8 hops in 251 ms wall. A footer says the model made one call per scene and got every hop as evidence.">

</details>

## The problem

An agent that reads, searches, fetches and re-checks spends most of its round trips moving
tokens rather than information. Every tool call is a full model pass — the model has to
remember the whole tool catalogue, emit a schema-valid argument object, and then read back
a result it did not need to see to decide the next step.

The work being done in those calls is mostly *decision*, not generation: which of these
nine operations, over which of these six candidate files, with which slot values. That is a
classification problem over a short state, and it does not need a frontier model.

toolInterfacer puts a decision model there instead. `useTool(intent, params)` is the only
thing the LLM calls; the service classifies the intent, resolves a step graph, fills slots
against real candidates, and runs the operations. The LLM pays one round trip for a whole
pipeline. A decision costs ~0.15 s on the hardware that served the sweep below (p50 over
the 150-row set), and zero generated tokens.

**Whether that is cheaper is a trade, and it is measured.** Nine fixed tasks, two arms, the
real service (`n=9` tasks): the layer removes **2 of 14 round trips (14.3%)** — that part is
counted, not modelled — and at an assumed 2 s agent pass it is **7.5 s slower**, because its
10 decisions cost 10.56 s against 0.14 s of actual tool work. The two arms cross at
**~5.8 s per model pass** (equivalently ~1.06 s per decision *at that pass cost*; at a 2 s
pass the break-even decision is ~0.30 s). So the layer pays for compound intents on slow
models with fast local decisions, and is pure overhead on single-call shapes. A re-run on a
box with no reachable decision backend does not reproduce the round-trip saving or any
wall-time figure — see the reproduction status in the source. Full table, method and
caveats: [`docs/roundtrips-2026-11-03.md`](docs/roundtrips-2026-11-03.md), raw
`bench/results/roundtrips-2026-11-03.json`.

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
                         ├─ classical      TF-IDF + linear SVM, no model server
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
| L0 passthrough (caller fully specifies the op) | works — validates params, executes, returns with no decision hop (1–18 ms measured, `docs/roundtrips-2026-11-03.md`) |
| L1 disk slot-fill (glob candidates → `decide_pick` → read) | works end to end model-free via `bench/replay.py` |
| L2 templates: `research` (queries → search → pick → fetch+verify loop → optional reddit cross-check → digest) and `reddit_digest` | `research` runs end to end **offline** via `bench/demo.py` — scene 3 pins the decisions to the fixture backend and both web CLIs to `tests/data/fake_cli.py`, and assembles a digest over the four stubbed sources. `bench/replay.py` runs the same plan against the *real* CLIs, so it needs them configured and takes minutes, not milliseconds. `reddit_digest` needs the reddit project (see gaps) |
| Nine registered ops | `decide_extract`, `decide_pick`, `digest_assemble`, `filesys_glob`, `filesys_grep`, `filesys_read`, `reddit_digest`, `web_fetch`, `web_search` |
| Decide backends and the pre-gate → cache → backend fast path | works — `heuristic_micro` pre-gate and the 256-entry decision cache are on by default |
| `pair` (local, escalating to hosted) and `openrouter` | implemented and measured by the author; running them needs a local llama.cpp server, and the hosted stage needs `TI_OPENROUTER_API_KEY` — [`docs/decision-pair-vs-hosted.md`](docs/decision-pair-vs-hosted.md) |
| Bounded decision state (structural, not conventional) | works — `session_digest.sanitize` projects each digest turn onto `{role, summary, mentions}` and each antecedent onto `{phrase, resolved}`, keeps the last 8 of each and clips every string that survives (240 chars per turn, 6 mentions, 1024 per path, `cwd` included), so a key the shape does not list is dropped rather than forwarded; `context.packer` is the backstop, stubbing the digest if the decide state still exceeds `conf.context.state_max_tokens` |
| pi extension: `useTool`, `/ti` command, service spawn on session start | complete — `npm ci && npx tsc --noEmit` exits 0, and `contract-check` passes every schema case plus the live leg against a stub service on a loopback port; the case counts and the case-by-case enumeration live in [`pi-extension/README.md`](pi-extension/README.md) |
| Test suite | **852 tests, 586 subtests**, no network, no model. This row is the only place the current count is written — every other document links here, tells you to run `pytest`, or states a floor (`700+`), so a track that adds tests edits one line |

### Honest gaps

These are not wired, and the README would be lying if it did not say so:

- **Intent classification is keyword rules, not the decision model.**
  `planner/inspector.py` picks L0/L1/L2 from ordered keyword frames — an explicit op, a
  reddit route, a named web channel, a published source, a research verb gated on nothing
  local — which is wider than the two words (`"research"`, `"compare"`) it shipped with
  and is still a rule list. The decision model is used *inside* a plan — candidate
  selection, slot fill, verification — not to choose the plan. Spec §2 calls the
  model-planned version **L3** and marks it *not built*, which is what this is; the code's
  **P2** label for that work is not the spec's phase of the same name
  ([`spec/1.0-spec.md`](spec/1.0-spec.md#divergences-from-this-spec), §8 row).
  `tests/test_route_traffic_shapes.py` is the table that holds the rules to the
  phrasings the hop log recorded — the shapes are the log's, the subjects on the
  seven off-disk rows are synthetic.
- **Ask-back works over HTTP, but only as a second call.** The transport has no channel
  mid-request, so `use_tool(..., ask_via_response=True)` returns `status: "need_input"`
  with the candidate list as answerable options, and the caller resolves it by calling
  again with `params.answer`. A direct caller that passes `on_ask` gets the callback form
  instead; `tests/test_loop.py` covers the callback form and
  `tests/test_live_e2e_regressions.py` the over-HTTP one (`need_input`, then the answer
  completing the call). `bench/live_e2e.py` carries the same ask-then-answer case against
  a live stack and does send the service token now, but its one recorded run predates
  that — every POST was refused `401` (`docs/live-e2e-2026-11-03.md`), so it is no
  evidence for this line yet.
- **`reddit_digest` cannot run out of the box.** The op is written against the
  `reddit-research` CLI contract but that project is not vendored, and its parser was
  inferred from the skill's docs rather than observed. `docs/ops.md` §6.
- **`web_search` / `web_fetch` shell out to CLIs you must configure.**
  `ops.web_search.script`, `ops.web_search_provider.script` and `ops.web_fetch.script`
  ship **empty** — they name a third-party CLI's install path, which is different on
  every machine, so a filled-in default would only ever be right on its author's. Point
  them at any CLI that honours the contract in [`docs/ops.md`](docs/ops.md); until you
  do, the ops raise `CliMissingBinary` naming the key to set.
- **L3 (model-planned DAG) is a placeholder** — `service/ti/planner/free.py` is a docstring.
  **L4 (browser)** is a row in the spec's level table and nothing else: no adapter, no code
  path, and no reference to one anywhere in `service/`.
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
# {"status": "ok", "ops": ["decide_extract", ..., "web_search"], "conf": true,
#  "decide": {"choice": "pair", "noul": "fixture", "extract": "rule_extract"},
#  "cascade": false}
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

`ops.filesys.roots` defaults to `["~", ".", "/tmp"]` — your home directory, the
directory the service was started in, and the temp directory — so the example above works
from a clone wherever it lives. A run root outside all three answers `{"error":
"scope_refused", ...}` and needs `roots` widened through `TI_CONF_OVERLAY` (below).

A *failed run* is still a `200`: the body says `status: "error"`, because the request was
accepted and the work is what failed. Read `status`, not the status line. The whole
contract — `401` / `403` / `400` / `411` / `413` / `503`, and why a run-level error is not
a `4xx` — is in [`docs/ops.md` §9](docs/ops.md).

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
echo '{"decide": {"backends_in_use": {"choice": "fixture"}}}' > /tmp/ti-fixture.json
TI_CONF_OVERLAY=/tmp/ti-fixture.json python3 bench/replay.py \
  --intent "read the file we discussed"
```

It runs the real engine, the real ops and the real gates, and prints the chain:

```
== summary ==
status=done wall_ms=18.6 hops=3
  filesys_glob       2.5ms conf=1.00  path_list[359]  e.g. TODO.md
  decide_pick       15.9ms conf=0.95  picked_items[1]  e.g. README.md
  filesys_read       0.1ms conf=1.00  file_content :: # toolInterfacer
  decide-max conf=0.95 (gate=0.7)
```

Three hops, 18 ms, and the only decision — which candidate file the intent meant — came
back at 0.95 against a 0.7 gate. `path_list[359]` is the one number here that is a property
of the **checkout** rather than of the code: the L1 slot fill globs the files it walks from
the repo root, so the count tracks the tree and moves whenever a file is added or removed.
An earlier recording of this same block read `path_list[179] e.g. LICENSE`; each is right
for its own tree, and neither is a claim the suite asserts. The `ms` and wall figures are
the other way round — they move with load on the same tree, which is why the latency rows
further down come from a sweep and not from this replay. (The script truncates its JSON dump
at 4000 characters, so on a long result body the block above the summary is cut mid-string
and the summary is the part that stays parseable.)

`TI_CONF_OVERLAY` names a file holding a partial config, deep-merged over the defaults
(`TI_CONF` selects a whole alternate file). It is how the tests and the bench repoint
`ops.*` and `decide.*` without copying `conf/defaults.json`.

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

#### Logging is off until you ask

Neither runtime log writes by default. `hops.jsonl` (one record per executed hop, for
the dataset pipeline) is gated by `conf.hops_log.enabled` and lives under
`~/.local/state/ti/logs/`. Opt in by writing a partial config and pointing the service at
it — `TI_CONF_OVERLAY` takes a **path**, not a JSON string:

```sh
printf '{"hops_log": {"enabled": true}}' > /tmp/ti-hops.json
TI_CONF_OVERLAY=/tmp/ti-hops.json ti-serve
```

That file is merged over `conf/defaults.json` at startup, so nothing else has to be
copied. (Setting the variable to the JSON text itself does not parse it — it opens it as a
filename and fails with `FileNotFoundError`.) The extension's `live.jsonl` (one record per
`useTool` call) is off until `/ti capture on`, which persists to
`~/.local/state/ti/capture.json` — the toggle survives a restart, and `/ti capture off`
stays off. What a record holds once enabled: the utterance, the argument *names*, sizes
and confidences — no working directory, no argument values, no artifact text, no error
messages. Both files are created `0600` in a `0700` directory, and a wider one is narrowed
rather than refused. Why: [`docs/logging-2026-11-04.md`](docs/logging-2026-11-04.md).

## Tests

```sh
python3 -m pytest tests/ -q
```

The last line prints the count, and the [Test suite row](#what-works-today) above is the
only place it is written down — there is nothing to update here when the suite grows.

Runs from a fresh clone with nothing installed: each test file puts `service/` on
`sys.path` itself. No network, no model, no CLI — the choice decision is the `fixture`
backend (recorded lanes in `tests/data/lanes/` for the cascade tests) and the subprocess ops
run against `tests/data/fake_cli.py`, which reproduces all three CLIs' result conventions.
`python3 -m unittest discover -s tests -p 'test_*.py'` works too.

One test is a policy guard rather than a behaviour test:
`tests/test_no_personal_paths.py` fails the suite if a machine-personal absolute path
appears anywhere in source, config, tests or docs. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

CI (`.github/workflows/ci.yml`) runs the suite on Python 3.10 and 3.12, runs
`bench/run.py selftest` and `bench/demo.py` offline, typechecks the extension on Node 22
and 24, and builds the wheel. The wheel check asserts a *set*, not a list of names: every
`.py` and `.json` under `service/ti` has to be inside it, nothing else may be, and
`ti/conf/defaults.json` is asserted separately. It then installs that wheel with
`--no-index` into a clean venv and starts `ti-serve` for a `/health` check — 9 ops and
`conf: true`, which is `defaults.json` resolving out of the wheel.

## Benchmarks

`bench/` holds the drivers; recorded results are in `bench/results/`.

| script | what it does |
|---|---|
| `bench/demo.py` | three scenes, a hop list as the receipt; offline by default, `--live` for the real backend |
| `bench/run.py` | score any decide arm on a frozen set, and compare two reports (McNemar exact + a latency delta); `selftest` runs the whole path offline |
| `bench/replay.py` | runs one plan through the real engine — no decision model once `TI_CONF_OVERLAY` points the choice decision at `fixture` (see the Quickstart); a plan with no web op then makes no network call, but an L2 `research` plan still runs the real `web_search` / `web_fetch` CLIs. The script itself uses the configured backend |
| `bench/live_probe.py` | routing probe over the frozen utterance set against a live llama.cpp server; `--expect-model <gguf>` asserts `/props` before any probe traffic |
| `bench/model_guard.py` | the `/props` served-model assertion on its own |
| `bench/serve_lane.py`, `bench/lane.sh`, `bench/lanes.json` | argv-safe llama-server lane launch, serve → assert → probe → teardown |
| `bench/band_summary.py` | per-band accuracy join over `bench/results/` |

### Corpora (`tools/`)

Two stdlib-only pipelines that produce the training and evaluation data. The
*generation* steps are derived-data producers and write to `var/`, which is
git-ignored; `certify.py` is the exception, freezing the audited artefact to the
committed `data/distill/` so a trainer has something checkable to fit on.

| tool | what it does |
|---|---|
| `python3 -m tools.dataset {ingest,validate,split,export,stats}` | normalises the runtime hop logs plus the frozen utterance sets into one validated record schema, with a leakage-aware split and an SFT export |
| `python3 tools/distill/utterances.py` | generates band-labelled routing rows beyond the frozen 150, with a similarity check that refuses anything restating the eval set |
| `python3 tools/distill/label.py` | labels those rows with the hosted teacher, resumably, under a cost gate; refuses a row whose reply is not schema-valid |
| `python3 tools/distill/audit.py` / `stats.py` | the human-check sample, and per-band counts with the teacher's own calibration |
| `python3 tools/distill/certify.py` | audits the label store against both frozen eval sets and freezes the committed training set at `data/distill/`; see `docs/distill-corpus-certification-2026-11-03.md` |

The hop log is *not* a decision log — see `docs/dataset-status.md` for the measured
count (14 trainable records out of 1398 hops, all one label), which is why the fine-tune
corpus is distilled rather than harvested. It is also **opt-in**: off unless
`hops_log.enabled` is set, written to `~/.local/state/ti/logs/hops.jsonl` (the directory
`tools/dataset` reads by default), `0600` in a `0700` directory. Both logs and the
reasoning behind them are in [`docs/logging-2026-11-04.md`](docs/logging-2026-11-04.md).

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

An 8-way classifier trained on the distilled corpus, frozen, scored on the same frozen 150
(`n=150`, per band t1 76.6% / t2 54.8% / t3 67.3%):

| signal | accuracy | 95% CI | vs `d1_3B` (McNemar exact) |
|---|---:|---|---|
| `rules` — `bench/local_clf.py`'s cue table, fitted on nothing | **78.7%** | 72.3–85.3 | — |
| `d1_3B` — archived per-row run | **74.0%** | 66.5–81.3 | — |
| distilled classifier (`tools/distill/`) | 68.7% | 59.6–77.0 | 28 fixed, 36 broken, p = 0.38 |

No evidence it beats the 3B, and it loses to the cue table at p = 0.040, so it is **not
shipped**. What it does show is the complement: 79.2% on `decide_extract` where the 3B gets
25.0%, and 63.6% on `filesys_grep` where the 3B gets 36.4% — the two classes
`bench/results/p1-diagnose/FINDINGS.md` diagnoses as noun-routed. Its report, including why the corpus's own
validation split ranks candidates backwards, is
[`docs/distill-classifier-2026-11-04.md`](docs/distill-classifier-2026-11-04.md).

On a CPU-only box, d1-3B-Q4 costs ~1.7 s per routing decision and ~50 ms on a
decision-cache hit; prefill dominates, which is why `decide.max_candidates` defaults to 6.
[`docs/testing.md`](docs/testing.md).

**These numbers are from specific hardware and do not reproduce on CI.** The suite asserts
behaviour, never latency.

## Docs

Grouped by what you are trying to do, not alphabetically. **Every `.md` under [`docs/`](docs/)
is in one of these four tables**, plus the spec, the architecture overview and the release
runbook — and the last table is dated material, a snapshot of one tree, not reference you
can read as current.

### The design of record

| file | contents |
|---|---|
| [`spec/1.0-spec.md`](spec/1.0-spec.md) | the agreed design: levels, tool contract, ask-back policy, the decision-model wall, phases — and a closing *Divergences* section listing where the code has since moved away from it |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | the overview: why the layer exists, the process view, the life of one `useTool` call, the two plugin walls, the bounded-context argument, the security surface. Read it before `docs/architecture.md` |
| [`docs/architecture.md`](docs/architecture.md) | the per-module contract: what each module owns and explicitly does not, so a change to a behaviour lands in the one module that owns it |
| [`docs/ops.md`](docs/ops.md) | the CLI contract the three subprocess ops rely on — invocation, locating the result, exit codes, op shapes — and the mismatches found against the live CLIs; §9 is the HTTP status contract, including why a run that fails is served `200` |
| [`docs/confidence-sources.md`](docs/confidence-sources.md) | every place a `confidence` is produced, each classified **measured** / **derived** / **asserted**, and the rule that came out of it: reported only when a hop measured one, the minimum over those, absent otherwise — the run-level `1.0` is `None` now |

### Running, testing, shipping

| file | contents |
|---|---|
| [`docs/testing.md`](docs/testing.md) | where the suite and the benchmarks run (bench-host over ssh; this box is compile-only), the d1-3B launch recipe, the enforced placeholders-not-paths rule, the CPU-only stack recipe, and how to prove the wire rather than the engine |
| [`RELEASE.md`](RELEASE.md) | the release runbook: the pre-flight that must exit 0, the wheel built *and installed* into a clean venv before anything is tagged, TestPyPI before PyPI, and the rollback order |

### What was measured

| file | contents |
|---|---|
| [`docs/roundtrips-2026-11-03.md`](docs/roundtrips-2026-11-03.md) | whether `useTool` saves round trips: 2 of 14 (14.3%) removed and 7.5 s *slower* at a 2 s agent pass, because 10 decisions cost 10.56 s against 138 ms of tool work; the crossover is ~5.8 s per pass (~0.30 s per decision), and the later re-run on a box with no reachable backend does not reproduce the saving |
| [`docs/decision-pair-vs-hosted.md`](docs/decision-pair-vs-hosted.md) | `pair` vs `openrouter` vs `d1_systemone` over 30 interleaved utterances: 3118 / 827 / 2261 ms p50 at 27, 28, 22 of 30 correct, McNemar p = 1.000 — and the decision that `pair` ships anyway, because hosted-only hard-fails without a key while the pair degrades to local |
| [`docs/model-sweep-consolidated.md`](docs/model-sweep-consolidated.md) | the October sweep on one frozen 150-row set, per band: d1-3B-Q4_K_M at 74% overall (t1 81 / t2 71 / t3 67, p50 153 ms) against the 4B challengers, plus the 2026-10-18 correction where the "Qwen3.5-4B" lanes turned out never to have served Qwen3.5-4B |
| [`docs/research-model-sweep-q4.md`](docs/research-model-sweep-q4.md) | the 4B-class scout that fed those lanes: three sourced candidates, downloaded and benched through `logit_read`, scored against the 84% d1-3B bar |
| [`docs/serving-fix-qwen35.md`](docs/serving-fix-qwen35.md) | why the qwen35 lanes served CPU-class in-process — **unresolved**: the flags, the builds, RPC, the driver and the GGUFs are each ruled out by measurement, and the surviving lead is a 6–7× compute-buffer split between `llama-bench` and `llama-server` |
| [`docs/classical-baseline-2026-11-03.md`](docs/classical-baseline-2026-11-03.md) | the TF-IDF + linear-SVM baseline rebuilt in stdlib: the recorded 93.19% reproduces exactly — but it is a **different task** (382 rows, 18 smart-home classes, empty intersection with this repo's 8 ops), so only the 62.7% against d1-3B's 74.0% on this repo's own 150 rows says anything about `useTool` — verdict: not the default; as a front gate, 79.3% at 95 ms against d1 alone at 74.0% and 169 ms |
| [`docs/local-classifier-cascade-2026-11-03.md`](docs/local-classifier-cascade-2026-11-03.md) | the same 150 rows and 8 labels asked a different question — how much can a free local model take off a **hosted** 95.3% fallback: 61.3% grouped CV, every coverage above 2.7% lowers blended accuracy, and the per-row gate fails because local confidence barely lifts (61.3% → 86.7% at 10%) and is non-monotone while the 3B's goes 74.0% → 97.6% at 28% coverage; the per-class alternative is **not** established — the report's 84.7% was an oracle applied on the row's true label (`bench/local_clf.py:1085`), 92.8% [88.5–96.6] is the union ceiling above it, and a deployable per-class figure is being measured on another track |
| [`docs/vulkan-igpu-2026-11-03.md`](docs/vulkan-igpu-2026-11-03.md) | the Iris Xe iGPU against the CPU build on the routing workload: 2.72× slower per decision (8943.6 vs 3283.7 ms p50, n=50), 6.9× slower at prefill, 0 of 50 paired rows won — the idle GPU is not compute, because routing is prefill-dominated |
| [`docs/eval-harness-2026-11-04.md`](docs/eval-harness-2026-11-04.md) | **the measurement instrument**: group-aware splits, repeated seeds, cluster-bootstrap intervals, and one `Predictor` interface — then every local signal re-measured through it. Headline: the 150-row set has **no families** to hold out (63.3% of rows are atomic, so a family split moves accuracy by −0.4 to +3.7 points, in both directions), the set that does have families loses **23.5 points** when they are held out (92.1% → 68.6%), and the repo's 84.7% per-class figure is an **oracle** applied on the true label |
| [`docs/number-sweep-2026-11-04.md`](docs/number-sweep-2026-11-04.md) | the sweep after those corrections: every place in the tree where 84.7 / 93.19 / 66.23 / 62.7 / 74.0 appears, each one marked **fixed**, **clean** or **coincidence**, with the reason — plus the eleven places the same four patterns appear without those digits |
| [`docs/dataset-status.md`](docs/dataset-status.md) | the hop log measured as training data: 14 of 1398 records trainable and all 14 carrying the same label (15 of 5243 on the re-checked snapshot) — it is an execution trace, so the fine-tune corpus is distilled rather than harvested |
| [`docs/distill-corpus-certification-2026-11-03.md`](docs/distill-corpus-certification-2026-11-03.md) | the distilled corpus certified as training data: **412** rows labelled (not 267 — that was a snapshot of a run still going), **0** collisions with either frozen eval set across all pairs, 410 frozen at `data/distill/` after two near-duplicate exclusions, the teacher's answer rejected as a label because it gives opposite answers to rows one filename apart at 0.9 confidence, and what a fair evaluation may claim from a corpus whose t1 band is saturated |
| [`docs/cost-reduction-glm-flash.md`](docs/cost-reduction-glm-flash.md) | the hosted routing bill cut, measured rather than projected: 38 endpoints of one model id on the same 150 rows, `$0.0202 → $0.0053` per 1k decisions (−73.8%) at 141/150 → 145/150, by pinning the cheapest endpoint and making three prompt/answer changes — and why `sort: "price"` does not get there |

**Correction — 2026-11-04.** The claims this table made that Phase 1 retired, in one place:

- **84.7% was an oracle, not a result.** It is the per-class `lr`-or-`3B` map applied on each
  row's ground-truth label (`bench/local_clf.py:1085`), so no router can reach it; 92.8%
  [88.5–96.6] family-held-out is the ceiling above it. The deployable numbers on that set are
  61.3% (`lr`), 50.0% (`1-NN`) and 74.0% (`d1-3B`). A deployable per-class figure is being
  measured on another track and is not quoted here — *(it has since landed: 79.3%
  [73.6–84.8], n=150, in [`docs/router-policy-2026-11-04.md`](docs/router-policy-2026-11-04.md),
  whose row is at the end of the "What was measured" table above)*.
- **`routing_382` is a different task.** 18 smart-home classes (`timer_create`,
  `thermostat_set`, …), its own registry, **empty intersection** with this repo's 8 ops. Its
  93.19% / 66.23% are not comparable to `useTool`; only the 62.7% measured on this repo's 150
  rows is.
- **`utterances_v2` cannot measure its own optimism.** 63.3% of its rows are atomic, holding
  families out moves accuracy −0.4 to +3.7 points in both directions, and the reported 8-point
  "leak" recomputes to 0.0 across seeds. The generalisation penalty this model class shows on a
  set that *can* be family-split is **23.5 points** (92.1% → 68.6%).

One correction has now been resolved rather than restated: the **74.0% → 82.0%** gain from
rewriting two op descriptions was in-sample, and it was run out of sample on
`heldout_v3.json` — 415 rows the rewrite never saw — where the same two strings score
**80.5% → 77.3%** (16 fixed, 29 broken, p = 0.0725). **82.0% is retired as an
improvement** — [`docs/rewrite-transfer-2026-11-04.md`](docs/rewrite-transfer-2026-11-04.md)
and the dated correction at the end of
[`bench/results/p1-diagnose/FINDINGS.md`](bench/results/p1-diagnose/FINDINGS.md).

The tempting salvage — that the `filesys_grep` line alone transfers, at +17.3 points and
9 fixed / 0 broken — **does not survive review and is not claimed here.** Three
independent passes killed it: its nine "fixes" are **three authored frames**, and at the
family unit the sign test is p = **0.25** with the project's own cluster bootstrap
**[+0.0, +36.5]**; the ledger was kept by *label* when the change is to an *option*, so
three of the twenty-nine breaks were captured by the new grep text and booked under
`web_search` (a real ledger of 9 fixed / 3 broken); and on the **test side alone**, the
only slice held out by construction, it is 3 fixed / 0 broken at p = **0.25**. There is
also **no "grep alone" arm** — both arms carried all eight options with both edits live.
[`docs/adversarial-review-2026-11-04.md`](docs/adversarial-review-2026-11-04.md),
[`docs/review-apparatus-2026-11-04.md`](docs/review-apparatus-2026-11-04.md),
`bench/verify_claims.py`.

Full derivations and the per-signal tables: [`docs/eval-harness-2026-11-04.md`](docs/eval-harness-2026-11-04.md).
Each affected document carries its own dated, appended correction.

### Dated audits, probes and plans — 2026-11-03 and 2026-11-04

One tree, not reference material. These were written on 2026-11-03 and 2026-11-04 against
the tree of the day — most name the commit — and none is rewritten afterwards except by an
appended Resolution section.

| file | contents |
|---|---|
| [`docs/audit-2026-11-03.md`](docs/audit-2026-11-03.md) | [`spec/1.0-spec.md`](spec/1.0-spec.md) audited module by module on a branch off `1a7c8fe`: four blockers — a wheel shipping 2 of 53 modules, 66 `tsc` errors, an L2 template naming ops that do not exist, a `pair` default needing a model and a key — with a 16:25 Resolution section saying which are fixed and which are still the owner's call |
| [`docs/review-2026-11-03.md`](docs/review-2026-11-03.md) | five blockers B1–B5 with the proof for each — `confidence: 1.0` on a run that measured nothing, a one-entry probability vector read as `p_max` 1.0, an unauthenticated POST reading `/etc/hostname` — and a Resolution recording what each later fix did and did not cover |
| [`docs/review2-2026-11-03.md`](docs/review2-2026-11-03.md) | the same five re-checked after twelve repair merges (B2 fixed at run level only, B4 structural with two holes still open) plus four new blockers NB1–NB4, among them a ReDoS guard a bounded quantifier walks through and `ops.filesys.roots` shipping as `~` |
| [`docs/doc-review-2026-11-03.md`](docs/doc-review-2026-11-03.md) | the docs read as a stranger at `a105d50`: the suite count wrong in seven places (715/349 against a green 722/388) and a digest bound the README calls structural although `sanitize` does not apply it — alongside the claims that held up |
| [`docs/quickstart-stranger-2026-11-03.md`](docs/quickstart-stranger-2026-11-03.md) | the README followed literally from a fresh clone: five wrong turns to a first working `useTool` — the L0 example 403s unless the clone sits under the reader's home directory, and `ti-serve` dead-ends on a held port with no recovery hint |
| [`docs/live-e2e-2026-11-03.md`](docs/live-e2e-2026-11-03.md) | the *shipped* path probed over a real socket, `5/5` cases — HTTP → inspector → plan → decide wall → ops wall with d1-3B answering — and ten defects found and fixed, from a picked path read against the service's cwd to a gate block that degraded instead of asking |
| [`docs/pre-publish-audit.md`](docs/pre-publish-audit.md) | the gate before a public push, taken at `ca7ffec`: what a stranger can see, download and infer from the clone — history, secrets, licence, the tracked tree — with a Resolution appendix saying which findings were closed and which are still the owner's call |
| [`docs/roadmap-2026-11.md`](docs/roadmap-2026-11.md) | the plan for the overnight run and its closing Outcome: which publish blockers were closed and re-probed rather than asserted, and the three measurements that changed the plan — the classical baseline, the iGPU, the hop log |
| [`docs/assets-inventory.md`](docs/assets-inventory.md) | what the sibling projects already hold, surveyed against this repo's three needs: five ranked import candidates with concrete paths and a recommendation for each, the cactus/needle fine-tune that came back negative, and two things worth not importing |
| [`docs/router-policy-2026-11-04.md`](docs/router-policy-2026-11-04.md) | the per-class routing figure measured without the answer: 84.7% was an oracle applied on the true label, its deployable replacement is **79.3% [73.6–84.8], n=150**, family-grouped 5-fold × 5 seeds — separately better than the cheap expert (+12.3 pts) and *not* separable from the 3B (+5.3 pts, CI −1.4 to +12.3), with the cascade's accuracy/cost curve (81.6% at 38% escalation), the finding that a top-2 margin gate is no better than top-1 probability, and the 14 of 22 confident 3B misses a class-level policy recovers where no confidence gate can — *not* the same 79.3% as the classical front gate in the baseline row above: same digits, different device, 30% coverage against 44% |
| [`docs/consistency-review-2026-11-04.md`](docs/consistency-review-2026-11-04.md) | the consistency pass over the whole night's corrections: a contradiction matrix for every retired number (84.7 / 93.19 / 66.23 / 62.7 / 82.0 / 74.0 / the 8.0-point "leak" / the "17-point gap"), which of the conflicts are a dated body doing its job and which are a reader being misled, the reachability ledger (a correction 342 lines below the number it retires is not a correction), the figures the first sweep's grep could not see, and the pointers this pass added |
| [`docs/logging-2026-11-04.md`](docs/logging-2026-11-04.md) | what the tool records by default, asked of the consumer first: one reader takes two fields out of `hops.jsonl`, so `artifact_summary` (200 chars of a file that was read) and `params` (verbatim argument values) had no reader at all — the conclusion is opt-in capture, a reduced record, a `~`-rooted path and `0600`/`0700` modes that narrow rather than refuse, plus what that costs |
| [`docs/security-surface-2026-11-04.md`](docs/security-surface-2026-11-04.md) | what a token holder can reach, written down because nothing said it: a valid `X-TI-Token` authorises reading **any file the operator owns**, and with the shipped config that is enumerated from `confine.py` rather than argued — `~/.ssh` (every private key, and the document's own end-to-end request returns one), `~/.gnupg`, `~/.aws`, `~/.local/state/ti` (the token file that authorised the request), `~/.kube` and `~/.docker`, `~/.netrc` / `~/.git-credentials`, shell histories, browser and keyring stores — three body fields and a header, no guard bypassed. It then says why the default is what it is: `roots: ["~", ".", "/tmp"]` closed a real `403` on a clone outside `$HOME` and was documented as a portability fix, never re-read as an authority grant. Recommendation: **keep the roots, ship a short `deny_roots` over the credential subtrees no `useTool` intent needs**, and write the incompleteness in as the headline, because a denylist reduces exposure and does not close it — with what a tighter default would cost and §8's account of the review that measured the widening in only one direction |

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
