# ARCHITECTURE

How a `useTool` call becomes executed work. This is the overview; the per-module
contracts live in [`docs/architecture.md`](docs/architecture.md) and the normative design
in [`spec/1.0-spec.md`](spec/1.0-spec.md).

The short version: **an LLM states an intent; a small local model makes the decisions; the
service runs the tools; the LLM gets one answer and a receipt.**

## Why there is a layer at all

An agent reading, searching and re-checking spends most of its round trips moving tokens
rather than information. Each of those calls is a full model pass in which the model holds
the whole tool catalogue, emits a schema-valid argument object, and reads back a result it
did not need to see in order to decide the next step.

The work inside those calls is mostly *classification* — which of these nine operations,
over which of these six candidate files, with which slot values. A 3B model with a
good scaffold can do that in one forward pass with zero generated tokens. So the layer
moves the deciding down and leaves the LLM with the thing only it can do: saying what it
actually wants.

**Whether that is cheaper is a trade, and it is measured** — see
`docs/roundtrips-2026-11-03.md`. It pays when a model pass is slow and the local decision is
fast; it is pure overhead on single-call shapes. This document describes the mechanism, not
a claim about the economics.

## Process view

Three processes, two of them optional.

```
   pi (the agent)
     │  useTool(intent, params?, mode?)
     ▼
   pi extension (TypeScript)            pi-extension/
     │  POST /v1/useTool  + X-TI-Token
     ▼
   decision service (Python, 127.0.0.1:1113)      service/ti/
     │
     ├──▶ llama.cpp  /v1/systemone                 optional, the local decision model
     ├──▶ hosted model (OpenRouter)                optional, the escalation stage
     └──▶ CLI tools (web-search, web-fetch, …)     optional, one subprocess each
```

The extension registers exactly one tool and owns no decisions. Everything testable lives
in the Python service, which is why the whole benchmark suite runs without pi.

## The life of one call

`service/ti/service_entry.py::use_tool` is the only place that wires these together, in
this order:

1. **Ingress bound.** `context.session_digest.sanitize` projects each digest turn onto
   `{role, summary, mentions}` and each antecedent onto `{phrase, resolved}`, keeps the last
   8 of each, and clips every string that survives (240 chars per turn, 6 mentions, 1024 per
   path, the `cwd` too) — a key the shape does not list is dropped, not forwarded. Behind it,
   `context.packer.trim` replaces a digest still over `conf.context.state_max_tokens` with a
   stub. So a caller cannot inflate the prompt either way: `sanitize` makes their digest
   arrive thin, and if it were bypassed `trim` would discard the digest rather than bound it.
2. **Run root.** `ops.filesys.confine.set_root(cwd)` pins every path the run may touch, for
   the duration of this call only, and is torn down in a `finally`. Relative paths resolve
   in the *caller's* frame, never the service's.
3. **Classify the intent** — `planner.inspector` picks a level and a template from ordered
   keyword frames (see "The levels" below).
4. **Apply the template** — `planner.templates.*` produces a list of `Step`s. A template
   publishes *shape*: which op, in what order, with which gates. It contains no thresholds.
5. **Run the plan** — `core.runner.run_plan` walks the steps: execute the op → store its
   artifact → evaluate the gate → pick the next step, under a hop/ask/clock budget.
6. **Map to a response** — `status`, `result`, `confidence`, `hops`, `degraded`,
   `assumptions`, `wall_ms`, `meta`, and for a gated run `question`. There is no
   top-level `sources` field: a digest's sources live inside its own artifact, which
   arrives as `result`.

## The levels

| level | what decides | example | round trips |
|---|---|---|---|
| **L0** passthrough | the caller | `params.tool = filesys_read` | 1 |
| **L1** one hop | the planner + one `choice` | "read the notes file" → glob → pick → read | 1 |
| **L2** template | an authored step graph | "research X" → extract → search → pick → fetch → digest | 1 |
| **L3** model-planned DAG | the decision model composes the plan | *not built* (`planner/free.py`) | — |
| **L4** browser | an adapter slot | *not wired* | — |

L0 and L1 are why the layer is cheap: L0 skips the decision wall entirely, and an L1 listing
terminates on the candidate list rather than reading anything.

## The two plugin walls

This is the design decision that makes the project testable, and it is worth stating
plainly: **the pipeline never names a backend, and a template never names a threshold.**

### Wall 1 — `decide` (what makes a decision)

Everything that chooses goes through four primitives in `service/ti/decide/primitives.py`:

| primitive | question | returns |
|---|---|---|
| `choice(state, instructions, criteria)` | pick one of these labelled options | `choice`, `probabilities`, `confidence` |
| `noul(state, question)` | is this true of that artifact | `p` |
| `extract(state, schema)` | pull literal values out of the state | `values` |
| `score(...)` | optional ranking | `scores` |

A backend implements them; `decide.resolve` picks one per call from
`conf.decide.backends_in_use`. Swapping the model is one config line, and every backend is
swappable for a recorded fixture — which is how the whole suite runs with no network and no
model.

Shipped backends: `d1_systemone` (llama.cpp `/v1/systemone`), `logit_read` (any local LLM,
A–Z letters read from `top_logprobs`), `pair` (local, escalating to hosted under a
confidence threshold), `cascade` (banded version of the same idea), `openrouter` (hosted),
`classical` (TF-IDF + linear SVM, no model server), `heuristic_micro` (a regex pre-gate that
mostly abstains), `rule_extract` (deterministic), `fixture` (replay).

### Wall 2 — `ops` (what a step does)

An op registers a schema and a function (`ops/op_api.py`) and receives validated parameters.
Nine are registered: six filesystem/web primitives, three decision steps surfaced as ops so
templates speak one vocabulary.

Ops that shell out go through **one** adapter, `ops/cli.py`, so the subprocess rules live in
exactly one place: argv lists and never shell strings, `--` before positionals, a timeout,
and three accepted conventions for locating a CLI's artifact (`docs/ops.md`).

## Bounded context, structurally

The decision model never sees an unbounded payload, and that is enforced rather than
practiced:

- **State** — `context.packer.guard_state` caps it at `conf.context.state_max_tokens`. It is
  shape-preserving and a no-op for normal-sized state, so the prompt the numbers were
  measured on is unchanged.
- **Candidates** — two bounds, because one is not enough: `decide.max_candidates` caps *how
  many* reach the model, `decide.max_criteria_chars` caps how much text each carries. The
  option set does not travel through `guard_state`, which is exactly how a 12 MB candidate
  payload once reached a backend configured for 2000 tokens.
- **Digest** — bounded at ingress, by the extension *and* again at the service.

## Confidence and gates

A gate blocks a step when its artifact's confidence is below a threshold; the runner then
either asks back, retries, or proceeds with `degraded: true` and an `assumptions[]` entry
naming what it settled for.

The governing rule is **a confidence is reported when — and only when — a hop measured one**,
and it is the absence of the field otherwise. That rule exists because the alternative was
measured: an error run with no hops reported `confidence: 1.0`, and a one-entry probability
vector was read as `p_max = 1.0`, so a 0.02-confident answer cleared the gate and executed
unattended. Every confidence source is classified measured / derived / asserted in
[`docs/confidence-sources.md`](docs/confidence-sources.md).

## Security surface

The service is a local HTTP endpoint that reads files on behalf of whoever asks, so:

- **Authentication.** Every POST carries a shared token (`http/auth.py`): `TI_HTTP_TOKEN`, or
  a `0600` runtime file the service creates. There is no unauthenticated mode. `/health`
  stays open because the extension polls it to decide whether to spawn.
- **Recording.** Nothing writes a log until asked: `hops_log.enabled` is false and the
  extension's capture is off until `/ti capture on` (a durable state file, not an in-memory
  flag). Enabled, both write the utterance, argument *names*, sizes and confidences to
  `~/.local/state/ti/logs/`, created `0600` in a `0700` directory — no working directory, no
  argument value, no artifact text, no error message — and a file that is wider than that is
  narrowed rather than refused. [`docs/logging-2026-11-04.md`](docs/logging-2026-11-04.md).
- **Confinement.** A run root must resolve inside `ops.filesys.roots` (default `~`, the
  service's start directory, and `/tmp`). Paths resolve component by component, refusing a
  `..` escape and a symlink whose target leaves the root.
- **The deny list.** `ops.filesys.deny_roots` ships empty and is refused *before the first
  `lstat`* and pruned before descent. It is empty because a deny root is a mount point on the
  operator's own machine (`~/mnt`, an sshfs target) and there is no portable value for it.
  This is not paranoia: an `os.walk` rooted above a dead network mount blocks in
  uninterruptible sleep on this box, measured from 22 minutes to 9.8 hours, and no timeout or
  signal releases it.
- **Bounds.** Body size, worker count, walk depth, walk hits, walk deadline, grep pattern
  length, grep line width, and three pattern *shapes* refused before compilation — a
  quantified group holding a quantified atom, two adjacent unbounded atoms that can match the
  same character, and an unbounded group whose alternation branches can start alike
  (`ops/filesys/regexguard.py`).

## What is deliberately not here

- **No model decides the plan.** Intent classification is keyword frames; the decision model
  is used *inside* a plan. Spec §2 calls the model-planned version L3 and lists it as not
  built; the **P2** label the code carries for that work is not the spec's phase of the same
  name — see [`spec/1.0-spec.md`](spec/1.0-spec.md#divergences-from-this-spec)'s §8 row.
- **No destructive tools.** `bash`/`write`/`edit` stay native; the v1 domain is read-only
  plus the web stack.
- **No orchestration.** This schedules one call. Multi-agent coordination is a different
  problem and a different repository.

## Where to read next

| if you want | read |
|---|---|
| to see it run | `python3 bench/demo.py` |
| the backlog and what the numbers imply | [`TODO.md`](TODO.md) |
| the normative design and the phase plan | [`spec/1.0-spec.md`](spec/1.0-spec.md) — including its closing "Divergences from this spec" |
| per-module ownership, what each owns and does not | [`docs/architecture.md`](docs/architecture.md) |
| the CLI contract the web ops depend on | [`docs/ops.md`](docs/ops.md) |
| what the economics actually are | [`docs/roundtrips-2026-11-03.md`](docs/roundtrips-2026-11-03.md) |
| which routing backend to use, with numbers | [`docs/classical-baseline-2026-11-03.md`](docs/classical-baseline-2026-11-03.md), [`docs/model-sweep-consolidated.md`](docs/model-sweep-consolidated.md) |
| why local + hosted, not one or the other | [`docs/decision-pair-vs-hosted.md`](docs/decision-pair-vs-hosted.md) |
