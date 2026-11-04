# Architecture: the per-module contract

This is the second half of the architecture, and it says one thing: **what each module owns,
and what it explicitly does not.** The overview — why the layer exists, the process view, the
life of one `useTool` call, the levels, the two plugin walls, the security surface — is
[`../ARCHITECTURE.md`](../ARCHITECTURE.md); read that first. Nothing here restates it.

The rule this file exists to make checkable: a change to one behaviour should land in the
module that owns it, and if it does not, the boundary is wrong.

## Module responsibilities

| Module | Owns | Does not own |
|---|---|---|
| `service_entry` | the one body `useTool` accepts: ingress sanitize, run root, route vs passthrough, response shape, `confidence` | planning, execution, any op |
| `planner.inspector` | intent → level + template (`L0` / `L1` / `L2`), by ordered keyword rules | anything below the label: no slots, no candidates, no model |
| `planner.shape`, `planner.scope`, `planner.pattern` | three independent reads of one disk intent: listing-or-read, which directory, which names | the plan itself |
| `planner.templates.*` | the step graph: which op, in what order, with which gates; its terminal artifact | every number in that graph — top-k, caps, gates, timeouts are `conf` |
| `planner.free` | nothing yet: L3 is a docstring | L3 |
| `core.runner` | step-graph execution: op → artifact → gate → next step, under a budget | deciding *which* step comes next (template), or running any op |
| `core.gates`, `core.budget` | gate arithmetic; hop/ask/clock ceilings | what the thresholds *are* (`policy`, `conf`) |
| `core.artifacts` | the artifact store and `Artifact.conf` (a measurement, or `None`) | judging a confidence |
| `decide` (`primitives`, `resolve`) | the primitive wall; backend selection from config; the `choice` fast path (pre-gate → cache → backend) | op execution; any backend's internals |
| `decide/backends/*`, `decide/pair.py`, `decide/cascade.py` | one decision strategy each | each other — a router backend calls stages through `make_backend` |
| `ops` (`op_api`, `cli`, `decisions`, `filesys/`, `web/`, `reddit/`, `research/`) | running one operation: validated schema in, `(kind, data, conf)` out | talking to a decision model, or knowing which op should run |
| `ops.cli` | the one subprocess adapter: argv lists, `--`, timeout, exit-code tolerance, artifact location | any CLI's semantics — that belongs to the op |
| `ops.filesys.confine`, `walk`, `regexguard` | the path policy, the bounded walk, the pattern guard | what a caller meant by a path |
| `context.packer` | what the decision model sees, within `conf.context.state_max_tokens` | execution |
| `context.session_digest` | the ingress bound on the caller's digest (≤ 8 turns × 240 chars) | resolving anything from it |
| `policy.askback`, `policy.confidence` | ask-back triggers and budget; the gate and confidence arithmetic | the model, the ops |
| `hops.log` | the JSONL trace log (opt-in, owner-only; `docs/logging-2026-11-04.md`) | gates, decisions |
| `http.api`, `http.auth`, `http.pool` | the route surface *and* the ingress guards the route cannot delegate: token, body bound, route, L0 allowlist, run-root scope | plan execution, or any op's behaviour |
| `conf` | every knob, one file | defaults that only one module could know |
| `pi-extension/src/index.ts` | the single registered tool, the `/ti` command, service lifecycle | every decision — the extension decides nothing |
| `pi-extension/src/bridge.ts`, `token.ts` | the HTTP client and the token it presents | retries or policy |
| `pi-extension/src/session_push.ts` | the rolling digest: last 8 turns, 240 chars each, ≤ 8 path antecedents | what the service does with it |

Two rows are worth spelling out because they moved:

- **`http.api` owns the ingress guards now.** It reads the body, checks the token, resolves
  the run root against `ops.filesys.roots`, and enforces the L0 allowlist *before*
  `service_entry.use_tool` is called (`service/ti/http/api.py`, module docstring). "The route
  surface, no logic" was true when the endpoint had no policy; it is not true now.
- **`service_entry` owns the response shape**, including the rule that `confidence` is
  present only when a hop measured one (`_route_conf`).

## Where the degrees live

Every number a plan uses is a `conf` key. `service/ti/planner/templates/research.py` names
its own in a docstring table (`research.query_count`, `pick_top_k`, `good_pages_target`,
`digest.claim_chars`, `policy.gate_slot`, `policy.gate_noul`); `disk_lookup` and
`reddit_digest` read the same way and carry no numbers of their own. A step can also declare
`over` (run once per item of an artifact, each with its own post-gate verdict) and
`stop_after` (end the loop once that many items verified) — both are degrees of the step, so
`research`'s per-page relevance check and its k-good-pages stop still get their numbers from
config.

A `TODO` with a number in it is a config key that has not been added yet.

## Benchmarking without pi or a model

`tests/` runs the real runner and the real ops with two substitutions: the `choice` decision
comes from the `fixture` backend (a fixture file keyed by a hash of
`(state, question, criteria)`, with a lexical fallback for the pick so a bespoke test needs no
recording — `service/ti/decide/backends/fixture.py`), and the three subprocess CLIs are
replaced by `tests/data/fake_cli.py`. `bench/replay.py` does the same for one plan end to
end.

The runner, the gates, the templates, the confinement and the response contract are the
production ones. The only difference between a test and a live run is which backend the
config names.
