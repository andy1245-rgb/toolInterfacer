# Architecture notes

The split: **decision service (Python, localhost HTTP) + pi extension (TypeScript)**.
The extension registers the `useTool` tool and nothing else — the entire
decision-making brain lives in the service, so experiments and benchmarks run
without pi.

## Process view

```
LLM ──useTool(intent, params)──▶ pi extension ──HTTP──▶ decision service
                                                          │
                              d1-3B (llama.cpp /v1/systemone) ◀── ┤
                              CLI ops (web-search/web-fetch/reddit) ◀── ┘
                              artifact store + gates + budget
◀──────── status + result + hops ─────────────┘
```

## Module responsibilities

| Module | Owns | Does not own |
|---|---|---|
| `core.runner` | step-graph execution: op → artifacts → gates → next | deciding *which* step comes next (planner), or any op |
| `planner` | intent → plan (L0 passthrough, L1 one-hop, L2 template) | execution |
| `decide` | the primitive wall; backend selection via config | op execution |
| `ops` | running one operation, validated schema in/out | talk to decision models |
| `context.packer` | what the decision model sees, within token budget | execution |
| `policy` | gate arithmetic + ask-back triggers/budget | models, ops |
| `hops` | the JSONL evidence log | gates |
| `http.api` | the route surface | any logic |

## The two plugin walls

1. **Decide backends** (`decide/backends/`) register against
   `decide/primitives.py`. The runner and templates address decisions as
   `choice/noul/score/extract` — never a backend name. Swapping d1-3B → jev
   API → qwen3.5-4b logit-read is a config line.
2. **Ops** (`ops/`) register via `ops/op_api.py`. Templates address steps as
   `{op, params, gate}` — abstract, no hardcoded k loops or thresholds. All
   tunables live in `conf/defaults.json`, so template files only publish the
   *shape* of a pipeline while config publishes its *degree*.

## Bounded context (the two-line guarantee)

The decision model never sees an unbounded payload, and that is structural, not
accidental:

- **extension side** — the session digest is `<= 8 turns x 240 chars`
  (`session_push.ts`: last 8 user/assistant turns, each compacted to 240 chars,
  plus <= 8 file-path antecedents and `cwd`).
- **service side** — `service_entry.use_tool` runs that digest through
  `context.session_digest.sanitize` at ingress (same 8 x 240 bound, `cwd` kept
  because the L1 bridge resolves every path from it), and every state handed to
  a decide backend passes `context.packer.guard_state`, which bounds it to
  `conf.context.state_max_tokens` (chars ~= tokens x 4).

`guard_state` is shape-preserving and a **no-op for normal-sized state** — the
same object comes back, so the prompt the decision model's validated numbers were
measured on is unchanged (`tests/test_context_bound.py` asserts byte-identical
serialization for the normal case). Over budget it trims in a fixed order: clip
long candidate descriptions, drop surplus candidate entries, stub the digest,
then clip any remaining long string — never dropping an outer key or changing a
key's type. Callers other than the extension therefore cannot inflate the prompt;
worst case they get a thinner state, not a bigger one.

## Benchmarking without pi or a model

`tests/` uses the fixture-player decide backend (replays recorded
d1 responses) and replayable op backends; `bench/replay.py` runs a plan
end-to-end with zero network. The same runner is what production uses — the
only difference between experiment and reality is which backend file the
config names.
