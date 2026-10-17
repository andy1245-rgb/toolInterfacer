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

## Benchmarking without pi or a model

`tests/` uses the fixture-player decide backend (replays recorded
d1 responses) and replayable op backends; `bench/replay.py` runs a plan
end-to-end with zero network. The same runner is what production uses — the
only difference between experiment and reality is which backend file the
config names.
