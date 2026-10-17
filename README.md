# toolInterfacer

A modular decision-model (Jev-style / systemone) tool-interfacing layer for the
pi coding agent.

Instead of the big LLM issuing every tool call itself (each costing a full model
round trip), the LLM states an **intent** — plain English plus whatever
parameters it already knows — and a local decision-model loop (d1-3B today,
fine-tuned qwen3.5-4b later) picks tools, fills slots, and executes the
underlying operations. The LLM pays seconds; the decision model pays
~250ms/decision with zero generated tokens.

- Core decision service: **Python** (`service/ti/`)
- pi extension adapter: **TypeScript** (`pi-extension/`)
- All decision backends behind one primitive wall (`choice` / `noul` / `score` /
  `extract`), selected by config: d1-3B via llama.cpp `/v1/systemone`, Jev API,
  logit-read (any LLM), heuristic micro-gate.

## Status

Skeleton + spec. See `spec/1.0-spec.md` for the phased plan with acceptance
criteria. Nothing is wired end to end yet.

## Layout

```
service/       Python core: engine, plan templates, decide backends, ops, policy
pi-extension/  TypeScript pi extension: useTool tool, bridge to the service
spec/          the written specification
docs/          architecture and usage notes
tests/         fixture-player tests (full engine, no model, no CLI)
bench/         benchmark drivers and probe scripts
```
