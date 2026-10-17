# pi-extension — `useTool` for pi

TypeScript extension: registers the `useTool` tool the LLM calls, bridges to the
decision service over localhost HTTP, pushes the rolling session digest, and
starts the service process on session start.

```
LLM ──useTool(intent, params, mode)──▶ this extension ──HTTP 127.0.0.1:8090──▶ service/ti/http.api
       ◀── one compact result: result, hops (op, ms, conf), assumptions, degraded flags ──
```

## Layout (`src/`)

| file | responsibility |
|---|---|
| `contract.ts` | the wire types (spec §3); mirrors `service/ti/service_entry.py` responses |
| `bridge.ts` | HTTP client: `call()` → `/v1/useTool`, `health()` → `/health`; base URL from `TI_BASE_URL` (default `http://127.0.0.1:8090`), 120s call timeout |
| `session_push.ts` | rolling digest from `ctx.sessionManager.getEntries()`: last 8 user/assistant turns (~240 chars each), file-path mentions per turn, user-mentioned paths that exist on disk as antecedents with their resolved absolute path, plus cwd |
| `lifecycle.ts` | on session start: `GET /health`; if down, spawns `python3 -m ti.http.api` detached (cwd `/home/andrew/toolInterfacer`, `PYTHONPATH=service`, overridable via `TI_SERVICE_DIR` / `TI_PYTHON`), polls /health up to 10s |
| `index.ts` | registers `useTool` (direct exposure), the `/ti` command, and the `session_start` hook |

## The tool

`useTool(intent: string, params?: object, mode?: "auto"|"plan"|"raw")` — the LLM
states a plain-English intent; the service classifies and runs the right op or
pipeline and returns one compact result. The result text is framed for the LLM:

- summary line (status, degraded flag, "decision-model service" attribution),
- the result body, truncated to ~1500 chars with a `…[truncated …]` marker that
  suggests `mode:"raw"` or a native read for the full content,
- a hops line (`op ms conf …`, max 6, plus route confidence),
- assumptions and, when degraded, a verification warning.

Ask-back: on `need_input` the question and its options are echoed verbatim with
an instruction to call `useTool` again with `params.answer` (option id or a
specific path/URL). The LLM answers and the loop resumes — no redo. Degraded
results (ask-back budget exhausted) come back as best-guess `done` with an
explicit verify-yourself warning.

Env passthrough of params is verbatim: LLM-supplied keys are injected as-is;
unknown keys ride to the service as soft context and never fail the call.

## `/ti` command

- `/ti` — service up/down at the base URL, registered ops, conf defaults ok,
  last `useTool` call this session (intent → status, hop count, route conf).
- `/ti doctor` — as above; if the service is down it attempts a spawn
  (`ensureService`) and reports whether /health recovered within 10s.

## Service process lifetime

The extension **starts** `python3 -m ti.http.api` detached (`stdio: "ignore"`,
unref'd) when /health is not answering, and deliberately **never kills it** —
not on `session_end`, not on exit. The service is a shared localhost service:
several pi sessions (and benchmark/replay runs) may live alongside it, and a
detached spawn would otherwise outlive the parent anyway. Stop it manually with
`kill <pid>` or `pkill -f "ti.http.api"`. If the spawn fails, /ti doctor and
session start surface it via notification (UI modes only).

## Development

```sh
ln -s $(ls -d ~/.pi/agent/install/releases/*/node_modules | tail -1) node_modules
npx --yes -p typescript@5.5.4 tsc --noEmit   # strict, zero errors
```

`node_modules` is a symlink to pi's own install tree (same pattern as
pi-edit3) — do **not** run `npm install` here; it would replace the symlink.
pi executes this TS directly via jiti, so no build step exists. Zero runtime
deps: only `@earendil-works/pi-ai` (TypeBox, aliased by pi's loader) and
Node builtins; HTTP goes through global `fetch`.

## Install (pi extension discovery)

```sh
ln -s /home/andrew/toolInterfacer/pi-extension/src ~/.pi/agent/extensions/toolinterfacer
```

pi loads each directory in `~/.pi/agent/extensions/` from its `index.ts`
entry, so the symlink targets `src/`.

## Verified

- `tsc --noEmit` (TS 5.5.4, strict): clean.
- Live print-mode run (`pi -p --no-session --mode json`): the model called
  `useTool` with an unpacked intent; the service routed
  `filesys_glob → decide_pick → filesys_read` (conf 0.95) and the final answer
  describes toolInterfacer with the hop trace.
- Ask-back stub run (canned `need_input` service): question + options rendered
  verbatim, the model re-called `useTool` with `params.answer` and got the
  resumed `done` result.
