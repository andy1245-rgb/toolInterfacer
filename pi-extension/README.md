# pi-extension — `useTool` for pi

TypeScript extension: registers the `useTool` tool the LLM calls, bridges to the
decision service over localhost HTTP, pushes the rolling session digest, and
starts the service process on session start.

```
LLM ──useTool(intent, params, mode)──▶ this extension ──HTTP 127.0.0.1:1113──▶ service/ti/http.api
       ◀── one compact result: result, hops (op, ms, conf), assumptions, degraded flags ──
```

The extension is a **hardened adapter**: the decision loop lives in Python
(`service/ti/decide/`, mode `auto|plan|raw` handled by `service_entry.py`) and
stays there. Nothing in this directory decides anything.

## Layout (`src/`)

| file | responsibility |
|---|---|
| `contract.ts` | TypeBox schemas for the wire response (`UseToolResponseSchema`, `HopsSchema`, `QuestionSchema`, `PlanStepSchema`) + the `UseToolRequest` interface; TS types are derived with `Static<…>`. Mirrors `service/ti/service_entry.py`. |
| `bridge.ts` | HTTP client: `call()` → `/v1/useTool`, `health()` → `/health`; base URL from `TI_BASE_URL` (default `http://127.0.0.1:1113`), timeout per call (120s, 240s for `mode:"raw"`). Classifies failures as `ServiceError` and validates the body against the schema before returning it. |
| `session_push.ts` | rolling digest from `ctx.sessionManager.getBranch()` (active branch, not `getEntries()`): last 8 user/assistant turns (~240 chars each), file-path mentions per turn, user-mentioned paths that exist on disk as antecedents with their resolved absolute path, plus cwd. Also `readLastCustom()` for state restored on resume. |
| `lifecycle.ts` | `ensureService()`: `GET /health`; if down, spawns `python3 -m ti.http.api` detached (cwd `TI_SERVICE_DIR`, default: the repo root, derived from the extension's own location, `PYTHONPATH=service`, interpreter from `TI_PYTHON`), polls /health up to 10s, returns `{ok, pid?, reason?, detail?}`. |
| `index.ts` | registers `useTool` (direct exposure, `outputSchema` + `structuredContent`, `onUpdate`, `executionMode:"sequential"`), the `/ti` command, and the `session_start` / `session_shutdown` hooks. |
| `scripts/contract-check.ts` | runtime check that every body the service can return passes the schema, and that malformed bodies do not. |

## The tool

`useTool(intent: string, params?: object, mode?: "auto"|"plan"|"raw")` — the LLM
states a plain-English intent; the service classifies and runs the right op or
pipeline and returns one compact result. The result text is framed for the LLM:

- summary line (status, degraded flag, "decision-model service" attribution),
- the result body, truncated to ~1500 chars with a `…[truncated …]` marker that
  suggests `mode:"raw"` or a native read for the full content,
- a hops line (`op ms conf …`, max 6, plus route confidence),
- assumptions and, when degraded, a verification warning.

`mode:"plan"` renders the resolved chain instead of executing it (level,
template, numbered steps with their params, final artifact key) and tells the
model how to run it. `mode:"raw"` gets the larger timeout.

Ask-back: on `need_input` the question and its options are echoed verbatim with
an instruction to call `useTool` again with `params.answer` (option id or a
specific path/URL). The LLM answers and the loop resumes — no redo. Degraded
results (ask-back budget exhausted) come back as best-guess `done` with an
explicit verify-yourself warning.

Env passthrough of params is verbatim: LLM-supplied keys are injected as-is;
unknown keys ride to the service as soft context and never fail the call.
`prepareArguments` parses a `params` argument the model sent as a JSON string
before pi validates it, which removes a common call failure.

## Contract validation

The response is parsed as `unknown` and checked with TypeBox's `Value.Check`
before the tool touches it. On a mismatch the tool returns a normal error result
naming up to three `Value.Errors` (`instancePath message`), so the model can
act on it; a malformed body never crashes the extension.

Two leniencies are deliberate, both because a false reject is worse than a
permissive accept:

- `status` includes `"plan"`. `_plan_view` (`service/ti/service_entry.py`) answers
  `mode:"plan"` and the `/v1/plan` route with a plan body, not a run result.
- `hops` is optional. The service now sends `hops: []` on plan bodies, but an
  older service that omits the key still parses.

Payload fields (`result`, `sources`, `meta`, step params/gates) are `Type.Any()`:
the schema enforces the envelope, not the payload.

## Failures

| condition | `ServiceError.kind` | what the model sees |
|---|---|---|
| caller aborted | `aborted` | `useTool: aborted after Nms by the caller` |
| client timeout fired | `timeout` | `…exceeded Ns without a response — retry with a narrower intent, or mode:"plan"…` |
| connect/DNS failure | `unreachable` | `…service unreachable — <base> (<code>) …; run /ti doctor` |
| non-2xx | `http` | `…service returned HTTP <status> — <body head>` |
| non-JSON body | `invalid` | `…service returned an unexpected shape — response is not JSON …` |
| wrong shape | `invalid` | `…unexpected shape — /status must be equal to constant; …` |

The fetch abort signal is `AbortSignal.any([caller signal, timeout])`, so a
cancelled turn cancels the HTTP request; the tool distinguishes caller abort
from timeout by checking which signal fired.

## Environment

| var | effect |
|---|---|
| `TI_BASE_URL` | service base (default `http://127.0.0.1:1113`) |
| `TI_SERVICE_DIR` | cwd for the spawned service (default: repo root, derived from the extension's own location) |
| `TI_PYTHON` | interpreter for the spawn (default `python3`) |
| `TI_DEV=1` | dev mode — see below |

## Dev mode (`TI_DEV=1`)

- `ensureService()` never spawns (returns `{ok:false, reason:"dev"}`); the session
  start notification says so. `/ti doctor` prints the exact command line it
  *would* run instead of running it.
- Every `useTool` result gets a `[dev] <base> · <elapsed>ms · <response JSON ≤600>`
  trailer, so routing/hops are visible in the transcript.
- `/ti` reminds you that `mode:"plan"` prints the decision chain without executing.

Canonical loop: run a stub or real service on some port, then launch pi with
`TI_BASE_URL=http://127.0.0.1:<port> TI_DEV=1`. Nothing is spawned or killed, and
the live service on 1113 (shared by other sessions and bench runs) is untouched.

## `/ti` command

- `/ti` — service up/down at the base URL (dev marker), registered ops, conf
  defaults, service pid if this session spawned it, last `useTool` call
  (intent → status, hop count, route conf).
- `/ti doctor` — as above; if the service is down it attempts a spawn
  (`ensureService`) and reports pid, the failure reason (`spawn-failed` with the
  child's error or exit status, or `timeout` after 10s).

## State and lifecycle

`session_start` restores the last `useTool` call from the active branch
(`ti.lastcall` custom entry) and records the spawned service pid
(`ti.service`). `session_shutdown` clears the `ti` status line. The service
process is deliberately **never killed** — it is a shared localhost service, and
several pi sessions (plus benchmark/replay runs) may live alongside it. The pid
is recorded only so it can be stopped deliberately: `kill <pid>` (shown by `/ti`).

## Development

```sh
ln -s $(ls -d ~/.pi/agent/install/releases/*/node_modules | tail -1) node_modules
npx --yes -p typescript@5.5.4 tsc --noEmit          # strict, zero errors
node scripts/contract-check.ts                       # 10/10 contract cases
```

`node_modules` is a symlink to pi's own install tree (same pattern as
pi-edit3) — do **not** run `npm install` here; it would replace the symlink.
`node_modules/` is gitignored. pi executes this TS directly via jiti, so no build
step exists. `scripts/contract-check.ts` runs under Node's built-in type
stripping (Node ≥ 22) and needs the same symlink.

Zero runtime deps: `@earendil-works/pi-ai` (TypeBox is re-exported from it; pi's
loader aliases `typebox`, `typebox/compile`, `typebox/value`), Node builtins, and
global `fetch`.

## Install (pi extension discovery)

```sh
ln -s <repo>/pi-extension/src ~/.pi/agent/extensions/toolinterfacer
```

pi loads each directory in `~/.pi/agent/extensions/` from its `index.ts`
entry, so the symlink targets `src/`.

## Not implemented on purpose

`ctx.modelRegistry.classify()` (pi's `typesafe-system-one` client for llama.cpp
decision models) is **not** wired here. The decision loop — inspect, templates,
gates, banded cascade, ask-back — is implemented and validated in Python
(`service/ti/decide/cascade.py`, config `decide.cascade`), and the service has no
"execute a plan decided elsewhere" endpoint, so a TS-side decision would still
need the Python executor. If that ever changes, `classify()` is the natural
fast-path for the narrow op-family choice (typed `choice` answers with
probabilities, confidence and usage); until then the extension stays a pure
adapter.

## Verified

- `tsc --noEmit` (TS 5.5.4 and 5.7.3, strict `noEmit`): clean on `src/` + `scripts/`.
- `node scripts/contract-check.ts`: 10/10 — auto (`done`, L0 passthrough),
  `need_input`, `error`, plan with and without `hops`, plus 4 malformed bodies
  that must be rejected.
- Live service on a spare port: `mode:"auto"` → `done` (3 hops, conf 0.95),
  `mode:"plan"` → `plan` (3 steps, `hops: []`), empty intent → `error`, and a
  1ms timeout → `ServiceError{kind:"timeout"}`.
- The B-plan schema (status without `"plan"`, `hops` required) rejects that live
  plan body — the regression this file documents was real.
- Live print-mode run (`pi -p --no-session --mode json`, pre-hardening build, not
  re-run since): the model called `useTool` with an unpacked intent; the service
  routed `filesys_glob → decide_pick → filesys_read` (conf 0.95) and the final
  answer describes toolInterfacer with the hop trace.
- Ask-back stub run (canned `need_input` service, pre-hardening build, not re-run
  since): question + options rendered verbatim, the model re-called `useTool`
  with `params.answer` and got the resumed `done` result.

Not verified (API read from the installed pi 1.2.0 tree, behaviour not observed):
`outputSchema`/`structuredContent` enforcement beyond codemode's documented
"resolve to `structuredContent`" path; how `onUpdate` partials render in
tui/json modes; `prepareArguments` against real model tool-call plumbing; the
`ti.lastcall` restore across an actual `/resume` or `/tree` navigation.
