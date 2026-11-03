# Documentation review — 2026-11-03

Read of `main @ a105d50` in worktree `d3-review`. Read-only: no source file, doc or config
edited; the only file written is this one. Every command below was run on this box, in this
worktree.

Scope: `README.md`, `ARCHITECTURE.md`, `TODO.md`, `CONTRIBUTING.md`, `CHANGELOG.md`,
`RELEASE.md`, all 20 files under `docs/`, `spec/1.0-spec.md`, `bench/README.md`,
`pi-extension/README.md`, `demo/README.md`, and the docstrings of `bench/demo.py`,
`bench/run.py`, `bench/replay.py`, `service/ti/http/api.py`.

---

## Verdict

A stranger would trust this repo: the honesty apparatus is real, not decorative — the
"Honest gaps" list held up on every item I could reproduce, `docs/confidence-sources.md`
classifies every confidence number by provenance, and the negative results reproduce.

The trust is at risk from number drift, not from overclaiming: the published suite count is
wrong in seven places (`715 tests, 349 subtests` against a green `722 passed, 388 subtests`),
the CHANGELOG's own correction table still quotes the number it was written to correct, and
two "bounded / offline" guarantees are contradicted by the code that implements them.

The single most costly item: **`README.md:123` and `ARCHITECTURE.md:54-56` claim a bound that
`sanitize()` does not apply** — a digest turn carrying unknown keys survives ingress at
100,100 characters (measured) — while the module docstring says those keys are "dropped rather
than forwarded". That is a security-shaped claim a reader will quote, and the one place the
repo says something is *structural* when it is not.

The presentation also buries resolutions the way it once buried limitations: three documents
in `docs/` still say "Not publishable today" / five blockers / "the guard cannot enforce its
own guarantee", all fixed, none marked superseded, and none reachable by a link.

---

## False or unverifiable claims

| document:line | what it says | what is true | evidence |
|---|---|---|---|
| `README.md:119` | "`research` runs end to end model-free via `bench/replay.py` and returns a sourced digest" | True only of `bench/demo.py` (which stubs the CLIs). `bench/replay.py` never stubs a CLI: in a clone it errors, and on this box with the author's skill scripts present it ran past 180 s without finishing | `TI_CONF_OVERLAY` fixture + `replay.py --intent "research lora rank for a 3b router"` → `"error": "CLI not found: /nonexistent/web-search/scripts/free-search"` (exit 0); with the real scripts: no completion in 180 s |
| `README.md:123` | "Bounded decision state (structural, not conventional) — digest ≤ 8 turns × 240 chars" | `sanitize` clips `turns[].summary` and the counts only; every other key of a turn is copied verbatim (`out = dict(turn)`), so arbitrary extra keys ride through. Measured: one turn `{"role":"user","summary":"x","blob":"A"*100000}` survives at 100,100 chars | `service/ti/context/session_digest.py:44-60`; `PYTHONPATH=service python3 -c "…sanitize(…)…"` → `turn keys: ['role','summary','blob']`, `blob len: 100000` |
| `service/ti/context/session_digest.py:26-29` | "Anything else survives only as far as those bounds allow — extra keys are dropped rather than forwarded" | Extra keys inside a turn are forwarded; only top-level extra keys are dropped | same probe: top-level `extra` dropped, per-turn `blob` kept |
| `ARCHITECTURE.md:54-56` | "`sanitize` clips the caller's digest to ≤ 8 turns × 240 chars … A caller cannot inflate the prompt by sending a bigger digest" | The prompt does stay bounded, but by `context.packer.trim` collapsing the digest to a stub, not by `sanitize`; the mechanism named is not the one that holds | `service/ti/context/packer.py:34-48` (`state["digest"] = {"summary": "trim: over budget"}`) |
| `bench/README.md:37-38` | "`replay.py` (P1) — runs a plan end-to-end through the fixture-player backend with zero network" | It uses the configured backend, default `pair`. Bare `python3 bench/replay.py` → `"error": "pair: no stage produced a choice"`; an L2 intent runs real `web_search`/`web_fetch` | `python3 bench/replay.py` (exit 0, error body); `README.md:322` already carries the corrected wording |
| `bench/replay.py:2-4` | docstring: "Runs a plan (fixture decide backend, zero network)" | Same as above — the fixture backend is only used when `TI_CONF_OVERLAY` points `decide.backends_in_use.choice` at it | `bench/replay.py:38-41` reads the configured backend via `load()` |
| `bench/README.md:60` | "`baseline.py` (P4) — scripted LLM-driver baseline for the research template comparison" | No such file; `bench/` has no `baseline.py` | `ls bench/` |
| `docs/testing.md:63-70` | Presents `bench/live_e2e.py --serve` as how you "prove the wire" | The probe sends no token (`headers = {"Content-Type": "application/json"}`) and every POST is refused 401, so all four cases fail | `bench/live_e2e.py:104`; `docs/live-e2e-2026-11-03.md:377-397` records `0/4 cases passed`, `HTTP 401` |
| `docs/ops.md:209-210` | "Two tests replay the recorded devskills fixtures … (`TI_DEVSKILLS_FIXTURES`, default `~/toolInterfacerExtras/devskills`); they skip when that tree is absent" | The default is repo-relative and ships: `ROOT/tests/data/devskills`. The two tests run and pass; they do not skip | `tests/test_ops_web.py:30-34`; `pytest tests/test_ops_web.py -k recorded -v` → `2 passed` |
| `docs/live-e2e-2026-11-03.md:349-351` | "`pi-extension/node_modules` is a symlink to pi's own install tree …, as the extension README documents" | The extension README says the opposite: "do not symlink it to pi's own install tree — that would bypass the pinned versions" | `pi-extension/README.md:136-139` |
| `docs/live-e2e-2026-11-03.md:337` | `node scripts/contract-check.ts` → "10/10 contract cases passed" | 11 static cases (`console.log(cases.length - failed / cases.length)`); 10 is the *stub* total (`served.size + 3`) | `pi-extension/scripts/contract-check.ts:29-121,136,249`; `pi-extension/README.md:132` says 11/11 |
| `README.md:312-314` | CI "builds the wheel, asserting the four service modules it names — `ti/http/api.py`, `ti/decide/resolve.py`, `ti/ops/op_api.py`, `ti/conf/defaults.json` — are inside it" | CI names no modules: it asserts the whole `service/ti` file set is present with no strays, plus `defaults.json` | `.github/workflows/ci.yml:106-128` |
| `docs/ops.md:14` | "An argv list … `--` precedes the positional so a query that starts with `-` is not read as a flag" | True for `web_search`/`web_fetch`; `reddit_digest` builds `["search", query, "--json", …]` with no `--` | `ops/web/search.py:127,132`, `ops/web/fetch.py:58`, `ops/reddit/digest.py:123` |
| `README.md:145` | ask-back: "`tests/test_loop.py` and the live probe both exercise it" | The live probe cannot exercise anything: it 401s on every case (see above) | `docs/live-e2e-2026-11-03.md:377` |

## Numbers with no source, or whose source now disagrees

| document:line | what it says | what the source says | evidence |
|---|---|---|---|
| `README.md:125`, `README.md:298`, `CONTRIBUTING.md:9`, `ARCHITECTURE.md:102`, `CHANGELOG.md:10`, `CHANGELOG.md:124`, `RELEASE.md:54`, `RELEASE.md:64`, `RELEASE.md:92` | "715 tests, 349 subtests" / "715 passed, 349 subtests passed" | **722 tests, 388 subtests**, green | `python3 -m pytest tests/ -q` → `722 passed, 388 subtests passed in 74.15s` (exit 0). A first run of mine showed `1 failed` — an artefact of my own `bench/demo.py` running concurrently (the test asserts no new `/tmp/ti-demo-*`); the isolated re-run and the clean full run are both green |
| `docs/testing.md:55-56` and `README.md:379` | "On a CPU-only box, d1-3B-Q4 costs ~1.7 s per routing decision and ~50 ms on a decision-cache hit" | No recorded run supports 1.7 s. The recorded CPU-arm figures are p50 3283.7 ms (`igpu-vulkan-v1`) and 3443–4311 ms (`run1/2/3-t8`); nothing anywhere records a cache-hit latency | `docs/vulkan-igpu-2026-11-03.md:6-7`; `bench/results/run1-t8-v1.json`, `run2-t8-v2-descriptions.json`, `run3-t8.json`, `igpu-vulkan-liveprobe-cpu-2026-11-03.json` |
| `README.md:343` (dataset paragraph) and `CHANGELOG.md:117` | "14 trainable records out of 1398 hops" / "1,398 logged lines yield 14" | The cited source says the log has grown: 5,243 lines now yield 15, all still `filesys_glob`. `CHANGELOG.md:141` says so in its own correction table, 24 lines below `:117` | `docs/dataset-status.md:12-23` |
| `spec/1.0-spec.md:10` | "Liquid d1-3B today: 91.4% tool routing at p50 243ms" | No source, no set named. The repo's own measurement of the same model on its own 150-row set is **74.0%**; 91.36% is a sibling-project figure on a different, easier 382-row set | `README.md:348-358`; `docs/classical-baseline-2026-11-03.md:24`; `docs/roadmap-2026-11.md:139-144` |
| `README.md:257` | Sample output: `filesys_glob 2.5ms conf=1.00 path_list[179] e.g. LICENSE` | The candidate count is a property of the working tree; this one yields `path_list[285]`. The README caveats latency ("do not reproduce on CI") but nothing here | `TI_CONF_OVERLAY=/tmp/ti-fixture.json python3 bench/replay.py --intent "read the file we discussed"` → `path_list[285] e.g. TODO.md` |
| `docs/roadmap-2026-11.md:9-11` | "Superseded numbers: the suite is 715 tests … `README.md` and `docs/dataset-status.md` carry the current ones" | The README carries the same stale 715; the body below still says `326 passed, 159 subtests` (`:16`) and `588 → 626 tests` (`:161`) | as above |

## Stale status

| defect | document that still describes it as open | fixed? |
|---|---|---|
| `bench/classical_baseline.py` crashes on 37 invalid-JSON result files | `CHANGELOG.md:139` ("now *crashes* … the documented command exits at its d1 load") | **Fixed.** All 61 JSON files under `bench/results/` parse; the script runs to completion, exit 0, and rewrites both `classical_382.json` and `classical_v2.json` (I restored both after the run) |
| same crash, stated in full | `docs/classical-baseline-2026-11-03.md:282-300` — "The first command does not run at HEAD (`ca7ffec`)", "`classical_v2.json` is never written" | **Fixed**; both files are written now |
| the portability guard is blind to the bare `/home/<name>` form | `docs/pre-publish-audit.md:411-416` — "Right now the repo advertises a portability guarantee that its own test cannot enforce" | **Fixed** in `7113ef8`/`a105d50`; the guard now matches the bare form and the Windows tree (`tests/test_no_personal_paths.py:16-31`). The doc's own §2.3 (`:123-126`) says so, contradicting §7 |
| "Not publishable today": wheel ships 2 of 53 modules; 66 `tsc` errors; research template names ops that do not exist | `docs/audit-2026-11-03.md:16-26` | **All three fixed** (`docs/roadmap-2026-11.md:145-157`; `CHANGELOG.md:81-88`; `RELEASE.md:96-128`). No superseded banner |
| five blockers (B1–B5) presented as live | `docs/review-2026-11-03.md:21-23` and following | **Fixed** — review2 has a "Status of B1–B5" section, review1 has none. Unlinked from every other document |
| "Suite: 361 passed, 159 subtests" | `docs/live-e2e-2026-11-03.md:5-7` | Stale (722/388). The doc is a dated run, but the count is not marked as-of |
| "the documented L0 example fails with `403 scope_refused` … and the README never says so" | `docs/quickstart-stranger-2026-11-03.md:15-17` (the Verdict, present tense) | **Fixed**: `ops.filesys.roots` ships as `["~",".","/tmp"]` and `README.md:224-228` states the default and that the example works from a clone wherever it lives. I ran the example: `{"status": "done", "result": ["pyproject.toml"], …}` |
| suite counts 326/159 and "588 → 626" | `docs/roadmap-2026-11.md:16` and `:161` | Stale; the header at `:9` claims superseded numbers are marked, and these two are not |

## Structural problems

1. **Orphaned documents.** `RELEASE.md`, `docs/pre-publish-audit.md` and
   `docs/review-2026-11-03.md` are named by no other file in the repo — not linked, not
   mentioned. `RELEASE.md` is the release runbook and `docs/pre-publish-audit.md` is the
   pre-publish gate; both are exactly what a maintainer needs and neither is reachable from
   the README's Docs table or `CONTRIBUTING.md`. `docs/audit-2026-11-03.md` is referenced only
   from `docs/review2-2026-11-03.md`.
2. **Dangling reference.** `docs/roadmap-2026-11.md:32` lists `docs/session-recall.md` as the
   Phase 0 "recall" deliverable. No such file exists.
3. **Two documents, opposite instructions, one citing the other.** `docs/live-e2e-2026-11-03.md:349`
   says `pi-extension/node_modules` is a symlink to pi's install tree "as the extension README
   documents"; `pi-extension/README.md:138` forbids exactly that. See also the 10/10 vs 11/11
   contract-check count (`docs/live-e2e-2026-11-03.md:337` vs `pi-extension/README.md:132`).
4. **One caveat, four times.** The P2 / L3 label collision is explained at `README.md:131-139`,
   `ARCHITECTURE.md:168-173`, `TODO.md:88-91` and `spec/1.0-spec.md:194`. A reader who
   reads three documents reads the same disambiguation three times, and none of them defers to
   the others.
5. **A sentence lost its link.** `bench/README.md:19-20`: "…is refused unless `--expect-model`
   names one — the guard `docs/model-sweep-consolidated.md` (CORRECTION) exists for." — the
   link is gone and the sentence no longer parses.
6. **`docs/testing.md` opens on a host no clone has.** Its first two sections are `ssh
   bench-host "powershell -Command ..."` (`:10`) and a Windows launch recipe (`:16-24`); the
   local recipe that actually works on a reader's box is at `:43-48`, underneath. Nothing says
   which of the two a stranger should use.
7. **Six unmerged branches are one `git push --all` from publishing.** `docs/pre-publish-audit.md:425-429`
   asks for them to be deleted or audited before publication; they are still there —
   `bench/ab-probe`, `bench/ab-run`, `bench/local-classifier`, `bench/or-models`,
   `bench/or-models-2`, `bench/qwen-family`, 9 commits not on `main`.
8. **The largest private disclosure is still in the repo.** `docs/pre-publish-audit.md:418-423`
   recommends cutting `docs/assets-inventory.md` to its conclusions; it still names ~35 private
   sibling directories by path (`:29-36`, `:53`, `:201`).

## The first dead end in the quickstart

`README.md:166`, the first command after the clone:

```sh
python3 -m venv .venv && . .venv/bin/activate   # Debian/Ubuntu also need `python3-venv`
```

```
The virtual environment was not created successfully because ensurepip is not
available.  On Debian/Ubuntu systems, you need to install the python3-venv
package using the following command.

    apt install python3.10-venv

You may need to use sudo with that command. ...
Failing command: /tmp/ti-venv-probe/bin/python3
```

The comment names the missing package but not what to do about it without root, and the
README gives no fallback. Two facts make this worse than it looks: the install line that
follows it (`:167`) does work here (`pip install --dry-run -e ".[dev]"` resolves in the
system interpreter), and `CONTRIBUTING.md:11-14` says the suite runs "from a fresh clone with
nothing installed" — so the venv step is the only obstacle and the README does not say you
may skip it.

Everything the quickstart promises *after* that point is true on this box, when run
directly: `PYTHONPATH=service python3 -m ti.http.api` serves, `/health` matches the documented
body byte for byte, the L0 `useTool` example returns
`{"status": "done", "result": ["pyproject.toml"], …}`, `/v1/plan` matches, the token file is
created at 0600 on first boot and a fresh `$HOME` gets one made for it, and the
`TI_CONF_OVERLAY` fixture replay works as documented. (Port 1113 was briefly held by another
instance during this read — `ti-serve` would have exited with `[Errno 98] Address already in
use`, still the one failure with no recovery hint in the message.)

## What is genuinely well presented

1. **The Honest gaps list is accurate, item by item.** Keyword routing
   (`/v1/plan` really does answer `"rule": "RESEARCH_VERB"`), `reddit_digest` unrunnable,
   `web_search`/`web_fetch` needing configured CLIs (`CLI not found: …`) and L3/L4 being
   placeholders all held up against the code.
2. **`docs/confidence-sources.md`** classifies every `conf` in the system as measured,
   derived or asserted, gives each one a disposition, and names the one prior it deliberately
   keeps. It is the best artifact in the repo and the model the other documents should follow.
3. **The negative results are real and reachable.** `bench/classical_baseline.py` prints
   `coverage 0.0%` at the shipped 0.70 gate and `62.7%` against d1-3B's 74.0%;
   `docs/vulkan-igpu-2026-11-03.md` names the evidence file per number; the round-trips report
   states its own non-reproduction before anyone asks.
4. **`spec/1.0-spec.md:186-210`** — a "Divergences from this spec" table that says, per row,
   whether the code or the spec is right, including one row where neither is.
5. **The quickstart's three samples are byte-accurate** (`/health`, the L0 `useTool` body,
   `/v1/plan`), every local markdown link in the repo resolves (0 broken of the 39 checked),
   and `bench/demo.py` runs offline in about a second with a header that states which decisions
   were recorded rather than made.
