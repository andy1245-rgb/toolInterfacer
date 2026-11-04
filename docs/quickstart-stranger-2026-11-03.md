# First contact: what a stranger hits

A cold read of `README.md`, followed literally, from a real `git clone` into
`/tmp/stranger-ti` (tracked files only: no `models/`, `extern/`, `service/logs/`,
`node_modules/`, `.venv/`). Clean venv, no system Python, branch `ov-quickstart`
from `main @ 8b385ee`.

Every command below was run. Output is quoted, not paraphrased.

---

## Verdict

**Yes — a stranger reaches a working `useTool` call, but not by following the README in
order.** The documented L0 example fails with `403 scope_refused` unless the repo happens
to be cloned under the reader's home directory, and the README never says so. Recovering
takes three wrong guesses at the request body before the fix (widen `ops.filesys.roots`)
becomes visible — the 403 text does name the config key, which is what saves the run.

Clone (`03:53`) → first successful `useTool` (`03:55:41`) was **five wrong turns**, of
which three were the scope gate and one was a port collision. The success required a
*second* clone into `$HOME`; the first clone never produced a working call without an
undocumented config overlay.

The install → console script → service → token → `/health` → `useTool` spine itself is
sound. **`pip install -e .` + `ti-serve` + a `useTool` call works from a clone**, with the
roots caveat above.

---

## Step-by-step log

| # | Command | What happened | Did the README prepare me? |
|---|---|---|---|
| 1 | `git clone <repo> /tmp/stranger-ti` | 246 tracked files. No `service/logs/`, no `node_modules/`, no `models/`. | Yes — implicitly; nothing claims those ship. |
| 2 | `python3 -m venv /tmp/venv-ti` | **Failed**: `ensurepip is not available`. Needs `python3-venv` (sudo). | No. README says "Python 3.10+ is the only requirement" and never mentions a venv. |
| 2b | `virtualenv /tmp/venv-ti` | Worked. Verified clean: no `pytest`, no `ti`. | n/a |
| 3 | `pip install -e ".[dev]"` | Exit 0. Editable wheel built; `pytest 9.2.0` pulled. | Yes. |
| 4 | `ls /tmp/venv-ti/bin/ti-serve` | Present, shebang = the venv. | Yes — README names the script. |
| 5 | `python3 -m pytest tests/ -q` | `588 passed, 349 subtests passed in 11.97s`. | Partly — README states **454 / 261** (and a different **433 / 276** in the capability table). Both stale. Fixed. |
| 6 | `ti-serve` | **Failed**: `ti: [Errno 98] Address already in use`. | No. Nothing about a port already held, and the message names no knob. |
| 7 | `ti-serve` with `TI_CONF_OVERLAY` on 11133, fresh `HOME` | Started. Created `~/.local/state/ti/http.token`, mode `600`. | Yes — the token contract is documented and true. |
| 8 | `curl /health` (no token) | `200`, `{"status": "ok", "ops": [...9 ops...], "conf": true, "decide": {...}, "cascade": false}` | Yes. Actual body is a superset of the sample; harmless. |
| 9 | `POST /v1/useTool` with **no** token | `401 {"error": "unauthenticated", "detail": "POST needs the service token in X-TI-Token (or Authorization: Bearer); see service/ti/http/auth.py"}` | Yes — actionable. |
| 10 | `POST /v1/useTool` with a **wrong** token | Same `401`. | Yes. |
| 11 | The README's L0 example, verbatim, correct token | **`403 {"error": "scope_refused", "detail": "run root '.' is outside the permitted roots ('/tmp/stranger-home'); ops.filesys.roots is what widens this"}`** | **No.** See blocker 1. |
| 12 | Guess 1: add `"digest": {"cwd": "/tmp/stranger-ti"}` | `403` — run root is outside the same single root. | No. |
| 13 | Guess 2: absolute `args.cwd` | `403`, identical body — the arg is not what is checked. | No. |
| 14 | Re-clone into `$HOME` (`/tmp/stranger-home/toolInterfacer`), same example | `200 {"status": "done", "result": ["pyproject.toml"], "degraded": false, "assumptions": [], "hops": [{"step": "L0", "op": "filesys_glob", "ms": 3.9, "conf": 1.0, "artifact_summary": "['pyproject.toml']"}]}` | The result shape now matches the README, except the documented top-level `"confidence": 1.0` is **absent**. Fixed. |
| 15 | `POST /v1/plan` with `{"intent":"research local inference latency"}` | `200` with `meta.level=L2`, `meta.template=research`, `meta.rule=RESEARCH_VERB`, and the four-step graph `queries → hits → picked → pages → digest` carrying `gate_pre` `conf_gte 0.7` and `gate_post` `is_relevant 0.75`. | Yes — matches the documented sample exactly. |
| 16 | `POST /v1/useTool` `{"intent":"research local inference latency"}` | `200` body with `"status": "error"`, `"error": "CLI not found: /tmp/stranger-home/.pi/profiles/registry/web-search/scripts/free-search"` | Yes — Honest gaps says the web ops shell out to CLIs you must point at. |
| 17 | `POST /v1/useTool` reddit intent | `200` with `{"status": "error", "error": "reddit-research is not configured: set ops.reddit.cwd to the reddit-research project directory (the clone holding .venv/ and reddit_research/)", …}`, `hops: []`. **Corrected 2026-11-04** — this row first recorded the message the service returned while `ops.reddit.cwd` still shipped a machine's own tree: `"reddit-research CLI unavailable: no project directory found; tried:\n  …/Documents/<private-project>/<project>\n  …/<project>\nSet ops.reddit.cwd (or ops.reddit.cwd_candidates) …"`. See the appended note. | Yes — gap 3, and the error is better than the README's own description. |
| 18 | `cd pi-extension && npm ci` | Exit 0, from a clone, `package-lock.json` tracked and consistent. `3 vulnerabilities (2 moderate, 1 high)`. | Mostly — `pi-extension/README.md` documents `npm ci`; neither README says it needs the network. |
| 19 | `npx tsc --noEmit` | Exit 0, silent. | Yes. |
| 20 | `node scripts/contract-check.ts` | `11/11 contract cases passed`. | Yes (bench of extension docs). |
| 21 | `python3 bench/run.py selftest` | `selftest PASSED — harness path is offline-clean and internally consistent` (6 `[ok]` lines). | Yes. |
| 22 | `python3 bench/run.py list` | `# 8 arm(s) from bench/arms.json (2 offline, 6 need an endpoint)`, then the arm table. | Yes. |
| 23 | `python3 -m tools.dataset ingest validate split export stats` | `ingest ... unrecognized arguments: validate split export stats` — five separate subcommands, not one chain. Run individually: all exit 0. `hops absent`, `live absent`; 496 records; `496 records pass the schema`; `split (seed=13) train=401, val=50, test=45`; `sft=200 rows, skipped=296`; stats table. | **No.** The word "dataset" appears nowhere in `README.md`. |
| 24 | `python3 bench/replay.py` (bare) | `"status": "error"`, `"error": "pair: no stage produced a choice"` — and **exit 0**. | No — the Benchmarks table sells this row as zero-network. See blocker 3. |
| 25 | `TI_CONF_OVERLAY=/tmp/ti-fixture.json python3 bench/replay.py --intent "read the file we discussed"` | Works: `status=done wall_ms=22.5 hops=3`, `filesys_glob 4.9ms conf=1.00 path_list[251]`, `decide_pick 17.3ms conf=0.95 picked_items[1] e.g. README.md`, `filesys_read 0.3ms`, `decide-max conf=0.95 (gate=0.7)`. | Yes — the Quickstart block is correct and the summary is exactly as advertised. Numbers differ from the sample (`[251] e.g. combined.json` vs `[179] e.g. LICENSE`) because they describe a specific working tree. |
| 26 | Token file at mode `644` | Refused to start: `ti: '.../http.token' has mode 644: group and other can read it, so it is not a token any more. `chmod 600 ...` (or delete it to have a fresh one made)` | Yes — and the message is exemplary. |
| 27 | `TI_HTTP_TOKEN=envwins` + a `644` file present | Started and authenticated on the env token: `/health` `200`, then `403 scope_refused` (i.e. past auth). Environment-first is real. | Yes. |
| 28 | `PYTHONPATH=service python3 -m ti.http.api` | Started and served (silent, as servers are). | Yes. |
| 29 | `pip install .` (non-editable) into a second clean venv | Ships `ti/conf/defaults.json`; imports from `site-packages`; `ti-serve` boots and answers `/health`. | Yes — packaging is correct. |
| 30 | Every `](...)` link in README, existence-checked | 13/13 local links resolve (`CONTRIBUTING.md`, 7 docs, `LICENSE`, `pi-extension/README.md`, `spec/1.0-spec.md`); the 4 misses are GitHub/shields URLs. | Yes. |
| 31 | `python3 -m unittest discover -s tests -p 'test_*.py'` | `Ran 588 tests ... OK`. | Yes — the alternative runner works and confirms 588 is the real count. |

---

## Blockers

### 1. The README's first `useTool` call returns `403` unless you cloned under `$HOME` — and that precondition is never stated

The single worst one: it is the first thing a reader runs to find out whether the product
works, and it fails for every clone outside the home directory (`/tmp`, `/opt`, `/srv`, a
second mount).

```sh
curl -s -X POST http://127.0.0.1:11133/v1/useTool \
  -H "Content-Type: application/json" \
  -H "X-TI-Token: $TI_HTTP_TOKEN" \
  -d '{"intent":"list the pyproject file",
       "params":{"tool":"filesys_glob",
                 "args":{"pattern":"pyproject.toml","cwd":"."}}}'
```

```
{"error": "scope_refused", "detail": "run root '.' is outside the permitted roots ('/tmp/stranger-home'); ops.filesys.roots is what widens this"}
```

Proximate cause: `ops.filesys.roots` defaults to `["~"]`
(`service/ti/conf/defaults.json`), and the run root — the service's cwd, or `digest.cwd`
if given — must resolve inside it. The README explains the confinement rule *after* the
example ("every path they touch is confined to the run's cwd … must itself resolve inside
`ops.filesys.roots`") but never states what roots defaults to, so the example reads as
unconditional. Two guesses that a reader would try first — adding `digest.cwd`, and making
`args.cwd` absolute — both 403 with the same body, because neither is what the gate reads.

Minimal fix (applied): one paragraph after the example stating the default is `["~"]`,
that the example therefore only works from a clone inside the home directory, and that
`TI_CONF_OVERLAY` is how `roots` is widened. Verified: with
`{"ops": {"filesys": {"roots": ["/tmp/stranger-pristine"]}}}` the same call returns
`{"status": "done", "result": ["pyproject.toml"], ...}`.

### 2. `ti-serve` dead-ends on an occupied port with no recovery hint

```sh
ti-serve
ti: [Errno 98] Address already in use
```

Exit 1. The message is honest and fast, but names no knob — not `conf.http.port`, not
`TI_CONF_OVERLAY`, not even "another instance may be running". The README does document
`conf.http.host` / `conf.http.port` and the overlay, in a different section, so the
information exists; it just is not reachable from the failure. A second session, or a
forgotten instance, is the ordinary way to hit this.

Minimal fix: one clause on the error path or in the quickstart, e.g. "if the port is
taken, set `conf.http.port` with `TI_CONF_OVERLAY`, or stop the other instance."
(Not applied — the message lives in `service/`, which this pass may not edit.)

### 3. `bench/replay.py` is not "zero network" as listed, and it exits 0 when it fails

The Benchmarks table says:

> `bench/replay.py` | runs one plan through the real engine with the fixture backend — zero network

The script uses the *configured* backend, which defaults to `pair`:

```sh
python3 bench/replay.py
```

```
{
  "status": "error",
  ...
  "error": "pair: no stage produced a choice"
}
```

```
=== EXIT 0 ===
```

Two defects in one: the zero-network property holds only with the `TI_CONF_OVERLAY`
fixture stanza from the Quickstart (which is correct and does work), and the failed run is
reported with exit status 0, so any wrapper, CI job or `&&` chain reads it as success. The
same "zero network" wording is repeated in `bench/README.md` §`replay.py`.

Minimal fix (applied): the README row now says zero-network holds once the overlay points
the choice decision at `fixture`, and that the script otherwise uses the configured
backend. The exit-0-on-error behaviour is a code change and is left as a finding.

### 4. `python3 -m venv` does not work on a Debian box without `python3-venv` (environment, not the repo)

```
The virtual environment was not created successfully because ensurepip is not
available.  On Debian/Ubuntu systems, you need to install the python3-venv
package using the following command:

    apt install python3.10-venv
```

No sudo available, so I used `virtualenv` instead. This is the box, not the project — but
it combines badly with the README's install line, which runs `pip install -e ".[dev]"`
against whatever `python3` is, with no venv step and no `--user`. On a host with the PEP
668 `EXTERNALLY-MANAGED` marker that same line is refused outright. This box has no such
marker (`python3 -m pip install --dry-run -e ".[dev]"` succeeds), so I could not reproduce
the refusal here; on stock Debian 12 / Ubuntu 24.04 it is the first thing a reader meets.

Minimal fix: one line under Quickstart — `python3 -m venv .venv && . .venv/bin/activate`
(or `pip install --user -e ".[dev]"`), with a note that Debian needs `python3-venv`.

---

## Gaps the README does not mention

Ordered by how much they cost a reader.

1. **`ops.filesys.roots` defaults to `["~"]`.** Covered as blocker 1, but it is equally a
   documentation hole: the security model is described as "confined to the run's cwd",
   which reads as a sandbox around the *repo*, not around the *home directory*. A reader
   who clones to `/tmp` concludes the tool is broken.
2. **Failed `useTool` calls return HTTP 200.** Both the CLI-missing and reddit-missing
   cases above came back `200 OK` with `"status": "error"` in the body. Nothing in the
   README states the status-code contract, so a client written from the extension
   description (`one compact result: result, hops, assumptions, degraded`) will treat a
   hard failure as a success. This is the most consequential undocumented behaviour for
   anyone building on the HTTP surface.
3. **`tools/` is invisible.** `python3 -m tools.dataset` has five working subcommands and a
   complete offline pipeline, and the string "dataset" does not occur in `README.md`; the
   Layout block lists `service/`, `pi-extension/`, `spec/`, `docs/`, `tests/`, `bench/` and
   omits `tools/` entirely. A reader learns it exists only by running `ls`. (It degrades
   correctly with no `service/logs/` — `hops absent`, `live absent`, exit 0 — so this is a
   discovery gap, not a quality one.)
4. **`npm ci` needs the network; the main README implies the extension needs nothing.**
   "pi executes the TypeScript directly via jiti, so there is no build step" is true and
   sits next to the typecheck instructions without mentioning that `tsc --noEmit` requires
   a populated `node_modules` first. `pi-extension/README.md` documents `npm ci`; neither
   file says it is a network operation. An offline reader cannot run the advertised check.
5. **`npm ci` reports `3 vulnerabilities (2 moderate, 1 high)`** and two packages with
   install scripts not covered by `allowScripts`. Not a blocker, but the first output a
   reader sees from the extension is a security warning with no guidance.
6. **Sample outputs are tied to one working tree.** The replay summary in the README shows
   `path_list[179] e.g. LICENSE`; a clone yields `path_list[251] e.g. combined.json`,
   because the candidate count depends on what is on disk. The prose is careful that
   latency numbers "do not reproduce on CI" but says nothing about the candidate count, so
   a reader diffing their output against the sample sees a mismatch and does not know it is
   expected.
7. **Two different test counts, both wrong.** The capability table said `433 tests, 276
   subtests`; the Tests section said `454 passed, 261 subtests passed`; reality is `588
   passed, 349 subtests passed` (confirmed independently by `unittest discover`). Fixed.
8. **One Honest-gaps entry is imprecise about its own mechanism.** The ask-back gap says
   "`http/api.py` calls `use_tool` with no ask channel". In fact `api.py:171` calls
   `use_tool(body, dry=dry, ask_via_response=not dry)` — there *is* a channel, a
   response-shaped one, and `service_entry.py` implements it. The stated consequence
   (gated steps degrade to `degraded: true` instead of asking) is the effect the code
   comment describes, but a reader who greps for `on_ask` in `http/api.py` will not find
   the gap the README says is there.

### Honest-gaps entries that are still accurate

All six hold up, and three of them predicted exactly what I hit:

- keyword-rule routing — confirmed live: `/v1/plan` reports `"rule": "RESEARCH_VERB"`.
- `reddit_digest` unrunnable out of the box — confirmed, with a better error than the
  README promises (it names both candidate paths and `ops.reddit.cwd`).
- `web_search` / `web_fetch` need configured CLIs — confirmed: `CLI not found:
  .../web-search/scripts/free-search`.
- `planner/free.py` is a one-line docstring; L3/L4 unwired — confirmed.
- destructive tools stay native — confirmed: `http.allowed_ops` is the three `filesys_*`.
- ask-back — accurate in effect, imprecise in mechanism (gap 8 above).

What the section does **not** warn about: the roots default (gap 1), HTTP 200 on error
(gap 2), the undocumented `tools/` package (gap 3), the stale counts (gap 7), and
`replay.py`'s default backend (blocker 3).

---

## What was frictionless

- **Clone → editable install → console script.** `pip install -e ".[dev]"` exit 0 into an
  empty venv, `ti-serve` on PATH, correct shebang, no dependency surprises. The service
  really is stdlib-only at runtime.
- **Packaging.** A non-editable `pip install .` ships `ti/conf/defaults.json`, imports from
  `site-packages`, and the wheel-installed `ti-serve` boots and serves. No packaging key
  was missing.
- **First-boot token.** No ceremony: the file appeared at mode `600` on first start. Both
  failure modes are better than most projects manage — the `401` names the header and the
  file, and the mode-`644` refusal prints the exact `chmod` remedy. Env-over-file precedence
  is documented and true.
- **`POST /v1/plan`.** Byte-for-byte the documented shape, including `meta.level`,
  `meta.template`, and step ids. Nothing to discover.
- **`bench/run.py selftest` and `list`.** Both offline in a clone, both meaningful
  (`selftest PASSED`, an 8-arm table that separates the 2 offline arms from the 6 that need
  an endpoint).
- **`tests/`.** 588 tests, green, no network, no model, in a clean venv, in 12 s — and they
  pass under `unittest discover` too, so the "nothing installed" claim is real.
- **`tools.dataset` robustness.** With no `service/logs/` it prints `hops absent` /
  `live absent` per source instead of crashing, and every subcommand exits 0.
- **The extension's offline checks.** `package-lock.json` is tracked and consistent, so
  `npm ci` is reproducible; `tsc --noEmit` is silent on the first try; `contract-check.ts`
  passes 11/11 without a server.
- **Doc hygiene.** Every local link in the README resolves, and the Honest-gaps section was
  accurate on every item I could reproduce — which is why the three gaps above stand out.

---

## Reproducing

Everything above ran from `/tmp/stranger-ti` and `/tmp/stranger-pristine` (clones) against
`/tmp/venv-ti` (clean, `virtualenv`) and `/tmp/venv-wheel` (non-editable install), with a
fresh `HOME` at `/tmp/stranger-home` so first-boot behaviour was genuine. Service ran on
11133–11137 via `TI_CONF_OVERLAY` because port 1113 was held by another instance; all of
my listeners were stopped and the temp trees removed afterwards.

---

## Note, appended 2026-11-04: row 17's response no longer exists

*Appended. Everything above is the 2026-11-03 record and has not been rewritten, with the
one exception marked inline: the captured body in row 17. This note says what moved and
why.*

**What changed.** `service/ti/conf/defaults.json` no longer ships a machine-specific
`ops.reddit.cwd` / `ops.reddit.cwd_candidates` (commit `812a0c7`, "conf: ship the CLI ops
unconfigured, and unpublish private project names" — the same pass that removed the last
`/home/<account>` paths). Both keys are empty now, so `resolve_project()` has nothing to
probe and takes its *other* branch:

```python
if not tried:
    raise cli.CliMissingBinary(
        "reddit-research is not configured: set ops.reddit.cwd to the "
        "reddit-research project directory (the clone holding .venv/ and "
        "reddit_research/)")
```

The message row 17 recorded — `reddit-research CLI unavailable: no project directory
found; tried: …` — is emitted only when there *are* candidates and none of them is a
directory. With the shipped conf there are no candidates, so that string is unreachable
from a fresh install. It is kept in row 17 as superseded history rather than deleted,
because the row's *finding* is unchanged and only its evidence went stale.

**Why it changed.** The old default named one operator's own `~/Documents/<project>`
tree, in package data, inside the wheel — so every install was told to look in one
person's documents folder. `tests/test_no_personal_paths.py`'s `_HOME_TREE` rule now fails
the suite if a `~`-relative path into the documents tree comes back. This is a disclosure
fix, not a behaviour fix: nothing about whether the op runs is different.

**What the service returns now.** Run, not derived: `PYTHONPATH=service python3 -m
ti.http.api` on a scratch port with `TI_CONF_OVERLAY` pointing only `http.port` and
`http.token_file` at temp locations (every `ops.*` value as shipped), then two `POST
/v1/useTool` calls — `{"intent": "what does r/MechanicalKeyboards say about switch lube"}`
and `{"intent": "what does reddit say about mechanical keyboards"}`. Both returned:

```
HTTP 200
{
    "status": "error",
    "result": null,
    "degraded": false,
    "assumptions": [],
    "hops": [],
    "wall_ms": 0.1,
    "meta": {
        "level": "L2",
        "route_gate": 0.7
    },
    "error": "reddit-research is not configured: set ops.reddit.cwd to the reddit-research project directory (the clone holding .venv/ and reddit_research/)"
}
```

Identical between the two intents apart from `wall_ms`. Four things about that body are
worth keeping straight, because they are what row 17 was really about:

- still **`200`** with `status: "error"` — gap 2 above is unchanged and is still the most
  consequential thing in this document for anyone building on the HTTP surface;
- **`hops: []`**: the failure lands before the single step records a hop, so a caller
  counting hops sees an empty plan rather than a failed step. The `error` string is the
  only evidence the plan ran at all;
- `meta.level` is `L2`, `meta.route_gate` `0.7`, and there is no `template` key on this
  route;
- the private tree path is **gone** from the body, which is the point of the change and
  also its cost: the old message told a stranger which two paths to fix, and the new one
  describes the shape of the directory to point at (`the clone holding .venv/ and
  reddit_research/`) without naming anywhere on their own disk.

**One more line in this document is superseded by the same change.** Under "Honest-gaps
entries that are still accurate", the `reddit_digest` bullet says the error arrives "with
a better error than the README promises (it names both candidate paths and
`ops.reddit.cwd`)". It still names `ops.reddit.cwd`; it no longer names candidate paths,
because there are none to name. The bullet's verdict — unrunnable out of the box, and
honest about it — stands.
