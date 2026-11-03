# Releasing toolInterfacer

A release is three artefacts, in this order: an annotated **git tag**, a **GitHub
release** pointing at it, and — only then — the **PyPI upload**. Each one is cheap to stop
before the next; none is cheap to undo after it.

## The two things a tag must never do

> **1. Never push a tag while CI is red — or before CI has run at all.** A tag is a
> promise that this exact tree installs and its suite passes. Push the commit to `main`
> first, watch the run for *that* commit go green, and only then push the tag.
>
> **2. Never tag a dirty tree.** If `git status --porcelain` prints anything, whatever you
> tagged is not what you tested. Uncommitted work is not in the archive, not in the sdist,
> and not in the wheel — and it will be the thing that was actually being tested.

Both are checked mechanically in step 1 and step 4 below. Do not skip them because the
diff "looks fine".

---

## 0. Prerequisites

- Push rights on `https://github.com/andy1245-rgb/toolInterfacer`.
- A PyPI account with a **project-scoped API token** for `toolinterfacer` (PyPI → Account
  settings → API tokens), and — for the first release — a **pending trusted publisher or
  a first manual upload** that claims the name `toolinterfacer` before anyone else does.
  A TestPyPI token is a separate token.
- `build` and `twine`. Step 3 installs them into the throwaway venv it builds in, so they
  never reach your runtime environment. Installed there with:

  ```sh
  python3 -m pip install --upgrade build twine
  ```

  On Debian/Ubuntu a venv needs `python3-venv` installed first; `python3 -m venv` fails
  with `ensurepip is not available` if it is missing.

Nothing else in this document adds a runtime dependency. The package installs with
`pip install toolinterfacer` and nothing else, and step 3 proves it.

## 1. Pre-flight — every command must exit 0

Run these from the repository root. If any one of them fails, stop here: there is nothing
to roll back yet, and that is the point of doing it first.

```sh
# the tree you are about to tag is the tree in the repo
git switch main
git pull --ff-only origin main
git status --porcelain          # must print NOTHING. Rule 2.

# the suite, offline: no network, no model, no CLI
python3 -m pytest tests/ -q     # expect: all pass — the count is on the README's Test suite row

# the benchmark harness, offline, end to end
python3 bench/run.py selftest   # expect: "selftest PASSED — harness path is offline-clean"

# the demo a stranger runs first
python3 bench/demo.py           # expect: exit 0, three scenes, all "status done"
```

`bench/demo.py` starts a real service on an ephemeral port. Run it on an otherwise idle
box: one `RemoteDisconnected` in scene 2 has been observed while the full test suite
was running concurrently, and not reproduced in 13 subsequent runs (10 solo, 3 concurrent).
If it fails, re-run it alone before assuming the tree is bad.

## 2. Version and changelog agree

```sh
grep '^version' pyproject.toml          # expect: version = "0.1.0"
sed -n '1,12p' CHANGELOG.md             # expect: an "## [0.1.0] — <today>" heading
grep -n 'Unreleased' CHANGELOG.md       # expect: no match, or an empty Unreleased section
```

The version string lives in exactly one place (`[project] version` in `pyproject.toml`).
If you bump it, bump the CHANGELOG heading in the same commit, and let that commit go
through CI like any other.

## 3. Build, inspect, and prove a clean tree installs

`git archive` exports exactly what was committed — no untracked files, no build output, no
`.git`. This is the closest thing to a stranger's clone you can do before publishing.

```sh
rm -rf /tmp/ti-rel && mkdir -p /tmp/ti-rel
git archive --format=tar HEAD | tar -x -C /tmp/ti-rel
cd /tmp/ti-rel

python3 -m venv .venv && . .venv/bin/activate
python3 -m pip install -e ".[dev]"
python3 -m pytest tests/ -q         # expect: all pass, same count as the pre-flight run
python3 bench/run.py selftest       # expect: exit 0
python3 bench/demo.py               # expect: exit 0
```

Then build the distributions and check what is actually in them. Install the build tools
into this same venv — the one that already has the package, so what you build is what you
tested:

```sh
python3 -m pip install --upgrade build twine
python3 -m build                    # writes dist/*.whl and dist/*.tar.gz
python3 -m twine check dist/*       # expect: PASSED for both files

python3 - <<'PY'
import glob, pathlib, zipfile
wheel = glob.glob("dist/*.whl")[0]
shipped = {n for n in zipfile.ZipFile(wheel).namelist() if n.startswith("ti/")}
src = pathlib.Path("service/ti")
expected = {str(p.relative_to("service")) for p in src.rglob("*")
            if p.is_file() and p.suffix in (".py", ".json") and "__pycache__" not in p.parts}
assert not expected - shipped, sorted(expected - shipped)
assert "ti/conf/defaults.json" in shipped
print(f"{wheel}: {len(shipped)} entries, {sum(x.endswith('.py') for x in shipped)} modules")
# 0.1.0: 63 entries, 62 modules
PY
```

And that the console script works from the wheel, with nowhere to fetch a dependency:

```sh
python3 -m venv /tmp/wheelcheck
/tmp/wheelcheck/bin/pip install --no-index --find-links dist toolinterfacer   # --no-index: dependencies = []
/tmp/wheelcheck/bin/ti-serve &            # or run it in a second terminal
curl -s http://127.0.0.1:1113/health      # expect: status ok, 9 ops, conf true
```

`conf: true` is the part that matters: it means `ti/conf/defaults.json` resolved out of
the installed wheel and not out of the source tree you are standing in. If the service
binds a port that is already taken, point it elsewhere — `TI_CONF_OVERLAY` takes a *path*
to a partial config, not a JSON string:

```sh
printf '{"http":{"port":1131}}' > /tmp/port.json
TI_CONF_OVERLAY=/tmp/port.json /tmp/wheelcheck/bin/ti-serve &
curl -s http://127.0.0.1:1131/health
```

If `--no-index` fails, a third-party import has crept in. `dependencies = []` is a policy,
not an accident — fix the import, do not add the dependency to `pyproject.toml`.

## 4. Tag — annotated, and only after green

```sh
cd /path/to/toolInterfacer
git switch main
git pull --ff-only origin main
if [ -n "$(git status --porcelain)" ]; then echo "DIRTY TREE - do not tag"; exit 1; fi

git push origin main          # CI runs on this commit
```

Wait for **that commit's** run to go green: the `pytest (3.10)`, `pytest (3.12)`,
`bench harness + demo (offline)`, `extension typecheck`, `extension checks` and
`wheel build, install and serve` jobs all passed. The tag is created *after* the wait, on
purpose — that is Rule 1 enforced by the order of the page rather than by your memory:

```sh
git tag -a v0.1.0 -m "toolInterfacer 0.1.0"
git show v0.1.0 --stat | head -20                     # the tag points where you think
git push origin v0.1.0
```

CI also runs on `v*` tags, so the tag itself gets a run. Confirm it is green before
uploading — that is the run you will point at if anyone asks whether 0.1.0 was tested.

## 5. Publish to PyPI — TestPyPI first

Keep the venv from step 3 active; `twine` and `dist/` both live in `/tmp/ti-rel`.

Do the whole upload against TestPyPI once. It costs a minute and it is the only place you
can practise.

```sh
python3 -m twine upload --repository testpypi dist/*        # paste the TestPyPI token

# install the artefact you just published into a venv that holds nothing else, so the
# test is of the upload rather than of the tree you are standing in
python3 -m venv /tmp/testpypi-check
/tmp/testpypi-check/bin/pip install --index-url https://test.pypi.org/simple/ \
        --no-deps toolinterfacer==0.1.0
/tmp/testpypi-check/bin/ti-serve &                          # expect: binds 127.0.0.1:1113
curl -s http://127.0.0.1:1113/health                        # expect: status ok, 9 ops
```

Then the real one:

```sh
python3 -m twine upload dist/*
```

`twine` prompts for the username and password. Username is `__token__`; the password is
the project-scoped token. Do **not** pass the token on the command line — argv lands in
your shell history and in the process table. If you must script it, put it in the
environment from a secret store, never in the command.

A version number on PyPI is permanent. `0.1.0` cannot be uploaded twice, and cannot be
re-uploaded with different bytes, ever. See §7 before you press enter.

## 6. GitHub release, and what to announce

```sh
gh release create v0.1.0 --title "toolInterfacer 0.1.0" --notes-file /tmp/notes.md
```

Build `/tmp/notes.md` from the `## [0.1.0]` section of `CHANGELOG.md`, and lead with the
number *and its condition* — the honest version is more useful and more credible than the
flattering one:

> **toolInterfacer 0.1.0** — the big model states an intent; a small local model decides
> which tools to run and with what arguments; the service runs them and returns the hop
> list it took.
>
> On nine fixed tasks it removes **2 of 14 round trips (14.3%)** and is **7.5 s slower**,
> because 10 local decisions cost 10.6 s against 0.14 s of tool work. The arms cross at
> **~5.8 s per model pass** or **~1 s per decision** — it pays for compound intents on slow
> models with fast local decisions, and is pure overhead on single-call shapes.
> Routing accuracy on the frozen 150-row set is **74.0%**. Both numbers come with their
> caveats: [`docs/roundtrips-2026-11-03.md`](docs/roundtrips-2026-11-03.md),
> [`docs/model-sweep-consolidated.md`](docs/model-sweep-consolidated.md).
>
> `pip install toolinterfacer` then `ti-serve`. No runtime dependencies, Python 3.10+.
> To see it without installing anything: `python3 bench/demo.py`.

Announce it where the audience already is — the pi agent's plugin/registry channel, the
README's own links, and whatever forum raised the question the project answers. Link the
README rather than pasting it. Say plainly what does not work yet: intent classification is
still keyword rules, `reddit_digest` has never run against a real CLI, `web_search` /
`web_fetch` default to skill scripts a fresh clone does not have, and L3 is a docstring.
`TODO.md` is the current list.

## 7. Rollback

Decide which of the three artefacts is wrong, and stop at the first one that is.

**The tag is wrong and was never pushed.**

```sh
git tag -d v0.1.0
```

Nothing else happened. Fix, commit, re-tag from a green commit.

**The tag is wrong and was pushed, but is not released yet.**

```sh
git push origin :refs/tags/v0.1.0     # delete the remote tag
git tag -d v0.1.0
```

Acceptable while nobody has fetched it. If someone has, they keep a stale `v0.1.0` that
now means something else. Prefer leaving it and shipping `0.1.1` — a moved tag is worse
than an extra version.

**The upload is wrong and PyPI has the files.**

A published version cannot be reused. Choose one:

- **Yank it** (PyPI → your project → Releases → `0.1.0` → *Yank*). Yanked releases are
  skipped by `pip install toolinterfacer` but still resolve when pinned exactly, so anyone
  who already installed it is not broken mid-build. This is the right choice for almost
  every case. Then release `0.1.1` with the fix.
- **Delete the release files**, only if it is minutes old and nothing has mirrored it.
  PyPI records file hashes and mirrors copy them; deleting does not un-copy.

Never re-upload `0.1.0` with different bytes. The version number is spent.

**The announcement is wrong.**

Edit the GitHub release body. It is a document, not an artefact — edit it in place, and fix
the CHANGELOG in the next commit so it does not disagree with what was published.

## 8. What this release does not publish

- **The pi extension does not go to npm.** It is installed from source
  (`cd pi-extension && npm ci`); `pi-extension/` is not in the wheel and should not be.
- **`bench/`, `tests/` and `docs/` are not installed** by `pip install toolinterfacer`.
  They ship in the **sdist**, so a distribution can rebuild and retest from it — but a
  wheel install gives you the `ti` package and `ti-serve`, nothing else.
- **No model and no CLI is vendored.** `d1_systemone` and `logit_read` need a llama.cpp
  server; `web_search`, `web_fetch` and `reddit_digest` need the scripts named in
  `service/ti/conf/defaults.json`. A fresh install answers `/health` and L0 file
  operations and nothing more, on purpose — see the honest-gaps list in `README.md`.
- **`dist/` is gitignored.** Never commit build output; the tag is the source of truth.
