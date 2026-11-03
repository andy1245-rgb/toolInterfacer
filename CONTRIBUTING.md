# Contributing

## Run the tests

```sh
python3 -m pytest tests/ -q
```

That is the whole suite: 700+ tests, no network, no model, no CLI. Run it and the last line
prints the exact count; the README's [Test suite row](README.md#what-works-today) is the
only place that number is written down, so it is the one to trust and the one to update. It
runs from a fresh clone with nothing installed — each test file puts `service/` on
`sys.path` itself. Installing the package (`pip install -e ".[dev]"`) also works and is
what CI does.

Every test is a `unittest` case, so either runner is fine:

```sh
python3 -m unittest discover -s tests -p 'test_*.py'
```

If you touch the pi extension, its checks are separate and Node-only:

```sh
cd pi-extension
npx --yes -p typescript@5.5.4 tsc --noEmit
node scripts/contract-check.ts
```

## Portability rule (enforced by a test)

Do not hardcode a machine-personal absolute path in source, config, tests or
docs — no `/home/<user>/...`, no `C:\Users\<user>\...`. Use an environment
variable, a `~`-relative default, a repo-relative path, or a `<placeholder>`.

`tests/test_no_personal_paths.py` scans the tree and fails the suite if one
appears. The fix is always to parameterise the path, never to add an exception.
Two trees are skipped by path, on purpose: `extern/` (vendored) and
`service/logs/` (runtime output); generated trees are skipped by directory name
(`node_modules`, `__pycache__`, `.venv`, `.git`, `build`, `build-vk`).
`bench/results/` used to be exempt and is not any more: the recorded model paths
are redacted to `$HOME/` and `C:\Users\<user>`, and the basenames that prove the
phantom-lane point are kept, so the evidence is still guarded.

Machine-specific bench settings go in `bench/lane.env`, which is gitignored —
copy `bench/lane.env.example`.

## Commit style

Conventional Commits, lower-case, imperative, with the affected module as the
scope when there is one:

```
feat(decide): add openrouter, a hosted backend behind a provider pin
fix(planner): a listing covers the directory it names, not the cwd
perf(decide): bound the candidate payload and stop duplicating it
docs: state the enforced portability rule
chore: remove machine-personal paths from portable files
```

The subject carries the change *and* its reason, usually as a comma or a `not`
clause — `fix(planner): a listing covers the directory it names, not the cwd`
says what was wrong, not just what file moved. Branch work lands as a
`merge: <the same sentence>` commit rather than a squash, so the reasoning stays
readable in history.

## Scope rules

- One concern per file. If a file stops being describable in one sentence, split
  it.
- Pipelines and the runner never import a decide backend directly. They address
  `choice` / `noul` / `score` / `extract` through `ti.decide.primitives`; the
  backend is a config line.
- Templates publish the *shape* of a pipeline. Every degree — top-k, caps, gate
  thresholds, timeouts, backend picks — lives in `service/ti/conf/defaults.json`.
  A `TODO` with a number in it is a config key that has not been added yet.
- Behaviour changes need a test that fails before the change. The suite is the
  only thing standing between a config edit and a silently degraded run.

## Before opening a pull request

```sh
python3 -m pytest tests/ -q     # green
python3 -m build --wheel        # build the wheel; CI then asserts the service
                                # modules it names are inside it
```

See [`docs/`](docs/) for the architecture, the ops CLI contract, the testing
notes and the measured decision-backend sweeps.
