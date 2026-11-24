# toolInterfacer

Tool routing for a coding agent: the agent says what it wants in plain language, a small decision
model picks one of nine operations and fills in its arguments, and the caller gets one answer plus a
receipt of every hop that produced it. The shipped backend is `pair` — a local 3B that escalates to a
hosted model only below τ = 0.9.

[![CI](https://github.com/andy1245-rgb/toolInterfacer/actions/workflows/ci.yml/badge.svg)](https://github.com/andy1245-rgb/toolInterfacer/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Python 3.10 – 3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)

## What it does

`useTool(intent, params?)` is the only tool the agent calls. Behind it:

- **Three levels of work.** The caller named the operation (**L0** — no decision, just validate and
  run), the intent is vague but names one thing (**L1** — glob, pick, read), or it is compound
  (**L2** — an authored template such as research or a reddit digest, with the model animating it).
- **One primitive wall.** Every decision inside a plan is a `choice`, `noul`, `score` or `extract`
  against `service/ti/decide/primitives.py`; no template names a backend, so swapping the model is
  one line in `service/ti/conf/defaults.json`.
- **Nine registered ops.** Three filesystem (`filesys_glob`, `filesys_grep`, `filesys_read`), two web
  (`web_search`, `web_fetch`), `reddit_digest`, and three decision steps surfaced as ops so a
  template speaks one vocabulary.
- **A receipt, not a transcript.** `hops[]`, `assumptions[]`, `degraded`, and a `confidence` only
  where a hop actually measured one — a run that measured nothing reports no number at all.

An intent and the answer it produces, offline, with no model, no network and no install:

```console
$ python3 bench/demo.py
scene 2 — a vague intent, resolved against real files
  intent  'read the notes file'
          L1 slot fill — the service globs the directory, picks, then reads

  step            op                       ms   conf  artifact
  candidates      filesys_glob            0.2   1.00  path_list[3] e.g. server-out.log
  picked_file     decide_pick           305.8   0.93  picked_items[1] e.g. notes.md
  content         filesys_read            0.1   1.00  file_content :: # Project notes Th

  status done   confidence 0.93   3 hops in 311 ms wall
  result # Project notes The decision model runs one forward pass.
```

Three hops behind one call, and the caller made one: the layer trades hop count against round-trip
count, which pays when a model pass is expensive and costs when the local decision is — measured
rather than asserted (round-trip rows below). The `ms` column is the service's own timing, and the
first decision in a fresh process pays a one-off backend warm-up.

That is the whole quickstart: Python 3.10+ and nothing else, since the package declares no runtime
dependencies.

```sh
git clone https://github.com/andy1245-rgb/toolInterfacer.git
cd toolInterfacer
python3 bench/demo.py            # three scenes, offline
python3 -m pytest tests/ -q      # 910 tests, 586 subtests, about 2 min
```

The same engine behind an HTTP socket. `--live` swaps the recorded decisions for the configured
backends and the real CLIs; neither is needed for anything above or below.

```sh
python3 -m venv .venv && . .venv/bin/activate   # Debian/Ubuntu also need `apt install python3-venv`
python3 -m pip install -e ".[dev]"
ti-serve                                       # 127.0.0.1:1113, from conf.http.port
```

`GET /health` answers without a token and reports the nine ops, `conf: true` once `defaults.json`
resolved, and the deciding backends. A POST is never served unauthenticated: with no `TI_HTTP_TOKEN`
the service mints 32 bytes at `~/.local/state/ti/http.token` (0600, a runtime path outside the repo)
and refuses a token file anyone but its owner can read. An L0 call names the operation itself:

```sh
export TI_HTTP_TOKEN="$(cat ~/.local/state/ti/http.token)"
curl -s -X POST http://127.0.0.1:1113/v1/useTool \
  -H "Content-Type: application/json" -H "X-TI-Token: $TI_HTTP_TOKEN" \
  -d '{"intent":"list the spec files",
       "params":{"tool":"filesys_glob","args":{"pattern":"*.md"}},
       "digest":{"cwd":"spec"}}'
# {"status": "done", "result": ["1.0-spec.md"], "degraded": false, "assumptions": [],
#  "hops": [{"step": "L0", "op": "filesys_glob", "ms": 0.2, "conf": 1.0, ...}]}
```

Three properties of that call are the contract, not the example:

| | |
|---|---|
| **The run root** | `digest.cwd` must resolve inside `ops.filesys.roots` — by default the account's home directory, the directory `ti-serve` started in, and `/tmp`. Outside them the answer is `403 scope_refused`; a tool's own path argument outside the run root is `403 path_refused`. Every path an op touches is resolved against that root, so a `..` climb or a symlink out stops before the first syscall. Widening it is `TI_CONF_OVERLAY`, not a request parameter. |
| **Who may be named** | At the HTTP boundary only the three filesystem ops may be named explicitly (`http.allowed_ops`). The CLI-spawning ops and the decision primitives are reachable only through the plan layer, `POST /v1/plan` (`mode: plan`), which chooses their arguments itself. |
| **A failed run** | Served `200` with `status: "error"` and an `error` string. The request was accepted and the work is what failed, so read `status`, not the status line. |

## Measured routing accuracy

On a frozen 415-row routing set a fused decision arm scores **84.1% (349/415)** against the recorded
3B baseline's **78.3% (325/415)**, clearing all four conditions written down before the run — and the
caveat belongs in the same breath: on the 207 held-out rows alone the arm gains **+3 rows at p = 0.45**,
and **it is not the router this repo ships**.

| condition, in `bench/results/fusion-option3-prereg.json` before the run | result |
|---|---|
| C1 Δ ≥ +20 rows | **pass** — +24 (26 fixed, 2 broken) |
| C2 exact McNemar p < 0.01 | **pass** — p = 3.0e-06 |
| C3 no label worse than −3 | **pass** — worst net 0 (`decide_pick` and `web_fetch`, both untouched) |
| C4 positive in ≥ 4 of 5 fold seeds | **pass** — 5 of 5 (+24, +22, +23, +24, +25) |

Both counts recompute from `bench/results/fusion-option3-2026-11-10/summary.json`
(`paired_vs_reference`: 325 → 349, `fixed` 26, `broke` 2, `p` 3.03e-06, plus a per-label
`ledger_vs_reference`). The conditions were in the pre-registration file before the run, and
`tests/test_fusion_option3_claim.py` re-derives C1–C4 from `rows.jsonl` with the metric functions the
run used, so a summary that disagrees with its own rows fails the suite.

What the number is, and is not:

| | the honest version |
|---|---|
| **Pooled, not held out** | The pre-registration covered all 415 rows. On the 207 held-out eval rows the same arm moves 165 → 168 (79.7% → 81.2%), five fixed and two broken, p = 0.45. Quoting +24 as a held-out gain is the specific error the test exists to prevent. |
| **Not attributed to fusion** | At the published fold rule and seed the cheap member alone also reaches 349/415 — p = 1.0000 against the fusion. What the fusion adds is stability across fold assignments, a 3-row spread against the member's 20, not rows. |
| **An authored corpus** | 415 rows written for this router, and 86% of them contain a literal slot value that also appears in `tools/distill/frames.py`, which makes them easier than real traffic. The transfer test to the 150-row set is +2.0 points at p = 0.375. |
| **Cheap to run** | p50 0.85 ms per decision with no model server in the loop, against p50 2714 ms for the recorded 3B it is compared with — both `latency` blocks of that same `summary.json`. |

| other measurements | number, and the record |
|---|---|
| round trips removed, 9 fixed tasks, two arms | **2 of 14 (14.3%)** — `bench/results/roundtrips-2026-11-03.json` |
| wall time at an assumed 2 s model pass | **7.5 s slower**; break-even ≈5.8 s per pass, same sweep; 3.58 hops per round trip |

## How it is measured

| command | what it does |
|---|---|
| `python3 bench/run.py list` | the arms in `bench/arms.json`, and what each one needs to run |
| `python3 bench/run.py run --arm d1 --set v2` | score one arm over one frozen set |
| `python3 bench/run.py compare a.json b.json` | exact McNemar and the latency delta for two reports |
| `python3 bench/run.py selftest` | the whole path offline: the CI smoke test |
| `python3 -m bench.harness.run --dataset … --arm … --reference … --aa` | the four-seam harness — any dataset, any predictor, one set of metrics |

One dataset seam, one predictor seam, one set of metrics, one set of controls
([`docs/harness.md`](docs/harness.md)): a scikit-learn pipeline, a fine-tuned encoder and a
llama-server are all the same kind of arm, so scoring a model is not a new script each time. Every
artifact records its fold rule, its fold seeds and its option order, because each of those moves the
answer — reversing the eight option descriptions alone moved 34 of 415 argmaxes and cost 9 rows
(p = 0.0784). Conditions go into a pre-registration file *before* the run, an A/A arm measures the
floor, and a per-label ledger shows who paid for the rows won. Exact McNemar throughout.

The baseline the headline compares against reproduces offline, with no model and no server:

```sh
python3 -m bench.harness.run --dataset tool_routing \
  --arm floor_a='recorded|arm=floor_a' --arm grep_only='recorded|arm=grep_only' \
  --reference floor_a --aa --out /tmp/repro
```

| arm | correct | accuracy | p50 |
|---|---:|---:|---:|
| `floor_a` — the recorded 3B | 325/415 | **78.3%** | 2714 ms |
| `grep_only` — an earlier rewrite arm | 328/415 | 79.0% | 2909 ms |

`+3 rows, 10 fixed, 7 broken, exact McNemar p = 0.6291` — the shape of every result here that did not
ship. `--aa` scores the reference twice and flips 0 of 415 choices: a determinism check, not a floor
for a served model.

Read next: [`docs/fusion-2026-11-10.md`](docs/fusion-2026-11-10.md) for the arm whose four conditions
passed and the sensitivity that undercuts it,
[`docs/closed-experiments-2026-11-10.md`](docs/closed-experiments-2026-11-10.md) for five experiments
shut with the number that closed each, [`docs/eval-harness-2026-11-04.md`](docs/eval-harness-2026-11-04.md)
for what an honest split does to the earlier local numbers, and
[`docs/roundtrips-2026-11-03.md`](docs/roundtrips-2026-11-03.md) for the cost side.

## Shadow mode

`decide.shadow` ships **on** in `service/ti/conf/defaults.json`, and it is the shape any other router
change should take: measure the counterfactual on real traffic before serving it. It returns exactly
the answer the service returned before — `choice()` hands back the inner backend's object, untouched —
and appends what the fused policy *would* have said on the same request, plus one `outcome` line per
executed op under the same `id`: which op ran, whether it errored, and how many attempts it took. So
"the shadow was right and the served answer was wrong" is a number, not an impression.

| | |
|---|---|
| **What it cannot change** | The served answer, the routing tier that produced it, the cache and the pre-gate. The counterfactual is computed after the answer exists and can only fail into the log; a raising member is recorded, not raised. |
| **What it never does** | Call a model a second time, escalate, retry, or spawn anything. The counterfactual is the cheap member's opinion blended with the answer already in hand, and `tests/test_shadow.py` asserts the model is asked exactly once. |

Read `vocab_match` first in any record. The only `choice` question the live path asks is
`decide_pick`, whose options are candidate ids and URLs, so the eight-op member abstains and
`{"why": "member abstained", "vocab_match": 0.0}` is the *normal* line — a limit of what this log can
answer, not agreement. The log goes to `$TI_SHADOW_LOG`, else `~/.local/state/ti/shadow.jsonl` (a
runtime path outside the repo). [`docs/shadow-mode-2026-11-10.md`](docs/shadow-mode-2026-11-10.md)
says which question it does measure, and what would make it measure the other.

## What it does not do

- **It is not the shipped router.** `decide.backends_in_use.choice` is `pair`; the fused arm above is
  measured, pre-registered and deliberately not enabled. Nothing in
  `service/ti/conf/defaults.json` was changed to make the number.
- **The headline does not survive the split.** +3 rows on the held-out half at p = 0.45, and the
  cheap member alone matches the fused arm at a fixed fold assignment: a corpus result, not a
  mechanism.
- **Intent classification is keyword frames, not the model.** The model is used *inside* a plan, never
  to choose one. L3, a model-planned DAG, is a docstring in `service/ti/planner/free.py`; L4 (browser)
  is a row in the spec's level table and nothing else.
- **Three of the nine ops need software this repo does not vendor.** `ops.web_search.script`,
  `ops.web_search_provider.script` and `ops.web_fetch.script` ship empty, and `reddit_digest` is
  written against a CLI that is not included; until you point them somewhere those ops raise
  `CliMissingBinary` naming the key. `--live` needs a llama-server and a GGUF, both gitignored local
  state rather than repo content.
- **The token is broad on purpose, and is the exposure to read first.** Any valid `X-TI-Token`
  authorises reading any file under `ops.filesys.roots` — by default the account's own `~/.ssh` and
  `~/.aws`, runtime paths outside the repo. Documented rather than hidden:
  [`docs/security-surface-2026-11-04.md`](docs/security-surface-2026-11-04.md).

## Layout

| path | what lives there |
|---|---|
| `service/ti/` | the Python decision service, stdlib only: planner, runner, ops, the decide wall, HTTP |
| `pi-extension/` | the TypeScript pi extension — the `useTool` tool, the `/ti` command, service lifecycle ([notes](pi-extension/README.md)) |
| `bench/` | the measurement: the arm runner, the four-seam harness, the demo, and the recorded results ([notes](bench/README.md)) |
| `tests/` | the suite: the real engine driven by recorded decisions, no model and no network |
| `docs/` | design, contracts and every measurement — indexed by purpose in [`docs/README.md`](docs/README.md) |
| `spec/` | the normative design, closing with a "divergences from this spec" section |
| `data/distill/` | the frozen, certified training corpus ([provenance](data/distill/README.md)) |
| `demo/` | the recorded demo session: the asciinema cast and the picture rendered from it |
| `tools/` | the dataset and distillation pipelines that produce `data/distill/` |
| `.github/` | CI: the suite on 3.10 and 3.12, the offline harness self-test and the demo, the extension typecheck plus its contract and capture checks, and a wheel that must install into a clean venv with `--no-index` and answer `/health` |

Flat at the top: [`ARCHITECTURE.md`](ARCHITECTURE.md) (how a call becomes executed work),
[`CONTRIBUTING.md`](CONTRIBUTING.md), [`CHANGELOG.md`](CHANGELOG.md), [`RELEASE.md`](RELEASE.md),
[`TODO.md`](TODO.md) (the open items), and [`pyproject.toml`](pyproject.toml) — `toolinterfacer` 0.1.0,
`requires-python >=3.10`, `dependencies = []`, entry point `ti-serve`. Gitignored, not repo content:
`models/`, `extern/` (GGUFs, a llama.cpp checkout) and `var/` (generated rows).

## Licence and attribution

Apache-2.0 — [`LICENSE`](LICENSE), and `license = "Apache-2.0"` in `pyproject.toml`.

| borrowed or external | how it is used | terms |
|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | the local decision model is served by it over HTTP; the source is **not** vendored here | MIT |
| the pi coding agent | the extension plugs into it; pi runs the TypeScript directly, so there is no build step | see [pi-extension](pi-extension/README.md) |
| the corpus in `data/distill/` | generated by `tools/distill/` from frames authored in this repo, then audited and frozen; the teacher's own answers are excluded as labels | [provenance](data/distill/README.md) |
| the 415-row routing set | authored for this router, not scraped | [`docs/fusion-2026-11-10.md`](docs/fusion-2026-11-10.md) |