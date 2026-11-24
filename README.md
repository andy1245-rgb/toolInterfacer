# toolInterfacer

Tool routing for a coding agent: the model says what it wants in plain language, a small local
decision model picks the operation and fills in its arguments, and the caller gets one answer plus
a receipt of every hop that produced it.

[![CI](https://github.com/andy1245-rgb/toolInterfacer/actions/workflows/ci.yml/badge.svg)](https://github.com/andy1245-rgb/toolInterfacer/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Python 3.10 – 3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)

## What it does

`useTool(intent, params?)` is the only tool the agent calls. Behind it:

- **Three levels of work.** The caller named the operation (**L0** — no decision at all, just
  validate and run), the intent is vague but names one thing (**L1** — glob the candidates, pick
  one, read it), or it is compound (**L2** — an authored template such as research or a reddit
  digest, with the decision model animating the choice nodes).
- **One primitive wall.** Every decision inside a plan is a `choice`, `noul`, `score` or `extract`
  against `service/ti/decide/primitives.py`, and no template names a backend. Swapping the decision
  model is one line in `service/ti/conf/defaults.json`.
- **Nine registered ops.** Three filesystem (`filesys_glob`, `filesys_grep`, `filesys_read`), two
  web (`web_search`, `web_fetch`), `reddit_digest`, and three decision steps surfaced as ops so a
  template speaks one vocabulary.
- **A receipt, not a transcript.** `hops[]`, `assumptions[]`, `degraded`, and a `confidence` only
  where a hop actually measured one.

An intent and the answer it produces — offline, no model, no network, no install:

```console
$ python3 bench/demo.py
scene 2 — a vague intent, resolved against real files
  intent  'read the notes file'
          L1 slot fill — the service globs the directory, picks, then reads

  step            op                       conf  artifact
  candidates      filesys_glob             1.00  path_list[3]
  picked_file     decide_pick              0.93  picked_items[1]  notes.md
  content         filesys_read             1.00  file_content :: # Project notes

  status done   confidence 0.93   3 hops
```

Three hops behind one call. The layer exists to decouple the hop count from the round-trip count:
that helps when a model pass is expensive and hurts when the local decision is, and which way it
lands is measured rather than asserted.

## Measured: routing accuracy

On a frozen 415-row routing set the fused decision arm scores **84.1% (349/415)** against the
recorded 3B baseline's **78.3% (325/415)** and passes all four pre-registered conditions (+24 rows,
exact McNemar p = 3.0e-06) — but that is the *pooled* corpus, on the 207 held-out rows alone the
same arm gains **+3 rows at p > 0.05**, and it is **not the router this repo ships**.

| condition, declared before the run | result |
|---|---|
| C1 Δ ≥ +20 rows | **pass** — +24 |
| C2 exact McNemar p < 0.01 | **pass** — 3.0e-06 |
| C3 no label worse than −3 | **pass** — worst net 0, because the two ops the base owns are protected |
| C4 positive in ≥ 4 of 5 fold seeds | **pass** — 5 of 5 (+24, +22, +23, +24, +25) |

Both counts recompute from `bench/results/fusion-option3-2026-11-10/summary.json`
(`paired_vs_reference`: 325 → 349, 26 fixed, 2 broken); the conditions were written to
`bench/results/fusion-option3-prereg.json` before the run; and `tests/test_fusion_option3_claim.py`
re-derives every condition from the rows, so a summary that disagrees with its own rows fails the
suite.

What that number is, and is not:

- **Pooled, not held out.** The pre-registration covered all 415 rows. The eval half moves
  165 → 168 at p > 0.05, and quoting +24 as a held-out gain is the specific error the test exists
  to prevent.
- **Not attributed to fusion.** At the published fold rule and seed the cheap member on its own
  also reaches 349/415, so the report carries the member's configuration as a live competing
  explanation rather than crediting the blend.
- **An authored corpus.** 415 rows written for this router; per `docs/fusion-2026-11-10.md`, 86% of
  them contain a literal slot value that also appears in the frame generator, which makes them
  easier than real traffic.
- **Cheap to run.** p50 0.85 ms per decision with no model server in the loop (`summary.json`),
  where the recorded 3B it is compared against replays at p50 2714 ms.

| other measurements | number | record |
|---|---|---|
| round trips removed, 9 fixed tasks, two arms | 2 of 14 (14.3%) | `bench/results/roundtrips-2026-11-03.json` |
| wall time at an assumed 2 s model pass | **7.5 s slower**; break-even ≈5.8 s per pass | same |
| decisions vs tool work in that run | 10.56 s of decisions against 0.14 s of tool execution | same |
| suite | 910 tests, 586 subtests, no network, no model | `python3 -m pytest tests/ -q` |

## Quickstart

Python 3.10+ is the only requirement for the demo, the harness and the suite. The first command
installs nothing:

```sh
git clone https://github.com/andy1245-rgb/toolInterfacer.git
cd toolInterfacer
python3 bench/demo.py      # three scenes; no model server, no network, no install
```

`--live` runs the same three scenes against the configured backend and the real CLIs instead of the
recorded fixtures.

The service is stdlib-only, so the install brings in nothing but the package:

```sh
python3 -m venv .venv && . .venv/bin/activate   # Debian/Ubuntu also need `apt install python3-venv`
python3 -m pip install -e ".[dev]"
ti-serve                                       # 127.0.0.1:1113, from conf.http.port
```

It refuses to start without a token, because it reads files on behalf of whoever asks and any local
process can open a loopback socket:

```sh
export TI_HTTP_TOKEN="$(cat ~/.local/state/ti/http.token)"   # mode 0600, created on first boot
curl -s http://127.0.0.1:1113/health
# {"status": "ok", "ops": [9 ops], "conf": true, "decide": {"choice": "pair", ...}, "cascade": false}
```

Then ask it for something. An L0 call names the operation itself, so no decision is made:

```sh
curl -s -X POST http://127.0.0.1:1113/v1/useTool \
  -H "Content-Type: application/json" -H "X-TI-Token: $TI_HTTP_TOKEN" \
  -d '{"intent":"list the markdown files",
       "params":{"tool":"filesys_glob","args":{"pattern":"*.md"}},
       "digest":{"cwd":"."}}'
# {"status": "done", "result": ["README.md", ...], "degraded": false,
#  "assumptions": [], "hops": [{"step": "L0", "op": "filesys_glob", "conf": 1.0, ...}]}
```

`digest.cwd` is the run root for the call, and it must resolve inside `ops.filesys.roots` — by
default `~`, the directory `ti-serve` was started in, and `/tmp`. `.` above is that start
directory, so run the service from the clone. A path outside all three answers `path_refused`;
widening it is a config change through `TI_CONF_OVERLAY`, not a request parameter. A *failed run*
is still served `200` with `status: "error"` — the request was accepted and the work is what
failed, so read `status` and not the status line.

Run the suite:

```sh
python3 -m pytest tests/ -q      # 910 passed, 586 subtests, ~78 s, no network, no model
```

The pi extension that exposes `useTool` to the agent — the tool registration, the `/ti` command and
the service lifecycle — lives in [`pi-extension/`](pi-extension/README.md), with its own install
notes and contract checks.

## How it is measured

One dataset seam, one predictor seam, one set of metrics, one set of controls ([`docs/harness.md`](docs/harness.md)):
a scikit-learn pipeline, a fine-tuned encoder and a llama-server are all the same kind of arm, so
scoring a model is not a new script each time. Every artifact records its fold rule, its fold seeds
and its option order, because each of those moves the answer — reversing the eight option
descriptions alone moved 34 argmaxes and cost 9 rows. Conditions are written to a pre-registration
file *before* the run, an A/A arm measures the floor, and a per-label ledger shows who paid for the
rows that were won. Exact McNemar throughout: at n = 415 the approximation is not worth the
argument.

The baseline the headline compares against reproduces offline, with no model and no server:

```sh
python3 -m bench.harness.run --dataset tool_routing \
  --arm floor_a='recorded|arm=floor_a' --arm grep_only='recorded|arm=grep_only' \
  --reference floor_a --aa --out /tmp/repro
```

| arm | correct | accuracy |
|---|---:|---:|
| `floor_a` — the recorded 3B | 325/415 | **78.3%** |
| `grep_only` — an earlier rewrite arm | 328/415 | 79.0% |

`+3 rows, 10 fixed, 7 broken, exact McNemar p = 0.6291`, and an A/A arm run twice flips 0 choices
in 415 rows — which is the shape of every result in this repo that did not ship.

Read in this order: [`docs/harness.md`](docs/harness.md) for the seams,
[`docs/eval-harness-2026-11-04.md`](docs/eval-harness-2026-11-04.md) for what the earlier local
numbers become under an honest split (including the 150-row set that cannot be split by family at
all), [`docs/fusion-2026-11-10.md`](docs/fusion-2026-11-10.md) for the arm whose four conditions
passed and the sensitivity that undercuts it, and
[`docs/closed-experiments-2026-11-10.md`](docs/closed-experiments-2026-11-10.md) for five
experiments shut with the number that closed them.

### Shadow mode

`decide.shadow` ships **on** in `service/ti/conf/defaults.json`. It returns exactly the answer the
service returned before, and afterwards writes what the fused policy *would* have said on the same
request, plus one `outcome` line per executed op under the same `id`. The model is asked exactly
once — `tests/test_shadow.py` asserts it — a raising member is recorded rather than failing the
request, and the log goes to `$TI_SHADOW_LOG` or `~/.local/state/ti/shadow.jsonl`, never a path in
the repository.

Read `vocab_match` first in any record. Today the only `choice` question the live path asks is
`decide_pick`, whose options are candidate ids, so the eight-op member abstains and
`{"why": "member abstained", "vocab_match": 0.0}` is the *normal* line — that is a limit of what
this log can answer, not agreement. [`docs/shadow-mode-2026-11-10.md`](docs/shadow-mode-2026-11-10.md)
says which question it does measure, and which routing change would make it measure the other.

## What it does not do

- **It is not the shipped router.** `decide.backends_in_use.choice` is `pair`; the fused arm above
  is measured, pre-registered and deliberately not enabled. Nothing in `conf/defaults.json` was
  changed to make the number.
- **The headline does not survive the split.** +3 rows on the held-out half, not significant, and
  the member alone matches the fused arm — so the gain is a real corpus result and not an
  established mechanism.
- **Intent classification is keyword frames, not the model.** The decision model is used *inside* a
  plan, never to choose the plan. L3 (a model-planned DAG) is a docstring,
  `service/ti/planner/free.py`; L4 (browser) is a row in the spec's level table and nothing else.
- **Two of the compound ops need software this repo does not vendor.** `ops.web_search.script`,
  `ops.web_search_provider.script` and `ops.web_fetch.script` ship empty — they name a third-party
  CLI whose install path differs per machine — and `reddit_digest` is written against a CLI that is
  not included. Until you point them somewhere, those ops raise `CliMissingBinary` naming the key.
- **It is not faster at every cost.** 2 of 14 round trips removed and 7.5 s slower at an assumed 2 s
  model pass; the two arms cross near 5.8 s per pass. On single-call shapes it is pure overhead.
- **The token is broad.** A valid `X-TI-Token` authorises reading any file under the configured
  roots, which by default includes `~/.ssh` and `~/.aws`. That is a documented exposure, not an
  oversight: [`docs/security-surface-2026-11-04.md`](docs/security-surface-2026-11-04.md).

## Layout

```
service/ti/       Python decision service — stdlib only: planner, runner, ops, decide wall, HTTP
pi-extension/     TypeScript pi extension: the useTool tool, the /ti command, service lifecycle
bench/            the measurement: harness seams, drivers, lane launch, and recorded results
tests/            the suite: the real engine with recorded decisions, no model, no network
docs/             design, contracts and every measurement — indexed in docs/README.md
spec/             the normative design, with a closing "divergences from this spec" section
data/distill/     the frozen, certified training corpus
demo/             the recorded demo session and the picture rendered from it
tools/            the dataset and distillation pipelines that produce data/distill
```

## Licence and attribution

Apache-2.0 — [`LICENSE`](LICENSE), and `license = "Apache-2.0"` in `pyproject.toml`.

| borrowed or external | how it is used | terms |
|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | the local decision model is served by it over HTTP; the source is **not** vendored here | MIT |
| the pi coding agent | the extension plugs into it; pi runs the TypeScript directly, so there is no build step | see `pi-extension/README.md` |
| the distilled corpus in `data/distill/` | generated by `tools/distill/` from frames authored in this repo, then audited and frozen; the teacher's own answers are excluded as labels | [`data/distill/README.md`](data/distill/README.md) |
| the 415-row routing set | authored for this router, not scraped | [`docs/fusion-2026-11-10.md`](docs/fusion-2026-11-10.md) |
