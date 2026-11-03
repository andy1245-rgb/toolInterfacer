# Cutting the GLM-5.3-Flash routing bill — measured

Everything below is a measured number from `bench/or_bench.py` on
`tests/data/utterances_v2.json`, not a price-sheet projection. Per-row provider
cost is the provider's own `usage.cost`; the endpoint that served each request is
the response's `provider` field, so no pin is assumed to have taken effect.

Run date 2026-10-26. Result files: `bench/results/or-bench-*-2026-10-26.json`
(see the map at the end). Total spent on the whole investigation: **$0.04944**.

## The one-paragraph version

Pinning the request to the cheapest endpoint (`Relace`, explicitly, with
`provider.only`) and making three prompt/answer changes — `reasoning.effort:
minimal`, a letter-code answer, and compacted operation descriptions — took the
routing bill from **$0.0202 to $0.0053 per 1k decisions (-73.8%)** while accuracy
went from 141/150 to 145/150. `sort: "price"` does not get you there: it never
left Wafer, because the cheapest endpoint does not advertise `response_format`
and JSON-mode routing steers around it. The `:batch` price is real but
unreachable from the live chat endpoint (HTTP 404).

## Headline: per-provider, 150 rows, identical request shape

One model id, 38 endpoints. Request shape is the untouched baseline here — full
prompt from `bench/live_probe.py`, JSON answer with `confidence`, default
reasoning — so the only variable is which endpoint answered. Every row is
`accuracy_pass` = one request per decision, so `$/1k` is directly comparable to
production (it excludes the 20 extra latency-pass requests).

| endpoint | quant | sheet $/M in / out | served | accuracy | $/1k decisions | $/1k correct | p50 | failures |
|---|---|---|---|---|---|---|---|---|
| **Relace** | unknown | 0.0413 / 0.0758 | 150/150 | **144/150 = 96.0%** | **$0.0174** | $0.0182 | 2575 ms | 0 |
| GMICloud | fp8 | 0.0750 / 0.2500 | 150/150 | 140/150 = 93.3% | $0.0180 | $0.0193 | 1562 ms | 0 |
| Wafer (control) | nvfp4 | 0.0540 / 0.2000 | 150/150 | 139/150 = 92.7% | $0.0195 | $0.0211 | 536 ms | 0 |
| unpinned (`auto`) | mixed | — | Wafer 149, StreamLake 1 | 141/150 = 94.0% | $0.0202 | $0.0215 | 598 ms | 0 |
| OpenInference | fp4 | 0.0767 / 0.0100 | 150/150 | 129/150 = 86.0% | $0.0139 | $0.0161 | 544 ms | 0 |
| StreamLake | fp8 | 0.0435 / 0.1450 | 130, 20 lost | 126/150 = 84.0% | $0.0242 | $0.0288 | 7942 ms | 20 × 429 |

Read it as: **pinning to the cheapest endpoint did not cost accuracy.** Relace
scored higher than Wafer on every run of the day (144–145/150 vs 139–141/150).
The two cheapest endpoints by sheet price are not the two cheapest in practice —
OpenInference's $0.01/M output is the lowest completion price in the catalogue,
but it emits 91–110 completion tokens per request (85–92 of them reasoning) and
its input price is the second highest, so it lands mid-table *and* loses 12 rows
of accuracy. StreamLake is cheap on paper and unusable: 20 hard failures and a
7.9 s p50.

`sort: "price"` is not a substitute for a pin. Measured three times — 50 rows
json, 20 rows json, 20 rows letter — it served **Wafer 100% of the time**, never
Relace. The mechanism: Relace does not declare `response_format` in its
`supported_parameters`, so it is filtered out of any request that carries
`response_format`; and with an ordered pin and fallbacks allowed
(`order: ["Darkbloom","Relace"]`) the fallback went to **Wafer 9/10** in JSON
mode but to **Relace 9/10** in letter mode. Explicit `only` is the only thing
that reliably reaches the cheapest endpoint. Cost of that difference on the same
10 rows: $0.00016 vs $0.00009.

### Noise floor: read the accuracy column with this in mind

The same endpoint, same request shape, run twice 30 minutes apart:

* unpinned (149/150 served by Wafer): 141/150
* pinned Wafer: 139/150

They **disagree on 12 of 150 individual rows** while the aggregate moves 2 rows.
So a single 150-row run resolves aggregate accuracy to about ±1.3 pp, and one row
is 0.67 pp. Every claim below about accuracy is stated against that. The earlier
baseline run of this dataset measured Wafer at 143/150 (95.3%); today's two Wafer
runs measured 139 and 141, so the endpoint's own spread spans 4 rows. A rule that
sits within ~2 pp of the 93.3% guardrail should be re-measured over 2–3 runs
before it is trusted.

## Ranked reductions, each measured

Cumulative, 150 rows each, all on the same day and dataset. "before" is the
unpinned baseline shape (full prompt, JSON + confidence, default reasoning).

| # | lever | accuracy | $/1k decisions | vs before | p50 | verdict |
|---|---|---|---|---|---|---|
| 0 | **before** — unpinned, default | 141/150 | $0.0202 | — | 598 ms | the baseline |
| 1 | pin `only: [Relace]`, nothing else | 144/150 | $0.0174 | **-13.9%** | 2575 ms | keeps accuracy, 4× the latency |
| 2 | + `reasoning.effort: minimal` | 145/150 | $0.0078 | **-61.4%** | 881 ms | biggest single win |
| 3 | + letter answer (`A`–`H`) | 141/150 | $0.0062 | -69.3% | 656 ms | accuracy-neutral within noise |
| 4 | + compact op descriptions | 145/150 | $0.0053 | **-73.8%** | 624 ms | recommended |

The factorial behind rows 2–4 (all with `Relace` + `reasoning.effort: minimal`),
so each lever's marginal effect is isolated:

| prompt | answer | accuracy | $/1k | completion tokens/req |
|---|---|---|---|---|
| full | json + confidence | 145/150 | $0.0078 | 26.4 |
| full | letter | 141/150 | $0.0062 | 10.5 |
| compact | json + confidence | 141/150 | $0.0067 | 25.5 |
| compact | letter | 145/150 | $0.0053 | 10.4 |

Compaction is worth -14% of the bill and shows no consistent accuracy direction
(+4 rows in the json cell, -4 rows in the letter cell) — i.e. neutral within the
noise floor, and it costs 29 of 172 prompt tokens. The answer format is worth
-20% and cuts completion tokens by 60%: a letter cannot be spelled wrong, so the
parse contract is also *more* robust than free-form JSON. Wafer, the endpoint the
baseline was billed on, gets the same treatment for comparison: $0.0105/1k
(compact + letter + minimal), still 2× Relace's price for the same request.

### 3. Completion tokens were reasoning tokens

Confirmed directly from `usage.completion_tokens_details.reasoning_tokens`:

* baseline shape: 56.1 completion tokens/request, 39.4 of them reasoning (70%)
* `reasoning: {"enabled": false}`: **rejected** — HTTP 400
  `"Reasoning is mandatory for this endpoint and cannot be disabled."` (50/50 rows)
* `reasoning: {"exclude": true}`: accepted, and **bills exactly what it hid** —
  134.7 completion tokens/request with `reasoning_tokens` reporting 139.3. Cosmetically
  quieter, 2.5× the cost of `effort: minimal`. Do not use it as a cost lever.
* `reasoning: {"effort": "minimal"}`: accepted, 10.2 reasoning tokens/request with
  the full JSON answer. This is the lever.

The `confidence` field is worth removing: it costs JSON scaffolding tokens, the
prior review already found it poorly calibrated, and nothing consumes it. No
backend in `service/ti/decide` calls a hosted model at all — the confidence that
reaches `policy.confidence.gate_route` is computed from the backend's own
probability vector (`conf_scale.py`), never parsed out of a reply — so in this
benchmark the field is read by the harness and by nothing downstream.

### 2a. Prompt caching: working, and worth almost nothing on the endpoint you were on

The ordering in the prompt is already right — the static block is the system
message, the ~8-token utterance is the last thing in the user message — and it
does get cached: `usage.prompt_tokens_details.cached_tokens` is non-zero on both
cheap endpoints, 58–85% of the prompt.

Whether that is worth anything depends entirely on the endpoint's cache-read
price, and this is where Wafer burned the baseline:

| endpoint | $/M input | $/M cache read | discount | measured cached share | effective $/M input charged |
|---|---|---|---|---|---|
| Relace | 0.0413 | 0.0300 | 27% | 58–84% | **$0.0318–0.0347** |
| GMICloud | 0.0750 | 0.0150 | 80% | 70% | $0.0329 |
| StreamLake | 0.0435 | 0.0087 | 80% | 23% | $0.0354 |
| Wafer | 0.0540 | 0.0530 | **2%** | 71–76% | **$0.0532** |
| OpenInference | 0.0767 | 0.0198 | 74% | 1.7–5% | $0.0667 |

Wafer charges its full input rate for a cache read, so the baseline's 73% cache
hit rate bought it approximately nothing. Caching is a real lever only where the
endpoint discounts it, and the cheapest endpoint is not the one with the deepest
cache discount (StreamLake is, and it is unusable).

### 4. The `:batch` variant

`z-ai/glm-5.3-flash:batch` is listed at **$0.05/M in, $0.18/M out** (catalogue,
and its `input_cache_read` at $0.01/M). It is not usable for live routing:

```
HTTP 404  z-ai/glm-5.3-flash:batch cannot be used with the chat/completions
          endpoint (adapter SailResearchBatchAdapter)
```

It is asynchronous batch-only, so it is not a fallback for a routing decision
that has to return in ~600 ms. Worth noting how little it offers anyway: the
batch price is 7% below Wafer's live price, and it is **more expensive than
Relace live on both axes**. Batching is not the lever here.

## Recommended configuration

```python
payload = {
    "model": "z-ai/glm-5.3-flash",
    "provider": {"only": ["Relace"], "allow_fallbacks": False},
    "reasoning": {"effort": "minimal"},
    "max_tokens": 64,
    "temperature": 0,
    "messages": [
        {"role": "system", "content": <compact static block, letters A-H>},
        {"role": "user", "content": utterance},
    ],
}
```

Full 150-row result (`bench/results/or-bench-recommended-2026-10-26.json`,
`order:Relace+Wafer` = same pin, order form; `or-bench-full150-2026-10-26.json`
holds the strict `only: [Relace]` cell):

| | value |
|---|---|
| accuracy | **145/150 = 96.7%** (0 malformed, 0 truncated, 0 failures) |
| by band | t1 98.4%, t2 93.5%, t3 96.4% |
| $ per run (150 decisions) | $0.00080 (strict only) / $0.00084 (ordered) |
| $/1k decisions | **$0.0053** / $0.0056 |
| $/1k correct | **$0.0055** |
| prompt tokens/req | 143.6 (83.9% served from cache) |
| completion tokens/req | 10.4 (8.4 of them reasoning) |
| effective rate charged | $0.0318/M in, $0.0758/M out |
| latency | p50 624 ms, p95 1080 ms (sequential, n=20) |
| served by | Relace 150/150 — the pin took effect, verified per row |

Against the baseline that is **-73.8% of the bill, +4 rows of accuracy, -39% p95**.
Against the previously recorded baseline run ($0.003321 for 170 requests, which is
$0.0195/1k on the same request-only basis) it is a 3.7× reduction.

Operational notes that came out of the sweep:

* Relace's p50 is 624 ms with `effort: minimal` but 2575 ms without it. If the
  reasoning setting is ever dropped, latency quadruples — the two levers are
  coupled, not independent.
* A strict `only` pin fails closed: if Relace is down, every request errors.
  `order: ["Relace","Wafer"]` behaved identically when Relace was healthy (150/150
  Relace, 96.7%, $0.0056/1k) and is the safer production shape. Do **not** rely on
  `order` alone to contain cost — see the Darkbloom probe above, where the
  fallback landed on Wafer at 2.4× the price with no error and `fallbacks` counting it.
* Relace does not advertise `response_format`; a strict `only` pin still served
  150/150 clean JSON, so it is accepted, but it is not guaranteed by capability
  and it is the reason `sort: price` skips Relace. The letter format removes the
  dependency entirely.

## Call-reduction curve (deterministic pre-filter)

From `bench/cost_curve.py` over the recommended run: the pre-filter
(`bench/prefilter.py`, 15 regex rules, no model) is applied to each row, rules are
ranked by measured precision, and absorbed intents never reach the API. Cost per
cascade point = measured per-decision cost × the calls it still makes. Precision
is in-sample on these 150 rows and is labelled as such.

| absorbed | model calls | pre-filter precision | blended accuracy | $/1k decisions |
|---|---|---|---|---|
| 0.0% | 150 | — | 96.7% | $0.0053 |
| **20.0%** | 120 | 100% | 97.3% | **$0.0043** |
| 36.0% | 96 | 100% | 98.0% | $0.0034 |
| **40.7%** | 89 | 100% | 98.0% | **$0.0032** |
| 52.0% | 72 | 98.7% | 98.7% | $0.0026 |
| **65.3%** | 52 | 96.9% | 97.3% | **$0.0019** |
| 82.7% | 26 | 88.7% | 90.0% | $0.0009 |

Above ~65% absorption accuracy starts to fall (90.0% at 82.7%) — one rule
(`listing_verb`, 24 rows, 58% precision) is where it breaks down, and it is
exactly the "which/what files" phrasing that also confuses the model. A cascade
bleeding 20–65% of intents into the deterministic stage costs **$0.0019–$0.0043
per 1k decisions with no accuracy loss measured**; sizing it past ~65% buys
cheapness by getting answers wrong.

Note the pre-filter's contribution is additive to the model-side reductions, and
the rules are cheap enough to run before any budget check.

## What was actually charged (effective per-M rates)

Recovered from `usage.cost_details.upstream_inference_prompt_cost` /
`upstream_inference_completions_cost` divided by their own token counts, so these
are the rates billed, not the sheet:

| endpoint | sheet in / out | effective in / out |
|---|---|---|
| Relace | 0.0413 / 0.0758 | **0.0318–0.0367 / 0.0758** |
| Wafer | 0.0540 / 0.2000 | 0.0532 / 0.1995 |
| GMICloud | 0.0750 / 0.2500 | 0.0329 / 0.2500 |
| StreamLake | 0.0435 / 0.1450 | 0.0354 / 0.1450 |
| OpenInference | 0.0767 / 0.0100 | 0.0667 / 0.0100 |

Completion rates are billed exactly as listed everywhere. Input rates land below
the sheet wherever the cache is both used and discounted, which is why the
catalogue's single price block is not a usable cost model for this model id: the
same nominal request costs 1.5–2.5× more depending on which of the 38 endpoints
answers it, and the baseline was on the one that discounts caching by 2%.

## Where the money went

| run | result file | $ |
|---|---|---|
| per-provider sweep, 50 rows × 6 pins | `or-bench-providers-2026-10-26.json` | 0.00808 |
| answer-format × reasoning sweep, 50 rows × 12 | `or-bench-tokens-2026-10-26.json` | 0.00615 |
| compaction sweep, 50 rows × 3 | `or-bench-compact-2026-10-26.json` | 0.00129 |
| `:batch` probe (404) | `or-bench-batch-probe-2026-10-26.json` | 0.00000 |
| control, 150 rows | `or-bench-control150-2026-10-26.json` | 0.00343 |
| pin × compaction × answer factorial, 150 rows × 8 | `or-bench-full150-2026-10-26.json` | 0.01298 |
| fp8/fp4 endpoints, 150 rows × 3 | `or-bench-providers150-2026-10-26.json` | 0.00952 |
| Relace baseline shape, 150 rows | `or-bench-relace-plain150-2026-10-26.json` | 0.00293 |
| Wafer baseline shape, 150 rows | `or-bench-wafer-plain150-2026-10-26.json` | 0.00331 |
| ordered-pin fallback probes, 10 rows × 2 | `or-bench-fallback-*-2026-10-26.json` | 0.00025 |
| `sort: price` probes, 20 rows × 2 | `or-bench-sortprice-probe-2026-10-26.json` | 0.00057 |
| recommended config, 150 rows | `or-bench-recommended-2026-10-26.json` | 0.00094 |
| **total** | | **0.04944** |

## Reproducing

```
export OPENROUTER_API_KEY=...        # the harness reads only this env var
python bench/or_bench.py --models z-ai/glm-5.3-flash \
  --providers Relace --variants compact --answers letter --reasoning minimal \
  --out bench/results/or-bench-recommended-$(date +%F).json
python bench/cost_curve.py bench/results/or-bench-recommended-$(date +%F).json
```

## What landed in the repo, and what did not

This doc was written on `bench/or-models` and merged to `main` as a **port, not a
merge**: the harness, the two analysis modules and this doc. One result file came
with it. The other fourteen did not.

| file | bytes | in the repo | why |
|---|---|---|---|
| `bench/or_bench.py` | 44,217 | **yes** | the instrument every number below came from (38,915 on the branch; the rest is the spend ledger and per-model token cap ported from `bench/or-models-2`, plus the retarget of its prompt import to `bench/ti_bench/prompts.py`) |
| `bench/prefilter.py` | 4,560 | **yes** | pure regex, no model, no network — the cascade it models needs it |
| `bench/cost_curve.py` | 7,391 | **yes** | reproduces the curve table from a recorded run |
| `bench/results/or-bench-recommended-2026-10-26.json` | 144,181 | **yes** | the recommended-config table, and the input `cost_curve.py` needs |
| `or-bench-full150-2026-10-26.json` | 1,136,725 | no | 8 configs × 150 rows of raw reply text |
| `or-bench-tokens-2026-10-26.json` | 767,252 | no | 12 configs × 50 rows |
| `or-bench-2026-10-26-run2.json` | 509,114 | no | the base 5 models at 2048 tokens |
| `or-bench-2026-10-26.json` | 482,469 | no | the same at 512 tokens |
| `or-bench-providers150-2026-10-26.json` | 441,964 | no | 3 endpoints × 150 rows |
| `or-bench-providers-2026-10-26.json` | 366,316 | no | 6 pins × 50 rows |
| `or-bench-compact-2026-10-26.json` | 182,494 | no | 3 configs × 50 rows |
| `or-bench-relace-plain150-2026-10-26.json` | 145,089 | no | one endpoint × 150 rows |
| `or-bench-wafer-plain150-2026-10-26.json` | 143,607 | no | one endpoint × 150 rows |
| `or-bench-control150-2026-10-26.json` | 143,567 | no | the unpinned baseline × 150 rows |
| `or-bench-sortprice-probe-2026-10-26.json` | 40,064 | no | 2 configs × 20 rows |
| `or-bench-fallback-letter-2026-10-26.json` | 12,980 | no | 1 config × 10 rows |
| `or-bench-fallback-probe-2026-10-26.json` | 12,894 | no | 1 config × 10 rows |
| `or-bench-batch-probe-2026-10-26.json` | 6,898 | no | 3 rows, all 404 |

The fourteen excluded files total **4,391,433 bytes**. The repo's tracked content
on `main` is 5,788,005 bytes, so committing them would have added 76% to the whole
repository for raw model transcripts of a single day's run. The one file that was
kept carries the config the doc recommends; the rest are regenerable from the
command in **Reproducing** above, which names every axis that distinguished them.
The port as merged — this doc, the harness, both modules and the one result file —
adds **227,643 bytes net**, 3.9% of the repository: 221,880 in five new files
and 5,763 of added `bench/README.md` prose.

**This does not weaken "every number carries its source file."** The source of
these numbers is a script plus a documented configuration, and both are in the
repo. A result file is a transcript of one run of that script, and the fourteen
that were left out are transcripts whose configuration line is written down in the
tables above. What would have broken the rule is merging this doc with *neither*
the script nor the command; that is not what happened.

## Corrections found while porting this doc

Recorded here rather than silently edited, because the numbers below were quoted
before they were checked against the files.

1. **The curve table's `$/1k decisions` column is the strict-pin basis, but the
   committed result file is the ordered-pin run.** Re-running
   `python3 bench/cost_curve.py bench/results/or-bench-recommended-2026-10-26.json`
   on the file in this repo gives **$0.0045 / $0.0033 / $0.0019** at 20.0% / 40.7%
   / 65.3% absorbed, not the $0.0043 / $0.0032 / $0.0019 in the table. The table is
   reproducible from the strict-`only` cell of `or-bench-full150`, which is a
   1,136,725-byte file that was not merged; the ordered pin it re-ran under costs
   $0.0056/1k rather than $0.0053/1k, which is the whole of the difference. The
   *shape* of the curve — 100% pre-filter precision through 40.7% absorption,
   breaking down at `listing_verb` — reproduces exactly. `prefilter.py`'s rules and
   their ranking are unchanged.

2. **`fallbacks` did not count the ordered-pin fallback in the recorded files.**
   The operational note above says the Darkbloom probe "landed on Wafer at 2.4×
   the price with no error and `fallbacks` counting it." Both probe files record
   `fallbacks: 0` while `providers_served` records `{Darkbloom: 1, Wafer: 9}` — the
   fallback happened, and the flag did not fire. `served_by_pin` in the merged
   `or_bench.py` counts `order` lists as a pin, so a re-run reports `fallbacks: 9`
   and the committed script and the committed data now disagree on this one field.
   The measurement (which endpoint answered) is unaffected; the derived flag in
   those two files is wrong. Do not quote `fallbacks` from `or-bench-fallback-*`.

3. **"Total spent on the whole investigation: $0.04944" is the total for the
   files in the table above**, which sum to exactly $0.049443. The branch also
   carried `or-bench-2026-10-26.json` ($0.124621) and `or-bench-2026-10-26-run2.json`
   ($0.150925) — the two base runs at 512 and 2048 tokens — which are part of the
   same investigation and are not in the table. All fifteen files sum to
   **$0.324989**. Read the headline as "the reductions cost $0.049", not "the
   sweep cost $0.049".

4. **The token/latency row of the Recommended-configuration table is the strict
   `only` cell, not the `or-bench-recommended` file it is headed with.** 143.6
   prompt tokens, 10.4 completion, 8.4 reasoning, 83.9% cached, p50 624.4 ms, p95
   1080.2 ms are the `Relace|compact|letter|r:minimal` cell of
   `or-bench-full150`. The committed ordered-pin file measures 143.6 / 10.8 / 9.0,
   70.3% cached, p50 571.1 ms, p95 1249.9 ms. Both are real runs; they are not the
   same run, and the doc's table gives two `$/1k` values without saying the token
   row belongs to one of them.

None of the four changes a conclusion. Items 1 and 4 mean a reader who reproduces
from the file in this repo will land within one tenth of a cent per 1k decisions of
the published figure and should expect the ordered-pin column, not the strict one.
