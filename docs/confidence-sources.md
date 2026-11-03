# Where every confidence number comes from

Review 2026-11-03 (B2, B3) and audit finding 14 named one bug class: a confidence
fabricated at exactly the moment there was no evidence. This file is the audit of
every place a `conf` is produced, and the rule the fix settles on.

**The rule, in one sentence:** a confidence is reported when — and only when — a
hop measured one, it is the minimum over those measurements, and it is absent
otherwise (`confidence` is `Type.Optional` in the contract), because "no
measurement" and "maximum certainty" must never be the same number.

Three classes are used below:

| class | means | example |
|---|---|---|
| **measured** | read off a distribution or a score the producing component actually computed | `logit_read`'s `1 - H/lnK` over renormalized logprobs |
| **derived** | a documented formula over measured inputs | `d1_conf = (p_max - 1/K)/(1 - 1/K)`; `web_search`'s coverage |
| **asserted** | a constant somebody typed | `_route_conf`'s old `1.0`; `filesys`'s `1.0` |

Only **asserted** sources are suspects. Each is either converted to a measurement,
converted to an honest absence, or kept with its provenance stated in code.

## Run level

| source | class | disposition |
|---|---|---|
| `service_entry._route_conf` | **derived** | Was `return 1.0` when no hop carried a measurement → now `None`, and `_run_steps` omits the field. Min over hops whose `conf is not None` and whose op is a decide step (or `askback.resolved`). Attached only when `status == "done"`: the number describes the delivered result, and an error run has none. |
| `service_entry._passthrough` (L0) | **absent** | Was a hardcoded `1.0` in both the envelope and the hop. L0 is the caller's own op and the caller's own args: the service makes no routing decision, so there is no route confidence to state. The envelope carries no `confidence`; the hop carries the op's own number, next to the op's name. |
| `core/runner.ASKBACK_CONF` = 0.9 | **asserted, kept on purpose** | See "The one retained prior" below. |

## Choice backends

| source | class | disposition |
|---|---|---|
| `backends/d1_systemone.py:37` | **measured** | The decision server's own `confidence_choice` over its distribution; `or 0.0` covers a reply that carries no number, which is "no answer", not a substitute. |
| `backends/logit_read.py:71` | **measured** | `1 - H(p)/ln K` over the renormalized top-20 logprobs of an actual token distribution. |
| `backends/openrouter.py:149` | **asserted by the model** | A chat completion carries no distribution, so `{choice: confidence}` is the single point the model asserted and `confidence` is its own self-report, clamped to `[0,1]`. Not ours to measure, and not gateable on a p_max — see below. |
| `backends/heuristic_micro.py:42` | **asserted** (routing prior) | `hits[pick]` = 0.99 is a keyword-pattern trust prior, and it is the pre-gate (`decide.routing`) whose whole purpose is to commit a trivial route without a model call. Kept: converting it to absence would make every heuristic route ask back, which is a routing-policy change, not a confidence fix. Flagged here so nobody reads it as a measurement. |
| `backends/fixture.py:60` | **asserted** (fixture) | `decide.fixture.default_choice_conf` = 0.95, recorded answers otherwise. Test/bench only; never a live default. |
| `backends/rule_extract.py` | **absent** | Was a hand-picked rubric (0.92 urls / 0.72 queries / 0.30 for a literal ask that found nothing / 0.10 empty). A rule extractor has no distribution: it matched a pattern or it did not. It now returns `basis` (`url_literals`, `path_literals`, `query_phrases`, `quoted_spans`, `distinctive_tokens`, `none`) — a fact about what happened — and no `conf`, and `ops/decisions.run_extract` maps that to `None` instead of its old `0.7` fallback. |
| `decide/pair.py` | **derived** | Passes the answering stage's number through. Stage1 is gated on the measured p_max; a stage1 answer with one probability entry expresses no distribution, so it is a stage error that escalates. Stage2 is terminal and is never gated, so its single-point answer is accepted as-is. |
| `decide/cascade.py` | **derived** | Commits report `path_conf` — the in-sample measured precision of each commit path (A2 §4.2: 0.875 / 0.781 / 0.80) — deferrals report the measured peak. Both stages are gated on p_max, so both must express a distribution. |
| `decide/conf_scale.py` | **derived** | `pmax_from_probs`, `d1_conf`, `entropy_conf`, and the inversion in `to_common_scale`; all pure functions of a distribution. `common_conf` is the gate-side entry point and refuses K < 2. |

## Ops

| source | class | disposition |
|---|---|---|
| `ops/decisions.run_extract` | **absent** | See `rule_extract` above; `_measured_conf` returns a float only when the backend actually reported one, so a future model-backed extractor can still supply a real number. |
| `ops/decisions.run_pick` | the backend's | Whatever `ChoiceAnswer.confidence` the resolved choice backend reported. |
| `ops/web/search.py:86` | **derived** | Coverage: 1.0 when every query answered, 0.5 otherwise, 0.0 when nothing came back. It is a measure of the search, documented as coverage and not as decision confidence. |
| `ops/web/fetch.py:65` | **derived** | Mean over URLs of 1.0 (content present, not degraded), 0.5 (meta-description shell), 0.0 (nothing usable). |
| `ops/reddit/digest.py:82` | **derived** (binary) | 1.0 with at least one post, 0.0 without: a fetch outcome. |
| `ops/filesys/impl.py:54,110,149` | **derived** (coverage) | *This row used to read "unconditional `1.0`, not edited here — another track owns that file".* The file now reports what the walk actually covered (`_confidence(state)`): `1.0` complete and nothing refused, `0.7` a cap or deny-list prune, `0.5` the deadline — and `filesys_read` returns `0.7` when it truncated at `ops.filesys.read.max_lines`/`max_bytes`. That is a measurement of coverage, not a claim about a decision, and it is what a `conf_gte` gate can legitimately compare. |

## Mechanism

| source | class | disposition |
|---|---|---|
| `core/artifacts.Artifact.conf` | **absent by default** | Was `1.0`, so any `put()` that forgot a conf passed every `conf_gte` gate. Now `None`. |
| `core/gates.check_pre` (`conf_gte`) | — | An artifact with `conf is None` raises `GateBlocked`: an absent measurement cannot be "above the threshold", so it routes to ask-back like any other gap. This makes the gate *more* likely to ask, never less. |

## The one retained prior

`runner.ASKBACK_CONF = 0.9` is still asserted. It is named, documented as a
policy prior in the module, and it appears in exactly one situation: a gate
raised a question, an answer came back and applied, and the artifact the gate
rejected is re-written so the retry can run. The prior is what lets the retry
clear the threshold that blocked it, so converting it to an absence would either
block every retry forever or require a gate protocol that distinguishes
"measured" from "answered" — a change outside this fix. It is not a model
measurement and the report above and `_route_conf`'s docstring both say so.

## What `p_max` means, and the K < 2 rule

`tau`, `tau_hi`, `tau_lo`, `tau2` and `policy.gate_route` are all thresholds on
`p_max`, the peak of a distribution over the offered options. A p_max needs a
distribution to be the peak *of*: with one option the vector sums to 1 whatever
the answer is, and every value of it is a peak, so there is no number to compare.

`conf_scale.d1_conf(p_max, k)` returns `1.0` for `k < 2` because that is what
`server-decision.cpp` computes for a degenerate question, and this module's job
is to reproduce the server's formula. Importing that sentinel into a *gate*
turned "the model expressed no distribution" into "maximum certainty": a stage
answering `{'only': 0.02}` read as p_max 1.0, cleared `tau`, never escalated, and
the cascade then reported the fitted commit precision (0.87) for it — above
`gate_slot` 0.7, so the next step ran unattended on a 0.02-confidence answer.

The fix is at the gate, not at the sentinel:

- `conf_scale.common_conf(probs)` is the gate-side p_max and raises for any
  vector shorter than two entries.
- `pair._require_choice(..., needs_distribution=True)` treats a one-entry stage1
  answer as a stage error (escalate, or raise when there is no fallback), exactly
  as it already treats a missing choice.
- `cascade._common` raises, so a cascade stage that expressed no distribution
  cannot commit on `path_conf`.
- `pair`'s stage2 keeps accepting a single-entry answer, because `tau` is never
  applied to stage2: it is a decision, not something a p_max gate scores.

Consequence for configuration: a stage whose backend asserts a single point
(`openrouter`) cannot serve as a pair stage1 or as a cascade stage — it will
escalate every time (pair) or raise (cascade). That is the honest outcome; the
backend that asserts a single number belongs in `backends_in_use.choice` itself,
where its own number is what the slot gate judges.

## Arithmetic, and why the direction is right

| formula | meaning | direction |
|---|---|---|
| `pmax_from_probs(p) = max(p)` | the peak of the distribution | non-decreasing in the peak; range `[1/K, 1]` |
| `common_conf(p)` | `pmax_from_probs` for a vector a gate may threshold | raises for `K < 2` |
| `d1_conf(p_max, K) = max(0, (p_max - 1/K)/(1 - 1/K))` | excess over uniform, rescaled | 0 at uniform, 1 at a point mass, increasing in `p_max`, floored at 0, never above 1 |
| `entropy_conf(p) = 1 - H(p)/ln K` | logit_read's reparameterized entropy | 0 at uniform, 1 at a point mass, decreasing in entropy, clamped to `[0, 1]` so floating-point noise cannot report a negative or above-certainty value |
| `to_common_scale(name, p)` | both of the above mapped onto p_max | exact for d1; entropy-preserving for logit_read |

Both formulas agree on both endpoints (uniform → 0, point mass → 1) and neither
can exceed 1 for a normalized vector: `d1_conf`'s numerator is
`p_max - 1/K <= 1 - 1/K`, and `entropy_conf`'s `H` is non-negative.
`tests/test_confidence_honesty.py` pins the endpoints, the monotonicity over a
sweep, the `[0, 1]` bounds, the round trip, and the gate-direction property
(raising a `conf_gte` threshold never admits more artifacts).

## What a caller should read instead of a number

When `confidence` is absent, the evidence is still there and attributed:

- `hops[]` — one record per step, with the op name, the op's own `conf` (or
  `null` when it measured nothing), and an artifact summary;
- `assumptions[]` — every place the run substituted a reading for the one it
  could not establish, including `gate unresolved, using the best guess`;
- `degraded` — true when a gate went unanswered or a budget stop truncated the
  run;
- `status` / `error` — an error run reports no confidence at all, by rule.
