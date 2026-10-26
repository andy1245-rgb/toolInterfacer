# Decision layer: pair vs hosted-only — measured, and which ships

Measured 2026-10-26 with interleaved calls (30 utterances x 3 arms, lead arm rotating
per utterance, one warm-up discarded per arm). Raw rows:
`bench/results/ab-interleaved-30-run2.json`.

Arms were built with `make_backend(...)` directly, bypassing `resolve()` on purpose:
`FastPathBackend` would otherwise answer some rows from the decision cache or the
`heuristic_micro` pre-gate with no network call, silently mixing 0 ms rows into an
arm's latency distribution.

## Latency (ms), 30 calls per arm

| arm | p50 | p90 | p99 |
|---|---:|---:|---:|
| pair (tau 0.90) | 3118 | 4768 | 8055 |
| openrouter | **827** | **2021** | **4706** |
| d1_systemone | 2261 | 2901 | 3302 |

## Accuracy, same 30 rows

| arm | correct | accuracy |
|---|---:|---:|
| pair | 27/30 | 90.0% |
| openrouter | 28/30 | 93.3% |
| d1_systemone | 22/30 | 73.3% |

McNemar pair vs openrouter: 0 discordant pair-only, 1 hosted-only, exact p = 1.000.
No detectable accuracy difference — structurally expected, since ~73% of the pair's
answers *are* the hosted answer.

## Why the pair is slow, structurally

`pair = local + escalation_rate x hosted`. It pays a full local call on 100% of
decisions and a hosted call on the escalated 73%. Escalation is additive: the residual
`pair - (local + hosted)` on escalated rows is p50 +196 ms, which is single-slot
queueing, not a hidden third call.

Observed escalation 22/30 = 73.3% (Wilson 95% 55.6-85.8) against a predicted 72.0%
offline (108/150, Wilson 64.3-78.6). All 22 carried `last_reason=low_confidence`;
zero `stage1_error`, zero `stage2_error`.

**The gate is not the defect.** The 8 locally-committed rows were 8/8 correct. The local
model simply cannot answer fast enough for committing early to be worth anything on
this hardware.

## Failure paths, induced for real

| induced | behaviour | cost |
|---|---|---|
| `TI_OPENROUTER_API_KEY` unset | degrades to stage1, does not raise | ~0 ms (key check precedes the socket) |
| stage1 base_url dead | hard-fallback to hosted | 674 ms |
| stage2 base_url dead, low-confidence stage1 | returns stage1's answer (policy rule 5) | 1478 ms |

## Decision

**Shipped default stays `pair` at tau 0.90 on this box, knowingly accepting a latency
regression**, because hosted-only hard-fails when the key is absent while the pair
degrades to local and keeps answering. Latency favours `openrouter`; availability
favours `pair`. Until `TI_OPENROUTER_API_KEY` is guaranteed in the service environment,
availability wins.

To switch when the key IS reliably present, one line in `service/ti/conf/defaults.json`:

```json
"backends_in_use": {"choice": "openrouter", ...}
```

**On fast-local hardware the pair is the right default.** At the local latency recorded
on dell321 (157-210 ms), expected pair p50 is `~200 + 0.733 x 827 ~= 0.81 s` — parity
with hosted-only — and then the pair buys what this bench could not measure: no
per-decision spend, no network dependency, no third-party data path, and a 100%-correct
commitment set. Keep tau at 0.90 there; raising it converts every decision into a hosted
call plus a wasted local call.

## Limits

- n=30 bounds the accuracy claims; the pair-vs-local trend (7:2, p=0.180) is a trend,
  not a result.
- The latency comparison does not need that power: the p50 gap is 2.3 s against a
  ~0.5 s load-induced swing.
- Direction (`pair costs escalation_rate x local_latency more than hosted-only`) is
  structural. Magnitude is load-dependent — local p50 moved 4.1 s to 2.0 s inside this
  run. Re-measure rather than re-reason when the box changes.
