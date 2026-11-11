# A tier that is silent rather than careful — and how to find the rest

> What this document is: the generalisable lesson from the classical backend's whole life of
> abstention, turned into an audit method — one row per symptom with the check that exposes it,
> plus the protocol that keeps a revived tier's measurement from being misread. The two records
> that came out of it are [`classical-revival-2026-11-10.md`](classical-revival-2026-11-10.md)
> and [`rot-audit-2026-11-10.md`](rot-audit-2026-11-10.md).

The lesson is one sentence: **a silent fallback is indistinguishable from a working one.** A tier
that always says "no opinion" returns, the expensive model answers, the answer is correct, and
nothing anywhere logs coverage. This document records the case that proved it and the method that
finds the others.

## The mistake, stated once

`decide.classical.corpus_path` points at `bench/registry_382.json` — a registry of **17
voice-assistant tools** (`timer_create`, `shoppinglist_add`, `calendar_today`, …). The backend is
then asked to choose among the **eight code/file ops** (`filesys_grep`, `web_search`, …). The label
sets have **zero overlap**, so the fitted model cannot name any candidate and abstains on every
request. It has never answered a single question.

It survived because **an abstaining tier is invisible**. Somebody did look at it — `temperature`
is set to `1.0` in the config — which is the fix for *bad confidence*, not for a *wrong vocabulary*,
so the attempt ended with the misdiagnosis intact. It is recorded here, rather than only in the
commit that changed the path, because a future reader needs to know the tier was dead for its whole
life and why nobody could tell, not merely that a path moved.

## Reviving a tier so that the revival is measurable

1. Build the corpus from the **v3 training split only** (`bench/data/heldout_v3.json`, rows whose
   `split` is not the eval split), as flat `{"utterance","label"}` rows. Training on the eval rows
   would be leakage and would produce a beautiful, meaningless number.
2. Point `corpus_path` at that file. This is the only production change, and it is one line — but
   it depends on step 1, which is not.
3. **Measure coverage, not just accuracy.** The failure mode was invisibility, so the measurement
   has to make the same failure loud: report `answered / total`, accuracy on those answered, and
   the confidence-threshold sweep. A tier that answers nothing must be impossible to miss in the
   report.
4. Assert it in the suite: a regression test that the tier **answers something** on a sample — the
   test that would have caught this two years ago. One line of assertion, permanently.

The measurement that matters is what the tier is worth as a front door: at what coverage does it
reach 90% precision, and what would that save per 1,000 requests at the recorded per-call latency.

## The audit method

The lesson generalises, so every backend in `service/ti/decide/backends/` and the compositions in
`pair.py` are audited for this pattern:

| symptom | how to detect it |
|---|---|
| a backend that always abstains | run it over the v3 rows, count answered; zero or near-zero is dead |
| a corpus or config pointing at a different vocabulary | compare the fitted label set against the option set, and fail if the intersection is empty |
| a path that does not resolve | every `corpus_path`/`base_url`/file config entry, resolved and asserted at construction |
| a backend that is registered but unreachable | enumerate the registry, and check each is reachable from at least one config route |
| a tier whose accuracy is quoted from a run nobody can reproduce | find the claim, then find the record; if the record is missing, mark the claim unverified |
| a threshold tuned while the mechanism was broken | `temperature`, `tau`, `p_max` — check each has a run that shows it moving the metric |
| dead code kept alive by tests that only assert it loads | tests that construct but never call |

The register that came out of it is [`rot-audit-2026-11-10.md`](rot-audit-2026-11-10.md): one row
per finding, each with the evidence command that shows it and a severity in the repo's own B/M/U/L
scale (`agentFleet`'s `docs/specs/LIMITATIONS.md` carries the calibration of what those letters
mean). Findings were fixed only where the fix was one line and obviously safe; what to spend on the
rest is a separate decision.

## Scope

No hosted model calls. The only production change was the single `classical.corpus_path` line.
