# The per-class router, measured without the answer

2026-11-04. Replaces the `84.7%` per-class routing figure with a *deployable*
router and measures what it actually buys. Branch `p2-router`, from `main`
(`fb64b87`). Artifact: `bench/results/router-policy-2026-11-04.json`.
Instrument: `bench/router_policy.py`. `bench/local_clf.py` now reports both the
oracle and the deployable number side by side; the oracle is kept and labelled,
not deleted.

**Verdict in one line:** the honest router is **79.3% [73.6–84.8], n=150** against
the 3B's 74.0% and the oracle's 84.5% — a real gain over the cheap expert
(+12.3 pts, paired CI +5.8 to +18.8) and over the 3B it is **not** separable
(+5.3 pts, paired CI −1.4 to +12.3). The oracle was hiding a genuine 5.2 points,
and the device that does work is the class-level one, not the confidence gate:
the router gets 14 of the 3B's 22 confident misses right where no gate can see
them at all.

## 1. The defect, and the fix

`bench/local_clf.py` computed the published figure as

```python
pred_nested[i] = (pred_lr[i] if choice_map[labels[i]] == "lr" else m3b_pred[i])
```

`choice_map` is learned on training folds — honestly — but it is indexed by
`labels[i]`, the **ground-truth class of the row being routed**. At inference the
class is the unknown being predicted, so the map can only be looked up with an
answer you do not have. It is an oracle, and the repo published it as a result.

The fix is one index:

```python
pred_routed[i] = (pred_lr[i] if choice_map[pred_lr[i]] == "lr" else m3b_pred[i])
```

where `pred_lr[i]` is the cheap model's own out-of-fold prediction of the class —
a value that exists at inference time. The map is still learned on training folds
only, from the same comparison of expert accuracy per class. `bench/local_clf.py`
now computes and prints both:

| routing, `local_clf.py`'s own split (single fold assignment) | accuracy | n |
|---|---:|---:|
| per-class map, dispatched on the **true** label — oracle, not deployable | 84.7% | 127/150 |
| per-class map, dispatched on the **predicted** class — deployable | 75.3% | 113/150 |
| static prior table, true label — oracle | 84.0% | 126/150 |
| static prior table, predicted class — deployable | 75.3% | 113/150 |

75.3% is one draw. Under the harness's five seeds the same policy ranges
**75.3–82.7%**, which is the point
[`docs/eval-harness-2026-11-04.md`](eval-harness-2026-11-04.md) §5(c) already
makes about single fold assignments: a 7-point spread across seeds means a single
split cannot rank two policies. Everything below is measured through
`bench/eval_harness.py`.

For a reader arriving from
[`docs/local-classifier-cascade-2026-11-03.md`](local-classifier-cascade-2026-11-03.md)(11,
224, 250) or the README line that repeats it: the 84.7% in that document is the
first row of the table above — the oracle. Its deployable replacement is the
second row of this document's §3 table, and the two are 5.2 points apart.

## 2. Protocol, and the caveat that governs every number here

| | |
|---|---|
| dataset | `tests/data/utterances_v2.json`, 150 rows, 8-way |
| split | `group` — the 11 declared t3 boundary families held out |
| groups | 106, of which **95 are singletons**: 63.3% of rows carry no family |
| folds | 5, repeated over 5 seeds (`0–4`), inner 3-fold grouped CV inside each training portion |
| interval | cluster bootstrap over groups, 2000 resamples, 95% |
| 3B | `d1-3B-Q4_K_M`, per-row choices and probability vectors from `bench/results/p1-diagnose/baseline_live.json`; frozen, never re-driven, **no server call in this run** |
| cheap expert | `lr` — tf-idf word+char, softmax regression, `min_df=1, l2=1e-3`, fixed a priori at the mid-range of `local_clf.GRID` |

**This set cannot measure generalisation, and every number below is optimistic by
an unknown amount.** 63.3% of its rows are atomic: the group split holds out the
t3 boundary families and nothing else, so most "held-out" rows were split as
rows. On the 382-row set, where 110 real phrasing families exist, the same model
class loses 23.5 points under a family split (§4 of the harness doc). Nothing
here establishes how much v2 would lose; the split method is not identifiable on
this set, which is itself the reason to distrust any single v2 figure — including
these.

## 3. Every policy on the same folds

5-fold × 5 seeds, cluster-bootstrap 95% CI over the 106 groups. "Sent to cheap"
is the share of rows the policy answers with the local model, i.e. the share that
does not pay the 3B's cost.

| policy | dispatch key | n | groups | acc | 95% CI | seeds (min–max) | sent to cheap |
|---|---|---:|---:|---:|---|---|---:|
| `lr` (cheap alone) | — | 150 | 106 | 67.1% | 59.6–73.9 | 65.3–70.0 | 100% |
| `1nn` (char 1-NN) | — | 150 | 106 | 51.3% | 44.4–57.7 | 48.0–56.0 | 100% |
| `rules` (**contaminated**) | — | 150 | 106 | 78.7% | 72.3–85.3 | — | 100% |
| `3b` alone | — | 150 | 106 | 74.0% | 66.5–81.3 | — | 0% |
| `router[static]` (**contaminated**) | cheap argmax | 150 | 106 | 79.7% | 73.8–85.0 | 78.0–82.7 | 44% |
| **`router[per-class, A]`** | cheap argmax | 150 | 106 | **79.3%** | **73.6–84.8** | 75.3–82.7 | 30% |
| `router[per-class, B]` | cheap argmax | 150 | 106 | 74.9% | 69.2–80.2 | 68.7–77.3 | 22% |
| `router[A, 1nn dispatch]` | 1nn argmax | 150 | 106 | 79.1% | 73.0–84.6 | 78.0–81.3 | — |
| `oracle[per-class]` | **ground truth** | 150 | 106 | 84.5% | 77.6–90.8 | 81.3–88.0 | — |

Two policies are marked contaminated and are not baselines:

- `rules` at 78.7% is `local_clf.RULES`, a keyword table written by an author who
  had read these 150 utterances. It is *higher* than the 3B and is evidence of
  nothing.
- `router[static]` uses `local_clf.STATIC_MAP`, the same hand-written prior table,
  indexed by predicted class. Dispatching it honestly costs nothing, but the
  table was authored against this set, so 79.7% inherits the same contamination.
  It is in the table because the *comparison* to the fitted map is the finding:
  the fitted map is not better than the hand table (−0.4 pts, paired CI −4.2 to
  +3.4, p = 0.70). At 15–24 rows per class the per-class selection is not
  learning anything the author did not already know.

`router[per-class, A]` is the primary deployable policy and the direct analogue
of the oracle: same learned map, honest dispatch. `router[per-class, B]` is a
variant that aligns the estimator with the deployed decision — it scores each
expert on the training rows the dispatcher would actually send it — and it does
worse (74.9%, −4.4 pts, CI −8.2 to −0.9). The reason is visible in the learned
maps: `B`'s estimator sees only the rows predicted into a class, which for the
small classes is a handful, and it flips `decide_extract` to the 3B in 11 of 25
folds, where `A` is unanimous. `router[A, 1nn dispatch]` confirms the result is
not an artefact of one dispatcher: swapping the class predictor for a
structurally different one (79.1%) moves nothing.

## 4. What the oracle was hiding

| policy | accuracy | n right | 95% CI |
|---|---:|---:|---|
| oracle union `lr` \| `3B` — ceiling for *any* selector | **92.8%** | 139.2/150 | 88.5–96.6 |
| `oracle[per-class]` — the published 84.7%, on the true label | **84.5%** | 126.8/150 | 77.6–90.8 |
| **`router[per-class, A]` — deployable** | **79.3%** | 119/150 | 73.6–84.8 |
| `3B` alone | 74.0% | 111/150 | 66.5–81.3 |
| `lr` alone | 67.1% | 100.7/150 | 59.6–73.9 |

Paired per-row differences, cluster-bootstrap CI over groups on the difference
itself (the interval is the instrument; the exact McNemar p beside it is
unadjusted for eleven comparisons):

| comparison | Δ acc | 95% CI on Δ | A only / B only | exact p |
|---|---:|---|---:|---:|
| `router[A]` vs `3B` | **+5.3 pts** | **−1.4 to +12.3** | 27 / 21 | 0.4709 |
| `router[A]` vs `lr` | **+12.3 pts** | **+5.8 to +18.8** | 44 / 14 | 0.0001 |
| `oracle[per-class]` vs `router[A]` | **+5.2 pts** | **+0.1 to +10.6** | 20 / 7 | 0.0192 |
| `router[B]` vs `router[A]` | −4.4 pts | −8.2 to −0.9 | 18 / 30 | 0.1114 |
| `router[A]` vs `router[static]` | −0.4 pts | −4.2 to +3.4 | 15 / 12 | 0.7011 |

Which gaps are real:

- **The oracle's premium over the deployable router is real: 5.2 points, CI
  excluding zero.** That is the quantity the published number concealed. It is
  the entire cost of dispatching on a 67.1%-accurate class predictor instead of
  on the truth — the oracle's map and the router's map are identical, only the
  key differs.
- **The router's gain over `lr` alone is real and large** (+12.3 points), and it
  is the one comparison that survives any correction for multiplicity.
- **The router's gain over the 3B alone is not established.** +5.3 points with an
  interval that crosses zero. The repo should not publish "79.3% beats 74.0%";
  it can publish 79.3% [73.6–84.8] against 74.0% [66.5–81.3] and say the
  difference is inside the noise at n=150.
- **The 92.8% union ceiling is 13.5 points above the router**, which is the
  headroom a per-row selector would need to find; nothing measured here finds
  any of it.

The class map `A` learns, over 25 folds: `decide_extract` → `lr` in 25/25,
`filesys_grep` → `lr` in 19/25, everything else → `3B` (with `filesys_read`
flipping once). Six of eight classes are stable, which is why the map is worth
having at all; `filesys_grep` is the unstable one, and it is also the class where
the cheap model is only modestly better than the 3B.

## 5. The cost-aware cascade

Commit the cheap answer when it is confident, escalate to the 3B when it is not.
Every row below is a policy with **no free parameter**: the threshold is fixed
a priori, so its accuracy is measured on rows no threshold ever looked at. The
two *nested* rows are the exception and are labelled: there τ is chosen on each
fold's training portion only, and applied blind to the scored rows.

Latency: the cheap model measured 0.32 ms/row in-process; the 3B's per-row cost
is taken at the **1.9 s** nominal figure the brief quotes (the recording's own
p50 is 3591 ms, but that was taken against the shared, loaded server and is not a
cost of the policy).

| τ | gate | committed | escalation | acc | 95% CI | latency (ms) |
|---:|---|---:|---:|---:|---|---:|
| — | `lr` only, no gate | 100% | 0% | 67.1% | 59.6–73.9 | ~0 |
| 0.20 | maxp | 95% | 5% | 68.5% | 61.7–75.1 | 94 |
| 0.25 | maxp | 78% | 22% | 78.4% | 72.0–84.3 | 424 |
| **0.30** | maxp | 62% | **38%** | **81.6%** | 75.4–87.4 | **722** |
| **0.35** | maxp | 48% | **52%** | **82.1%** | 76.0–88.1 | **994** |
| 0.40 | maxp | 37% | 63% | 80.5% | 73.6–87.3 | 1204 |
| 0.50 | maxp | 21% | 79% | 78.0% | 71.3–84.6 | 1505 |
| 0.60 | maxp | 9% | 91% | 75.7% | 68.7–82.8 | 1733 |
| 0.70 | maxp | 1% | 99% | 74.1% | 66.6–81.4 | 1880 |
| — | `3B` only | 0% | 100% | 74.0% | 66.5–81.3 | 1900 |
| 0.10 | margin | 62% | 38% | 81.9% | 75.8–87.5 | 722 |
| 0.20 | margin | 43% | 57% | 81.9% | 75.6–87.6 | 1080 |
| 0.30 | margin | 29% | 71% | 79.7% | 72.7–86.5 | 1353 |
| *nested* (median τ 0.40) | maxp | 34% | 66% | **78.9%** | 72.3–85.4 | 1252 |
| *nested* (median τ 0.40) | margin | 38% | 62% | **79.5%** | 73.3–85.4 | 1176 |

Reading it:

- **The cheap model's useful gate is not at 0.5.** `lr`'s probability scale is
  compressed — at τ = 0.5 it commits only 21% of rows and lands at 78.0%. The
  curve's peak is at **0.30–0.35, where it commits 48–62% of rows and reaches
  81.6–82.1%** for 722–994 ms. A 0.5-anchored search — the anchor the repo's own
  gate curve used — would have reported 78.0% at 79% escalation and missed that
  the same gate at 0.30 buys 3.6 more points for half the escalation.
- **The gain is not certified, and the two instruments disagree about how
  close it is.** `cascade[maxp≥0.30]` vs the 3B alone is +7.6 points, bootstrap
  CI on Δ −0.5 to +15.9, p = 0.49. The curve's best point (τ = 0.35) is +8.1
  points with a bootstrap CI of **+1.4 to +15.2 — excluding zero — while its
  paired sign test gives p = 0.117**. A bootstrap interval that excludes zero
  beside a sign test that does not is exactly the situation in which a number
  should not be published as an improvement: the two instruments disagree
  because the seed-mean outcome smooths rows the sign test counts individually,
  and at n=150 with 63% atomic rows there is not enough data to break the tie.
  The nested rows exist for the same reason: **78.9% [72.3–85.4] at 66%
  escalation** is what the policy delivers when τ is picked blind on the training
  fold, and that is +4.9 points with a CI of −0.3 to +10.2 (p = 0.050).
- **The cascade and the router are the same device measured twice**: +2.3 points
  apart, CI −2.8 to +7.9, p = 1.0, at the same cost (the router leaves 70% of
  rows paying the 3B, ≈1330 ms; the nested cascade escalates 66%, ≈1252 ms).
  Neither dominates. The router has the better story, because it escalates on
  *which class* a row is in rather than *how unsure* the model is — and §6 shows
  that distinction is the one that matters.

## 6. The 3B's confident misses: which signal catches them

This is the question the phase-1 finding raises: 22 of the 3B's 39 misses arrive
at max-probability ≥ 0.5, so the maxprob gate commits them by construction.

**What is not better: the margin.** As a gate on the 3B's own answer, top-1 minus
top-2 is indistinguishable from top probability — AUC 0.799 against 0.803 for
detecting a miss, and at a matched escalation budget they catch the same rows
(30% budget: 24 misses caught by each, of 39). A better summary of one model's
own distribution is not what is missing here.

| detector of a 3B miss (n=150, 39 misses) | AUC |
|---|---:|
| `3B` max-probability, low = suspicious | 0.803 |
| `3B` margin (top-1 − top-2), low = suspicious | 0.799 |
| cheap/3B **disagreement**, 1 = suspicious | **0.786** |
| cheap max-probability, low = suspicious | 0.415\* |
| cheap margin, low = suspicious | 0.411\* |
| cheap model wrong, 1 = suspicious | 0.465 |

\* Below 0.5 means the sign runs the useful way: a *confident* cheap model
predicts a 3B miss (equivalently 0.585 as a positive detector). The cheap
model's own uncertainty is a weak detector of the 3B's errors — it is the cheap
model's *opinion*, not its confidence, that carries information.

The disagreement flag is strong where it matters: it catches **36 of 39 misses
and 20 of the 22 confident misses**. But it fires on 50% of all rows, so as a
decision rule its precision is a coin flip:

| arbitration: 3B always answers, override on disagreement | overrides | acc | 95% CI | Δ vs 3B | exact p |
|---|---:|---|---|---:|---:|
| override only if cheap maxp ≥ 0.25 (fixed τ) | 35% (52) | 78.4% | 72.0–84.3 | +4.4 | 0.9007 |
| override only if cheap maxp ≥ 0.30 (fixed τ) | 28% (42) | 81.6% | 75.4–87.4 | +7.6 | 0.4885 |
| override only if cheap maxp ≥ 0.35 (fixed τ) | 21% (31) | **82.1%** | 76.0–88.1 | +8.1 | 0.1173 |
| **override at a τ chosen on the training fold** | **13% (19)** | **78.7%** | 72.0–85.2 | +4.7 | **0.0708** |
| no override (3B alone) | 0% | 74.0% | 66.5–81.3 | — | — |

(How good the overrides are, from the single seed-0 pass: at τ = 0.30 it overrides
40 rows and 26 of those cheap answers are right — 65% precision on 27% of the
traffic; at τ = 0.50 it overrides 7 rows and all 7 are right. The second number
is why the small-τ end of this curve looks attractive and the first is why it is
not a decision.)

The honest reading: **the flag exists, the decision does not.** You can detect
20 of the 22 confident misses with one cheap inference — the highest-value
detector measured here — but the override's *net* is +4.7 points when τ is
chosen blind (p = 0.071), and the accuracy-maximising τ is a selection made on
the rows it is then scored against. It also costs the 3B on every row, so it is
an accuracy lever, not a cost one.

### The class-level answer, which does work

Cut the same 22 rows by what class they are in:

| the 3B's 22 confident misses, by true class | n |
|---|---:|
| `decide_extract` | 11 |
| `filesys_grep` | 9 |
| `web_fetch` | 1 |
| `filesys_read` | 1 |

**20 of 22 are in exactly the two classes the per-class map assigns to the cheap
expert.** And the accuracy on those rows, measured over the same five seeds:

| policy | right on the 3B's 22 confident misses |
|---|---:|
| `3B` | 0/22 — by construction |
| `lr` alone | **16.4/22 (75%)** |
| `router[static]` (contaminated) | 16.2/22 (74%) |
| `router[per-class, A]` | **14/22 (64%)** |
| `oracle[per-class]` | 14/22 (64%) |
| `cascade[nested τ]` | 7.4/22 (34%) |
| `override[nested τ]` | 7.0/22 (32%) |

That is the result worth keeping. A per-row confidence gate cannot address this
failure mode *at all* — the cascade converts only a third of these rows, because
the ones it escalates are exactly the ones the 3B is confidently wrong about.
The class-level policy converts two thirds of them without asking the 3B, and it
converts the same 14 the oracle does: on these rows the oracle's knowledge adds
nothing over the map's two entries. `lr` alone does better still (16.4), because
the 2.4-row difference is exactly the rows the router diverts to a 3B that has
nothing to contribute on them — routing buys nothing on this subset and can lose,
which is the same trade the paired test reports over the whole set.

## 7. Is per-class routing worth building?

**Yes as a cost device, no as an accuracy claim — and the version worth building
is the one the repo did not have.**

- The published 84.7% is an oracle. The deployable analogue is **79.3% [73.6–84.8],
  n=150, family-grouped 5×5**, and the 5.2 points between them are real
  (paired CI +0.1 to +10.6). Anyone quoting 84.7% as a routing result is quoting
  a policy that needs the answer to look up the answer.
- Against the 3B's 74.0% the router's +5.3 points **do not clear the paired
  test** (CI −1.4 to +12.3). The claim that survives is narrower: the router is
  much better than the cheap expert alone (+12.3, CI +5.8 to +18.8), and it adds
  nothing measurable over a hand-written class table (itself contaminated here)
  — so what it buys is the cheap expert's accuracy on the two classes where it
  wins, for 30% of the traffic, not a new accuracy ceiling.
- The concrete win, if a router ships, is the confident-miss set: 14 of 22 rows
  the 3B gets confidently wrong, recovered for the price of a 0.3 ms
  classification. No confidence gate reaches them.
- Everything here is a 150-row, single-author, adversarially-authored set with
  63% atomic rows. **The numbers are optimistic by an unknown amount and this
  set cannot certify the direction of a 5-point difference.** The next step is
  a held-out routing set, not another threshold.

## 8. What this does not settle

1. **How optimistic.** v2 cannot measure generalisation (§2). The 23.5-point
   penalty measured on 382 is the same model class, not this set.
2. **Whether the router beats the 3B.** The point estimate says yes by 5.3
   points; the paired interval includes zero. At n=150 and 63% atomic rows this
   is the strongest statement the data supports.
3. **Whether `filesys_grep` belongs to `lr`.** It is the only unstable entry in
   the learned map (19 of 25 folds) and the cheapest place for the map to be
   wrong.
4. **The 3B's numbers are the *baseline* description set, not the targeted
   rewrite** that phase 1 found moves the same rows to 82.0%. That rewrite was
   written by reading these 150 rows, so it is in-sample; using it here would
   have put a tuned number in both the numerator and the denominator. Note what
   it would do to this document's conclusion if it holds up on a fresh set: the
   rewrite repairs `decide_extract` (25% → 79%) and `filesys_grep` (36% → 50%) —
   exactly the two classes this map sends to `lr` — so the router's mix barely
   moves while its baseline rises, and **the 79.3% router stops beating the 3B
   at all**. Any per-class routing claim has to be re-measured against the fixed
   3B, not against the broken one. *(The fresh set exists: applied unchanged to
   `heldout_v3` the whole edit is net negative, and the `filesys_grep` line alone
   is the part that transfers — `docs/rewrite-transfer-2026-11-04.md`. The
   conditional above is still unmeasured, because no arm restored `decide_extract`'s
   original text.)*

## 9. Reproduce

```
python3 bench/router_policy.py                       # every table, ~2 min idle
python3 bench/router_policy.py --seeds 1 --boot 200  # fast smoke
python3 bench/router_policy.py --split row           # the mirage split, for contrast
python3 bench/local_clf.py                           # oracle + deployable side by side
python3 -m pytest tests/test_router_policy.py -q     # the no-ground-truth guard
```

`tests/test_router_policy.py` asserts the defect directly: it perturbs the
*labels of the scored fold* and requires every deployable policy's decisions to
be byte-identical — and requires the oracle's to change, so the test cannot pass
by perturbing nothing.

No server is contacted. The 3B's per-row answers come from
`bench/results/p1-diagnose/baseline_live.json`, byte-identical to
`bench/results/dell/v2-d1-3B-Q4.json` (0 of 150 argmax disagreements across two
llama.cpp builds).
