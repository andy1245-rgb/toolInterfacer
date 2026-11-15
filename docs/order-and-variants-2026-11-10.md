# Shadow mode, the variant sweep, and order selection on half the corpus

2026-11-10, third job. Three things at once: a way to measure the fusion on real traffic without
switching it on, a wide sweep of the variants (including the ones that lose), and an order
experiment that is honest about what selecting on the scored rows buys.

`service/ti/conf/defaults.json` is untouched. Nothing here is on.

---

## 1. Shadow mode — the fusion, measured on requests instead of on 415 authored rows

`service/ti/decide/shadow.py` wraps the backend `resolve()` already built. It returns that
backend's answer object unchanged, and afterwards works out what the fused policy *would* have
said on the same request, then appends both to a log beside the outcome that actually happened.

Why this shape:

* **It cannot change behaviour.** The counterfactual is computed after the served answer exists,
  from the cheap member's opinion blended with the answer already in hand. The model is asked
  exactly once — a test asserts it — and nothing is retried or escalated.
* **It cannot fail a request.** A raising member is recorded as `shadow_error`; an inner failure
  still propagates as itself; a log write that fails is swallowed.
* **It cannot go unnoticed.** Every decision is one JSON line with `served`, `member`, `fused`,
  `why`, the routing `source` (pre-gate / cache / primary), and every op execution appends an
  `outcome` line under the same id: which op ran, whether it errored, and `attempt`, so a retried
  step reads as attempt 2.

### Turning it on

```diff
     "backends_in_use": {
       "choice": "pair",
       "noul": "fixture",
       "extract": "rule_extract"
     },
+    "shadow": {
+      "enabled": true,
+      "member": "classical",
+      "weights": 0.5,
+      "protect": ["web_search", "web_fetch"],
+      "log_path": ""
+    },
```

`log_path` empty means `$TI_SHADOW_LOG`, else `~/.local/state/ti/shadow.jsonl` — never a path in
the repository, because the log contains traffic. Off is the default: without the block,
`resolve()` returns exactly the backend it returned before, and `wrap()` is the identity function.

### What to read after a few days

Join `decision` and `outcome` on `id`, then ask four questions, in this order:

1. **How often did the two sides disagree?** `fused != served`. If this is under a few percent, the
   corpus's 415 rows were describing a disagreement rate that real traffic does not have.
2. **When they disagreed, who was right?** The served side's correctness needs the outcome: an op
   that ran, succeeded, and produced the artifact the request wanted. The shadow's side is judged
   the same way only when you can see that it *would* have — which is the honest limit of shadow
   mode, and it is still far better than an authored corpus: the disagreement rate and the served
   side's failure rate are both real numbers.
3. **Did the failures cluster on the ops the protect rule hands to the model?** If a retired
   `web_search`/`web_fetch` decision is where the retries land, the protect list is too wide.
4. **What did a disagreement cost?** `attempt > 1` on the outcome is a retry; a retry is the cost
   of being wrong, measured in the operator's time rather than in rows.

That is the experiment the operator asked for, and it is the only one that can settle the
shipping question: the 415 rows are the corpus the arm was tuned against.

---

## 2. The variant sweep — seven arms, three fold seeds, every loss on the table

All on the 415 rows, grouped CV over 133 families, stratified, seeds 0/1/2, member spec
`sklearn_text|vectorizer=tfidf_word|clf=logreg|features=rules` unless the row says otherwise.
Artifact: `bench/results/fusion-variants-2026-11-10/`.

| arm | correct | delta | fixed/broke | exact McNemar | worst label | per-seed deltas |
|---|---:|---:|---:|---:|---:|---|
| `floor_a` — the 3B alone | 325 = 78.3% | — | — | — | — | — |
| **`f_opt3`** — the pre-registered arm: mean@0.5, protect two ops | **349 = 84.1%** | **+24** | 26 / 2 | 3.0e-06 | **0** | 24, 22, 23 |
| `f_w40` — same, base weight 0.4 | **353 = 85.1%** | **+28** | 31 / 3 | 7.7e-07 | 0 | 28, 28, 23 |
| `f_w60` — same, base weight 0.6 | 339 = 81.7% | +14 | 15 / 1 | 5.2e-04 | 0 | 14, 16, 14 |
| `f_noprot` — no protect list (the +26 that failed C3) | 351 = 84.6% | +26 | 37 / 11 | 2.2e-04 | **−5** | 26, 23, 26 |
| `f_scaled` — member with `feature_scale=maxabs` | 337 = 81.2% | +12 | **12 / 0** | 4.9e-04 | 0 | 12, 12, 13 |
| `f_char` — member on word+char TF-IDF with sigmoid calibration | 343 = 82.7% | +18 | 21 / 3 | 2.8e-04 | 0 | 18, 17, 17 |
| `f_gate` — gate on the base's confidence at t=0.38 | 328 = 79.0% | +3 | 4 / 1 | **0.38 (FAIL)** | 0 | 3, 4, 3 |

Four things this table says, including the uncomfortable ones:

1. **`f_w40` beats the shipped arm by 4 rows and must not be adopted.** The weight was chosen by
   looking at `w=0.5` and `w=0.6` on these same rows; a knob read off the scored rows is not a
   result, and adopting it would repeat the error that killed the earlier cascade. The
   pre-registered arm remains `w=0.5`. If the weight is worth having, it gets the treatment the
   order experiment gets below: select on one half, validate on the other.
2. **The protect rule is what fixes C3, not the weight or the member.** Every protected arm has a
   worst-label net of **0 or better**; the only arm that fails C3 is the one without it (−5). That
   is the mechanism, isolated.
3. **`f_scaled` is the only arm that broke nothing at all** — 12 fixed, 0 broken. It is not the
   best arm, but it is the safest, and on a system that pays for retries the two are different
   products.
4. **The gate form still fails.** +3 rows at p = 0.38, which is the third independent time the
   confidence-gate mechanism has failed to reach significance on this corpus.

The sweep also fixes a claim from earlier in the night: `f_scaled` was reported as "the member that
beats the 3B on its own" at 349/415. That number was the **member alone** at seed 0, not this arm;
as a fusion member it scores 337. The single-arm number stands (`docs/fusion-2026-11-10.md`,
`member_scaled`), and the two must not be quoted as each other.

### The mask variants, on top of the protect rule

Same protocol, three more arms, artifact `bench/results/fusion-mask-variants-2026-11-10/`.

| arm | correct | delta | fixed/broke | exact McNemar | worst label |
|---|---:|---:|---:|---:|---:|
| `m_auto` — the competence mask alone | 330 = 79.5% | +5 | 14 / 9 | 0.41 (FAIL) | **−6** |
| `m_auto_prot` — mask + protect | 337 = 81.2% | +12 | **12 / 0** | 4.9e-04 | 0 |
| `m_auto_prot_w40` — mask + protect at base weight 0.4 | 340 = 81.9% | +15 | **15 / 0** | 6.1e-05 | 0 |

The mask never adds anything on top of the protect rule: `m_auto_prot` (337) sits below `f_opt3`
(349), and it lands on the same numbers as `f_scaled`, which is a coincidence of which rows the two
rules happen to remove and not a mechanism. `m_auto_prot_w40` is the best **zero-break** arm in the
sweep, and it inherits the weight caveat from `f_w40`: 0.4 was read off these rows.

So the mask direction is now closed twice over — refuted on mechanism (it cannot see the
`reddit_digest`-on-`web_search` pair) and beaten on result (it adds nothing once the base owns the
two ops it was meant to protect).

---

## 3. Order selection, on one half and validated on the other

Choosing the best of N option orders on the 415 rows you also score on is overfitting with extra
steps: N orders are N chances to fit the same rows. `bench/order_selection.py` therefore selects on
one half of the corpus and reports what the selection does on the other half, and it prints its own
verdict — including "the selection did not transfer" when that is the answer.

Two live orders are available from the same server and the same 415 rows, both recorded from
served-model runs in `bench/results/p3-transfer/rows.jsonl`:

| order | selection half (208 rows, `split == train`) | other half (207 rows, `split == test`) | vs frozen on the other half |
|---|---:|---:|---:|
| **frozen** — the shipped order | **160/208 = 76.9%** | **165/207 = 79.7%** | — |
| `rev` — the same eight descriptions reversed | 160/208 = 76.9% | 156/207 = 75.4% | **−9 rows** |

The selection half cannot tell them apart — 160 against 160 — and the other half separates them by
nine rows. `order_selection.py`'s verdict:

```
selected on the split half: floor_a (0.769 against floor_a's 0.769, +0.0% there)
that order on the other half: 0.797 against 0.797 = +0.0% realised
verdict: the frozen order won the selection too — no order change is supported.
```

**So there is no order result here, and that is the honest outcome.** The reversal ties on the half
it would be selected on and loses nine rows on the half it would be validated on. Anyone who had
tuned on the pooled 415 would have seen −9 and been right for the wrong reason; anyone who tuned on
the selection half would have seen a tie and learned nothing.

### What cut the breadth short — and the first explanation was wrong

Five permutations were planned (`rev`, seeded shuffles 11/23/37, plus the paired rewrite). Two were
abandoned mid-run. The reason first written here — "the box is four times slower than the
recording" — **is wrong, and the correction matters more than the original claim.**

Measured under load: 10.3–13 s per row. Measured with the box quiet: **2.3 s per row per slot,
which is the recorded pace.** The difference was contention, and the contention was largely mine:
load average 31 on 12 cores, 383 MB of free RAM, 8.8 GB in swap, with `ladder_port.py` and
`needle_probe.py` from another session in this repository **and my own ten-arm offline sweep** all
running while the live arms did. The box did not get slower. I filled it up.

Three concrete consequences, all self-inflicted:

1. **Running an offline arm sweep and a live order arm at the same time measures the scheduler, not
the model.** The 7.8 s single call that produced the wrong claim was taken while nine other python
processes were running.
2. **The live arms competed with each other** on the same two server slots: the server logged 928
releases against the 830 requests two arms could produce, and the surplus was my own probe plus a
third client left running.
3. **I killed the server mid-run.** A `pkill` aimed at the process I had started also killed the
server a delegated run was using, aborting its arms. The delegated run's own report caught both the
foreign client and the kill and said so.

The planning number, corrected: **≈2.3 s per row per slot on a quiet box**, so a 415-row live arm is
~16 minutes and four arms fit in an hour — *if nothing else is running*. That is a scheduling
constraint, not a hardware one, and it is the reason the breadth request needs a quiet box rather
than a faster one.

The two orders reported above are enough to demonstrate the protocol and answer the ordering
question as it currently stands: the reversal ties on the half it would be selected on and loses
nine rows on the half it would be validated on. A wider permutation sweep is now cheap — four arms
in an hour — and it is the obvious next step, on an otherwise idle machine.
