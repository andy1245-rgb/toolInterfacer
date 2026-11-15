# Closed experiments — do not re-run these without a reason that is written down here

2026-11-10. Each entry states the number that closed it, the shape of the claim that would reopen
it, and where the full record lives. The point of the file is that the next person (or the next
session) does not spend a night rediscovering a negative.

A closed experiment is not a failed one. Every entry here produced a number that is now known and
does not need to be measured again.

---

## 1. Option-order optimisation — CLOSED

**The claim that was tested:** some permutation of the eight option descriptions routes better than
the shipped order, so it is worth searching for one.

**What was measured.** Live, all 415 rows, on the documented server command:

| arm | correct | vs recording | fixed / broke | exact McNemar |
|---|---:|---:|---:|---:|
| live, shipped order (the control) | **326/415 = 78.6%** | **+1** | 1 / 0 | 1.0000 |
| live, eight descriptions reversed | **315/415 = 75.9%** | **−10** | 6 / 16 | 0.0525 |
| recorded reversal (`p3-transfer`) | 316/415 = 76.1% | −9 | 6 / 15 | 0.0784 |

The control matters: the live model reproduces the recording to one row, so the −10 is the order
and not the box. The reversal has now been measured twice, independently, at −9 and −10.

**And the search does not pay.** Applying the select-on-one-half / validate-on-the-other protocol:

| order | selection half (208) | other half (207) | vs frozen on the other half |
|---|---:|---:|---:|
| frozen | **161 = 77.4%** | **165 = 79.7%** | — |
| reversed | 159 = 76.4% | 156 = 75.4% | **−9** |

The selection half separates the two orders by **two rows** — inside the fold-noise band this repo
has measured at ±4 points — while the validation half separates them by nine, against the reversal.

**Why it is closed:** the effect size of a permutation (~9 rows) is no larger than the noise of
choosing one (selection halves differ by 2), and the one permutation anyone has a *reason* to try
loses. Picking the best of N orders on the rows being scored is N chances to fit those rows; the
protocol above is what it costs to notice.

**What would reopen it:** a corpus where the order effect exceeds 3× the fold noise **and** a
selection that transfers — the chosen order must beat the frozen one by ≥ 3 rows on the validation
half at p < 0.05. Not measured, not plausible, not worth a night.

**Record:** `docs/order-and-variants-2026-11-10.md` §3–§4, `bench/results/live-orders-2026-11-10/`,
`bench/results/p3-transfer/rows.jsonl` (arm `order_rev`).

---

## 2. Fusion weighting — CLOSED

**The claim:** `w = 0.4` (weight on the base) beats the shipped `w = 0.5`, because it scores 353
against 349 on the 415 rows.

**What was measured.** Five weights, grouped CV, 133 families, stratified, seeds 0/1/2, plus the
per-half split:

| weight on the base | whole corpus | selection half (208) | other half (207) |
|---|---:|---:|---:|
| 0.3 | 344 | 182 | 162 |
| **0.4** | **353** | **185** | **168** |
| 0.5 (shipped) | 349 | 181 | 168 |
| 0.6 | 339 | 171 | 168 |
| 0.7 | 332 | 166 | 166 |

Selection picks 0.4 by four rows (185 against 181). Validation: **168 against 168, +0 rows.**

**Why it is closed:** the entire four-row advantage sits on the half the weight was chosen on. On
the half it was not chosen on, 0.4 and 0.5 are the same arm to the row. That is the exact shape of
a knob, and it is why 0.4 was refused when it first appeared.

**What would reopen it:** a weight that wins selection **and** holds ≥ 3 rows of that gain on the
validation half. The curve is also flat-topped at 0.5–0.6 on validation (168, 168), so the honest
reading is that the optimum is a plateau, not a point.

**Record:** `docs/order-and-variants-2026-11-10.md` §2 and §5,
`bench/results/fusion-weights-2026-11-10/`, `bench/results/fusion-variants-2026-11-10/`.

---

## 3. The competence mask — CLOSED (refuted twice)

Not an optimisation but a hypothesis, and it is here so nobody reads the two entries above and
assumes only knobs have been swept.

**The claim:** masking the text member's candidate set to labels it can answer repairs the fusion's
C3 failure on `web_search`.

**Why it is closed:** the mechanism is wrong — the member never votes `web_search` on a row that
breaks (0 of 415); the damage is `reddit_digest` claimed on `web_search` rows, and `reddit_digest`
is a label the member is *good* at, so every computable competence rule keeps it. And on result the
mask loses: 330/415 and 331/415 masked forms against 351/415 unmasked, with `web_search` at −8 and
−6 against −5.

**Record:** `docs/fusion-2026-11-10.md`, section headed *REFUTED: the mask hypothesis*.

---

## Still open

* **Shadow mode on real traffic** — collecting. The reading note is `docs/shadow-mode-2026-11-10.md`,
  and the first thing it says is that `vocab_match: 0.0` is the normal record today.
* **A wider permutation sweep** — only reopenable under the condition in §1, which is not met.

---

## 4. The paired rewrite — CLOSED, and it is the most interesting failure in the register

**The claim:** one arm that strengthens `filesys_grep`'s description and compensates in
`web_search`'s takes grep's ten rows without paying them out of `web_search`. Order frozen, so the
wording is the only difference from the reference. Pre-registered in
`bench/results/paired-rewrite-prereg.json` before the run; the artifact also carries the exact
paraphrase and its sha (`paraphrase_sha: 6c99bd077a106e09`).

| condition, written before the run | result |
|---|---|
| C1 delta ≥ 4 rows (3× the A/A floor) | **FAIL — +2** |
| C2 exact McNemar p < 0.05 | **FAIL — 0.8145** |
| C3 no label worse than −3 | **FAIL — −4** |
| C4 `filesys_grep` nets ≥ 0 | **PASS — +10** |
| C5 `web_search` nets ≥ 0 | **PASS — 0** |

| label | before | after | net |
|---|---:|---:|---:|
| **filesys_grep** | 36 | **46** | **+10** |
| **web_search** | 49 | 49 | **0** |
| reddit_digest | 32 | 28 | **−4** |
| decide_extract | 21 | 19 | −2 |
| filesys_read | 43 | 42 | −1 |
| web_fetch | 49 | 48 | −1 |

**Verdict against its own conditions: FAIL.** Overall 327/415 against 325/415, fixed 10 / broke 8.

**The pairing worked exactly as designed and the experiment still failed.** `filesys_grep` gains
all ten rows the earlier one-of-eight rewrite gained, and `web_search` pays **nothing** — the
compensating line did its job. The cost simply moved to a *third* label: `reddit_digest` loses four
rows, which is the same label that broke the fusion's C3 and the same confusion the mask could not
see. Strengthening grep on content-search wording pulls reddit-flavoured asks into it.

**Why it is closed:** the paired rewrite's local objective was met (+10 and 0) and its global
condition was not (−4 elsewhere), and the displacement lands on `reddit_digest` — a label this
corpus has now broken three separate experiments on. Rewriting the descriptions is not the lever for
that confusion; distinguishing “what people say in a subreddit” from “what is inside a file” is not
a wording problem.

**What would reopen it:** a third description change that also holds `reddit_digest` at ≥ 0 **and**
a net of ≥ 4 rows at p < 0.05. Equivalently: one arm that edits three descriptions in a coordinated
way, pre-registered the same way.

**Record:** `bench/results/paired-rewrite-2026-11-10/` (summary, rows, and the paraphrase inside
the artifact), `docs/order-and-variants-2026-11-10.md`.
