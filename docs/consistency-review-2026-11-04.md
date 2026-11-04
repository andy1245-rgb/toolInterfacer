# Where the corrections contradict each other

2026-11-04. One night retired or qualified eight numbers. Every correction was
**appended to its dated document**, which is the right policy: a dated document
that is silently rewritten teaches a future reader nothing. But appending creates
two failure modes that the append-only policy does not cover by itself:

1. **The correction is not reachable from the number.** A reader who stops at the
   Verdict, the summary table, or the summary row never learns the claim above
   them was withdrawn.
2. **The correction and the body are read by different readers.** An entry point
   (`README.md`, `CHANGELOG.md`, `TODO.md`, `spec/`) carries its own copy of the
   claim, and the copy does not inherit the correction.

This document is the audit of both, for the eight numbers the night touched, and
for the numbers the previous sweep did not reach. Nothing here changes a number.
Where the defect was a missing pointer, the pointer was added — that is all that
was added.

Method: every occurrence of each figure was opened and read in its own sentence,
not grepped and counted. `docs/number-sweep-2026-11-04.md` is the earlier
inventory of the same figures; where this document disagrees with it, the
disagreement is stated with the line that was read.

**On line numbers.** A line number is a cache and it goes stale — that is §3(b)'s
finding about the earlier sweep, and this document is not immune to it. The
`file:line` references in [§1](#1-the-contradiction-matrix) are as of the tree
this review was taken against, *before* the pointers in §7 were added. The
[§3](#3-reachability--the-pointer-ledger) ledger is recomputed against the tree
this document ships with. On any later tree, neither is the durable handle: the
section name and the quoted sentence in the same row are, and
`git log -S"<the quoted words>"` resolves either.

## Contents

1. [The contradiction matrix](#1-the-contradiction-matrix)
2. [Harmful, harmless, and why](#2-harmful-harmless-and-why)
3. [Reachability — the pointer ledger](#3-reachability--the-pointer-ledger)
4. [The numbers the earlier sweep missed](#4-the-numbers-the-earlier-sweep-missed)
5. [The newest claim: what `+17.3` does and does not say](#5-the-newest-claim-what-173-does-and-does-not-say)
6. [Contradictions outside the number list](#6-contradictions-outside-the-number-list)
7. [What was fixed here, and what was only reported](#7-what-was-fixed-here-and-what-was-only-reported)
8. [The single most misleading sentence currently live](#8-the-single-most-misleading-sentence-currently-live)

---

## 1. The contradiction matrix

Verdict key: **live** (asserted as current truth, no condition attached) ·
**dated** (asserted inside a dated document whose frame says it is a record) ·
**corrected** (the document carries its own appended correction naming it) ·
**pointer-missing** (corrected, but nothing at the place the number is read says so).

### 84.7 — the oracle

| file:line | what it currently asserts | verdict |
|---|---|---|
| `docs/local-classifier-cascade-2026-11-03.md:11` | "per-class routing, decided on training rows, gets **84.7%** against the 3B's **74.0%**" — the document's verdict | **dated, corrected** (C1 at `:353`) — **pointer-missing** |
| same `:223` | "per-class choice, static prior table (no fitting) \| 84.0% \| 126/150" | **dated, corrected** (C1) — labelled nothing inline |
| same `:224` | "**per-class choice from training portion (nested)** \| **84.7%** \| **127/150**" | **dated, corrected** (C1) — no inline caveat |
| same `:225` | "per-class choice from pooled OOF (optimistic bound) \| 84.7% \| 127/150" | **dated, corrected** (C1). The body's own "(optimistic bound)" is the only inline warning on any of the three |
| same `:226` | "oracle union \| 92.7% \| 139/150" | **dated** — the row is *labelled* "oracle", which is the one clear inline signal in the table |
| same `:229` | "The nested choice keeps **57%** of the `lr\|3b` oracle gap" | **dated, corrected** (C1) — pointer-missing |
| same `:250` | "Per-class routing is the right device for this pair" | **dated, corrected** (C1) — pointer-missing |
| `README.md:454` | the docs-table row: "the report's 84.7% was an oracle applied on the row's true label, 92.8% is the union ceiling above it" | **live and correct** |
| `README.md:464–467` | the 2026-11-04 block: "84.7% was an oracle, not a result … no router can reach it" | **live and correct** |
| `CHANGELOG.md:161–168` | "Retired: an oracle quoted as a deployable accuracy … no router can reach it" | **live and correct** |
| `TODO.md:162` | Known-limits row: "`nested_selection_acc` is an oracle … 84.7% applies the per-class map on the row's true label" | **live and correct** |
| `docs/eval-harness-2026-11-04.md:253, 271, 290` | the report that establishes the defect | **live and correct** |
| `docs/vulkan-igpu-2026-11-03.md:204` | `84.7 ms/tok` | coincidence, per-token latency |

**No two documents assert different values for 84.7.** The defect is entirely
reachability: the four entry-point copies are right and the source document's own
headline is wrong without a pointer.

### 93.19 / 66.23 — the different task

| file:line | what it currently asserts | verdict |
|---|---|---|
| `docs/classical-baseline-2026-11-03.md:18` | summary table: `TF-IDF + LinearSVC, 382 rows, 18-way \| 93.19%` immediately above `v2 set, 150 rows, 8-way \| 62.7%` | **dated, corrected** (C1 at `:397`) — **pointer-missing**; C1 names this table as the defect and the table does not name C1 |
| same `:8–13` | the verdict already says "it was measured on a task this repo does not have" | **dated, qualified inline** — this is why the file is not actively misleading at the top |
| same `:124` | "That is the number to quote next to 93.19%" | **dated, corrected** (C1) — **pointer-missing**; C1 names this sentence too |
| same `:95, 114, 118, 251, 252, 256` | statements about the recorded protocol on its own set, each carrying §8/§9 | **dated, clean** |
| `docs/assets-inventory.md:77` | "A 3B model at **74.7%** reads as a decent small router until a classical model at **93.19%** is on the same board … a 400× latency gap at better accuracy" | **dated, corrected** (C2 at `:500`) — **pointer-missing** |
| same `:390` | "Until those are on the board, '74.7% for a 3B' reads as good; against a 0.61 ms classical model at 93.19% it reads as the wrong instrument" | **dated, corrected** (C2) — **pointer-missing** |
| same `:424` | "**91.4% @ 243 ms** with none-detection 1.00, which is **17 points** above the 74.7% this repo currently treats as the incumbent. It reframes the whole live-probe table" | **dated, corrected** (C2) — **pointer-missing** |
| same `:468` | the Status row that runs 93.19 / 66.23 / 62.7 / 74.0 in one sentence | **dated, corrected** (C1) — C2 at `:500` says how it should be read |
| `docs/roadmap-2026-11.md:141` | Finding 1: "TF-IDF + LinearSVC **93.19% at 0.61 ms**, d1-3B **91.36% at 242.7 ms** … That is a 400x latency gap at equal-or-better accuracy" | **dated, corrected** (C1 at `:244`) — **pointer-missing** |
| same `:204` | Outcome: "holding out whole *phrasing families* instead of rows drops it to **66.23%**" | **dated** — the claim is true of that task and C1 supplies the scope |
| `docs/local-classifier-cascade-2026-11-03.md:59–60` | opens by citing 93.19% → 66.23% as a split precedent for *this* task | **dated, corrected** (C3 at `:469`) |
| `README.md:453` | "the recorded 93.19% reproduces exactly — but it is a **different task** … only the 62.7% … says anything about `useTool`" | **live and correct** |
| `CHANGELOG.md:113` | "the sibling project's 93.19%" — the released bullet, left as published | **dated** — retired by name at `:178–190` |
| `TODO.md:161, 163` | "only the 62.7% on `utterances_v2` is comparable"; "(n=382 — a different task, see above)" | **live and correct** |
| `docs/distill-corpus-certification-2026-11-03.md:81` | the document that states the empty intersection | **live and correct** |
| `tests/data/routing_382_PROVENANCE.md:16, 65, 76, 96` | provenance, with "Caveat carried forward" | **live and correct** |

**Same quantity, different values?** No — 93.19% and 66.23% are asserted
identically everywhere. What differs is *what they are asserted to be evidence of*,
and the disagreement is between documents, not inside one: `docs/assets-inventory.md`
frames them as reframing this project (`:77`, `:424`) while
`docs/classical-baseline-2026-11-03.md:8` and `CHANGELOG.md:178` say they measure a
different task. Both are still live sentences in the repo. The assets survey is
self-described as a dated reading of a private tree (`:3–6`), which is the
mitigation — but only for a reader who starts at line 3.

### 62.7 — the figure that *is* comparable

| file:line | asserts | verdict |
|---|---|---|
| `docs/classical-baseline-2026-11-03.md:10, 23, 157, 317, 352` | the v2 transfer figure with its n | dated, clean |
| `docs/local-classifier-cascade-2026-11-03.md:163, 238` | reconciled as the same-task transfer figure | dated, clean |
| `docs/roadmap-2026-11.md:205` | "on this repo's own set it scores 62.7% against d1-3B's 74.0%" | dated, clean |
| `README.md:453`, `CHANGELOG.md:113`, `TODO.md:161`, `data/distill/README.md:142` | same-task, with the set named | clean |
| `docs/distill-corpus-certification-2026-11-03.md:200` | the two recorded baselines | clean |

**No contradiction.** This is the one figure in the pair that survived intact and
is used identically in nine documents.

### 74.0 — the 3B on this repo's task

| file:line | asserts | verdict |
|---|---|---|
| `README.md:375` | the Measured table, 150 rows, per band | live and correct |
| `docs/eval-harness-2026-11-04.md:171, 176, 191, 222` | re-measured, frozen, split-invariant, `[66.5, 81.3]` | live and correct |
| `docs/classical-baseline-2026-11-03.md:13, 23, 157, 196, 198, 320` | comparator in every cascade row | dated, clean |
| `docs/local-classifier-cascade-2026-11-03.md:12, 49, 106, 141, 151, 197, 222, 249, 251` | comparator | dated, clean |
| `docs/router-policy-2026-11-04.md:11, 94, 132, 157, 194, 263, 321` | comparator, and "the repo should not publish '79.3% beats 74.0%'" | live and correct |
| `docs/rewrite-transfer-2026-11-04.md:43, 87` | the in-sample baseline | live and correct |
| `TODO.md:35, 79`, `CHANGELOG.md:108`, `RELEASE.md:219`, `docs/distill-classifier-2026-11-04.md:11, 148, 261, 308` | comparator with n | clean |

**No contradiction.** But see §6: the *bar* this figure is compared against in
`docs/research-model-sweep-q4.md` is 84%, not 74.0%, in two places still live.

### 82.0 / 74.0 → 82.0 — the in-sample improvement

| file:line | what it currently asserts | verdict |
|---|---|---|
| `bench/results/p1-diagnose/FINDINGS.md:20–24` | Verdict: "moves the same 150 rows from **74.0% to 82.0%** … recovers 17 of the 32 misses" | **dated, corrected twice** (`:250`, `:302`) — **pointer-missing**; the first correction says the caveat "belongs beside the number … including the Verdict at the top of this file", and the Verdict does not carry it |
| same `:160–162` | §3 table, 74.0% → 82.0%, 18/6, p = 0.0227 | dated, correct as an in-sample paired result |
| same `:188` | "Treat 82.0% as an upper bound on a fresh set" | dated — the caveat §3 always had |
| `docs/rewrite-transfer-2026-11-04.md:9` | "**Verdict: the whole rewrite fails out of sample. The 82.0% must be retired.**" | live and correct — the authoritative statement |
| same `:88` | "targeted (archived) \| **82.0%** (123/150) \| 18 / 6 \| p = 0.0227", labelled "(archived)" | live and correct |
| same `:115` | "The 82.0% does not reproduce anywhere on v3" | live and correct |
| `README.md:478–485` | "**82.0% is retired as an improvement**", whole edit net negative | live and correct |
| `TODO.md:86–96` | "[x] … Done, and it fails … **82.0% is retired as an improvement**" | live and correct |
| `CHANGELOG.md:205–214` | "Retired: an in-sample improvement described as a finding about the model … **Another track is validating it on a held-out set; this changelog does not pre-empt that result.**" | **live and wrong** — the track landed; `docs/rewrite-transfer-2026-11-04.md` is in the tree and `README.md:478` records the result |
| `docs/distill-classifier-2026-11-04.md:320–325` | "**A 3B description fix already exists and is worth more than this model.** … moves the same 3B from 74.0% to **82.0%** … **That is a larger, better-evidenced gain than anything measured here**" | **live and wrong** — see §8 |
| `docs/router-policy-2026-11-04.md:344–353` | §8.4: "Note what it would do to this document's conclusion **if it holds up on a fresh set**" | **live and unanswered** — the fresh set was run the same day |
| `docs/serving-fix-qwen35.md:75` | `82.02 MiB` | coincidence, memory |

**This is the matrix's worst row.** The same quantity is described three ways in
three live 2026-11-04 documents: retired (`README`, `TODO`, `rewrite-transfer`,
`FINDINGS`), still being validated (`CHANGELOG`), and a "larger, better-evidenced
gain" (`distill-classifier`).

### "8.0-point leak"

| file:line | asserts | verdict |
|---|---|---|
| `docs/local-classifier-cascade-2026-11-03.md:89–93` | "Grouped: **61.3%**. The **8.0-point** difference is the leak" | dated, corrected (C2 at `:430`) |
| same `:241–243` | "The leak on this set is 8.0 points, and it is not only in t3" | dated, corrected (C2) |
| same `:433, 448–449` | C2 itself, naming both passages | dated, correct |
| `docs/classical-baseline-2026-11-03.md:433–449` (C2) | the v2 grouped/random pair read as a leak; retired | dated, correct |
| `README.md:473–476` | "the reported 8-point 'leak' recomputes to 0.0 across seeds" | live and correct — the phrase itself is gone from the table |
| `CHANGELOG.md:192–203` | "Retired: a split that measured nothing … recomputes to **0.0 points** across 5 seeds" | live and correct |
| `docs/eval-harness-2026-11-04.md:270–277` | §5(c), the recomputation | live and correct |
| `TODO.md:160` | the known-limits row | live and correct |

**No live assertion of the 8.0-point leak survives.** The only residual is
arithmetic: 8.0 also names two unrelated quantities (`a 3B at 74.7%` is 8.0 points
above nothing in particular; `docs/eval-harness-2026-11-04.md:270` quotes "8.0
points" as the *size of the retired claim*, which is correct usage).

### "17-point gap"

| file:line | asserts | verdict |
|---|---|---|
| `docs/assets-inventory.md:424` | "**91.4% @ 243 ms** … which is **17 points** above the 74.7% this repo currently treats as the incumbent. It reframes the whole live-probe table" | dated, corrected (C2 at `:500`) — **pointer-missing** |
| same `:523–525` | C2: "There is no 17-point gap on any single board in this repo" | dated, correct |
| `docs/rewrite-transfer-2026-11-04.md:21, 132` | `+17.3` — a different quantity entirely | live, correct |
| `docs/classical-baseline-2026-11-03.md:226` | "93.19 vs 91.36" — the real 1.83-point same-board difference | dated, correct |

**One live assertion, one retirement, no pointer between them.** And note the
collision §6 records: `+17.3` (grep transfer, 2026-11-04) and `17 points` (the
retired cross-board gap, 2026-11-03) are one digit apart and about different
things.

---

## 2. Harmful, harmless, and why

### Harmless — a dated body recording what was believed

These are working as intended. The policy is append-only, the document's frame
says it is a record, and a reader who arrives expects to read a record:

- The Verdict, tables and prose of `docs/classical-baseline-2026-11-03.md`,
  `docs/local-classifier-cascade-2026-11-03.md`, `docs/assets-inventory.md`,
  `docs/roadmap-2026-11.md`, `bench/results/p1-diagnose/FINDINGS.md` and
  `docs/model-sweep-consolidated.md` — each has an appended correction and each
  header states the date it was true.
- `CHANGELOG.md:99–150`, the released `### Measured` section. Deliberately
  preserved so a reader can see what `0.1.0` claimed; retired by name at `:151`
  with a pointer at `:10`.
- `docs/local-classifier-cascade-2026-11-03.md:225`'s "(optimistic bound)" label —
  the body's own inline warning, which is the pattern this repo should use more.
- Numbers that only share digits with the retired ones: `84.7 ms/tok`
  (`docs/vulkan-igpu-2026-11-03.md:204`), `84.0%` at n=50
  (`docs/vulkan-igpu-2026-11-03.md:172`, `:282`, correctly caveated as "two rows at
  n=50"), `92.7%` / `84.0%` for the hosted endpoint
  (`docs/cost-reduction-glm-flash.md:34, 37`), `82.02 MiB`
  (`docs/serving-fix-qwen35.md:75`), `68.0%` for Kev-4B
  (`docs/research-model-sweep-q4.md:131`).

### Harmful — a reader acting on a retired number, or two live documents disagreeing

| # | defect | why it is harmful |
|---|---|---|
| H1 | `docs/distill-classifier-2026-11-04.md:320–325` presents 74.0% → 82.0% as "a larger, better-evidenced gain than anything measured here" | A live 2026-11-04 document, in a decision section ("What would change the answer"), restates a retired improvement as the best available gain. Nothing in the file mentions the out-of-sample run |
| H2 | `CHANGELOG.md:212–214` says the held-out validation "is being run" and "this changelog does not pre-empt that result" | The changelog is the release record. It says an open question is open when it is closed, and names only `FINDINGS.md` as corrected — not `docs/rewrite-transfer-2026-11-04.md`, not the README |
| H3 | `spec/1.0-spec.md:10` asserts d1-3B at "91.4% tool routing at p50 243ms" with no set named | The "source of truth" for the design, first table in the README's Docs section. The repo's own measurement of that model on its own task is 74.0%. Flagged on 2026-11-03 in `docs/doc-review-2026-11-03.md:67` and never fixed; outside every sweep's grep path (§3) |
| H4 | `docs/research-model-sweep-q4.md:7, 135` assert a d1-3B "bar" of **84%** and the scout's pick of Qwen3.5-4B on the strength of 80/75% lanes | Both are contradicted by `docs/model-sweep-consolidated.md`: the same model on the same 150 rows is 74.0% (`:47`), and the Qwen lanes "never served Qwen3.5-4B" (`:17–30`), the genuine figure being 58.0% (`:50`). The scout carries no correction and no pointer, and `README.md:451` repeats the 84% bar |
| H5 | Four documents say the deployable per-class figure is still being measured, while a fifth quotes it | `README.md:454` ("being measured on another track"), `README.md:467–469` ("is not quoted here"), `CHANGELOG.md:173` ("will carry a further correction when it lands"), `docs/local-classifier-cascade-2026-11-03.md:415–427` ("No number is quoted here, and none should be inferred") — against `docs/router-policy-2026-11-04.md:131` (79.3%, 119/150) and `README.md:508`, which quotes it in the same file that denies it |
| H6 | `docs/router-policy-2026-11-04.md:344–353` leaves the rewrite as a live conditional | "if it holds up on a fresh set … **the 79.3% router stops beating the 3B at all**". The fresh set was run the same day: the grep line transfers, the extract line does not, and the grep-alone arm was not run. Neither horn of the conditional is measured, and the item reads as open |
| H7 | `79.3` names two different devices, in two documents, with the same interval | The classical-front cascade (`docs/classical-baseline-2026-11-03.md:199`: gate ≥ 0.20, 44% coverage, 16 fixed / 8 broken, p = 0.152, 95 ms) and the per-class router (`docs/router-policy-2026-11-04.md:96`: 30% coverage, +5.3 pts against the 3B, CI −1.4 to +12.3). Both are 119/150. `README.md:453` carries one and `README.md:508` the other, 55 lines apart in one table. `TODO.md:35` calls the first "Best configuration anyone has measured here", which is a different claim from the second's "+5.3 points do not clear the paired test" |
| H8 | `README.md:125` states "**786 tests, 388 subtests**" and that this row "is the only place the current count is written" | `python3 -m pytest tests/ -q` at this commit prints **826 passed, 551 subtests**. The one row the whole repo defers to is stale, so every "run pytest" instruction now contradicts the number it defers to |
| H9 | `docs/roadmap-2026-11.md` asserts two values for the corpus | `:88` "**412** teacher-labelled, $0.0043" against `:219` "414 rows, **199** teacher-labelled, $0.002" and `:229` "199 rows against the 300-row floor". Same document, same quantity, 131 lines apart. `TODO.md:57` marks the work done at 412 |
| H10 | `docs/eval-harness-2026-11-04.md:22` and `:212` give the same gap opposite signs | `:22` "**−23.5 pts**", `:212` "**+23.5 pts**" for row-wise 92.1% / held-out 68.6%. Every other document says "loses 23.5 points". The `gap` column has no stated convention, so one of the two tables is miscounted |
| H11 | `CHANGELOG.md:10` says "Two claims in the `### Measured` section below were retired on 2026-11-04" | `:153` says "Four claims published in `0.1.0` were retired", and only one of the four (the 382-row bullet) is actually in `### Measured`. Three counts in two lines and a section |

### The one reachability failure that matters most

Every item in H1–H11 except H8, H9 and H10 is, mechanically, a **missing
pointer**. The content defect was already repaired by whoever appended the
correction; what was never done is the half the append-only policy depends on —
putting a line at the number that says *this was retired, read X*.

---

## 3. Reachability — the pointer ledger

For each place a retired number is **read**, is there anything at that place
pointing at its correction?

| number read at | correction lives at | distance | pointer at the number? |
|---|---|---|---|
| `docs/local-classifier-cascade-2026-11-03.md:11` (84.7%, the verdict) | `:362` | 351 lines | **no** — fixed here |
| same `:232–234` (the oracle rows) | `:362` | 130 lines | **no** — fixed here |
| same `:250` ("the leak on this set is 8.0 points") | `:445` | 195 lines | **no** — fixed here |
| `docs/classical-baseline-2026-11-03.md:18` (the summary table) | `:404` | 386 lines | **no** — fixed here |
| same `:129` ("quote next to 93.19%") | `:404` | 275 lines | **no** — fixed here |
| `docs/assets-inventory.md:77` | `:507` | 430 lines | **no** — fixed here |
| same `:392` | `:507` | 115 lines | **no** — fixed here |
| same `:428` (the "17 points") | `:507` | 79 lines | **no** — fixed here |
| `docs/roadmap-2026-11.md:145` (Finding 1) | `:255` | 110 lines | **no** — fixed here |
| same `:208–211` (the Outcome) | `:255`, `:281` | 44 lines | **no** — fixed here |
| `bench/results/p1-diagnose/FINDINGS.md:20–30` (the verdict) | `:257`, `:309` | 233 lines | **no** — fixed here |
| `docs/distill-classifier-2026-11-04.md:320–325` | does not exist in that file | — | **no — no correction exists at all** (H1) |
| `docs/router-policy-2026-11-04.md:344–353` | does not exist in that file | — | **no** (H6) |
| `CHANGELOG.md:211–229` | in the same section, but named only `FINDINGS.md` | — | **partly** — fixed here |
| `spec/1.0-spec.md:10` | nowhere in that file | — | **no** (H3) |
| `docs/research-model-sweep-q4.md:7, 141, 148` | `docs/model-sweep-consolidated.md:17` — a different file | — | **no** (H4) — fixed here |
| `README.md:468, 471` | `docs/router-policy-2026-11-04.md` — a different file | — | **no** (H5) — fixed here |
| `docs/local-classifier-cascade-2026-11-03.md:437–449` | `docs/router-policy-2026-11-04.md` — a different file | — | **no** (H5) — fixed here |

Two structural findings behind that ledger:

**(a) The earlier sweep's definition of "fixed" does not include reachability.**
`docs/number-sweep-2026-11-04.md:32` defines the verdict as "**fixed** (a dated
correction was appended, named)". Under that definition `docs/assets-inventory.md:77`
is fixed, because C2 at `:507` names it. A reader at `:77` is still told the
comparison is the reason to reframe the project. Fourteen instances were certified
"fixed" by that rule and twelve of them have no pointer at the number.

**(b) The sweep's own line numbers no longer resolve.** It was written before the
last round of edits, and its index is now wrong where the documents grew:

| sweep says | the line now reads |
|---|---|
| `TODO.md` 135 (the `routing_382` known-limits row) | `:135` is "`web_search` / `web_fetch` need configured CLIs" — the row it means is `:163` |
| `README.md` 437, 438 (the classical and local-classifier rows) | they are `:453` and `:454`; `:437` is a table header |
| `number-sweep` `:14` promises a sweep of `82.0` | there is no 82.0 section; the figure appears only as pattern P4 and in one FINDINGS row |

The sweep does print the `grep` command that regenerates the list, so the inventory
is recoverable — but the line references, which are what a reader follows, are not.

**(c) The sweep's search path does not cover the whole repo.** `:13–15` runs
`grep -rn --include=*.md … docs README.md CHANGELOG.md TODO.md RELEASE.md bench
data tests`. `spec/`, `ARCHITECTURE.md` and `CONTRIBUTING.md` are outside it. That
is how `spec/1.0-spec.md:10` — already flagged by name in
`docs/doc-review-2026-11-03.md:67` on 2026-11-03 — was missed for a second
consecutive pass.

---

## 4. The numbers the earlier sweep missed

### The ten on the second list

| number | used consistently? | carries its n? | verdict |
|---|---|---|---|
| **79.3** | **no** — two different quantities, see H7 | yes in both | **harmful** — the digits collide across `docs/classical-baseline-2026-11-03.md:199` and `docs/router-policy-2026-11-04.md:96`, and `README.md` carries both without disambiguating |
| **78.9** | yes — `docs/router-policy-2026-11-04.md:198, 218` only | yes (150 rows, 66% escalation, coverage stated) | clean; it is the nested-threshold row of the same gate curve, and nothing else in the tree uses it |
| **68.7** | yes — `README.md:404`, `TODO.md:79`, `docs/distill-classifier-2026-11-04.md:11, 146, 259, 308, 337` | yes — `[59.6–77.0]`, n=150, 28/36, p = 0.38 | clean. `docs/router-policy-2026-11-04.md:97` uses `68.7–77.3` as a *seed range*, and `:192` as a CI bound — different axes, correctly labelled |
| **80.5** | **ambiguously** — `docs/rewrite-transfer-2026-11-04.md:111` (v3 baseline, 334/415) vs `:146` (the six untouched classes, targeted, 251/312 = 80.45%) vs `docs/router-policy-2026-11-04.md:190` (a gate row at 37% coverage) | yes — each carries its own n and column | coincidence, not defect: three unrelated 80.5s, all in n-bearing tables. Worth knowing before anyone greps it |
| **91.36** | yes — d1-3B on the 382-row set, in `docs/classical-baseline-2026-11-03.md:22, 226, 426`, `docs/assets-inventory.md:67, 493, 514, 521, 526, 536, 538`, `docs/roadmap-2026-11.md:141, 245` | yes — "382 rows, 18 smart-home classes", and C1's table marks it **not comparable** | clean, and the same value everywhere. Note it is precisely the figure that makes H4's "84% bar" untenable: the repo already records the 382-row figure for that model |
| **92.1** | yes — `docs/eval-harness-2026-11-04.md:22, 206` and six downstream files | yes — 382 rows, 382 groups | clean in value; see H10 for its **sign** |
| **68.6** | yes as the 382 held-out figure (`docs/eval-harness-2026-11-04.md:205, 212, 242`, `README.md:456, 476`, `CHANGELOG.md:199`, `docs/classical-baseline-2026-11-03.md:471`, `docs/local-classifier-cascade-2026-11-03.md:460`, `docs/model-sweep-consolidated.md:241`, `docs/roadmap-2026-11.md:275`) | yes | clean, except the sign at `:212` (H10). `docs/distill-classifier-2026-11-04.md:152` also has "`d1_3B` 68.6%" as a per-band figure — different axis, labelled |
| **92.8** | yes — the union ceiling, `[88.5–96.6]` family-held-out, in `README.md:454, 465`, `CHANGELOG.md:170`, `TODO.md:162`, `docs/router-policy-2026-11-04.md:129, 160`, `docs/eval-harness-2026-11-04.md:260, 290`, `docs/local-classifier-cascade-2026-11-03.md:405` | yes, with the interval and the split named in every instance read | clean. The one place it needs care is `docs/eval-harness-2026-11-04.md:262`'s note that the repo's own §5 reports **92.7%** for the nested-hyperparameter variant — stated as the same quantity at a different config, which is honest |
| **"5.2-point"** | yes — `docs/router-policy-2026-11-04.md:13, 60, 143, 149, 318` | yes — paired CI +0.1 to +10.6, 20/7 rows, p = 0.0192 | clean. It is *also* the first command's "hidden" figure ("The oracle was hiding a genuine 5.2 points"), which reads correctly against 84.7 → 79.3 |
| **−23.5** | **no** — see H10 | yes | **harmful** — one sign flip inside `docs/eval-harness-2026-11-04.md` |

### The numbers on neither list, still live and still wrong

| figure | where | what is wrong |
|---|---|---|
| **91.4** | `spec/1.0-spec.md:10` | the design-of-record's premise figure; sibling-project, 382-row set, no set named. H3 |
| **84% / 86%** | `docs/research-model-sweep-q4.md:7, 135`; repeated as "the 84% d1-3B bar" in `README.md:451` | a d1-3B bar that does not match the repo's own 74.0% on the same model and set. H4 |
| **80.0% / 74.7%** | `docs/research-model-sweep-q4.md:133–134, 140` | lanes that were answered by a stale server; retired in `docs/model-sweep-consolidated.md:17–30` and not here. H4 |
| **786 / 388** | `README.md:125` | actual 826 / 551. H8 |
| **412 vs 199** | `docs/roadmap-2026-11.md:88` vs `:219, :229` | same quantity, two values, one file. H9 |
| **"Two" vs "Four"** | `CHANGELOG.md:10` vs `:153` | retired-claim count. H11 |
| **+23.5 vs −23.5** | `docs/eval-harness-2026-11-04.md:212` vs `:22` | same gap, opposite signs. H10 |
| **199 / $0.002** | `docs/roadmap-2026-11.md:219` | a spend and a count that `TODO.md:57` and `docs/distill-corpus-certification-2026-11-03.md:94` supersede at 412 / $0.0043 |

---

## 5. The newest claim: what `+17.3` does and does not say

The claim under test, as it stands in the two entry points:

- `README.md:481–483`: "The diagnosis holds and the `filesys_grep` line **transfers**
  (+17.3 points, 9 fixed / 0 broken); the `decide_extract` line was fitted to the
  150 and the whole edit is net negative."
- `TODO.md:91–93`: "`filesys_grep`'s **line alone transfers** (+17.3 pts, 9/0,
  p = 0.0039); the extract line was fitted to the 150."

**Is it restated as a general improvement?** No. Every place that carries `+17.3`
also carries the whole-edit failure in the same sentence or the next one:
`README.md:483` ("the whole edit is net negative"), `TODO.md:88` ("Done, and it
fails"), `docs/rewrite-transfer-2026-11-04.md:9` ("the whole rewrite fails out of
sample"), `bench/results/p1-diagnose/FINDINGS.md:302` ("the rewrite was run out of
sample, and it fails"). That part is clean and is the strongest thing the night
produced.

**Is the post-hoc nature of the per-class split visible wherever it is claimed?**
No. Four defects, in order of how far a reader can be misled:

1. **The words "alone" and "on its own" assert an arm that was not run.**
   `TODO.md:92` says the grep line "alone transfers";
   `bench/results/p1-diagnose/FINDINGS.md:320` says "the `filesys_grep` line
   transfers **on its own**". The report is explicit that this arm does not exist:
   `docs/rewrite-transfer-2026-11-04.md:289–293`, "**Whether the grep line alone is
   net positive out of sample.** That is one arm this experiment deliberately did
   not run". What was measured is the two-line edit evaluated *per class*; the
   grep-alone configuration — new grep text, original extract text — is a different
   experiment, and the report says so in the same document that reports +17.3.
   A reader of `TODO.md` or `FINDINGS.md` alone would ship the grep line believing
   it had been scored as a shipping candidate.
2. **The population qualifier disappears outside the report.**
   `docs/rewrite-transfer-2026-11-04.md:259` states the number should be read as
   "+17.3 points on **non-package-noun** grep phrasing, which is the family the
   rewrite repairs; on package-noun phrasing it is unchanged, 0/4 on v2". Neither
   `README.md:482` nor `TODO.md:92` nor `FINDINGS.md:320` carries that qualifier,
   and §8 of the report exists precisely because it cuts *against* the favourable
   reading (v3 has 0 package-noun rows, so it slightly overstates the transfer for
   a package-noun-heavy population).
3. **The selection is invisible.** The rewrite targeted two of eight classes. Of
   those two, one transferred (grep, p = 0.0039) and one did not (extract,
   p = 0.1797), and the survivor is the one now nominated to ship. That is a
   reasonable reading of the data and it is also a post-hoc selection between two
   candidates, on one set, at n = 52. The phrase "the one durable finding"
   (`docs/rewrite-transfer-2026-11-04.md:152`) is the selection being made, but no
   document says "this is one of two classes, named after the other failed". The
   report also runs eight per-class McNemar tests (`:132–139`) and does not say so
   or adjust for it; one of the eight is at p = 0.0039.
4. **The word "transfers" is doing more work than the design supports.** What was
   run is: the *same two-line edit*, scored on 415 rows authored after both
   strings were written. That is a genuine out-of-sample test of the edit. It is
   not a test of the grep description in isolation, and §9 says so.

**Verdict on §5:** the newest claim is not overstated as a *whole-edit* claim —
that is stated correctly and prominently. It is overstated as a *shipping*
recommendation: "the `filesys_grep` line alone transfers" is the one sentence that
has crossed from a measured per-class effect into an unmeasured deployment claim,
and the report's own §8 and §9 both withhold it.

---

## 6. Contradictions outside the number list

Recorded because a consistency pass that only counts digits would miss all of these:

1. **Two documents disagree about whether a measurement exists at all.** H5 —
   "being measured on another track" versus a quoted 79.3%. This is worse than a
   stale number: a reader is told to wait for something that is in the tree.
2. **`docs/number-sweep-2026-11-04.md` is the repository's inventory of these
   figures and its line numbers no longer resolve** (§3b). The inventory is right
   about what exists and wrong about where.
3. **`docs/roadmap-2026-11.md:88` vs `:219`** (H9) — one document, two corpus
   states. The Phase 2 verdict was updated to 412 and the Outcome was not, and the
   document's own 2026-11-04 correction says "two of its inferences do not survive"
   without mentioning that a third number has moved.
4. **`README.md:125`'s "only place the current count is written"** is now the only
   place it is written *wrong* (H8). Every other document's deferral to that row is
   currently pointing at a stale figure — the failure mode the row was designed to
   prevent.
5. **`spec/1.0-spec.md:10` was flagged on 2026-11-03 and is still there (H3),**
   because `docs/doc-review-2026-11-03.md` is a dated audit that reports rather than
   repairs and no later pass picked the item up. The sweep's search path excludes
   `spec/`, so the second pass could not find it either.
6. **`docs/research-model-sweep-q4.md` publishes a conclusion its own successor
   retired (H4)** and is reachable from `README.md:451` under a description that
   presents it as a live measurement.

---

## 7. What was fixed here, and what was only reported

### Pointers added (no number changed, no claim rewritten)

| file | pointer added |
|---|---|
| `docs/local-classifier-cascade-2026-11-03.md` | at the Verdict (`:11`), at the oracle table, and at C1's "What replaces it" — the last of these now names the measurement that landed |
| `docs/classical-baseline-2026-11-03.md` | under the summary table and at §4's "number to quote next to 93.19%" |
| `docs/assets-inventory.md` | at lines 77, 390 and 424 |
| `docs/roadmap-2026-11.md` | in the document frame, at Finding 1, and at the Outcome's corpus count |
| `bench/results/p1-diagnose/FINDINGS.md` | at the Verdict — the sentence the first correction asked for and did not get |
| `docs/distill-classifier-2026-11-04.md` | at the 82.0% claim: retired, out of sample, see the report |
| `docs/router-policy-2026-11-04.md` | at §8.4: the fresh-set question was run, and the answer |
| `docs/research-model-sweep-q4.md` | at the bar, and at the pick: superseded by the consolidated correction |
| `spec/1.0-spec.md` | at `:10`: retired, with the two documents that establish it |
| `README.md` | at the "being measured on another track" sentence; a disambiguating clause on the second 79.3% in the docs table; and this document's row in the Docs table |
| `CHANGELOG.md` | the "Two claims" count reconciled by reference, the fourth correction referenced forward to the out-of-sample report, and the "further correction when it lands" sentence now names it |
| `TODO.md` | the cascade item now names the other 79.3% |

### Reported, not repaired

These need a decision, not a pointer, because fixing them means changing a claim
or a number:

- **H1** `docs/distill-classifier-2026-11-04.md:320–325` — needs its own appended
  correction. A pointer was added; the sentence remains the most misleading one in
  the repo (§8).
  **Narrowed 2026-11-04** — that document now retires the 82.0% sentence inline, in the same
  bullet (`:324-326`), so what H1 asked for exists and what remains is weaker. The rest of
  H1's claim, and §8's premise, are corrected in the appendix at the end of this document.
- **H8** `README.md:125` — the test count has to be re-read from a live run and
  the row updated.
  **Closed 2026-11-04** — repaired in `12a127b`, which re-read the count from a live run and
  updated the row to `826 tests, 551 subtests`. That row is its own index (it is the only place
  the count is written), so closure needs the commit and not a pointer. It has moved again since
  for the reason the row itself predicts: the logging track added tests, and a live run at the
  tree this note was appended to reads `847 passed, 579 subtests`, so the row wants one more
  edit — in `README.md`, which this document does not own.
- **H9** `docs/roadmap-2026-11.md:88` vs `:219` — one of the two has to be marked
  superseded; both are currently presented as current.
- **H10** `docs/eval-harness-2026-11-04.md:22` vs `:212` — one of the two signs is
  wrong; which one is a question about the column's definition, not about the data.
- **H11** `CHANGELOG.md:10` "Two claims" — reconciled by reference here, but the
  word itself is still there and wants to be "one" or "four" depending on whether
  the sentence means the section or the release.
- **H4** `docs/research-model-sweep-q4.md` — the 84% bar and the scout's pick need
  a dated, appended correction of their own, not only a pointer.
- **§3b** the sweep's stale line numbers — the durable fix is to stop using line
  numbers as the index and lean on the `grep` command it already prints.

---

## 8. The single most misleading sentence currently live

`docs/distill-classifier-2026-11-04.md:320–325`:

> **A 3B description fix already exists and is worth more than this model.**
> `p1-diagnose` moves the same 3B from 74.0% to **82.0%** with two description
> strings (18 fixed, 6 broken, p = 0.0227). That is a larger, better-evidenced
> gain than anything measured here, and it is the natural first comparison for any
> future arm.

Why this one and not the others:

- It is **live**. The document is dated 2026-11-04, is not framed as a record, and
  carries no correction. There is no reading order in which a reader meets the
  out-of-sample result first.
- It is **in a decision section**. "What would change the answer" is where a reader
  goes to find out what to do next, and this is the first bullet.
- It calls the retired improvement **"better-evidenced"** than the negative result
  it sits inside. Measured on phrasing nobody had written, the same edit is
  −3.1 points over 415 rows with 29 broken rows against 16 fixed
  (`docs/rewrite-transfer-2026-11-04.md:9–15`). The word is exactly inverted.
- It draws the **opposite conclusion** from the one the newest report draws — "the
  natural first comparison for any future arm" against "the description-edit
  shortcut to raising its accuracy is now a negative result"
  (`docs/rewrite-transfer-2026-11-04.md:296–298`).
- It has **no competitor for the fault**. The nearest runners-up —
  `spec/1.0-spec.md:10`'s 91.4% premise and `docs/research-model-sweep-q4.md:7`'s
  84% bar — are wrong about a *level*, and a level error is visible the moment a
  reader reaches the README's measured table. This one is wrong about the
  *direction of a result that was measured*, in the same tree, on the same day, and
  it says so in the comparative form ("better-evidenced") that a reader has no way
  to check without reading three other files.

A pointer has been added above it. The sentence itself is left as written, because
retiring it is the job of that document's own dated correction, not of this one.

---

## Appendix — appended 2026-11-04: §8's premise is not true of the tree this ships with

*Appended, not rewritten. §7's H1 bullet and §8 both say the 82.0% sentence carries no
correction and that the sentence is left as written. Both were false when they were written,
and are false now.*

`docs/distill-classifier-2026-11-04.md` carries an inline retirement of that sentence, in the
same bullet, immediately beneath it — *"(Retired: that gain is in-sample and does not survive a
held-out set — 80.5% → 77.3% over 415 rows, 16 fixed / 29 broken. See
`docs/rewrite-transfer-2026-11-04.md`.)"* (`:324-326`). It arrived in `691ea50`, the commit
that wrote this section, so §8's first reason — "carries no correction … no reading order in
which a reader meets the out-of-sample result first" — was wrong on the day it was written
rather than stale a day later.

What survives from §8, and what does not:

- **Survives.** The sentence still calls the retired gain "better-evidenced", and a reader who
  stops at the bullet's first line still meets an inverted comparative. An inline retirement is
  weaker than deletion, and it is the strongest form append-only allows.
- **Does not survive.** "The sentence itself is left as written", two lines above this appendix;
  the heading's "currently live"; and H1's "needs its own appended correction" in §7 — the
  correction is there, which is what H1 asked for.
- **Moot.** "No competitor for the fault" ranked this sentence above `spec/1.0-spec.md:10` and
  `docs/research-model-sweep-q4.md`'s 84% bar on the ground that those two are marked and this
  one is not. All three now carry a marking — §7's own pointer table is the ledger for the
  other two — so the ranking no longer separates them. What §8 was right about is the class:
  a retired *direction of a measured result* is worse than a retired *level*.
