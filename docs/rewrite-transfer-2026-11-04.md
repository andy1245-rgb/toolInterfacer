# The description rewrite, out of sample: the extract half was fitted to v2

2026-11-04. Runs the two-description rewrite from
[`bench/results/p1-diagnose/FINDINGS.md`](../bench/results/p1-diagnose/FINDINGS.md) §3 —
unchanged, no third iteration — against `bench/data/heldout_v3.json`, a set authored
after the rewrite was written and independent of the 150 rows it was derived from. Probe:
`bench/rewrite_transfer.py`. Branch `p3-rewrite`.

**Verdict: the whole rewrite fails out of sample. The 82.0% must be retired.**

* **On the full v3 set the rewrite is 3.1 points *worse*, not 8 points better:**
  80.5% → **77.3%** (334/415 → 321/415). 16 rows fixed, **29 broken**, exact McNemar
  **p = 0.0725**; paired delta **−3.1 pts [−8.5, +1.9]** over 133 families.
* **The gain on the two classes it targets is real and transfers:** grep + extract
  54.4% → **68.0%** (56/103 → 70/103), 16 fixed / **2** broken, p = **0.0013**,
  +13.6 pts [+3.5, +25.6]. That part of the diagnosis survives.
* **The cost is 27 broken rows on the six classes it does not touch** — 0 fixed, 27
  broken, p < 0.0001, −8.6 pts [−14.2, −4.0]. 27 > 16, which is the whole verdict.
* **The extract half was the fit.** On v2 it gained **+54.2** points; out of sample it
  gains **+9.8** (p = 0.18, n = 51). The grep half is the one that generalises: +13.6 on
  v2, **+17.3** on v3 with 9 fixed and 0 broken (p = 0.0039).
* **Class mix explains at most 2.7 of the 11.1-point swing.** Reweighting v3's per-class
  deltas to v2's class weights gives −1.7 pts (against v2's +8.0); reweighting v2's to
  v3's gives +5.3. The flip is the effect, not the composition.

## Contents

1. [What was asked, and why it became answerable](#1-what-was-asked-and-why-it-became-answerable)
2. [What was run](#2-what-was-run)
3. [The in-sample claim, reproduced](#3-the-in-sample-claim-reproduced)
4. [Out of sample](#4-out-of-sample)
5. [Per class: the two targets and the six](#5-per-class-the-two-targets-and-the-six)
6. [What the trade is, mechanically](#6-what-the-trade-is-mechanically)
7. [Composition or effect](#7-composition-or-effect)
8. [The `{mod}` caveat, measured rather than assumed](#8-the-mod-caveat-measured-rather-than-assumed)
9. [What this settles, and what it does not](#9-what-this-settles-and-what-it-does-not)
10. [Reproduce](#10-reproduce)

## 1. What was asked, and why it became answerable

`p1-diagnose` found that d1-3B routes on the object noun rather than the operation, and
that changing two of the eight option descriptions — `filesys_grep` and
`decide_extract` — moved 74.0% to 82.0% on the 150-row frozen set. It also stated, in
the same file, that those two strings were written by reading the errors of the rows
they were then scored on, so the gain was in-sample and 82.0% was an upper bound.

`bench/data/heldout_v3.json` did not exist when they were written. It is 415 rows over
8 ops in 133 authored families, with the train/test assignment made from the frame
inventory **before any utterance was rendered** (`bench/data/heldout_v3_NOTES.md`), and
it shares no row with `tests/data/utterances_v2.json`. So applying the unchanged rewrite
to v3 is a real out-of-sample test of the rewrite, and that is what this document
reports.

The rewrite was imported, not restated: `bench/rewrite_transfer.py` takes
`routing_diagnose.TARGETED` directly. Nothing was tuned, extended or re-derived — if it
failed, it was going to fail.

## 2. What was run

| arm | option set | source | rows |
|---|---|---|---:|
| baseline | `ti_bench.prompts.OP_DESCRIPTIONS` (frozen instrument) | `bench/results/p2-heldout/v3-d1-3B-Q4.json`, recorded 2026-11-04 | 415 |
| targeted | `routing_diagnose.TARGETED` (2 of 8 changed) | measured here, all 415 rows | 415 |

Protocol is the archived one: one `/v1/systemone` `choice` question per row over the
bare utterance, the same `INSTRUCTIONS` string, the same eight options in the same
order, served model asserted against `/props` before any probe traffic
(`--expect-model d1-3B-Q4_K_M.gguf`).

**Coverage is complete, so no row is skipped:** 415/415 on the targeted arm, of which
207 are v3's declared test side and 208 its train side. The plan was built to run the
two targeted classes first and both sides of the split, so a partial run would still
have answered the question; the server was at 3–5 s/call rather than the expected
14–18 s, so the sample was extended to the whole set and the sampling question
disappeared. Per-row output, including the probability vector, is
`bench/results/p3-transfer/rows.jsonl`.

The server is shared and was not started, stopped or rebound by this run; it was
attached to at `127.0.0.1:8080` only.

## 3. The in-sample claim, reproduced

Re-derived offline from the two archived recordings, not re-typed from the report:

| v2, 150 rows | accuracy | 95% Wilson | fixed / broken | exact McNemar |
|---|---|---:|---|---:|
| baseline (archived) | 74.0% (111/150) | [66.4, 80.4] | — | — |
| targeted (archived) | **82.0%** (123/150) | [75.1, 87.3] | 18 / 6 | **p = 0.0227** |

Paired delta **+8.0 pts [+2.0, +14.0]**, cluster bootstrap over families. This is
exactly the number `p1-diagnose` published, so the claim under test is the claim that
was made, and the run below is a comparison against a reproduced baseline.

The archive is only usable if the endpoint still answers as it did. Two checks, both
live, both against the archived answers:

* **24 hash-chosen v2 rows re-asked** on the targeted arm: **23/24 identical argmax**,
  **24/24 identical correctness**. The one argmax flip is a row that is wrong in both
  runs, decided 0.2549 against 0.2512 — 0.4% of probability mass. The quantity being
  compared here is correctness; it did not move.
* The v3 baseline arm was re-run live alongside this experiment as a whole-set check
  (below), because a 415-row pairing against an archive is a stronger claim than a
  24-row spot check.

## 4. Out of sample

The rewrite, unchanged, on the set it has never seen:

| v3 | baseline | targeted | fixed / broken | exact McNemar | paired delta (family CLT) |
|---|---:|---:|---|---:|---|
| **all 415 rows** | 80.5% (334) [76.4, 84.0] | **77.3% (321)** [73.1, 81.1] | **16 / 29** | 0.0725 | **−3.1** [−8.5, +1.9] |
| **test side, 207** | 81.6% (169) [75.8, 86.3] | 79.7% (165) [73.7, 84.6] | 7 / 11 | 0.4807 | −1.9 [−8.4, +4.3] |
| train side, 208 | 79.3% (165) [73.3, 84.3] | 75.0% (156) [68.7, 80.4] | 9 / 18 | 0.1221 | −4.3 [−12.9, +3.9] |

The 82.0% does not reproduce anywhere on v3. The point estimate is *below* baseline on
all 415 rows and on both sides of the split, and no slice reaches significance in either
direction — which is the honest reading: the rewrite is not a measurable improvement on
this set, and it is close to a measurable harm.

The test-side number is the one that matters for the original claim, because it is the
only slice whose phrasings were held out by construction. **81.6% → 79.7%: −1.9 points,
7 fixed against 11 broken, p = 0.48.** The claim to retire is not "+8 points on fresh
phrasing"; there is no +8 points on fresh phrasing.

## 5. Per class: the two targets and the six

The overall number is a sum, and the sum is where the answer is. Per class, v3 (all 415
rows) beside the v2 in-sample figure for the same class:

| class | v2 n | v2 Δ | v3 n | v3 baseline | v3 targeted | v3 fix/brk | v3 Δ | p |
|---|---:|---:|---:|---|---|---|---:|---:|
| **`filesys_grep`** *(rewritten)* | 22 | +13.6 | 52 | 69.2% (36) | **86.5% (45)** | **9 / 0** | **+17.3** | **0.0039** |
| **`decide_extract`** *(rewritten)* | 24 | **+54.2** | 51 | 39.2% (20) | 49.0% (25) | 7 / 2 | **+9.8** | 0.1797 |
| `reddit_digest` | 15 | −6.7 | 52 | 80.8% (42) | **61.5% (32)** | 0 / 10 | **−19.2** | 0.0020 |
| `web_search` | 18 | 0.0 | 52 | 94.2% (49) | **78.8% (41)** | 0 / 8 | **−15.4** | 0.0078 |
| `filesys_read` | 17 | 0.0 | 52 | 82.7% (43) | 76.9% (40) | 0 / 3 | −5.8 | 0.2500 |
| `filesys_glob` | 21 | 0.0 | 52 | 84.6% (44) | 80.8% (42) | 0 / 2 | −3.8 | 0.5000 |
| `web_fetch` | 15 | −6.7 | 52 | 94.2% (49) | 90.4% (47) | 0 / 2 | −3.8 | 0.5000 |
| `decide_pick` | 18 | −11.1 | 52 | 98.1% (51) | 94.2% (49) | 0 / 2 | −3.8 | 0.5000 |

Aggregated the way the rewrite was aimed:

| | n | baseline | targeted | fixed / broken | exact McNemar | paired delta |
|---|---:|---|---|---|---:|---|
| v3, the two rewritten classes | 103 | 54.4% (56) [44.8, 63.7] | **68.0% (70)** [58.4, 76.2] | **16 / 2** | **0.0013** | **+13.6** [+3.5, +25.6] |
| v3, the six untouched classes | 312 | 89.1% (278) [85.2, 92.1] | 80.5% (251) [75.7, 84.5] | **0 / 27** | **< 0.0001** | **−8.6** [−14.2, −4.0] |
| v2, the two rewritten classes *(in-sample)* | 46 | 30.4% (14) | 65.2% (30) | 17 / 1 | 0.0001 | +34.8 |
| v2, the six untouched classes *(in-sample)* | 104 | 93.3% (97) | 89.4% (93) | 1 / 5 | 0.2188 | −3.9 |

Three readings, in order of how much they matter:

1. **`filesys_grep` is fixed, and this is the one durable finding.** 9 fixed, 0 broken,
   p = 0.0039 on held-out families, and it is better out of sample (+17.3) than it was
   in sample (+13.6). The grep half of the diagnosis — "the description claims no noun,
   so the noun decides" — is confirmed on phrasing nobody had written when the fix was
   written.
2. **`decide_extract` was the fit.** +54.2 points in sample collapses to +9.8 out of
   sample and stops being significant. A single description whose measured effect falls
   by a factor of 5.5 on fresh phrasing is a description that was written against the
   24 rows it scored on.
3. **The trade is not a wash — it is 27 rows, and it is net negative.** In sample the
   rewrite broke 5 rows outside the two classes; on v3 it breaks 27. Nothing outside the
   two classes was fixed. Phase 1's hope that touching only two descriptions would avoid
   the equilibrium shift is falsified: the shift simply moved to classes the v2 sample
   could not see it move in.

## 6. What the trade is, mechanically

29 broken rows, by where they went and by which family they came from. Clusters matter
more than rows here: v3's rows are slot-filled variants of one authored sentence, so a
family of 6 contributes 6 rows but only one observation.

| rows | families | baseline → targeted | example row |
|---:|---:|---|---|
| 6 | 1 | `reddit_digest` → `decide_extract` | "the reddit picture on {topic}, not a single post" |
| 5 | 3 | `web_search` → `decide_extract` | "is {topic} still a problem in the wild?" |
| 4 | 1 | `reddit_digest` → `web_search` | "search reddit for {topic}" |
| 3 | 2 | `filesys_read` → `decide_extract` | "pull up the file service/ti/decide/resolve.py" |
| 3 | 1 | `web_search` → `filesys_grep` | "find writing about {topic} outside the community threads" |
| 2 | 2 | `decide_pick` → `decide_extract` | "which should we act on first?" |
| 2 | 2 | `filesys_glob` → `decide_extract` | "give me the shape of the tree under service/ti" |
| 2 | 1 | `decide_extract` → `web_search` | "turn that into three distinct search queries" |
| 2 | 2 | `web_fetch` → `decide_extract` | "go and get it, all of it" |

29 rows in 15 family-clusters. **20 of the 29 land on `decide_extract`.**

Two mechanisms, both second-order consequences of editing description text, both of them
the thing `p1-diagnose` §3 warned about when a *global* rewrite destroyed `decide_pick`:

* **`decide_extract` became the sink.** Its new text names "queries, keywords, paths,
  names" and asserts "the user's own message", so any utterance that is a question or a
  possessive with no operation cue now matches extract before it matches the option that
  owns the operation: "is there a test harness anywhere in here?", "give me the shape of
  the tree under `service/ti`", "pull up the file service/ti/decide/resolve.py", "which
  should we act on first?". The same sentence that fixes 7 of extract's own rows breaks
  20 rows belonging to other classes — the description is not doing two things, it is
  doing one thing across a wider surface than it was fitted to.
* **The untouched descriptions re-normalised, and the grep line leaks too.**
  "search reddit for {topic}" (4 rows, one family) went to `web_search` although neither
  `reddit_digest` nor `web_search` was edited, and "find writing about {topic} outside the
  community threads" (3 rows, one family) went to `filesys_grep` — the rewritten grep
  line claiming a web-shaped search. "Two lines changed, six unchanged" does not mean
  "six classes insulated": the options compete for every row, so changing two changes
  the equilibrium all eight are read in.

10 of the 29 breaks are the two `reddit_digest` families above, and v2's sample contains
neither shape in any form: every v2 `reddit_digest` row names the venue ("what's hot on
r/LocalLLaMA today?", "digest the comments on the llama.cpp release post", "check the
boards for reactions") and never says "search reddit for {topic}". That is the
composition point from §7 seen at the row level — but it cuts against the rewrite, not
for it: the frames v2 could not see were the frames the edit damages.

## 7. Composition or effect

v3's baseline is 80.5%, not 74.0%, so any comparison of levels between the sets is a
comparison of difficulty mixes. Paired deltas are immune to that, so the question is
whether the *delta* difference is composition:

| | v2 | v3 | note |
|---|---:|---:|---|
| measured delta, own class mix | **+8.0** | **−3.1** | 11.1-point swing |
| delta at v2's class weights | +8.0 | **−1.7** | v3's per-class deltas reweighted to v2's mix |
| delta at v3's class weights | **+5.3** | −3.1 | v2's per-class deltas reweighted to v3's mix |

Class mix is worth **1.4 to 2.7 points** of the 11.1-point swing, depending on which
set's per-class effects are held fixed while the weights move (the two Oaxaca orderings,
which differ here because the effect vectors are so far apart). The remaining **8 to 10
points are the effect**, and the per-class table in §5 shows where: only `filesys_grep`
holds or improves out of sample, while `decide_extract` falls from +54.2 to +9.8 and the
classes that were at or near 100% on v2 — `reddit_digest` (−6.7 → −19.2) and
`web_search` (0.0 → −15.4) — are where the loss lands.

Two composition facts go the *other* way and should be stated with the above:

* **v3's untouched classes are less question-shaped than v2's**, not more: 75 of 312
  (24%) end in a question mark against 39 of 104 (38%) on v2. So the sink mechanism is
  not being helped by v3 being interrogative; it is firing on frames, not on punctuation.
* **v3 is class-balanced (25% in the two weak classes) where v2 reserves 31%** for them,
  which is the 1.4 points above. A rewrite whose gain is confined to two classes will
  therefore look worse on v3 for a reason that has nothing to do with the rewrite — and
  it still comes out negative after that is corrected for.

## 8. The `{mod}` caveat, measured rather than assumed

`p2-heldout` warned that the v3 grep families use `{sym}`, `{pat}`, `{topic}` and `{path}`
and **not one `{mod}` frame**, so the package-noun confusion that dominated v2's grep
misses is under-represented. That caveat is real and it is measurable, and it cuts in a
specific direction:

| | grep rows | of which package-noun (`urllib`, `requests`, `fastapi`, `pydantic`, ...) |
|---|---:|---:|
| v2 | 22 | **4** |
| v3 | 52 | **0** |

On v2, the rewrite fixed **0 of those 4** package-noun rows and did all of its grep work
on the other 18 (3 fixed, 0 broken). So the family v3 cannot measure is the family the
rewrite provably does not repair — which means **v3 slightly over-states the grep
transfer for a package-noun-heavy population, not under-states it.** The grep number in
§5 should be read as "+17.3 points on non-package-noun grep phrasing, which is the family
the rewrite repairs; on package-noun phrasing it is unchanged, 0/4 on v2".

This is the opposite of the hypothesis that the v3 effect might be an understatement of
the real-phrasing effect. It is not a hypothesis that survives contact with the v2
package-noun rows.

## 9. What this settles, and what it does not

**Settled.**

* **74.0% → 82.0% does not generalise.** On held-out families the same rewrite scores
  80.5% → 77.3% over 415 rows, and 81.6% → 79.7% on the 207-row test side. Any document
  quoting 82.0% as an improvement should quote this instead, or quote 82.0% only as the
  in-sample fit it is.
* **The diagnosis survives; the fix does not.** Noun-routing is still the mechanism
  (v3's misses are the same shape as v2's), but a description edit is a global change to
  the option competition, and two lines changed out of eight was enough to move 27 rows
  the wrong way. The "targeted" framing was the hypothesis that it wouldn't; that
  hypothesis is now falsified on a set that could test it.
* **`filesys_grep`'s new description transfers** (+17.3 pts, 9 fixed / 0 broken,
  p = 0.0039, 52 rows). If anything from `p1-diagnose` ships, it is that one line —
  and it should ship with `decide_extract`'s original text restored, which is a
  different experiment from this one and is **not** measured here.
* **The keyword rule table (78.7% on v2) is untouched by any of this.** It is still the
  production router, and nothing in this document argues against it.

**Not settled.**

* **Whether the grep line alone is net positive out of sample.** That is one arm this
  experiment deliberately did not run — it would be a new rewrite, and the brief is to
  score *this* one. The evidence points that way (grep's 9 fixes are pure and its 0
  breaks are independent of the extract change), but a claim needs its own run.
* **Anything about real user phrasing.** Every v3 row is a slot-filled variant of 384
  authored sentences. This document measures transfer to unseen *frames* in this repo's
  inventory, not to a session transcript.
* **The size of the extract effect on a population where extract's noun list is right.**
  v3's extract rows are the t3 families that hand a path to an extract request; a
  population of "pull the keywords out of this email" rows was not sampled, and v2's
  +54.2 remains the only number for it — in-sample, and now known to be roughly 5x too
  large for anything else.

**Consequence for the roadmap.** The 8-way choice router is the planned P2 replacement
and the SFT target; the description-edit shortcut to raising its accuracy is now a
negative result, and the next change to it should be measured on v3-test with the same
paired protocol rather than on the 150.

## 10. Reproduce

```bash
# both arms on heldout_v3 (targeted is measured live; baseline is archived in p2-heldout)
python3 bench/rewrite_transfer.py --arm targeted \
    --expect-model d1-3B-Q4_K_M.gguf --base-url http://127.0.0.1:8080
python3 bench/rewrite_transfer.py --arm baseline \
    --expect-model d1-3B-Q4_K_M.gguf --base-url http://127.0.0.1:8080

# live check that the archived v2 targeted recording still describes this endpoint
python3 bench/rewrite_transfer.py --spot-v2 24 --arm targeted \
    --expect-model d1-3B-Q4_K_M.gguf

# every table above, offline, from the committed per-row artifacts
python3 bench/rewrite_transfer.py --report
```

Artifacts: `bench/results/p3-transfer/rows.jsonl` (415 rows, targeted arm, per-row
probabilities), `bench/results/p3-transfer/summary.json` (the numbers above),
`bench/results/p3-transfer/v2_spot_check_targeted.json` (24 rows).
