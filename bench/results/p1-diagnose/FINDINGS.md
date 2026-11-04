# Why d1-3B loses `filesys_grep` and `decide_extract`

Probe: `bench/routing_diagnose.py`. Artifacts: this directory (one JSON per
experiment, per-row with the full probability vector). Every number below is
re-derived by `python3 bench/routing_diagnose.py --report`, which reads only
these files and never touches the server.

Server: `127.0.0.1:8080`, `models/d1-3B-Q4_K_M.gguf`, llama.cpp build
`b1-f74e2fa`, shared with other work (p50 3591 ms against the archived lane's
155 ms — latency only; the endpoint is deterministic).

## Verdict

**Presentation bug, not a capability limit.** The 3B routes on the *object noun*
in the utterance and ignores the *operation* the noun is an object of. It fails
`decide_extract` and `filesys_grep` because those two are the only labels whose
description does not claim the nouns their rows are phrased around, so the model
answers with whichever other option claims the noun instead.

Changing **two of the eight description strings** — leaving the other six
byte-identical — moves the same 150 rows from **74.0% to 82.0%**
(111/150 → 123/150, 18 fixed / 6 broken, exact McNemar **p = 0.0227**), and
recovers **17 of the 32** misses in the two classes. No retraining, no new
model, no service change.

**The 82.0% is retired as an improvement — read the two corrections at the end of
this file before quoting it.** It is an in-sample fit (first correction), and the
rewrite was subsequently run out of sample, where it scores 80.5% → 77.3% over
415 rows with 29 broken against 16 fixed (second correction, and
`docs/rewrite-transfer-2026-11-04.md`). The mechanism above stands. The size of
the gain does not, and no part of it should be quoted as a gain.

## 1. Reproduction, and what it answered instead

`bench/live_probe.py` over `tests/data/utterances_v2.json`, the frozen
instrument: `backend.choice(utterance, INSTRUCTIONS, OP_DESCRIPTIONS)`.

| | archived lane (`b11512`) | this run (`b1-f74e2fa`) |
|---|---|---|
| accuracy | 74.00% | **74.00%** |
| per-class, all 8 labels | identical | identical |
| argmax disagreement row by row | — | **0 / 150** |

The finding is the model's, not the instrument's or a build's.

### The 14 `filesys_grep` misses

```
→ web_fetch      conf=0.80  | which functions call urllib in the test suite?
→ web_fetch      conf=0.88  | find any use of the requests module in src
→ web_fetch      conf=0.74  | where is the http client wired up in service/net?
→ web_fetch      conf=0.53  | locate each place we catch HTTPError
→ web_fetch      conf=0.47  | where is handle_stream defined?
→ web_fetch      conf=0.22  | does anything here still call the v1 endpoint?
→ filesys_glob   conf=0.71  | any references to wallet.json left in the tree?
→ filesys_glob   conf=0.53  | which files under src discuss the shard map?
→ filesys_glob   conf=0.36  | what files make use of the old env var?
→ filesys_glob   conf=0.22  | does anything in bin/ reference the legacy CLI?
→ decide_extract conf=0.49  | is fastapi referenced anywhere in the tree?
→ decide_extract conf=0.23  | where do we define the hop budget?
→ filesys_read   conf=0.57  | which modules import yaml.safe_load?
→ web_search     conf=0.09  | do any modules pull in pydantic here?
```

### The 18 `decide_extract` misses

```
→ filesys_glob   conf=0.93  | write out the file names the request refers to
→ filesys_glob   conf=0.88  | extract the file paths mentioned in the last message
→ filesys_glob   conf=0.87  | before we dump anything, write out the paths it wants
→ filesys_glob   conf=0.83  | name the dirs the request refers to
→ filesys_glob   conf=0.75  | what literal paths show up in that instruction?
→ filesys_glob   conf=0.75  | pull out the dirs the user pointed at
→ filesys_glob   conf=0.55  | turn the user's wording into the exact filenames to print
→ filesys_glob   conf=0.36  | list the exact identifiers the user mentioned
→ web_search     conf=0.65  | what search terms fit this ask?
→ web_search     conf=0.57  | give me a fresh batch of angles to try
→ web_search     conf=0.51  | give me keyword variants for the upcoming lookup
→ web_search     conf=0.43  | give me three phrasings of this question for the engine
→ web_search     conf=0.39  | what keywords best capture what I'm after?
→ web_search     conf=0.31  | what words should I throw at it to get hits?
→ web_search     conf=0.15  | come up with alternate wordings for my lookup
→ web_search     conf=0.29  | come up with five candidate titles from the brief
→ decide_pick    conf=0.45  | pull the topic keywords from my message
→ decide_pick    conf=0.15  | generate a few phrasings of the lookup before we hit the web
```

**The errors are structured, and the structure is the noun.** In `decide_extract`
the split is almost exact: 8 rows whose object is a *path, name or directory* go
to `filesys_glob`; 8 whose object is a *keyword, term or phrasing* go to
`web_search`. In `filesys_grep`, 6 of 14 go to `web_fetch` and every one of those
six carries a network noun (`urllib`, `requests`, `http client`, `HTTPError`,
`endpoint`); 4 go to `filesys_glob` and every one of those carries a
*listing*-shaped object (`which files`, `in the tree`, `in bin/`).

A cue test over the 32 misses: **27 of 32 carry a noun from another option's
domain**. The model is not reading the eight descriptions as operations; it is
matching the salient noun to whichever description mentions that kind of thing.

The predictions in the brief were partly wrong and should be retired: the errors
are **not** "`decide_extract` answered as `filesys_read`" (that bucket is 0 rows
for extract; the reverse, `filesys_read` → `decide_extract`, is 4 rows) and they
are not one specific wrong label (`decide_extract` splits 8/8 between two).

## 2. Hypotheses: which died

**Truncation — dead.** `/props` reports `n_ctx = 8192`; a real call reports
`input_tokens = 124`. The state is the bare utterance (≤ ~90 chars) plus a fixed
option block. No row is within two orders of magnitude of a limit.

**`/v1/systemone` request shape — dead, and structurally so.** The instrument
builds one request for every row: `state` = the utterance string, `instructions`
= one constant, `criteria` = the same eight options in the same order
(`bench/live_probe.py:109` → `service/ti/decide/backends/d1_systemone.py:29`).
There is no per-class instruction, no per-class option set and no per-class
candidate list to be malformed. The only per-row variable is the utterance.

**K dependence — real but minor, and not the cause.** Asked as a 3-way question
(truth + that class's two favourite wrong answers, nothing else), the 46 rows go
from 14/46 to 20/46 (8 fixed, 2 broken). Halving K to 6 by dropping two
non-competitors buys **one row** (30.4% → 32.6%). And at K=3 the same wrong
answers survive *confidently* — `which functions call urllib in the test suite?`
→ `web_fetch` at conf 0.90; `extract the file paths mentioned in the last
message` → `filesys_glob` at 0.92. The model is not crowded out of the right
answer; it prefers the wrong one in a two-way contest.

**Option order / position bias — real, but not a lever.** Reversing the option
order flips the argmax on **8 of 46** rows (17%), so the model is order-sensitive
— but accuracy does not improve (30.4% → 28.3%); the flips move rows within the
noise, in both directions. Presentation *order* is a second-order knob;
presentation *content* is the first-order one.

**Post-hoc remap — impossible, not merely untried.** The wrong labels are shared
across truths, so no function of the answer alone can repair it: `filesys_glob`
stands for 8 `decide_extract` rows and 4 `filesys_grep` rows. Every candidate
`answer → truth` remap is net negative, because it sacrifices the rows that same
label gets right. Best achievable net over all 12 candidate remaps: **−2 rows**.
A remap that worked would have to read the utterance — which is what the keyword
rules already do, and why rules beat the 3B here.

**Description text — held.** See §3.

## 3. The fix, and what it recovers

Two description strings change; the other six stay byte-identical
(`TARGETED` in `bench/routing_diagnose.py`):

```diff
-filesys_grep   Search inside file contents for a regular expression.
+filesys_grep   Search inside local file contents for a pattern or string; the
+               answer is which code or text contains it.
-decide_extract Extract query strings or parameter values from the user command.
+decide_extract Pull the literal strings out of the user's own message — queries,
+               keywords, paths, names. Answers with words from the request.
```

| label | n | baseline | targeted (95% Wilson) |
|---|---|---|---|
| decide_extract | 24 | 25.0% [12,45] | **79.2% [60,91]** |
| filesys_grep | 22 | 36.4% [20,57] | **50.0% [31,69]** |
| decide_pick | 18 | 100.0% | 88.9% −2 |
| reddit_digest | 15 | 100.0% | 93.3% −1 |
| web_fetch | 15 | 93.3% | 86.7% −1 |
| filesys_read | 17 | 70.6% | 70.6% ±0 |
| filesys_glob | 21 | 95.2% | 95.2% ±0 |
| web_search | 18 | 100.0% | 100.0% ±0 |
| **overall** | **150** | **74.0% [66,80]** | **82.0% [75,87]** |

18 rows fixed, 6 broken, exact McNemar **p = 0.0227**. 82.0% clears the keyword
rules (78.7%) on the same set, which the unmodified 3B did not.

Recovery by error family — the two families the rewrite addresses collapse, the
one it does not address does not move at all:

| family | n | recovered |
|---|---|---|
| extract → filesys_glob (path/name object) | 8 | **7 / 8** |
| extract → web_search (query-word object) | 8 | **6 / 8** |
| extract → decide_pick | 2 | 1 / 2 |
| grep → filesys_glob (listing shape) | 4 | 3 / 4 |
| **grep → web_fetch (network noun)** | 6 | **0 / 6** |
| grep → other | 4 | 0 / 4 |
| **total** | **32** | **17 / 32** |

Gain by band: t1 81.2% → 84.4%, t2 71.0% → 77.4%, t3 67.3% → 81.8%. Inside the
two failing classes every band improves (t1 6/16→10/16, t2 3/8→6/8, t3 5/22→14/22),
so the gain is not confined to the boundary rows the set was built around.

### Caveats, stated rather than buried

- **These numbers are in-sample.** The two replacement strings were written by
  reading these 150 rows' errors. There is no held-out routing set in the repo
  (`heldout_phase3/4.json` are the *shape* task, not routing), so no held-out
  figure is available. The band breakdown above is the only robustness read, and
  it is not a substitute. Treat 82.0% as an upper bound on a fresh set.
- **The brief's "19 rows" does not reproduce.** The two classes account for **32**
  of the 39 misses; 25 of them are rows the keyword rules get right and the 3B
  gets wrong. No cut I tried (band, error family, rules-right) yields 19.
- The rewrite must **not** be applied to all eight descriptions. A first attempt
  that reworded all of them fixed 15 rows and broke 20 (111 → 106, p = 0.50):
  `decide_pick` collapsed 18/18 → 5/18 because its original text was doing real
  work. The eight descriptions compete for rows, so a global rewrite measures a
  new equilibrium rather than the two classes it means to repair.

## 4. Why this looked unfixable: the gate cannot see it

17 of the 32 misses are emitted at **conf ≥ 0.5**, so the maxprob gate that the
cascade relies on commits them. At τ = 0.5 the gate keeps 111 rows at 83.8%
precision, and 24 of those rows are in the two classes with only 7 correct —
these classes are the dominant source of *confident* error, which is why they
cost more than their 32/150 share of the accuracy column.

## 5. Where this sits in the product

Scope note, because it changes the urgency but not the diagnosis:

- Production routing is still the keyword rule layer
  (`service/ti/planner/inspector.py`), not the 8-way choice; the choice router is
  the planned P2 replacement and the SFT target (`tools/dataset/criteria.py`
  names `routing.ops_v1` as the default SFT task). This bench is measuring that
  future router.
- Production `decide_extract` never asks this question at all: it calls
  `backend.extract(state, {"type": "value_list"})`
  (`service/ti/ops/decisions.py:run_extract`), and the shipped extract backend is
  `rule_extract`. The 8-way "which op is this?" question is not how the product
  runs an extract today.

Neither point makes the routing benchmark wrong — it is the target the project
intends to train and ship — but the 25.0% should be read as "the 3B is a bad
router", not "extract is broken in production".

## 6. The residual lever, if someone wants it

`filesys_grep → web_fetch` is untouched (0 of 6). Its rows all carry a network
noun, and the grep description still does not deny the reading "this mentions
`urllib`, so it must be a fetch". A third iteration would add that denial to
`filesys_grep` alone and re-measure; the more valuable version of the experiment
would first build a held-out routing set, so the next change can be scored
out-of-sample instead of tuned.

## Reproduce

```
python3 bench/live_probe.py --utterances tests/data/utterances_v2.json \
    --expect-model d1-3B-Q4_K_M.gguf --base-url http://127.0.0.1:8080 \
    --out bench/results/p1-diagnose/baseline_live.json

python3 bench/routing_diagnose.py --only k3,neutral6,orders,minimal_pairs,rewrite2
python3 bench/routing_diagnose.py --report      # every number above, from the JSON
```

The frozen instrument (`bench/ti_bench/prompts.py`) is unmodified;
`tests/test_dataset_pipeline.py` pins it and it still matches the dataset's copy.

---

## Correction — 2026-11-04: the 82.0% is in-sample

Appended, not edited. Every number above is unaffected — they are recorded per-row
artifacts, re-derived by `python3 bench/routing_diagnose.py --report`. What follows
is a constraint on how the headline may be read, stated at the top level because
**the Verdict is the part that gets quoted** and §3's caveat is the part that does
not.

**74.0% → 82.0% is an in-sample improvement.** The two replacement description
strings were written by reading the errors of the same 150 rows they are then
scored on. §3 already says so ("These numbers are in-sample ... Treat 82.0% as an
upper bound on a fresh set"); the correction is that the sentence belongs beside
the number wherever the number appears, including the Verdict at the top of this
file and any downstream summary of it.

What that does and does not leave standing:

- **The mechanism is not in doubt.** The model routes on the object noun and not
  the operation — that was established by the raw miss lists in §1, by the K=3
  two-way contests in which the wrong answer wins at conf 0.90, and by the remap
  analysis (no `answer → truth` function can repair it). None of that depends on
  the rewrite.
- **The size of the gain is in doubt.** 18 fixed / 6 broken at exact McNemar
  p = 0.0227 is a real paired difference *on these rows*; it is not an estimate of
  the effect on phrasing nobody has written. The rewrite is a fit to this set.
- **The band breakdown in §3 is not a robustness check.** t1 81.2 → 84.4, t2 71.0 →
  77.4, t3 67.3 → 81.8 was read as evidence the gain is "not confined to the
  boundary rows the set was built around". It is not a substitute for a held-out
  set: `utterances_v2` has **63.3% atomic rows** (95 of 150), 55 rows in 11 declared
  boundary families, and holding families out moves accuracy by **−0.4 to +3.7
  points, in both directions, inside the seed-to-seed spread** — so no split of
  this set can test unseen phrasing. See `docs/eval-harness-2026-11-04.md` §1–§2
  and §4.
- **"82.0% clears the keyword rules (78.7%)" (§3) compares an in-sample figure
  against a contaminated one.** The rules table was written after reading these
  same 150 utterances, which the local-classifier report states in its §7 item 3.
  Both numbers are fits to this set; the comparison between them carries no
  information about a fresh one.

**Status.** A held-out validation of the description rewrite is being run on
another track. This file does not pre-empt it, holds no held-out number, and
should not be updated with one except as a further dated correction. Until then,
quote 82.0% only as "in-sample, upper bound", and 74.0% as the model's measured
accuracy.

**Evidence.** §3 above ("Caveats, stated rather than buried"), and
`docs/eval-harness-2026-11-04.md` §1–§2 (the atomic-row fraction and the three
failed family-reconstruction routes), §4 (the row-versus-family gap for every
signal on this set), §6 ("Not settled").

---

## Correction — 2026-11-04 (second): the rewrite was run out of sample, and it fails

The first correction said a held-out validation was running on another track and
that this file holds no held-out number. It has one now.
`docs/rewrite-transfer-2026-11-04.md` applies `TARGETED` **unchanged** to
`bench/data/heldout_v3.json` — 415 rows, 133 families, authored after these two
strings were written — with the archived v2 run as the in-sample reference.

| set | n | baseline | targeted | fixed / broken | exact McNemar |
|---|---:|---|---|---|---:|
| v2 (in-sample, §3 above) | 150 | 74.0% | 82.0% | 18 / 6 | 0.0227 |
| **v3, all rows** | **415** | **80.5%** | **77.3%** | **16 / 29** | **0.0725** |
| v3, test side | 207 | 81.6% | 79.7% | 7 / 11 | 0.4807 |
| v3, the two rewritten classes | 103 | 54.4% | 68.0% | 16 / 2 | **0.0013** |
| v3, the six untouched classes | 312 | 89.1% | 80.5% | 0 / 27 | **< 0.0001** |

**The 82.0% does not generalise and must not be quoted as an improvement.** The
mechanism in §1–§2 stands and the `filesys_grep` line transfers on its own
(+17.3 pts, 9 fixed / 0 broken, p = 0.0039), but `decide_extract`'s +54.2 falls to
+9.8 (p = 0.18) out of sample, and the edit breaks 27 rows in the six classes it
does not touch — 20 of them by turning `decide_extract` into a sink for
question-shaped utterances. Net: −3.1 points over 415 rows.

Two specific claims above are retired with it:

- **§3's "the rewrite must not be applied to all eight descriptions" implied that
  changing two would avoid the equilibrium shift.** It does not. Editing two
  descriptions re-normalises the other six: 10 broken rows are two
  `reddit_digest` families that moved to `web_search` and `decide_extract`
  although neither description was touched.
- **§6's residual lever** (`filesys_grep → web_fetch`, 0 of 6 on v2) is not
  exercised by v3 at all: v3's grep families contain no package-noun frames. What
  v3 does show is that the grep line fixes the *other* families strongly enough to
  be significant on held-out phrasing.

§3's tables and per-row artifacts are unchanged and still re-derive exactly
(`python3 bench/routing_diagnose.py --report`). What changed is their status: they
are the in-sample fit, and the out-of-sample measurement is now on record.
