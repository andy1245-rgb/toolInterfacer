# A classifier trained on the distilled corpus: 68.7%, and not shippable

2026-11-04. Fits an 8-way op classifier on `data/distill/` — the corpus certified in
[`distill-corpus-certification-2026-11-03.md`](distill-corpus-certification-2026-11-03.md)
— and measures it on the corpus's own held-out split and then, as a transfer test, on
`tests/data/utterances_v2.json`. Stdlib only: no dependency was added to
`pyproject.toml`. Branch `p2-distill`.

**Verdict: a negative result, cleanly measured — with one positive finding inside it.**

* **It does not beat the 3B.** 68.7% [59.6–77.0] against `d1_3B`'s 74.0% [66.5–81.3] on
  the same 150 rows. The 5.3-point gap is **not significant** (exact McNemar over the 64
  discordant rows: 28 fixed, 36 broken, **p = 0.38**). At n=150 that gap is noise.
* **It does not beat a keyword table.** `bench/local_clf.py`'s hand-written cue table —
  frozen, fitted on nothing — scores **78.7%** [72.3–85.3] on the same rows, and its
  advantage over the classifier **is** significant (16 fixed, 31 broken, **p = 0.040**). A
  corpus, a fit and a recipe buy less than a 35-rule cue table.
* **It does beat the 3B exactly where the 3B is broken.** `decide_extract` 79.2% against
  25.0% (14 rows fixed, 1 broken) and `filesys_grep` 63.6% against 36.4% (10 fixed, 4
  broken). This is the failure `bench/results/p1-diagnose/FINDINGS.md` diagnosed — the
  3B routes on the object noun — and a model trained on the operation's frames does not
  make it. It
  pays for that on `web_search` (**38.9%** against 100%), `filesys_glob` (52.4% against
  95.2%) and `decide_pick` (77.8% against 100%).
* **The corpus cannot select the model, and that is the reusable finding.** Across the
  seven candidates measured, family-grouped CV on the corpus ranks them **backwards**
  against transfer (Spearman ρ = **−0.71**): the model CV likes best transfers worst
  (77.4% CV → 60.0% v2) and the one it ranks fourth of seven transfers best (76.3% →
  68.7%). §6.

Per the certification's own rule, no number below is pooled alone: every table is per
band, with n.

## Contents

1. [What was trained](#1-what-was-trained)
2. [Per band, on the corpus](#2-per-band-on-the-corpus)
3. [The transfer to v2](#3-the-transfer-to-v2)
4. [Per class: where it wins and where it loses](#4-per-class-where-it-wins-and-where-it-loses)
5. [Where it fails, and why](#5-where-it-fails-and-why)
6. [The selection problem](#6-the-selection-problem)
7. [Decision](#7-decision)
8. [Reproduce](#8-reproduce)

## 1. What was trained

**Data.** `data/distill/train.jsonl`, 329 rows, 239 families. The label is the row's
`intended` op as shipped (`label`) — never the teacher's answer, which is not in the
frozen rows at all: the artefact carries no `target`/`teacher` field, so
`tests/test_distill_clf.py` can assert the teacher's answer is structurally unavailable to
the trainer rather than merely discouraged. `val.jsonl` (81 rows) is the corpus's declared
family-disjoint holdout; `utterances_v2` is used for one thing only, the transfer
measurement in §3, and never for fitting.

**Features** (`tools/distill/clf.py`), two views and a shape block:

* the raw word unigrams and bigrams of the utterance;
* the same, over an **object-abstracted** copy: `abstract()` rewrites the utterance's
  nouns into their kind — `<url>`, `<sub>`, `<path>`, `<file>`, `<ident>`, `<lit>`,
  `<num>` — so `print the whole pyproject.toml for me` and `open service/ti/registry.py`
  keep their operation and lose their noun. The patterns are properties of the surface
  form, so an object the corpus never saw (`wallet.json`, `tests/data/utterances.json`)
  is typed the same way at inference time. This is the feature-space version of the
  diagnosis in `bench/results/p1-diagnose/FINDINGS.md`: route on the operation, not
  the object;
* a shape block (`s:url`, `s:sub`, `s:ext`, word count, first token, …).

**Learner.** Closed-form dual ridge on one-hot targets (`alpha = (K+λI)⁻¹Y`, λ=1.0,
`K = XXᵀ` over l2-normalised idf rows), via a pure-Python Cholesky. One hyperparameter,
no seed, no epoch count. `data/distill/clf_recipe.json` carries the configuration, the
corpus hashes and the artefact sha256; `train.py --check` re-fits and asserts it
reproduces byte-identically.

**Why ridge and not the eight-seed ensemble I first shipped.** The first model was an
unbounded Adagrad softmax ensemble. It fits the corpus in-sample at **100%**, its weights
reach |w| = 4.4e3, its eight seeds disagree on **109 of 150** v2 rows, and its averaged
"confidence" on v2 sits at multiples of 1/8 — it is not an ensemble of a fit, it is eight
random memorisers whose mean is a vote. It transfers at 60.0%. Ridge, on the same
features, fits at 97.9% in-sample with mean max-probability 0.53 and transfers at 68.7%.
§6 records what this did to model selection.

## 2. Per band, on the corpus

Three protocols, all corpus-only. `val` is the certified family-disjoint split and ships
with the artefact; `family-CV` is 5-fold grouped CV over all 329 train rows with three
seed-draws of the fold assignment, and is the in-corpus number worth quoting because a
val cell is 21–32 rows wide.

| band | val (n) | family-CV (n) | object-CV (n) |
|---|---:|---:|---:|
| t1 literal | 84.4% (32) | 85.7% (100) | 93.3% (100) |
| t2 veiled | 52.4% (21) | 63.1% (85) | 69.0% (85) |
| t3 boundary | 75.0% (28) | 77.5% (144) | 84.7% (144) |
| **pooled** | **72.8% (81)** | **76.3% (329)** | **83.3% (329)** |
| **t2+t3 only** | 65.3% (49) | 71.6% (229) | 79.8% (229) |

`object-CV` is a third protocol added for §6: the same 5-fold grouped CV with the
split unit replaced by the **slot value** a row names (`open <path>`, `grep for <sym>`), so
a fold tests the frame on nouns it never saw. It scores **7.0 points higher** than the
family split and every band rises. That is the opposite of what a generalisation test
should do, and §6 is about why.

Per class on val: `filesys_grep` 10/10, `filesys_read` 11/13, `web_fetch` 11/13,
`decide_extract` 8/11, `filesys_glob` 7/10, `decide_pick` 5/6, `reddit_digest` 2/2,
**`web_search` 5/16**.

**Which band carries it? t3.** 144 of the 329 CV rows are boundary rows, they are the only
band where the corpus's structure is a real test (a frame that cues both the label and the
confusable rival), and they are held at 77.5%. t1 is the opposite: it is 100 of the 329
rows, its utterances name the operation, and the model is *worse* on it than the
unregularised ensemble was (85.7% against 92.3%) — ridge trades t1 accuracy for +9.0
points on t2, which is the trade the certification asks for
("t1 teaches keyword matching and will inflate every headline number").

**t2 is the band that does not work.** 63.1% on 85 rows and 52.4% on 21. It is also the
band the certification flagged: 76.4% teacher agreement, every cell under 15 rows, and
labels the teacher contradicts on 25 of 106 rows. Two readings are available and this
measurement cannot separate them — the classifier is weak on veiled phrasing, or ~24% of
the t2 labels are genuinely ambiguous so that a model agreeing with the teacher would be
*supposed* to score ~0.76 there. A ceiling near 0.76 would make 63.1% less alarming, but
this run cannot tell the difference, and the audit CSV that would is still blank.

## 3. The transfer to v2

Frozen evaluation: fit on `data/distill/train.jsonl`, then score all 150 rows of
`tests/data/utterances_v2.json`. Nothing fits on v2 — `eval_harness`'s `frozen` path skips
the refit loop exactly as it does for a recorded 3B run, so both signals are scored once,
on the same rows, with the same cluster bootstrap over v2's 106 groups.

**What this number cannot measure.** v2's declared families are its t3 boundary pairs, so
**95 of 150 rows are atomic** (63%): for those rows a "family-held-out" split holds out
nothing and the interval is honest about dependence it cannot remove. The number is
therefore **optimistic by an unknown amount**, and it is a *transfer* test — a different
authoring pass over the same 8 ops — not a generalisation test.

Seven candidates, fixed in `bench/distill_transfer.py` as one declared set, so this
is a table rather than a search (§6 says what that is worth):

| candidate | v2 acc | 95% CI | t1 (64) | t2 (31) | t3 (55) |
|---|---:|---|---:|---:|---:|
| `C1` ens, raw | 60.0% | 53.1–66.7 | 56.2% | 61.3% | 63.6% |
| `C2` ens, both views | 60.7% | 53.9–67.4 | 59.4% | 58.1% | 63.6% |
| `C3` ens, abstract | 66.0% | 58.9–73.4 | 65.6% | 61.3% | 69.1% |
| `C4` ens, decayed | 59.3% | 52.2–66.0 | 56.2% | 58.1% | 63.6% |
| `C5` ridge, raw | 63.3% | 54.4–71.4 | 67.2% | 51.6% | 65.5% |
| **`C6` ridge, both views (shipped)** | **68.7%** | **59.6–77.0** | 76.6% | 54.8% | 67.3% |
| `C7` ridge, abstract | 66.7% | 57.6–74.7 | 68.8% | 61.3% | 67.3% |
| — `d1_3B` (archived run) | 74.0% | 66.5–81.3 | 81.2% | 71.0% | 67.3% |
| — `rules` (cue table, fitted on nothing) | 78.7% | 72.3–85.3 | 85.9% | 67.7% | 76.4% |

t2+t3 only, which is what the certification asks a reader to look at: shipped **62.8%**
(86), `d1_3B` 68.6%, `rules` 73.3%.

**Paired tests** — exact McNemar over the rows two signals disagree on, so a 2-point gap
on 150 rows is reported as the noise it is:

| a | b | a fixes | a breaks | net | p |
|---|---|---:|---:|---:|---|
| shipped | `d1_3B` | 28 | 36 | −8 | **0.38** — no evidence of a difference |
| shipped | `rules` | 16 | 31 | −15 | **0.040** — the cue table wins |
| `C1` (CV's pick) | `d1_3B` | 28 | 49 | −21 | 0.022 |

So: **no evidence that the corpus-trained classifier beats the 3B, and positive evidence
that it loses to a keyword table that needs no training data.** The point estimate is 5.3
points below the 3B, and 150 rows cannot resolve a gap that size.

## 4. Per class: where it wins and where it loses

Against the 3B on the same rows, with the fix/break split so the trade is visible:

| op | n | shipped | `d1_3B` | `rules` | fixed | broke |
|---|---:|---:|---:|---:|---:|---:|
| `decide_extract` | 24 | **79.2%** | 25.0% | 83.3% | **14** | 1 |
| `decide_pick` | 18 | 77.8% | 100.0% | 94.4% | 0 | 4 |
| `filesys_glob` | 21 | 52.4% | 95.2% | 71.4% | 0 | 9 |
| `filesys_grep` | 22 | **63.6%** | 36.4% | 72.7% | **10** | 4 |
| `filesys_read` | 17 | 64.7% | 70.6% | 82.4% | 3 | 4 |
| `reddit_digest` | 15 | 100.0% | 100.0% | 80.0% | 0 | 0 |
| `web_fetch` | 15 | 80.0% | 93.3% | 73.3% | 1 | 3 |
| `web_search` | 18 | **38.9%** | 100.0% | 72.2% | 0 | **11** |

The first two bold rows are the finding worth keeping, and they line up exactly with
`bench/results/p1-diagnose/FINDINGS.md`: the 3B's description-strings make it answer
the noun, so
`decide_extract` (which claims no noun) collapses to 25.0% and `filesys_grep` (same)
collapses to 36.4%. A model trained on frames — where "give me three phrasings of this
question" and "write out the file names the request refers to" are `decide_extract` rows —
gets those back:

```
decide_extract  3B said web_search    | give me three phrasings of this question for the engine
decide_extract  3B said filesys_glob  | list the exact identifiers the user mentioned
decide_extract  3B said decide_pick   | pull the topic keywords from my message
filesys_grep    3B said web_fetch     | where is handle_stream defined?
filesys_grep    3B said filesys_glob  | any references to wallet.json left in the tree?
filesys_grep    3B said web_fetch     | which functions call urllib in the test suite?
```

and it loses the ones the 3B gets right because its noun semantics are excellent:

```
web_search    mine said filesys_grep | find the official docs on httponly cookie flags
web_search    mine said reddit_digest| any chatter elsewhere about the sigill crash?
filesys_glob  mine said filesys_grep | list every file ending in .toml anywhere in the repo
filesys_read  mine said reddit_digest| check the top of CHANGELOG.md
```

**The union is 139/150 = 92.7%** — the rows where either signal is right. That is the
ceiling for any row-by-row selector between them, and it is the case for the complement
being real. It is also not special: `union(rules | d1_3B)` is the same 92.7%, and the best
candidate reaches 94.7% (`C3`). A per-class selector over a broken-on-`extract` 3B and a
broken-on-`web_search` classifier has plenty of room, and nothing here shows that *this*
classifier is the right second opinion — the cue table is cheaper and no worse.

## 5. Where it fails, and why

**`web_search` collapses into `reddit_digest`, and it does so on the corpus too.**
11 of the 22 misses on the corpus's own val split are `web_search` → `reddit_digest`:
"search online for benchmarks on the cascade thresholds", "who's talking about the retry
budget these days?", "i don't know enough about the x86 lane numbers yet". So this is not
a transfer artefact — the corpus teaches it. Its `web_search` frames are built to be
veiled (no "search" keyword allowed by `bands.py`) and its `reddit_digest` frames are its
chattiest register, so a veiled ask and a chatty ask land on the same features; when the
transfer target's search rows are veiled a different way, the classifier has nothing left
to separate them, and all 11 of its v2 misses go to a neighbour (5 `reddit_digest`, 4
`filesys_grep`, 1 `filesys_glob`, 1 `filesys_read`), 0 fixed against the 3B.

**`reddit_digest` is a magnet: 20 of the 47 misses land on it,** including
`filesys_read` ("check the top of CHANGELOG.md", "pull up the cron config for me") and
`decide_pick` ("which of those two forum threads is more on-topic?"). The corpus's reddit
frames are the corpus's chattiest register ("what does r/selfhosted say about {topic}?",
"catch me up on that megathread"), so chatty-and-indirect — which is what v2's t2 rows
are, by construction — points at `reddit_digest`. The corpus's register mix, which
`frames.py` deliberately varies, is doing harm here: t2 in the corpus *is* reddit-shaped
more often than v2's t2 is.

**`filesys_glob` at 52.4% is the object abstraction misfiring.** "list every file ending
in .toml anywhere in the repo" keeps `.toml` as a bare token that the corpus's own glob
frames rarely use alone, and "how many places do we still do it the old way?" has no
object to type at all. Nine rows broken, none fixed, and the 3B is at 95.2% — this is
where retail noun knowledge beats a frame model.

**Nothing here is a bug in the instrument.** The per-row dumps are
`bench/results/distill_transfer_sweep.json`; `--sweep` re-derives every table above from
the committed corpus and the archived 3B run with no live calls.

## 6. The selection problem

This is the part worth keeping even though the classifier is not.

The seven candidates, ranked by family-grouped CV on `train.jsonl` (the corpus's own
criterion) and by v2 transfer:

| candidate | family-CV | v2 | in-sample | mean max-P on v2 | seed disagreement |
|---|---:|---:|---:|---:|---|
| `C1` ens, raw | **77.4%** | 60.0% | 100% | 0.70 | 109/150 |
| `C4` ens, decayed | 77.2% | **59.3%** | 100% | 0.71 | 112/150 |
| `C2` ens, both | 77.0% | 60.7% | 100% | 0.71 | 112/150 |
| `C6` ridge, both | 76.3% | **68.7%** | 98% | 0.53 | — |
| `C3` ens, abstract | 75.2% | 66.0% | 100% | 0.72 | 108/150 |
| `C5` ridge, raw | 74.0% | 63.3% | 98% | 0.52 | — |
| `C7` ridge, abstract | 73.2% | 66.7% | 97% | 0.56 | — |

**Spearman ρ = −0.71 between the two rankings.** The corpus's criterion prefers the four
models that memorise (100% in-sample, saturated outputs, seeds that disagree on 109–112 of
150 out-of-corpus rows) and ranks the best transferrer fourth. The mechanism is the one
the certification warned about in its own words — "object vocabulary is shared with the
eval set… rows do not duplicate, but nouns recur": **family-held-out CV leaves the nouns
in place.** A frame's slot values recur across families (`topic`, `path` and `sym` values
are drawn from a 69-value inventory reused by every frame), so holding a family out still
leaves every noun in the training fold, and a model that has learned "tau_lo → grep" is
rewarded by the split that was supposed to punish it. Object-held-out CV, which I also
built (`train.grouped_cv(unit="object")`), does not fix this either: the memorising
model scores **83.4%** on it and the shipped model **83.3%** — both *higher* than their
family CV — because an unseen object inside a seen frame is the easy case, not the hard
one.

What this means, stated plainly: **an in-corpus number on this corpus cannot rank models
for this task, and the certification's recommended protocol (§5, "fit on train.jsonl,
select on val.jsonl") inherits the defect.** val ranks the candidates the same way family
CV does (74.1% for the three ensembles, 72.8% for the shipped ridge) and would have picked
`C1`, costing 8.7 points of transfer.

Two honest caveats, because this is where a reader should press:

* **The bounded-vs-unbounded split was decided after `C1`'s transfer number was seen.**
  The declared rule was "highest family-CV among candidates whose fit is bounded", and
  the evidence for calling the ensembles unbounded is the in-sample/mean-max-P/disagreement
  columns above, measured independently of v2 — but it is not a rule I wrote before looking.
  The conclusion does not depend on it: every candidate loses to the 3B and to the cue
  table except `C6`, which ties the 3B within noise.
* **v2 has now been read once for diagnosis and once for measurement**, so it is spent as
  an unbiased benchmark for this model family. The sweep above is a diagnostic table, not
  seven independent transfer estimates. The conclusion that survives is the *relative* one
  — nothing here beats 78.7% — and the certification's own instruction not to tune on v2 is
  why there is no eighth candidate.

## 7. Decision

**Do not ship it as the routed classifier, and do not put it behind the inspector.**

The four reasons, in order of weight:

1. **It is worse than something already in the repo that needs no training data.** The cue
   table in `bench/local_clf.py` scores 78.7% [72.3–85.3] and beats this classifier at
   p = 0.040. Shipping a corpus, a fit and a 138 kB artefact to lose to a 35-rule cue table is
   a regression dressed as progress.
2. **It is not distinguishable from the 3B overall** (68.7% vs 74.0%, p = 0.38), so it buys
   nothing on the aggregate decision it was built for.
3. **Its win is concentrated in two classes and its loss in three others.** 92.7% of rows
   are answerable by the pair, but the pair is `filesys_grep`+`decide_extract` on one side
   and *everything noun-driven* on the other. A per-class selector is the interesting
   design, and this classifier is not yet the right component for it, because a cue table
   covers the same rows for free.
4. **The corpus's selection protocol is broken for model choice (§6)**, so "train on this
   corpus and pick the best model" is not yet a reliable procedure. That is a property of
   the corpus, not of this classifier, and it is the thing to fix first.

**What would change the answer**, and each of these is a measurement, not a hunch:

* **A 3B description fix already exists and is worth more than this model.** `p1-diagnose`
  moves the same 3B from 74.0% to **82.0%** with two description strings (18 fixed, 6
  broken, p = 0.0227). That is a larger, better-evidenced gain than anything measured here,
  and it is the natural first comparison for any future arm. *(Retired: that gain is
  in-sample and does not survive a held-out set — **78.3% → 77.4%** over 415 rows against a
  live baseline measured in the same session, 16 fixed / 20 broken, p = 0.618, and 165 → 165
  on the held-out side: a wash, not a failure. An earlier archive-paired reading of
  80.5% → 77.3%, 16 fixed / 29 broken, was corrected on 2026-11-05. See
  `docs/rewrite-transfer-2026-11-04.md`.)*
* **Object-held-out selection is the protocol to add.** It exists as a diagnostic here
  (`train.object_key` over `frames.SLOTS`); what the corpus needs is a *split* that holds slot
  values out, so an in-corpus number stops rewarding memorised nouns. Until then the
  corpus's own val split cannot select a model.
* **t2 needs the audit.** Half the residual error is in a band where 24% of the labels are
  contested by the teacher. Annotating the audit CSV would tell a future run whether 63%
  is a model failure or the band's ceiling — and the certification already blocks a
  confidence target on the same annotation.
* **A two-view model that keeps the noun for the classes that need it** (glob, extract) and
  drops it for the classes that must not use it (grep, search) is the obvious next shape.
  That is a per-class feature decision, and §4 names which classes it would have to flip.

None of that is a reason to ship today. The negative result is the deliverable: **68.7%
[59.6–77.0], not better than the 3B within noise, worse than the cue table at p = 0.040,
better than the 3B on exactly the two classes the 3B is documented to fail, and trained
under a corpus protocol that ranks candidates backwards.**

## 8. Reproduce

```bash
python3 tools/distill/train.py --cv --object-cv  # val + both CV splits, per band
python3 tools/distill/train.py --check           # the recipe re-fits to the same sha
python3 bench/distill_transfer.py --sweep \
    --json bench/results/distill_transfer_sweep.json     # this document's §3–§6 tables
python3 -m pytest tests/test_distill_clf.py -q
```

No server is touched and no network call is made: the 3B column is the archived per-row
run at `bench/results/p1-diagnose/baseline_live.json` (deterministic; `p1-diagnose` shows
two runs agreeing on 150/150 rows). The `rules` row needs `numpy`, which
`bench/local_clf.py` already imports; if it is absent the row is skipped with a note and
nothing else changes.
