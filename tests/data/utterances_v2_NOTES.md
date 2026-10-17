# utterances_v2 — design notes

Deliverable: `tests/data/utterances_v2.json` — 150 rows, v1 schema (`{"utterance","label"}`) plus `"band"` on every row and `"tier_target"` on every t3 row. v1 (`utterances.json`) is untouched and stays canonical for the 50-row live probe.

## Bands

- **t1 — unambiguous (64)**: 8 per op. Direct intent, may contain signature vocabulary ("grep the tree", "search the web"). Sanity floor for the harness.
- **t2 — veiled (31)**: intent stated with no op-signature keyword. No "grep/search/mention", no "pick/choose/rank", no "queries/keywords/terms", no "fetch/url/link", no "r//reddit/sub" on the reddit rows. Phrasing leans on the *object* ("pull up the cron config", "of the results we just pulled, which one matters?").
- **t3 — boundary pairs (55)**: each row sits between two ops where the other op is a defensible reading; the label is the semantically correct one, `tier_target` is the confusable rival. t3 rows stress the sibling boundaries run2's misses came from.

Per-band per-label counts:

| label | t1 | t2 | t3 | total |
|---|---|---|---|---|
| filesys_read | 8 | 4 | 5 | 17 |
| filesys_glob | 8 | 4 | 9 | 21 |
| filesys_grep | 8 | 4 | 10 | 22 |
| decide_pick | 8 | 4 | 6 | 18 |
| decide_extract | 8 | 4 | 12 | 24 |
| web_search | 8 | 4 | 6 | 18 |
| web_fetch | 8 | 4 | 3 | 15 |
| reddit_digest | 8 | 3 | 4 | 15 |
| **total** | **64** | **31** | **55** | **150** |

Every op ≥ 15. decide_extract is deliberately heaviest in t3 (12) because run2's worst op (50% accuracy) lost rows in three different directions (web_search, filesys_glob, and via "display … .py").

## t3 boundary pairs

| pair | rows | label split | miss family it stresses |
|---|---|---|---|
| glob ↔ grep | 7 | grep 4 / glob 3 | "which files mention X" → misrouted to glob (idx 14) |
| read ↔ extract | 5 | read 3 / extract 2 | "display service/ti/service_entry.py" → misrouted to extract (idx 7) |
| pick ↔ glob | 5 | pick 3 / glob 2 | "among the files found…" → misrouted to glob (idx 25) |
| extract ↔ web_search | 7 | extract 4 / search 3 | "what search terms…" (idx 29) and "generate query variations" (idx 33) both → web_search |
| extract ↔ glob | 6 | extract 3 / glob 3 | "extract the file paths mentioned…" → misrouted to glob (idx 30) |
| read ↔ grep | 5 | grep 3 / read 2 | "does any file import fastapi?" → misrouted to read (idx 20) |
| grep ↔ web (search/fetch) | 5 | grep 3 / search 1 / fetch 1 | "which functions call urllib in tests" → misrouted to web_fetch (idx 18) |
| pick ↔ extract | 6 | pick 3 / extract 3 | latent decide-internal boundary (no run2 miss, probe showed 0.46 conf on extract rows) |
| reddit ↔ web_search | 4 | reddit 2 / search 2 | latent: forum/community phrasing ("check what the sub says" sat at 0.46 with 0.15 on search) |
| reddit ↔ web_fetch | 4 | reddit 2 / fetch 2 | latent: "digest the thread at that url" drew 0.13 on fetch |

Within each pair the two directions are both sampled (some rows lean toward the trap, some lean toward the true op while the trap stays implicit), so the label cannot be guessed from phrasing style alone. All 8 run2 misses are covered from both sides this way.

## Miss-paraphrase coverage (each ≥ 3 rows)

| run2 miss | probe row | paraphrases in v2 |
|---|---|---|
| display .py → decide_extract | idx 7 | "show ops/matcher.py on screen", "pull up the full listing of core/registry.py", "what's written in prompts/system.txt?" (pair read↔extract) |
| which files mention X → filesys_glob | idx 14 | "which files mention the scheduler hooks?", "which files talk about the eviction policy?", "which files under src discuss the shard map?", "what files make use of the old env var?" (pair glob↔grep) |
| urllib call-sites → web_fetch | idx 18 | "which functions call urllib in the test suite?", "find any use of the requests module in src", "where is the http client wired up in service/net?" (pair grep↔web) |
| does any file import fastapi → filesys_read | idx 20 | "do any modules pull in pydantic here?", "is fastapi referenced anywhere in the tree?", "where does the codebase touch the stage registry?" (pair read↔grep) |
| among files found → decide_pick | idx 25 | "among the files found, which one should we ship to the fetcher?", "among the earlier matches, pick the one answering the ask", "of the dozen files that came up, which is the right one to open?" (pair pick↔glob) |
| what search terms → decide_extract | idx 29 | "what search terms fit this ask?", "turn this into three distinct ways to ask around" (pair extract↔web_search) |
| extract file paths → decide_extract | idx 30 | "extract the file paths mentioned in the last message", "pull out the dirs the user pointed at", "what literal paths show up in that instruction?" (pair extract↔glob) |
| query variations → web_search | idx 33 | "generate a few phrasings of the lookup before we hit the web", "give me keyword variants for the upcoming lookup" (pair extract↔web_search) |

Note: idx 29 and idx 33 stress the *same* boundary from two angles; the extract↔web_search pair (7 rows) is the shared family, with "terms"-style and "variants"-style rows both present.

## Controls

- JSON parses (`python3 -m json.tool`), 150 rows exactly, no duplicate utterances after normalization (lowercase, whitespace collapse), zero verbatim reuse of v1 strings.
- Labels restricted to the 8 v1 op names; bands t1/t2/t3 only; `tier_target` present iff band == t3 and always ≠ label.
- Utterances 10–90 chars, user-voice, mixed imperative/interrogative/snippet; no self-referential phrasing ("run the grep op…"), no keyword-stamped t2 rows.
- Residual t2 nouns that overlap op vocabulary, judged generic rather than giveaway: "page" (2 fetch rows — v1's own fetch rows use "page"), "thread" (1 reddit row — generic forum noun), "hits" (1 pick row).

## Known risks / semantics flagged for review

1. **"open" is context-dependent.** v1 maps "open config/defaults.json" → filesys_read but "open the pending URL" → web_fetch. The verb carries no op signal; the object does. Several v2 rows rely on this (fetch t1 "open that blog post…"). A keyword-heuristic system cannot survive this — good, that's the point.
2. **"find" is a three-way ash heap.** find-by-name → glob ("find the README"), find-where-used → grep ("find every call site…"), find-info → web_search ("find recent news…"). v2 keeps all three under their v1-consistent labels; if a mid-tier model grades on "find ⇒ grep" it will fail ~6 rows. That is intentional pressure, not labeling error.
3. **reddit_digest without an "r/".** t2 reddit rows ("what's the community saying…", "check the boards…") lean on community language instead of the signature token. If the grader requires an "r/"-anchor, these will look mislabeled; treating reddit_digest as *reddit-native scraping* (per spec §7) rather than token-matched is the defensible reading.
4. **"cfmm"/"thread" rows straddling reddit↔fetch**: "give me the gist of that thread in r/MachineLearning" (reddit) vs "load the full thread at that forum url" (fetch). The split rule used: an in-sentence `r/…` anchor = reddit_digest; a *specific resolvable URL/adjective form* ("that … url", "linked post") = web_fetch. Both are contestable; the labels follow that stated rule consistently.
5. **c6-type rows** ("any config files lying around that look relevant?" → filesys_glob): dual reading — a name/entity query over the tree (glob) vs triage of already-surfaced items (pick). Labeled glob because no earlier candidate set is referenced; "lying around" is existential, not triage.
6. **"which functions call urllib…"-family** labels are grep even though the topic word (urllib, requests, http) is web-flavored. Deliberate: run2's idx-18 failure is exactly vocabulary-priming ("web word ⇒ web op"); these rows are the corrective stimulus.
7. **Length floor**: 7 rows sit under 25 chars ("find the README" is 15) — matching the user's own sanctioned t2 example; keep the 10-char floor as-is in the harness.
