# Overnight handoff — 2026-11-10, toolInterfacer

Operator is asleep and has authorised autonomous work: commits on the project's own branch, **no
pushes**, **no deleting data**, **$0** (no paid API calls), nothing in `service/ti/conf/defaults.json`.

## What is already done (do not redo)

The grep experiment ran: 415 rows, one session, one served model (`d1-3B-Q4_K_M.gguf`), four
arms. The numbers are in `docs/grep-arm-2026-11-10.md` and the analysis lives in
`bench/grep_arm_report.py`. Read both before touching anything.

Summary of the result:

| arm | correct | fixed | broke | note |
|---|---:|---:|---:|---|
| `floor_a` | 325/415 | — | — | A/A half 1 |
| `floor_b` | 326/415 | — | — | A/A half 2, **1 flip in 415** |
| `grep_only` | 328/415 | 10 | 7 | McNemar p = 0.6291 |
| `grep_control` | 321/415 | 10 | 14 | paraphrase, same length |

Per-option ledger: `filesys_grep` **+10 with 0 of its own rows broken**, `web_search` −3,
`decide_extract` −2, `filesys_glob` −1, `filesys_read` −1. Untouched net **−4**, so
pre-registered conditions 1 (p < 0.05) and 3 (untouched not net-negative) **FAIL**.
Eight of the ten `filesys_grep` rows fixed by `grep_only` are also fixed by `grep_control`,
so the gain on that option is robust to wording; the collateral damage is not.

**Verdict recorded: do not ship as a whole-dictionary edit.**

## Your tasks, in this order

1. **A `llama-server` is already running** on `127.0.0.1:8080` serving `d1-3B-Q4_K_M.gguf` with
   `-fa on -c 4096 -t 8`. Use it. Do not start another one and do not change the model.
2. **Run the `order_rev` arm**, which is already added to `ARMS` and has never been run. It is
   the same eight descriptions in the opposite order — no string changes — so it measures
   position alone (E3 of the plan):

   ```
   python3 bench/rewrite_transfer.py --arm order_rev --expect-model d1-3B-Q4_K_M.gguf
   ```

   About 19 minutes, 415 rows, 2.6–3.1 s/row measured tonight.
3. **Add its result to `docs/grep-arm-2026-11-10.md`** under an E3 heading, and state plainly
   what position alone is worth. If position moves accuracy as much as text does, say that —
   it changes how every earlier comparison in this repo should be read.
4. **Extend `bench/grep_arm_report.py`** so `order_rev` is compared against `floor_a` alongside
   the other arms.
5. **Kill the `llama-server` process when you no longer need it.** Port 8080 must not be left
   occupied. This is mandatory, not optional.
6. **Commit on the branch `overnight/2026-11-10`**, which already exists and carries the
   uncommitted harness change. Two commits: one for `bench/rewrite_transfer.py` +
   `bench/grep_arm_report.py`, one for the docs (`docs/grep-arm-2026-11-10.md`,
   `docs/plan-post-rewrite-2026-11-09.md`). Never push. Never rebase. Never touch `main`.
7. Leave `service/ti/conf/defaults.json` untouched. The operator decides that after he reads
   the doc.

## The open question you should leave ready for him

The grep gain is real and wording-robust, but it is paid for by `web_search` — all three lost
rows are phrased "find writing about X". The next experiment is therefore a **paired** rewrite:
`filesys_grep` plus a compensating `web_search` description in the same arm, both measured
together. Do **not** run it tonight unless you have finished everything above; it is a product
decision about what the descriptions say, and it is his.
