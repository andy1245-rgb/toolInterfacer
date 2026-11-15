# How to read the shadow log after a few days

One page. Shadow mode is **on** (`decide.shadow.enabled: true` in `defaults.json`, 2026-11-10). It
serves exactly what it served before; it only writes. Log: `$TI_SHADOW_LOG`, else
`~/.local/state/ti/shadow.jsonl`.

## The shape of a record

Every decision writes one line, and every op execution writes another under the **same `id`**:

```json
{"kind":"decision","id":"1a1ddcebe68-10","served":"bench/data/heldout_v3.json",
 "served_conf":0.41,"source":"primary","fused":"bench/data/heldout_v3.json",
 "member":null,"member_answered":false,"vocab_match":0.0,"why":"member abstained",
 "n_options":6,"options":["…"],"protect":["web_search","web_fetch"],"ts":1794722639.1}
{"kind":"outcome","id":"1a1ddcebe68-10","op":"decide_pick","step":"picked_file",
 "ok":true,"ms":3234.0,"attempt":1}
{"kind":"outcome","id":"1a1ddcebe68-10","op":"filesys_read","step":"content",
 "ok":true,"ms":2.8,"attempt":2}
```

`source` is which tier answered: `pre_gate`, `cache`, or `primary`. `attempt` counts outcomes per
decision, so **`attempt: 2` on the same `id` is a retry** — the cost of being wrong, in the
operator's time.

## Read it in this order

1. **`vocab_match` first. It decides whether the rest of the line means anything.** On live traffic
   the only `choice` question asked is `decide_pick`, whose options are candidate ids; the member
   is an eight-op classifier, so `vocab_match: 0.0` and `why: "member abstained"` is the *normal*
   record today. It says the shadow had no opinion, not that the fusion agrees.
2. **Then `member_answered: true` records.** Those are the only rows where a counterfactual exists.
   `agrees` / `fused != served` on them is the disagreement rate the fusion would produce.
3. **Then join to outcomes.** For a disagreement, did the served op succeed? `ok: false`, or a
   later `attempt: 2`, is the served answer's cost. The shadow's own choice has no outcome — it
   never ran — so a disagreement is *evidence of risk*, not proof of a mistake. Both sides need the
   served side's outcome to be judged, and only one of them has one.
4. **Then filter by `protect`.** On rows where `served` is `web_search` or `web_fetch`, `why` is
   `protected` and the member was not consulted. If retries cluster there, the protect list is too
   wide.

## What this log cannot tell you yet

The eight-op routing question — `filesys_grep` versus `web_search` — is **not** the question
`decide_pick` asks. Its options are candidate files and URLs, so the member abstains by
construction and the log will fill with `member abstained` records that look like perfect agreement.

That is the first thing to fix before the log can answer the fusion question, and it is a routing
change, not a shadow change: the op choice has to pass through `resolve_backend("choice")` with the
eight ops as its option set. Until it does, this log measures the `decide_pick` decision, which is a
real decision with a real downstream outcome and worth reading on its own terms — it just is not the
fusion's question.

## The one-line queries

```bash
# how often did the member actually have a say
jq -r 'select(.kind=="decision") | .member_answered' ~/.local/state/ti/shadow.jsonl | sort | uniq -c

# disagreements, with what was served
jq -r 'select(.kind=="decision" and .member_answered and (.fused != .served))
       | [.id, .served, .fused, .member_conf] | @tsv' ~/.local/state/ti/shadow.jsonl | head

# served-side failures and retries, per op
jq -r 'select(.kind=="outcome" and (.ok|not)) | .op' ~/.local/state/ti/shadow.jsonl | sort | uniq -c
jq -r 'select(.kind=="outcome" and .attempt > 1) | .op' ~/.local/state/ti/shadow.jsonl | sort | uniq -c
```
