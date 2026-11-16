# The consequence question is answered by the op set: every op is a read

2026-11-10, third job. The operator answered the open question by intuition — *"recoverable ops are
probably anything read-only"* — and the audit says the intuition is stronger than he thought: **there
is no mutating op in the set at all.**

## What the audit found

`service/ti/ops/` contains four families — `filesys`, `reddit`, `research`, `web` — and the eight
public ops map onto them:

| op | what it does | mutates? |
|---|---|---|
| `filesys_read` | returns one file's contents | no |
| `filesys_glob` | returns paths | no |
| `filesys_grep` | returns matching lines | no |
| `decide_pick` | returns a choice from a list | no |
| `decide_extract` | returns strings from the request | no |
| `web_search` | returns results | no |
| `web_fetch` | returns one page | no |
| `reddit_digest` | returns a summary | no |

And the single subprocess path, `ops/cli.py`, builds `argv` as a **list** (`launcher(script) + args`)
and calls `subprocess.run(argv, capture_output=True, ...)` with no `shell=True`. So there is no shell
injection surface and no command string assembled from model output.

**Therefore a misroute costs one wasted call and a retry.** It cannot damage anything, because no op
can damage anything. 84.58% is not "good enough if it's safe" — it is directly usable, and the
remaining questions are cost and latency, not safety.

Document this in `docs/` as the *consequence model*, because it is the argument that licenses the
accuracy level, and it is currently implicit.

## The concrete job: make the fusion pass its own condition

The fusion arm failed only C3 — "no label worse than −3" — because `web_search` nets −5. The members
are a calibrated text classifier and the 3B, averaged. The failure is not the fusion arithmetic, it is
that the text member is allowed to vote for an op it has no business answering.

Options, in order of preference:

1. **Mask the member's candidate set.** A member votes only among the labels it has demonstrated
   competence on (from its own confusion matrix), never among all eight. This is the principled fix:
   the member's weakness is known, so do not consult it there.
2. **Weight the mean by per-label member skill**, measured out-of-fold.
3. **Exclude `web_search` from the fusion and route it by the 3B alone.**

Try (1) first. Re-run the same pre-registered conditions on the same folds: C1 delta ≥ +20 rows,
C2 p < 0.01, C3 no label worse than −3, C4 positive in ≥4 of 5 fold seeds. Report all four, and if
C3 still fails, say what the residual loss is and stop rather than tuning until it passes.

## Also, from the rot register and not yet closed

- **`heuristic_micro` is dead** — its path regex requires a leading `/` and 0 of 415 rows match. Decide
  whether to fix or retire it, and if fixed, prove it with coverage, not accuracy.
- **The fresh-install error message recommends `classical`**, which answered 0/415 until this week.
  It should recommend whatever the coverage table says is alive.

## Hard limits

Branch commits only, never push. No hosted calls. `service/ti/conf/defaults.json` untouched — the
fusion is an arm, not a config change, until the operator says otherwise.
