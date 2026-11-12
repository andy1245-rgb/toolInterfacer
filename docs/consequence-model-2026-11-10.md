# The consequence model — why a misroute costs one call and not an incident

2026-11-10. This is the argument that licenses a router in the mid-80s. It is short because the
property it rests on is structural rather than statistical: **no operation in this system can
change anything.**

## The claim

A routing error — the model picks `filesys_grep` where the operator meant `filesys_read` — costs:

1. **one wasted op call**, and
2. **a retry**.

It cannot damage state, cannot execute a command, and cannot read outside the run root. The
remaining risks are cost and latency, not consequence, which is why an accuracy in the mid-80s is
directly usable rather than "usable if it is safe". The operator's intuition — *recoverable ops
are probably anything read-only* — is stronger than that: there is no mutating op in the set to
recover from.

## What it rests on, verified

**Every registered op is a read.** `service/ti/ops/` registers nine: `filesys_read`,
`filesys_glob`, `filesys_grep`, `decide_pick`, `decide_extract`, `web_search`, `web_fetch`,
`reddit_digest`, and the internal `digest_assemble`. A scan of the whole package for
`write_text`, `.write(`, `open(…'w')`, `os.remove`, `os.unlink`, `mkdir`, `shutil.rm/move/copy`,
and `rename` returns nothing:

```bash
grep -rn "write_text\|shutil\.\(rm\|move\|copy\)\|os\.remove\|os\.unlink\|mkdir\|rename" \
  service/ti/ops/ --include=*.py
# (no output)
```

**There is one subprocess path, and it has no shell.** `ops/cli.py:132` builds
`launcher(script) + mode_args + [str(a) for a in args]` — a list — and `:141` calls
`subprocess.run(argv, capture_output=True, text=True, …)` with `shell` left false. A query or URL
therefore cannot be re-read as a flag or a command separator, and no command string is assembled
from model output. A scan for `shell=True`, `os.system` and `popen(` across `service/ti` returns
nothing.

**Filesystem access is confined before the first syscall.** `filesys/confine.py` resolves every
path against the run root and refuses anything that leaves it, walking one component at a time so
that the check itself never enters a denied directory:

```
$ python3 -c "... confine.resolve('/etc/passwd') ..."
/etc/passwd        -> REFUSED OutsideRoot '/etc/passwd' resolves outside '<repo>'
../../etc/passwd   -> REFUSED OutsideRoot '../../etc/passwd' resolves outside '<repo>'
bench/data/heldout_v3.json -> <repo>/bench/data/heldout_v3.json
```

So a misroute cannot turn into an unintended read of a file the operator was not already working
in.

## What it does **not** cover

* **Added ops.** The guarantee is a property of the op set as it stands. One `filesys_write` would
  end it, and nothing about the surrounding code would notice — the same shape as the classical
  tier pointing at the wrong vocabulary. So the property is enforced by a test, not by a comment:
  `tests/test_consequence_model.py` fails if a mutating call appears in `service/ti/ops/`, if the
  subprocess adapter gains a shell, or if the eight router ops stop being read-only.
* **Cost.** A misrouted `filesys_grep` can walk a large tree, and a misrouted `web_fetch` can fetch
  a page. The consequence is time and money, which the budget and the hop log already bound.
* **Correctness of the answer.** This model says a mistake is cheap, not that it is invisible. A
  router that is wrong one time in five still wastes one call in five.

## Why it licenses the number

Without this property, an 84% router needs a human in the loop, because the 16% is a live hazard
and its tail is unbounded. With it, the 16% is a queue of retries, and the decision becomes an
ordinary cost question: how many retries per thousand requests, at what latency, against what the
expensive model costs. That is the frame the numbers in `docs/fusion-2026-11-10.md` and
`docs/classical-revival-2026-11-10.md` should be read in.
