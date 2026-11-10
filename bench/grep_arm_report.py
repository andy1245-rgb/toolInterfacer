"""Analyse the p3-transfer rows for the grep ship decision.

Reads bench/results/p3-transfer/rows.jsonl, filters the arms run on 2026-11-10
(floor_a, floor_b = the A/A floor; grep_only = the treatment; grep_control = the
worded control; order_rev = the same eight descriptions reversed), applies the
repo's own P2 gate test, and reports the four things P5 asks for: rows with an
exact McNemar, the per-option ledger, and whether the delta clears 3x the floor.
"""
import json
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

ROWS = Path(__file__).resolve().parents[1] / "bench/results/p3-transfer/rows.jsonl"
ARMS = ("floor_a", "grep_only", "grep_control", "floor_b", "order_rev")
TARGET = "filesys_grep"

# The baseline order, from ti_bench.prompts.OP_DESCRIPTIONS. The order_rev arm is
# this list reversed, so position is read as (1-based slot in this list).
BASELINE_ORDER = ("filesys_read", "filesys_glob", "filesys_grep", "decide_pick",
                  "decide_extract", "web_search", "web_fetch", "reddit_digest")
MS_FLOOR = 1000.0


def is_gate(rec):
    """P2: a gate answer has ms under the floor, or a vector that is one hot 1.0
    with seven 1e-6 -- that is the resolver answering, not the model."""
    if rec.get("ms", 0) < MS_FLOOR:
        return True
    probs = rec.get("probabilities") or {}
    if not probs:
        return False
    vals = sorted((float(v) for v in probs.values()), reverse=True)
    return len(vals) >= 2 and abs(vals[0] - 1.0) < 1e-6 and all(abs(v - 1e-6) < 1e-6 for v in vals[1:])


def mcnemar_exact(b, c):
    """Two-sided exact binomial on the discordant pairs, as the repo's protocol asks."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def load():
    by_arm = defaultdict(dict)
    gate = Counter()
    for line in ROWS.open():
        rec = json.loads(line)
        arm = rec["arm"]
        if arm not in ARMS:
            continue
        if is_gate(rec):
            gate[arm] += 1
            continue
        by_arm[arm][rec["id"]] = rec
    return by_arm, gate


def order_section(base, rev, grep_arm=None):
    """E3: the same eight descriptions reversed. No string changes, so whatever
    moves here is position and nothing else.

    Read the tables together: `flips` is the size of the position effect, the
    per-slot ledger says whether it has a direction (a slot that wins rows when it
    sits early) or is just churn, and the last block asks whether reversal buys the
    same rows the grep wording buys -- the two effects are not the same mechanism
    if it does not.
    """
    if not rev:
        return
    shared = sorted(set(base) & set(rev))
    print(f"\n{'='*72}\nE3: position alone -- the same eight descriptions, reversed")
    print(f"    compared on {len(shared)} rows present in both floor_a and order_rev")
    if not shared:
        return

    cb = sum(base[k]["correct"] for k in shared)
    cr = sum(rev[k]["correct"] for k in shared)
    fixed = [k for k in shared if not base[k]["correct"] and rev[k]["correct"]]
    broke = [k for k in shared if base[k]["correct"] and not rev[k]["correct"]]
    flips = [k for k in shared if base[k]["choice"] != rev[k]["choice"]]
    p = mcnemar_exact(len(broke), len(fixed))
    print(f"    floor_a   {cb}/{len(shared)} = {100*cb/len(shared):.1f}%   "
          f"order_rev {cr}/{len(shared)} = {100*cr/len(shared):.1f}%   "
          f"delta {100*(cr-cb)/len(shared):+.2f} pts")
    print(f"    McNemar   fixed {len(fixed)}  broke {len(broke)}  p={p:.4f}")
    print(f"    argmax moved on {len(flips)}/{len(shared)} rows = "
          f"{100*len(flips)/len(shared):.1f}%")
    print("    (that is the position effect; the A/A floor printed above is the same")
    print("     measurement with the order unchanged, so the two are directly comparable)")

    # direction, not just size: per slot in the baseline order, net rows gained when
    # the reversal moves that description to the other end of the list
    print(f"\n    {'label':<16}{'slot':>5}{'rev slot':>9}{'n':>5}{'base':>7}"
          f"{'rev':>7}{'fix':>5}{'brk':>5}{'net':>5}")
    print(f"    {'-'*16}{'-'*5}{'-'*9}{'-'*5}{'-'*7}{'-'*7}{'-'*5}{'-'*5}{'-'*5}")
    for i, lab in enumerate(BASELINE_ORDER, 1):
        ks = [k for k in shared if base[k]["label"] == lab]
        if not ks:
            continue
        lcb = sum(base[k]["correct"] for k in ks)
        lcr = sum(rev[k]["correct"] for k in ks)
        lf = sum(1 for k in ks if not base[k]["correct"] and rev[k]["correct"])
        lb = sum(1 for k in ks if base[k]["correct"] and not rev[k]["correct"])
        print(f"    {lab:<16}{i:>5}{9-i:>9}{len(ks):>5}{lcb:>7}{lcr:>7}"
              f"{lf:>5}{lb:>5}{lcr-lcb:>+5}")

    # where the flips went: a position effect that shuffles rows between labels looks
    # different from one that buys correct answers
    moved = Counter((base[k]["choice"], rev[k]["choice"]) for k in flips)
    print("\n    the largest move pairs (baseline choice -> reversed choice):")
    for (src, dst), n in moved.most_common(6):
        gained = sum(1 for k in flips
                     if base[k]["choice"] == src and rev[k]["choice"] == dst
                     and not base[k]["correct"] and rev[k]["correct"])
        lost = sum(1 for k in flips
                   if base[k]["choice"] == src and rev[k]["choice"] == dst
                   and base[k]["correct"] and not rev[k]["correct"])
        print(f"      {src:<16}-> {dst:<16}{n:>4} rows  ({gained} fixed, {lost} broke)")

    # do the two effects overlap? the 10 grep rows the wording bought are the test
    g = grep_arm or {}
    bought = [k for k in shared if k in g and base[k]["label"] == TARGET
              and not base[k]["correct"] and g[k]["correct"]]
    if bought:
        same = sum(1 for k in bought if rev[k]["correct"])
        print(f"\n    the {len(bought)} {TARGET} rows the wording bought: order_rev also gets "
              f"{same} of them")
        for k in bought:
            print(f"      {k}  order_rev chose {rev[k]['choice']:<16} "
                  f"{'ok' if rev[k]['correct'] else 'miss'}  {base[k]['utterance'][:52]}")
    return {"n": len(shared), "a": cb, "b": cr, "fixed": len(fixed),
            "broke": len(broke), "flips": len(flips), "p": p}


def main():
    by_arm, gate = load()
    for arm in ARMS:
        print(f"{arm:<10} rows={len(by_arm[arm]):>4}  gate-excluded={gate[arm]:>3}")

    a, b = by_arm["floor_a"], by_arm.get("floor_b", {})
    order_section(a, by_arm.get("order_rev", {}), by_arm.get("grep_only"))

    g = by_arm.get("grep_only", {})
    if not g:
        print("\ngrep_only arm has not finished; stopping at the floor.")
        return

    base = {k: v for k, v in a.items() if k in g}
    tgt = {k: v for k, v in g.items() if k in a}
    shared = sorted(base)
    print(f"\n{'='*72}\ncompared on {len(shared)} rows present in both floor_a and grep_only\n")

    # --- P4: measure the floor in the same session
    if b:
        bshared = [k for k in shared if k in b]
        flips = [k for k in bshared if a[k]["choice"] != b[k]["choice"]]
        acc_a = sum(a[k]["correct"] for k in bshared)
        acc_b = sum(b[k]["correct"] for k in bshared)
        print(f"A/A floor   floor_a {acc_a}/{len(bshared)}  floor_b {acc_b}/{len(bshared)}  "
              f"choice-flips {len(flips)} = {100*len(flips)/len(bshared):.2f}%")
        print(f"            McNemar on the A/A discordants: "
              f"b={sum(1 for k in bshared if a[k]['correct'] and not b[k]['correct'])} "
              f"c={sum(1 for k in bshared if b[k]['correct'] and not a[k]['correct'])} "
              f"p={mcnemar_exact(sum(1 for k in bshared if a[k]['correct'] and not b[k]['correct']), sum(1 for k in bshared if b[k]['correct'] and not a[k]['correct'])):.4f}")
    else:
        flips = []
        print("A/A floor   floor_b has not finished yet")

    # --- rows: accuracy and McNemar
    ca = sum(a[k]["correct"] for k in shared)
    cg = sum(tgt[k]["correct"] for k in shared)
    fixed = [k for k in shared if not a[k]["correct"] and tgt[k]["correct"]]
    broke = [k for k in shared if a[k]["correct"] and not tgt[k]["correct"]]
    p = mcnemar_exact(len(broke), len(fixed))
    print(f"\nrows        floor_a {ca}/{len(shared)} = {100*ca/len(shared):.1f}%   "
          f"grep_only {cg}/{len(shared)} = {100*cg/len(shared):.1f}%   "
          f"delta {100*(cg-ca)/len(shared):+.2f} pts")
    print(f"McNemar     fixed {len(fixed)}  broke {len(broke)}  p={p:.4f}")
    if flips:
        print(f"vs floor    delta/floor = {abs(cg-ca)/max(1,len(flips)):.1f}x  "
              f"(P4 wants >3x; floor is {len(flips)} row(s))")

    # --- per-option ledger: the unit that was actually edited
    print(f"\n{'label':<18}{'n':>5}{'base':>7}{'treat':>7}{'fix':>5}{'brk':>5}{'net':>5}")
    ledger = {}
    for lab in sorted({base[k]["label"] for k in shared}):
        ks = [k for k in shared if base[k]["label"] == lab]
        cb = sum(base[k]["correct"] for k in ks)
        cg2 = sum(tgt[k]["correct"] for k in ks)
        f = sum(1 for k in ks if not base[k]["correct"] and tgt[k]["correct"])
        r = sum(1 for k in ks if base[k]["correct"] and not tgt[k]["correct"])
        ledger[lab] = (len(ks), cb, cg2, f, r)
        print(f"{lab:<18}{len(ks):>5}{cb:>7}{cg2:>7}{f:>5}{r:>5}{cg2-cb:>+5}")

    # --- the three pre-registered pass conditions
    ws = ledger.get("web_search", (0, 0, 0, 0, 0))
    untouched = {l: v for l, v in ledger.items() if l not in (TARGET, "web_search")}
    untouched_net = sum(v[2] - v[1] for v in untouched.values())
    grep_net = ledger.get(TARGET, (0, 0, 0, 0, 0))[2] - ledger.get(TARGET, (0, 0, 0, 0, 0))[1]
    print(f"\n{'='*72}\npre-registered conditions")
    print(f"  1. p < 0.05                        -> p={p:.4f}  {'PASS' if p < 0.05 else 'FAIL'}")
    print(f"  2. web_search loses <= 3 rows      -> {ws[4]} lost  {'PASS' if ws[4] <= 3 else 'FAIL'}")
    print(f"  3. untouched tools not net-negative-> net {untouched_net:+d}  {'PASS' if untouched_net >= 0 else 'FAIL'}")
    print(f"  (context) filesys_grep net {grep_net:+d} on its own option")

    # --- E2: is the effect the clarification, or just different words at length?
    ctl = by_arm.get("grep_control", {})
    if ctl:
        cshared = [k for k in shared if k in ctl]
        cc = sum(ctl[k]["correct"] for k in cshared)
        gc = sum(tgt[k]["correct"] for k in cshared)
        bc = sum(base[k]["correct"] for k in cshared)
        fc = [k for k in cshared if not base[k]["correct"] and ctl[k]["correct"]]
        rc = [k for k in cshared if base[k]["correct"] and not ctl[k]["correct"]]
        print(f"\n{'='*72}\nE2 control: same meaning, different words, same length")
        print(f"            floor_a  {bc}/{len(cshared)}")
        print(f"            control  {cc}/{len(cshared)}  ({100*(cc-bc)/len(cshared):+.2f} pts)  "
              f"fixed {len(fc)} broke {len(rc)} p={mcnemar_exact(len(rc), len(fc)):.4f}")
        print(f"            grep_only {gc}/{len(cshared)}  ({100*(gc-bc)/len(cshared):+.2f} pts)")
        agree = sum(1 for k in cshared if ctl[k]["choice"] == tgt[k]["choice"])
        print(f"            control and grep_only chose identically on {agree}/{len(cshared)} rows")
        if len(fc) + len(rc) == 0:
            print("            control moved nothing: the effect is not 'any rewording'.")
        elif gc - bc > 3 * (len(fc) - len(rc)):
            print("            control moved less than the treatment: the clarification carried it.")
        else:
            print("            control moved comparably: length/novelty may explain the treatment.")

    # --- where the bad rows went
    print(f"\nfilesys_grep rows that grep_only still gets wrong:")
    wrong = [k for k in shared if base[k]["label"] == TARGET and not tgt[k]["correct"]]
    for k in wrong[:12]:
        print(f"  {k}  chose {tgt[k]['choice']:<18} {base[k]['utterance'][:64]}")
    print(f"web_search rows that grep_only breaks:")
    wbroke = [k for k in shared if base[k]["label"] == "web_search" and not tgt[k]["correct"] and base[k]["correct"]]
    for k in wbroke[:12]:
        print(f"  {k}  chose {tgt[k]['choice']:<18} {base[k]['utterance'][:64]}")


if __name__ == "__main__":
    main()
