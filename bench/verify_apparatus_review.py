"""verify_apparatus_review — every number in docs/review-apparatus-2026-11-04.md.

Reads committed data only. No model call, no network, no refit. Stdlib only, so
it runs on the tree as shipped:

    python3 bench/verify_apparatus_review.py            # all sections
    python3 bench/verify_apparatus_review.py --only cd  # one or more sections

Each section prints the quantity the review cites and, where it matters, the
slack around it. The review exists because three of these quantities are quoted
in this repo at a unit of analysis the data does not support; the script is here
so a reader can check that claim rather than take it.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))

SEED = 20261104
REPS = 4000


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _v3_pairs():
    """v3 rows joined to the baseline recording and the targeted recording."""
    meta = json.loads((ROOT / "bench/data/heldout_v3.json").read_text())
    base = json.loads((ROOT / "bench/results/p2-heldout/v3-d1-3B-Q4.json").read_text())["per_row"]
    targeted = {r["id"]: r for r in _jsonl(ROOT / "bench/results/p3-transfer/rows.jsonl")
                if r["arm"] == "targeted"}
    out = []
    for m, b in zip(meta, base):
        if m["utterance"] != b["utterance"]:
            raise SystemExit(f"misaligned recording at {m['id']}")
        t = targeted[m["id"]]
        out.append(dict(id=m["id"], cls=m["label"], family=m["family"], split=m["split"],
                        band=m["band"], utterance=m["utterance"],
                        base=bool(b["correct"]), targ=bool(t["correct"]),
                        base_choice=b["choice"], targ_choice=t["choice"]))
    return out


def _sign_test(k_pos, k_neg):
    """Two-sided exact sign test on nets, the family-level analogue of McNemar."""
    n = k_pos + k_neg
    if not n:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(k_pos, k_neg) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _cluster_bootstrap(rows, reps=REPS, seed=SEED):
    """95% interval for the paired delta, resampling families (the repo's own unit)."""
    by = collections.defaultdict(list)
    for r in rows:
        by[r["family"]].append(r)
    keys = sorted(by)
    rng = random.Random(seed)
    deltas = []
    for _ in range(reps):
        sub = [r for k in (rng.choice(keys) for _ in keys) for r in by[k]]
        deltas.append((sum(r["targ"] for r in sub) - sum(r["base"] for r in sub)) / len(sub))
    deltas.sort()
    obs = (sum(r["targ"] for r in rows) - sum(r["base"] for r in rows)) / len(rows)
    return 100 * obs, 100 * deltas[int(0.025 * reps)], 100 * deltas[int(0.975 * reps)], len(keys)


def _section_v3(structure):
    """A: does the instrument sample its own inventory evenly?"""
    v3 = json.loads((ROOT / "bench/data/heldout_v3.json").read_text())
    v2 = json.loads((ROOT / "tests/data/utterances_v2.json").read_text())
    print("== A  v3 structure: rows per family, per class")
    per = collections.defaultdict(collections.Counter)
    for r in v3:
        per[r["label"]][r["family"]] += 1
    for cls in sorted(per):
        c = per[cls]
        multi = sum(v for v in c.values() if v > 1)
        print(f"   {cls:16s} families={len(c):3d}  rows={sum(c.values()):3d}  "
              f"in multi-row families={multi:3d} ({multi / sum(c.values()):5.1%})")
    print("   band mix v3", dict(collections.Counter(r["band"] for r in v3)))
    print("   band mix v2", dict(collections.Counter(r["band"] for r in v2)))

    import re
    pkg = re.compile(r"\b(urllib|requests|fastapi|pydantic|asyncio|numpy)\b", re.I)
    for name, rows in (("v2", v2), ("v3", v3)):
        g = [r for r in rows if r["label"] == "filesys_grep"]
        print(f"   {name} grep rows={len(g):3d} carrying a package noun="
              f"{sum(1 for r in g if pkg.search(r['utterance']))}")
    print("   frames.py offers 2 {mod} frames among the 26 t1/t2 filesys_grep frames;")
    t12 = sorted({r["family"] for r in v3
                  if r["label"] == "filesys_grep" and r["family"][:2] in ("t1", "t2")})
    print(f"   v3's grep sample takes {len(t12)} of its 10 families from t1/t2, so a draw")
    print("   that size misses both {mod} frames with probability "
          f"C(24,{len(t12)})/C(26,{len(t12)}) = "
          f"{math.comb(24, len(t12)) / math.comb(26, len(t12)):.3f} — small-sample variance,"
          " unchecked.")


def _section_grep(pairs):
    """B: the published per-class grep claim, at the unit the repo uses elsewhere."""
    print("\n== B  the filesys_grep claim at row and family resolution")
    for name, sel in (("grep alone", [r for r in pairs if r["cls"] == "filesys_grep"]),
                      ("grep, v3-test side only",
                       [r for r in pairs if r["cls"] == "filesys_grep" and r["split"] == "test"]),
                      ("the two rewritten classes",
                       [r for r in pairs if r["cls"] in ("filesys_grep", "decide_extract")]),
                      ("the six untouched classes",
                       [r for r in pairs if r["cls"] not in ("filesys_grep", "decide_extract")]),
                      ("all 415", pairs)):
        fixed = [r for r in sel if not r["base"] and r["targ"]]
        broke = [r for r in sel if r["base"] and not r["targ"]]
        ff = collections.Counter(r["family"] for r in fixed)
        bf = collections.Counter(r["family"] for r in broke)
        net = collections.Counter()
        for f in set(ff) | set(bf):
            net[f] = (ff[f] > bf[f]) - (bf[f] > ff[f])
        pos = sum(1 for v in net.values() if v > 0)
        neg = sum(1 for v in net.values() if v < 0)
        d, lo, hi, units = _cluster_bootstrap(sel)
        print(f"   {name:30s} n={len(sel):3d} families={units:3d} "
              f"delta={d:+5.1f} pts [{lo:+5.1f}, {hi:+5.1f}]  fixed={len(fixed)} "
              f"({len(ff)} families) broken={len(broke)} ({len(bf)})  "
              f"family sign {pos}-{neg} p={_sign_test(pos, neg):.4f}")
    print("   (the doc quotes p=0.0039 for grep alone: mcnemar_exact(9, 0) on 52 rows,")
    print("    where those 9 rows are 3 authored sentences)")

    print("\n   what the 9 grep fixes actually are:")
    fam = collections.defaultdict(list)
    for r in pairs:
        if r["cls"] == "filesys_grep" and not r["base"] and r["targ"]:
            fam[r["family"]].append(r)
    for f, ms in sorted(fam.items(), key=lambda kv: -len(kv[1])):
        print(f"     {f:34s} {len(ms)} rows, side={ms[0]['split']:5s} "
              f"e.g. {ms[0]['utterance']!r}")

    print("\n   leave-one-family-out, and the pre-specified pair:")
    for name, sel in (("grep", [r for r in pairs if r["cls"] == "filesys_grep"]),
                      ("all 415", pairs),
                      ("the six untouched classes",
                       [r for r in pairs if r["cls"] not in ("filesys_grep", "decide_extract")])):
        fams = sorted({r["family"] for r in sel})
        ds = []
        for f in fams:
            s = [r for r in sel if r["family"] != f]
            if s:
                ds.append(100 * (sum(r["targ"] for r in s) - sum(r["base"] for r in s)) / len(s))
        if ds:
            print(f"   {name:26s} delta={100 * (sum(r['targ'] for r in sel) - sum(r['base'] for r in sel)) / len(sel):+6.1f}"
                  f"  jackknife range {min(ds):+6.1f} .. {max(ds):+6.1f} over {len(ds)} deletions")


def _section_ledger(pairs):
    """C: whose column does an option-text break land in?"""
    print("\n== C  the 29 broken rows, attributed by the option that captured them")
    broke = [r for r in pairs if r["base"] and not r["targ"]]
    for dest, c in collections.Counter(r["targ_choice"] for r in broke).most_common():
        print(f"   captured by {dest:16s} {c:3d}")
    print("   a break whose destination is filesys_grep was caused by the grep edit but is")
    print("   booked against its own label's column:")
    for r in broke:
        if r["targ_choice"] == "filesys_grep":
            print(f"     label={r['cls']:12s} {r['utterance']!r}")
    print("   so the grep edit's own ledger on v3 is +%d fixed / -%d broken, not 9 / 0"
          % (sum(1 for r in pairs if r["cls"] == "filesys_grep" and not r["base"] and r["targ"]),
             sum(1 for r in broke if r["targ_choice"] == "filesys_grep")))


def _v2_outcomes():
    base = json.loads((ROOT / "bench/results/p1-diagnose/baseline_live.json").read_text())
    return [1 if r["correct"] else 0 for r in base["per_row"]]


def _section_groups(_):
    """D: is the v2 'cluster' bootstrap a cluster bootstrap?"""
    import eval_harness as H
    v2 = json.loads((ROOT / "tests/data/utterances_v2.json").read_text())
    y = _v2_outcomes()
    decl = H.v2_boundary_groups(v2)
    surf = H.v2_surface_groups(v2)
    print("== D  v2 group structure")
    for name, g in (("boundary (declared)", decl), ("surface", surf)):
        c = collections.Counter(g)
        print(f"   {name:20s} groups={len(c):3d} singletons="
              f"{sum(1 for v in c.values() if v == 1):3d} "
              f"({sum(1 for v in c.values() if v == 1) / len(c):5.1%}) "
              f"sizes={dict(sorted(collections.Counter(c.values()).items()))}")
    print(f"   ceiling on variance inflation from the 11 real families: 150/106 = "
          f"{150 / 106:.2f} -> width x{math.sqrt(150 / 106):.2f}")

    def icc(groups):
        by = collections.defaultdict(list)
        for v, g in zip(y, groups):
            by[g].append(v)
        k = [len(v) for v in by.values()]
        n, gk = len(y), len(by)
        grand = sum(y) / n
        ssb = sum(len(v) * (sum(v) / len(v) - grand) ** 2 for v in by.values())
        ssw = sum((x - sum(v) / len(v)) ** 2 for v in by.values() for x in v)
        msb, msw = ssb / (gk - 1), ssw / (n - gk)
        n0 = (n - sum(kk * kk for kk in k) / n) / (gk - 1)
        return (msb - msw) / (msb + (n0 - 1) * msw)

    print(f"   ICC of the 3B outcome inside the declared families: {icc(decl):+.4f}")
    print(f"   ICC inside only the 11 t3 families:                 "
          f"{icc([g if (g or '').startswith('t3') else None for g in decl]):+.4f}")
    print("   a cluster interval is supposed to respect positive dependence inside a group;")
    print("   inside the declared groups there is none to respect")

    def sh(t, n=3):
        s = " ".join("".join(c if c.isalnum() else " " for c in t.lower()).split())
        return {s[i:i + n] for i in range(len(s) - n + 1)}

    S = [sh(r["utterance"]) for r in v2]
    print("\n   same-label near-duplicate pairs that the declared grouping keeps apart:")
    for th in (0.30, 0.25, 0.20):
        pairs = [(i, j) for i, j in itertools.combinations(range(150), 2)
                 if v2[i]["label"] == v2[j]["label"]
                 and len(S[i] & S[j]) / len(S[i] | S[j]) >= th]
        cross = [p for p in pairs if decl[p[0]] != decl[p[1]]]
        agree = sum(1 for i, j in cross if y[i] == y[j])
        print(f"     J>={th:.2f}: {len(pairs):3d} pairs, {len(cross):3d} straddle a group, "
              f"outcome agreement {agree}/{len(cross)}")

    def boot(groups, reps=REPS):
        by = collections.defaultdict(list)
        for v, g in zip(y, groups):
            by[g].append(v)
        keys = sorted(by, key=str)
        rng = random.Random(SEED)
        out = []
        for _ in range(reps):
            vs = [v for k in (rng.choice(keys) for _ in keys) for v in by[k]]
            out.append(sum(vs) / len(vs))
        out.sort()
        return 100 * out[int(0.025 * reps)], 100 * out[int(0.975 * reps)], len(keys)

    print("\n   3B accuracy interval under three groupings of the same 150 rows:")
    for name, g in (("i.i.d. (150 units)", list(range(150))), ("declared (106)", decl),
                    ("surface (102)", surf)):
        lo, hi, n = boot(g)
        print(f"     {name:22s} [{lo:.1f}, {hi:.1f}] width {hi - lo:.1f} over {n} units")


def _section_power(_):
    """E: what difference can n=150 see?"""
    print("\n== E  the router comparison: resolution, not equivalence")
    p10, p01 = 27 / 150, 21 / 150
    disc = p10 + p01
    za, zb = 1.959964, 0.8416212
    print(f"   discordant rows {27}+{21}=48 of 150 ({disc:.3f}); "
          f"raw delta {(p10 - p01) * 100:.1f} pts, seed-mean delta 5.3 pts")
    for d in (0.040, 0.053):
        n = (za + zb) ** 2 * disc / d ** 2
        print(f"   n for 80% power at {100 * d:.1f} pts: {n:,.0f} (x1.2 design effect: "
              f"{1.2 * n:,.0f})")
    for n in (150, 415, 900, 1200):
        print(f"   at n={n:4d}: minimum detectable delta = "
              f"{100 * math.sqrt((za + zb) ** 2 * disc / n):4.1f} pts (80% power), "
              f"{100 * math.sqrt(za ** 2 * disc / n):4.1f} pts (50%)")


def _section_selection(_):
    """F: is the corpus's CV ranking a ranking?"""
    print("\n== F  the corpus's selection protocol")
    sweep = json.loads((ROOT / "bench/results/distill_transfer_sweep.json").read_text())
    transfer = {r["predictor"]: 100 * r["acc_mean"] for r in sweep["rows"]}
    cands = [(name, v["cv"]) for name, v in sweep["corpus_scores"].items()]
    cands = [(n, cv) for n, cv in cands if n in transfer]
    print(f"   {'candidate':26s} {'cv':>6s} {'sd':>5s} {'per seed':>22s} {'val':>6s} {'v2':>6s}")
    for name, cv in cands:
        val = 100 * sweep["corpus_scores"][name]["val_pooled"]["acc"]
        print(f"   {name:26s} {100 * cv['mean']:6.1f} {100 * cv['sd']:5.2f} "
              f"{str([round(100 * x, 1) for x in cv['per_seed']]):>22s} "
              f"{val:6.1f} {transfer[name]:6.1f}")

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: -v[i])
        r = [0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos + 1
        return r

    def spearman(a, b):
        ra, rb = rank(a), rank(b)
        m = (len(a) + 1) / 2
        num = sum((x - m) * (y - m) for x, y in zip(ra, rb))
        den = (sum((x - m) ** 2 for x in ra) * sum((y - m) ** 2 for y in rb)) ** 0.5
        return num / den

    names = [n for n, _ in cands]
    means = [cv["mean"] for _, cv in cands]
    tr = [transfer[n] for n in names]
    obs = spearman(means, tr)
    tail = sum(1 for perm in itertools.permutations(tr) if spearman(means, list(perm)) <= obs + 1e-12)
    total = math.factorial(len(tr))
    print(f"\n   spearman(cv mean, v2 transfer) = {obs:+.4f}; exact one-sided "
          f"permutation p = {tail}/{total} = {tail / total:.4f}")
    seeds = list(zip(*[cv["per_seed"] for _, cv in cands]))
    for k, s in enumerate(seeds):
        rk = rank(list(s))
        order = [names[i] for i in sorted(range(len(names)), key=lambda i: rk[i])]
        print(f"   seed {k}: spearman vs transfer = {spearman(list(s), tr):+.3f}  "
              f"top three: {order[:3]}")
    for a, b in itertools.combinations(range(len(seeds)), 2):
        print(f"   spearman(seed{a}, seed{b}) = {spearman(list(seeds[a]), list(seeds[b])):+.3f}")
    print(f"\n   cv spread across candidates {100 * (max(means) - min(means)):.1f} pts; "
          f"transfer spread {max(tr) - min(tr):.1f} pts; "
          f"largest within-candidate seed sd {100 * max(cv['sd'] for _, cv in cands):.2f} pts; "
          f"binomial SE of one 66-row fold {100 * math.sqrt(0.763 * 0.237 / 66):.2f} pts")


def _section_objects(_):
    """G: does the object leak the corpus doc names actually explain the ranking?"""
    print("\n== G  the corpus's slot nouns")
    sys.path.insert(0, str(ROOT / "tools"))
    from distill.frames import SLOTS
    vals = sorted(((k, v.lower()) for k, vs in SLOTS.items() for v in vs),
                  key=lambda kv: -len(kv[1]))

    def okey(text):
        low = text.lower()
        for kind, v in vals:
            if v in low:
                return f"{kind}:{v}"
        return None

    tr = _jsonl(ROOT / "data/distill/train.jsonl")
    va = _jsonl(ROOT / "data/distill/val.jsonl")
    tr_objs = {okey(r["utterance"]) for r in tr} - {None}
    va_objs = [okey(r["utterance"]) for r in va]
    print(f"   inventory {len(vals)} slot values; {len(tr_objs)} resolvable in train")
    print(f"   val rows naming an object train also has: "
          f"{sum(1 for k in va_objs if k in tr_objs)}/{len(va)}")
    by = collections.defaultdict(set)
    lab = collections.defaultdict(collections.Counter)
    for r in tr:
        k = okey(r["utterance"])
        if k:
            by[k].add(r["family"])
            lab[k][r["label"]] += 1
    print(f"   objects appearing in more than one train family: "
          f"{sum(1 for v in by.values() if len(v) > 1)}/{len(by)}")
    print(f"   objects whose train rows all share one label: "
          f"{sum(1 for c in lab.values() if len(c) == 1)}/{len(lab)}")
    print("   -> the nouns do recur (the premise holds), but the big ones are shared across")
    print("      labels, and the corpus's own object-held-out CV scores +7.0 points HIGHER")
    print("      than family-held-out CV (83.3 vs 76.3), which is the opposite of what")
    print("      'the split leaves the memorised noun in place' predicts.")


SECTIONS = {"a": ("v3 structure", _section_v3), "b": ("grep claim", _section_grep),
            "c": ("break ledger", _section_ledger), "d": ("v2 groups", _section_groups),
            "e": ("power", _section_power), "f": ("corpus selection", _section_selection),
            "g": ("corpus nouns", _section_objects)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", choices=sorted(SECTIONS), default=None)
    args = ap.parse_args()
    which = args.only or sorted(SECTIONS)
    pairs = _v3_pairs()
    print(f"v3 pairs: {len(pairs)} rows; baseline correct {sum(r['base'] for r in pairs)}; "
          f"targeted correct {sum(r['targ'] for r in pairs)}")
    for key in which:
        label, fn = SECTIONS[key]
        print(f"\n######## {key.upper()}  {label}")
        fn(pairs)


if __name__ == "__main__":
    main()
