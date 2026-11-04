#!/usr/bin/env python3
"""verify_claims — every headline number in this tree, recomputed from its artefact.

Three published claims were found wrong in one night, each by someone recomputing
instead of reading: an accuracy that was an oracle, an accuracy measured on a
different task, and an improvement that failed out of sample. This script exists so
the next one is found by a test rather than by a reader.

What it does
------------
For each claim it takes the figure as *published* (the doc or the artefact field a
reader would quote), derives the same quantity from the rows underneath it, and
prints both. Nothing here reads an `accuracy` field and calls it verified: the
provenance column says where the number came from — `per_row` sums, joins of two
recording arms, a refit from the committed corpus, or a re-derivation of the whole
sweep from `tests/data/utterances_v2.json` plus the archived 3B answers.

Cost and isolation: no network, no model call, no spend. The 3B is read out of
`bench/results/p1-diagnose/baseline_live.json` and
`bench/results/p2-heldout/v3-d1-3B-Q4.json`, exactly as the probes that produced
these numbers did. The only compute is `bench/router_policy.run()` — the same
estimator, re-run from raw rows — which is why a full pass takes minutes rather
than seconds.

Exit code is non-zero if any claim FAILS. Two other statuses exist:

* `CAVEAT` — the arithmetic is right but the claim's scope is not what its wording
  says (a slice, a coverage, an arm that was never run). Left for a human to word.
* `SKIP` — the claim was not run (`--skip-router`).

Tolerances are stated per row: a percentage published to one decimal is compared at
one decimal, half a unit in the last place either way, so a display rounding
(66.449 → "66.5") is not reported as a defect. Everything else is compared exactly
against the float the artefact stores.

Usage::

    python3 bench/verify_claims.py                  # every claim, ~15 min
    python3 bench/verify_claims.py --skip-router    # everything except claim 2
    python3 bench/verify_claims.py --seeds 1        # smoke: claim 2 on one seed only
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "service"), str(ROOT / "bench"), str(ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

D1_V2 = ROOT / "bench/results/p1-diagnose/baseline_live.json"
D1_V3 = ROOT / "bench/results/p2-heldout/v3-d1-3B-Q4.json"
ROUTER = ROOT / "bench/results/router-policy-2026-11-04.json"
P3_SUMMARY = ROOT / "bench/results/p3-transfer/summary.json"
P3_ROWS = ROOT / "bench/results/p3-transfer/rows.jsonl"
DISTILL = ROOT / "bench/results/distill_transfer.json"
V3 = ROOT / "bench/data/heldout_v3.json"
V3_FAMILIES = ROOT / "bench/data/heldout_v3_families.json"
DISTILL_DIR = ROOT / "data/distill"

TARGET_LABELS = ("decide_extract", "filesys_grep")

# docs/rewrite-transfer-2026-11-04.md §5, transcribed as (n, a, b, fixed, broke, p).
# Transcribing a table is the only way to check it: the doc is the artefact that
# would be quoted, and nothing in the tree generates it.
DOC_PER_CLASS = {
    "filesys_grep": (52, 36, 45, 9, 0, 0.0039),
    "decide_extract": (51, 20, 25, 7, 2, 0.1797),
    "reddit_digest": (52, 42, 32, 0, 10, 0.0020),
    "web_search": (52, 49, 41, 0, 8, 0.0078),
    "filesys_read": (52, 43, 40, 0, 3, 0.2500),
    "filesys_glob": (52, 44, 42, 0, 2, 0.5000),
    "web_fetch": (52, 49, 47, 0, 2, 0.5000),
    "decide_pick": (52, 51, 49, 0, 2, 0.5000),
}
DOC_AGG_TARGETS = (103, 56, 70, 16, 2, 0.0013)
DOC_AGG_OTHERS = (312, 278, 251, 0, 27, 1.0e-8)   # published "< 0.0001"

# --------------------------------------------------------------------------- #
# what is published, and where
# --------------------------------------------------------------------------- #

# Figures as a reader finds them. The artefact field that carries the same number
# is named where one exists, so a drift between the doc and its artefact is also a
# failure rather than a second source of truth.
PUB = {
    # 1. d1_3B on utterances_v2 — README "archived per-row run | 74.0% | 66.5–81.3"
    "d1_acc": 74.0, "d1_ci": (66.5, 81.3), "d1_n": 150,
    # 2. the deployable router — README, router-policy row of the index table
    "router_acc": 79.3, "router_ci": (73.6, 84.8),
    "oracle_acc": 84.5, "lr_acc": 67.1, "b3_acc": 74.0,
    "router_vs_3b": {"delta": 5.3, "ci": (-1.4, 12.3), "separable": False},
    "router_vs_lr": {"delta": 12.3, "ci": (5.8, 18.8), "separable": True},
    # 3. the rewrite fails out of sample — README + docs/rewrite-transfer §4
    "rewrite": {"a": 80.5, "b": 77.3, "fixed": 16, "broke": 29, "p": 0.0725,
                "n": 415, "delta": -3.1, "delta_ci": (-8.5, 1.9), "units": 133},
    # 4. the grep line alone transfers — README + docs/rewrite-transfer §5, §9
    "grep": {"n": 52, "a": 36, "b": 45, "fixed": 9, "broke": 0, "p": 0.0039,
             "delta": 17.3, "a_pct": 69.2, "b_pct": 86.5},
    # 5. the distill classifier — README index + bench/results/distill_transfer.json
    "distill_acc": 68.7, "distill_ci": (59.6, 77.0),
    "distill_vs_rules": {"p": 0.040, "fixed": 16, "broke": 31},
    "distill_vs_3b": {"p": 0.38, "fixed": 28, "broke": 36},
    "rules_acc": 78.7,
    # 6. heldout_v3 — bench/data/heldout_v3_NOTES.md + the families artefact
    "v3_rows": 415, "v3_families": 133, "v3_crossing": 0,
    # 7. the overlap joins — heldout_v3_families.json distill_overlap
    "join_test_rows": 160, "join_trainonly_test_rows": 145,
}

METHOD_CLUSTER = "cluster bootstrap over groups (percentile, seed 0, 2000 reps)"
METHOD_CLUSTER_FAM = ("cluster bootstrap over families (percentile, seed 20261104, "
                      "4000 reps)")


# --------------------------------------------------------------------------- #
# the recomputation primitives — implemented here, not imported from the probe
# --------------------------------------------------------------------------- #

def mcnemar(fixed: int, broke: int) -> float:
    """Two-sided exact McNemar: 2 * P(X <= min) for Binomial(n, 1/2), capped at 1."""
    n = fixed + broke
    if n <= 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(fixed, broke) + 1)) / 2 ** n
    return min(1.0, 2.0 * tail)


def cluster_bootstrap(outcomes: list[float], units: list[str], *, reps: int = 2000,
                      seed: int = 0, level: float = 0.95,
                      index: str = "harness") -> tuple[float, float]:
    """Percentile interval for a mean, resampling whole units (families/groups).

    `index` selects between the two percentile conventions the tree actually ships:
    `harness` is `eval_harness.cluster_bootstrap_ci` (interpolated index over
    reps-1), `rw` is `rewrite_transfer.cluster_bootstrap` (`int(p * reps)`). They
    differ by one order statistic; a claim is checked under the convention of the
    artefact that published it.
    """
    by_unit: dict[str, list[float]] = {}
    for o, u in zip(outcomes, units):
        by_unit.setdefault(u, []).append(o)
    keys = sorted(by_unit)
    if not keys:
        return (0.0, 1.0)
    rng = random.Random(seed)
    means = []
    for _ in range(reps):
        pick = [keys[rng.randrange(len(keys))] for _ in keys]
        total = hits = 0.0
        for u in pick:
            vals = by_unit[u]
            hits += sum(vals)
            total += len(vals)
        means.append(hits / total if total else 0.0)
    means.sort()
    if index == "rw":
        return (means[int((1 - level) / 2 * reps)], means[int((1 + level) / 2 * reps)])
    lo = means[int(math.floor((1 - level) / 2 * (len(means) - 1)))]
    hi = means[int(math.ceil((1 + level) / 2 * (len(means) - 1)))]
    return (lo, hi)


def paired(a: dict[int, bool], b: dict[int, bool], ids: list[int]) -> dict:
    """`a` is baseline, `b` is the intervention: fixed = b right & a wrong.

    This is `rewrite_transfer`'s convention (a row the rewrite fixed). The distill
    probe uses the mirror image with the candidate first, so claim 5 counts inline
    rather than borrowing this.
    """
    fixed = sum(1 for i in ids if b[i] and not a[i])
    broke = sum(1 for i in ids if a[i] and not b[i])
    hits_a = sum(1 for i in ids if a[i])
    hits_b = sum(1 for i in ids if b[i])
    return {"n": len(ids), "a": hits_a, "b": hits_b, "fixed": fixed, "broke": broke,
            "p": mcnemar(fixed, broke)}


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

@dataclass
class Check:
    cid: str
    claim: str
    published: str
    computed: str
    method: str
    status: str          # PASS | FAIL | CAVEAT | SKIP
    note: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, *args, **kwargs) -> None:
        self.checks.append(Check(*args, **kwargs))

    def verdict(self, cid: str, claim: str, published: str, computed: str,
                method: str, ok: bool | None, note: str = "",
                status: str | None = None) -> None:
        if status is None:
            status = "PASS" if ok else "FAIL"
        self.add(cid, claim, published, computed, method, status, note)

    def render(self) -> str:
        head = (f"{'':<6}{'claim':<43}{'published':<26}{'recomputed':<33}"
                f"{'status':<9}method")
        lines = [head, "-" * len(head)]
        for c in self.checks:
            lines.append(f"{c.cid:<6}{c.claim[:42]:<43}{c.published[:25]:<26}"
                         f"{c.computed[:32]:<33}{c.status:<9}{c.method}")
            if c.note:
                lines.append(f"    ↳ {c.note}")
        return "\n".join(lines)


def _pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def _close(pub: float, got: float, places: int) -> bool:
    """Equal at the precision the claim is published to.

    `got` is a percentage. Artefacts store the *fraction* to four decimals and
    documents round that again, so a bound of 0.6644737 becomes 0.6645, is printed
    66.45, and is displayed "66.5". The comparison walks that chain — round to the
    stored precision, then to the displayed one — and allows half a unit in the last
    displayed place. That is a rounding tie, not a licence to be approximate: a
    figure wrong by more than half its last digit still fails.
    """
    stored_pct = round(got / 100, 4) * 100
    return abs(pub - stored_pct) <= 0.5 * 10 ** -places + 1e-9


def _close_raw(pub: float, got: float, places: int) -> bool:
    """The same tolerance for a value already in its final units — a p, a ratio."""
    return abs(pub - got) <= 0.5 * 10 ** -places + 1e-9


# --------------------------------------------------------------------------- #
# claim 1 — d1_3B on utterances_v2
# --------------------------------------------------------------------------- #

def claim_1(rep: Report, ds) -> None:
    rec = json.loads(D1_V2.read_text())
    rows = rec["per_row"]
    outcomes = [1.0 if r["correct"] else 0.0 for r in rows]
    hits, n = int(sum(outcomes)), len(rows)
    lo, hi = cluster_bootstrap(outcomes, ds.groups)

    rep.verdict("1.1", "d1_3B accuracy on utterances_v2",
                f"{PUB['d1_acc']:.1f}%", f"{hits}/{n} = {_pct(hits / n)}",
                "sum of per_row.correct", _close(PUB["d1_acc"], 100 * hits / n, 1))
    rep.verdict("1.2", "d1_3B 95% CI",
                f"[{PUB['d1_ci'][0]:.1f}, {PUB['d1_ci'][1]:.1f}]",
                f"[{100 * lo:.2f}, {100 * hi:.2f}]", METHOD_CLUSTER,
                _close(PUB["d1_ci"][0], 100 * lo, 1) and _close(PUB["d1_ci"][1], 100 * hi, 1))
    rep.verdict("1.3", "d1_3B rows scored", str(PUB["d1_n"]), f"{n} rows scored",
                "len(per_row)", n == PUB["d1_n"])
    rep.verdict("1.4", "the artefact's own accuracy field agrees with its rows",
                f"{rec['accuracy']:.4f}", f"{hits / n:.4f}", "per_row vs field",
                abs(rec["accuracy"] - hits / n) < 5e-5)

    # the interval is a cluster bootstrap, not Wilson: state what Wilson would say,
    # because the two are not interchangeable and a reader cannot tell from the text
    p = hits / n
    z = 1.959963984540054
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    wl, wh = 100 * (c - h), 100 * (c + h)
    rep.verdict("1.5", "the published interval is the stated method, not Wilson",
                f"[{PUB['d1_ci'][0]:.1f}, {PUB['d1_ci'][1]:.1f}]",
                f"Wilson would give [{wl:.1f}, {wh:.1f}]",
                "cluster bootstrap reproduces; row-Wilson does not",
                not _close(PUB["d1_ci"][1], wh, 1),
                f"row-Wilson puts the upper bound {PUB['d1_ci'][1] - wh:.1f} points "
                "lower, so the method statement is load-bearing, not decoration")


# --------------------------------------------------------------------------- #
# claim 2 — the deployable router
# --------------------------------------------------------------------------- #

def claim_2(rep: Report, ds, seeds: tuple[int, ...]) -> None:
    import router_policy as RP  # noqa: PLC0415 — offline; the 3B comes from the archive

    three_b = RP.ThreeB().align(ds)
    sweep = RP.run(ds, three_b, split="group", k=5, seeds=seeds, k_inner=3, n_boot=2000)
    res = sweep["results"]
    art = json.loads(ROUTER.read_text())

    lead = "" if len(seeds) == 5 else f" (only {len(seeds)} seeds — not comparable)"

    def _status(ok: bool) -> str:
        return "CAVEAT" if lead else ("PASS" if ok else "FAIL")

    def acc_row(cid: str, name: str, pub: float, label: str) -> None:
        got = res[name]["acc"]
        rep.verdict(cid, f"{label} accuracy", f"{pub:.1f}%",
                    f"{got * 100:.2f}%", f"re-derived: {name}" + lead,
                    _close(pub, 100 * got, 1), status=_status(_close(pub, 100 * got, 1)))

    acc_row("2.1", "oracle[per-class]", PUB["oracle_acc"], "oracle[per-class]")
    acc_row("2.2", "router[per-class, A]", PUB["router_acc"], "router[per-class, A]")
    acc_row("2.3", "lr", PUB["lr_acc"], "lr")
    acc_row("2.4", "3b", PUB["b3_acc"], "3b")

    lo, hi = res["router[per-class, A]"]["ci95"]
    ci_ok = (_close(PUB["router_ci"][0], 100 * lo, 1)
             and _close(PUB["router_ci"][1], 100 * hi, 1))
    rep.verdict("2.5", "router 95% CI",
                f"[{PUB['router_ci'][0]:.1f}, {PUB['router_ci'][1]:.1f}]",
                f"[{100 * lo:.2f}, {100 * hi:.2f}]", METHOD_CLUSTER + lead,
                ci_ok, status=_status(ci_ok))

    rep.verdict("2.6", "router n", "150", f"{res['router[per-class, A]']['n']}",
                "re-derived", res["router[per-class, A]"]["n"] == 150)

    # paired comparisons, from the per-row outcomes of the re-derived sweep
    def pair_row(cid: str, a: str, b: str, pub: dict, label: str) -> None:
        oa, ob = res[a]["outcomes"], res[b]["outcomes"]
        diff = [x - y for x, y in zip(oa, ob)]
        delta = sum(diff) / len(diff)
        dlo, dhi = cluster_bootstrap(diff, ds.groups)
        fixed = sum(1 for x, y in zip(oa, ob) if x > y)
        broke = sum(1 for x, y in zip(oa, ob) if y > x)
        pval = mcnemar(fixed, broke)
        separable = not (dlo <= 0.0 <= dhi)
        stored = art["pairs"][f"{a} vs {b}"]
        rep.verdict(f"{cid}a", f"{label}: paired delta",
                    f"{pub['delta']:+.1f} pts", f"{100 * delta:+.2f} pts",
                    "re-derived per-row difference",
                    _close(pub["delta"], 100 * delta, 1),
                    status=_status(_close(pub["delta"], 100 * delta, 1)))
        rep.verdict(f"{cid}b", f"{label}: delta 95% CI",
                    f"[{pub['ci'][0]:+.1f}, {pub['ci'][1]:+.1f}]",
                    f"[{100 * dlo:+.2f}, {100 * dhi:+.2f}]", METHOD_CLUSTER + lead,
                    _close(pub["ci"][0], 100 * dlo, 1)
                    and _close(pub["ci"][1], 100 * dhi, 1),
                    status=_status(_close(pub["ci"][0], 100 * dlo, 1)
                                   and _close(pub["ci"][1], 100 * dhi, 1)))
        rep.verdict(f"{cid}c", f"{label}: A-only / B-only, exact McNemar",
                    f"{stored['a_only']} / {stored['b_only']}, p={stored['p']:.4f}",
                    f"{fixed} / {broke}, p={pval:.4f}",
                    "elementwise over the same 150 rows",
                    (fixed, broke) == (stored["a_only"], stored["b_only"])
                    and abs(pval - stored["p"]) < 5e-5)
        rep.verdict(f"{cid}d", f"{label}: separable (CI excludes 0)",
                    "not separable" if not pub["separable"] else "separable",
                    f"{'separable' if separable else 'not separable'} "
                    f"(p={pval:.4f})",
                    "CI contains 0 ⇔ not separable",
                    separable == pub["separable"],
                    status=_status(separable == pub["separable"]))

    pair_row("2.7", "router[per-class, A]", "3b", PUB["router_vs_3b"], "router vs 3B")
    pair_row("2.8", "router[per-class, A]", "lr", PUB["router_vs_lr"], "router vs lr")

    # the artefact's stored fields, against the same re-derivation
    for cid, name, key in (("2.9", "router[per-class, A]", "acc"),
                           ("2.10", "oracle[per-class]", "acc"),
                           ("2.11", "lr", "acc"),
                           ("2.12", "3b", "acc")):
        stored = art["results"][name][key]
        got = res[name][key]
        rep.verdict(cid, f"artefact field results[{name!r}].acc",
                    f"{stored:.4f}", f"{got:.4f}", "re-derived", abs(stored - got) < 5e-5)
    for cid, name in (("2.13", "router[per-class, A]"), ("2.14", "oracle[per-class]")):
        stored = art["results"][name]["ci95"]
        got = res[name]["ci95"]
        rep.verdict(cid, f"artefact field results[{name!r}].ci95",
                    f"[{stored[0]:.4f}, {stored[1]:.4f}]",
                    f"[{got[0]:.4f}, {got[1]:.4f}]", METHOD_CLUSTER,
                    max(abs(stored[0] - got[0]), abs(stored[1] - got[1])) < 5e-5)


# --------------------------------------------------------------------------- #
# claims 3 and 4 — the rewrite on heldout_v3
# --------------------------------------------------------------------------- #

def _v3() -> tuple[list[dict], dict[int, bool], dict[int, bool]]:
    rows = json.loads(V3.read_text())
    for i, r in enumerate(rows):
        r["idx"] = i
    base_rows = json.loads(D1_V3.read_text())["per_row"]
    if len(base_rows) != len(rows):
        raise SystemExit(f"{D1_V3.name}: {len(base_rows)} rows vs {len(rows)}")
    for i, (b, r) in enumerate(zip(base_rows, rows)):
        if b["utterance"] != r["utterance"]:
            raise SystemExit(f"{D1_V3.name}: row {i} is not this dataset")
    a = {i: bool(b["correct"]) for i, b in enumerate(base_rows)}
    tgt = {}
    for line in P3_ROWS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["arm"] == "targeted":
            tgt[rec["idx"]] = bool(rec["correct"])
    return rows, a, tgt


def claim_3(rep: Report, rows: list[dict], a: dict[int, bool], b: dict[int, bool]) -> None:
    pub = PUB["rewrite"]
    ids = [r["idx"] for r in rows]
    comp = paired(a, b, ids)
    rep.verdict("3.1", "v3, all 415 rows: baseline accuracy",
                f"{pub['a']:.1f}%", f"{comp['a']}/{comp['n']} = {_pct(comp['a'] / comp['n'])}",
                "sum over p2 archive per_row", _close(pub["a"], 100 * comp["a"] / comp["n"], 1))
    rep.verdict("3.2", "v3, all 415 rows: targeted accuracy",
                f"{pub['b']:.1f}%", f"{comp['b']}/{comp['n']} = {_pct(comp['b'] / comp['n'])}",
                "sum over rows.jsonl arm=targeted", _close(pub["b"], 100 * comp["b"] / comp["n"], 1))
    rep.verdict("3.3", "v3 fixed / broken rows",
                f"{pub['fixed']} / {pub['broke']}", f"{comp['fixed']} / {comp['broke']}",
                "join on idx, both arms", (comp["fixed"], comp["broke"]) == (pub["fixed"], pub["broke"]))
    rep.verdict("3.4", "v3 exact McNemar p", f"{pub['p']:.4f}", f"{comp['p']:.4f}",
                "exact binomial on discordant pairs",
                _close_raw(pub["p"], comp["p"], 4))
    rep.verdict("3.5", "v3 rows paired", str(pub["n"]), f"{comp['n']}",
                "ids present in both arms", comp["n"] == pub["n"])

    diff = [1.0 if b[i] else 0.0 for i in ids]
    base = [1.0 if a[i] else 0.0 for i in ids]
    d = [x - y for x, y in zip(diff, base)]
    units = [rows[i]["family"] for i in ids]
    lo, hi = cluster_bootstrap(d, units, reps=4000, seed=20261104, index="rw")
    delta = sum(d) / len(d)
    rep.verdict("3.6", "v3 paired delta",
                f"{pub['delta']:+.1f} pts", f"{100 * delta:+.2f} pts",
                METHOD_CLUSTER_FAM, _close(pub["delta"], 100 * delta, 1))
    rep.verdict("3.7", "v3 paired delta 95% CI",
                f"[{pub['delta_ci'][0]:+.1f}, {pub['delta_ci'][1]:+.1f}]",
                f"[{100 * lo:+.2f}, {100 * hi:+.2f}]", METHOD_CLUSTER_FAM,
                _close(pub["delta_ci"][0], 100 * lo, 1)
                and _close(pub["delta_ci"][1], 100 * hi, 1))
    rep.verdict("3.8", "v3 resampling units (families)",
                str(pub["units"]), str(len(set(units))), "distinct family ids",
                len(set(units)) == pub["units"])

    summ = json.loads(P3_SUMMARY.read_text())["v3_all"]
    same = ((summ["n"], summ["a"], summ["b"], summ["fixed"], summ["broke"])
            == (comp["n"], comp["a"], comp["b"], comp["fixed"], comp["broke"]))
    rep.verdict("3.9", "p3 summary's own v3_all fields match its rows",
                f"{summ['a']}/{summ['b']} {summ['fixed']}/{summ['broke']} "
                f"p={summ['p']:.4f}",
                f"n={comp['n']} {comp['a']}/{comp['b']} "
                f"{comp['fixed']}/{comp['broke']} p={comp['p']:.4f}",
                "summary fields vs recomputed",
                same and abs(summ["p"] - comp["p"]) < 5e-5)

    # the claim's *stated* source is rows.jsonl alone, two arms joined on idx. That
    # join is not the pairing the figure comes from: rows.jsonl carries only 233
    # baseline rows (the whole test side, 26 of 208 train rows), so the 415-row
    # pairing needs the p2 archive as its baseline arm. Say so with the number the
    # wrong join actually produces.
    only_base, only_tgt = {}, {}
    for line in P3_ROWS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            (only_tgt if rec["arm"] == "targeted" else only_base)[rec["idx"]] = rec
    shared = sorted(set(only_base) & set(only_tgt))
    if len(shared) != len(ids):
        c = paired({i: bool(only_base[i]["correct"]) for i in shared},
                   {i: bool(only_tgt[i]["correct"]) for i in shared}, shared)
        rep.verdict("3.10", "claim 3's stated source (rows.jsonl, both arms) is the "
                    "pairing it was computed from",
                    "415 rows (rows.jsonl)",
                    f"n={c['n']} {_pct(c['a'] / c['n'])}→{_pct(c['b'] / c['n'])} "
                    f"{c['fixed']}/{c['broke']} p={c['p']:.4f}",
                    "join rows.jsonl on idx alone", None, status="CAVEAT",
                    note=("the 415-row figure needs "
                          "bench/results/p2-heldout/v3-d1-3B-Q4.json as the baseline "
                          "arm (as the doc's §2 table says); rows.jsonl's own "
                          "baseline arm is 233 rows, and joining it gives a "
                          "different, non-significant number"))


def claim_4(rep: Report, rows: list[dict], a: dict[int, bool], b: dict[int, bool]) -> None:
    pub = PUB["grep"]
    lab = {r["idx"]: r["label"] for r in rows}
    split = {r["idx"]: r["split"] for r in rows}
    ids = [i for i in sorted(a) if lab[i] == "filesys_grep"]
    comp = paired(a, b, ids)
    rep.verdict("4.1", "filesys_grep rows on v3", str(pub["n"]), f"{comp['n']}",
                "label filter over v3", comp["n"] == pub["n"])
    rep.verdict("4.2", "filesys_grep baseline accuracy",
                f"{pub['a_pct']:.1f}% ({pub['a']})",
                f"{comp['a']}/{comp['n']} = {_pct(comp['a'] / comp['n'])}",
                "sum over p2 archive per_row",
                _close(pub["a_pct"], 100 * comp["a"] / comp["n"], 1) and comp["a"] == pub["a"])
    rep.verdict("4.3", "filesys_grep targeted accuracy",
                f"{pub['b_pct']:.1f}% ({pub['b']})",
                f"{comp['b']}/{comp['n']} = {_pct(comp['b'] / comp['n'])}",
                "sum over rows.jsonl arm=targeted",
                _close(pub["b_pct"], 100 * comp["b"] / comp["n"], 1) and comp["b"] == pub["b"])
    rep.verdict("4.4", "filesys_grep fixed / broken",
                f"{pub['fixed']} / {pub['broke']}", f"{comp['fixed']} / {comp['broke']}",
                "join on idx, both arms",
                (comp["fixed"], comp["broke"]) == (pub["fixed"], pub["broke"]))
    rep.verdict("4.5", "filesys_grep exact McNemar p", f"{pub['p']:.4f}", f"{comp['p']:.4f}",
                "exact binomial on discordant pairs",
                _close_raw(pub["p"], comp["p"], 4))
    rep.verdict("4.6", "filesys_grep delta",
                f"{pub['delta']:+.1f} pts", f"{100 * (comp['b'] - comp['a']) / comp['n']:+.2f} pts",
                "difference of the two sums",
                _close(pub["delta"], 100 * (comp["b"] - comp["a"]) / comp["n"], 1))

    # the same slice, cut by the one partition the artefact calls held out
    for cid, side in (("4.7", "test"), ("4.8", "train")):
        sub = [i for i in ids if split[i] == side]
        c = paired(a, b, sub)
        rep.verdict(cid, f"filesys_grep, v3 {side} side only",
                    "held out (p=0.0039)" if side == "test" else "—",
                    f"n={c['n']} {c['a']}→{c['b']} {_pct(c['a'] / c['n'])}→"
                    f"{_pct(c['b'] / c['n'])} {c['fixed']}/{c['broke']} p={c['p']:.4f}",
                    "same arithmetic, split filter", None, status="CAVEAT",
                    note=("the published figure is the 52-row slice, which is 26 test "
                          "rows + 26 train rows; on the test side alone it is "
                          f"{100 * (c['b'] - c['a']) / c['n']:+.1f} points and not "
                          "significant" if side == "test" else
                          "the train half of the 52-row slice, for contrast"))

    # "the grep line alone transfers" — was a grep-only arm ever run?
    from ti_bench.prompts import OP_DESCRIPTIONS  # noqa: PLC0415
    from routing_diagnose import TARGETED  # noqa: PLC0415
    changed = sorted(k for k in TARGETED if TARGETED[k] != OP_DESCRIPTIONS[k])
    arms = sorted({json.loads(line)["arm"] for line in
                   P3_ROWS.read_text(encoding="utf-8").splitlines() if line.strip()})
    rep.verdict("4.9", "'the grep line alone transfers' — is that arm in the artefact?",
                "grep line alone", f"arms={arms}, descriptions changed={changed}",
                "both arms carry all 8 options; no single-line arm exists",
                None, status="CAVEAT",
                note=("no recording isolates the grep edit: every row was asked with "
                      "both changed descriptions, and the doc's own §9 says the "
                      "grep-only arm 'is a different experiment from this one and is "
                      "not measured here'. The 9/0 split is evidence about grep rows "
                      "under the full rewrite, not about the grep line alone"))

    # per class, so a single class cannot carry the headline
    for pos, lab_name in enumerate(sorted({lab[i] for i in a})):
        sub = [i for i in sorted(a) if lab[i] == lab_name]
        c = paired(a, b, sub)
        marked = " (rewritten)" if lab_name in TARGET_LABELS else ""
        doc = DOC_PER_CLASS[lab_name]
        rep.verdict(f"4.{10 + pos}", f"per class {lab_name}{marked}",
                    f"{doc[0]}: {doc[1]}→{doc[2]} {doc[3]}/{doc[4]} p={doc[5]:g}",
                    f"n={c['n']} {c['a']}→{c['b']} {c['fixed']}/{c['broke']} "
                    f"p={c['p']:.4f}", "per-class slice of all 415 rows",
                    (c["n"], c["a"], c["b"], c["fixed"], c["broke"]) == doc[:5]
                    and _close_raw(doc[5], c["p"], 4))

    # the two aggregates below the per-class table, same source
    for cid, name, keep, doc in (
            ("4.18", "v3, the two rewritten classes", True, DOC_AGG_TARGETS),
            ("4.19", "v3, the six untouched classes", False, DOC_AGG_OTHERS)):
        sub = [i for i in sorted(a) if (lab[i] in TARGET_LABELS) == keep]
        c = paired(a, b, sub)
        rep.verdict(cid, name,
                    f"{doc[0]}: {doc[1]}→{doc[2]} {doc[3]}/{doc[4]} p={doc[5]:.4g}",
                    f"{c['n']} rows, {c['a']}→{c['b']}, {c['fixed']}/{c['broke']}, "
                    f"p={c['p']:.4g}", "aggregate over class slices",
                    (c["n"], c["a"], c["b"], c["fixed"], c["broke"]) == doc[:5]
                    and (c["p"] < 1e-4 if doc[5] <= 1e-4 else _close_raw(doc[5], c["p"], 4)))


# --------------------------------------------------------------------------- #
# claim 5 — the distill classifier
# --------------------------------------------------------------------------- #

def claim_5(rep: Report, ds) -> None:
    from distill import clf  # noqa: PLC0415 — the committed corpus is the training set
    import local_clf as lc  # noqa: PLC0415

    art = json.loads(DISTILL.read_text())
    cfg = {"learner": "ridge", "mode": "both", "lam": 1.0}   # C6-ridge-both, shipped
    model = clf.DistillClassifier.from_config(cfg).fit(clf.load_split("train"))
    labels = [e.label for e in ds.examples]
    c6 = [1.0 if model.predict(e.text)[0] == lab else 0.0 for e, lab in zip(ds.examples, labels)]
    rules = [1.0 if lc.rules_predict(e.text)[0] == lab else 0.0
             for e, lab in zip(ds.examples, labels)]
    d1 = [1.0 if r["correct"] else 0.0
          for r in json.loads(D1_V2.read_text())["per_row"]]

    hits, n = int(sum(c6)), len(c6)
    lo, hi = cluster_bootstrap(c6, ds.groups)
    rep.verdict("5.1", "distill (C6-ridge-both) accuracy on v2", f"{PUB['distill_acc']:.1f}%",
                f"{hits}/{n} = {_pct(hits / n)}",
                "refit on data/distill/train.jsonl, scored on v2", _close(PUB["distill_acc"], 100 * hits / n, 1))
    rep.verdict("5.2", "distill 95% CI", f"[{PUB['distill_ci'][0]:.1f}, {PUB['distill_ci'][1]:.1f}]",
                f"[{100 * lo:.2f}, {100 * hi:.2f}]", METHOD_CLUSTER,
                _close(PUB["distill_ci"][0], 100 * lo, 1)
                and _close(PUB["distill_ci"][1], 100 * hi, 1))
    rep.verdict("5.3", "rules accuracy on v2 (the bar it must clear)", f"{PUB['rules_acc']:.1f}%",
                f"{int(sum(rules))}/{n} = {_pct(sum(rules) / n)}", "keyword cue table",
                _close(PUB["rules_acc"], 100 * sum(rules) / n, 1))

    fixed = sum(1 for x, y in zip(c6, rules) if x > y)
    broke = sum(1 for x, y in zip(c6, rules) if y > x)
    p_rules = mcnemar(fixed, broke)
    fixed3 = sum(1 for x, y in zip(c6, d1) if x > y)
    broke3 = sum(1 for x, y in zip(c6, d1) if y > x)
    p_3b = mcnemar(fixed3, broke3)
    rep.verdict("5.4", "distill vs rules: fixed / broken",
                f"{PUB['distill_vs_rules']['fixed']} / {PUB['distill_vs_rules']['broke']}",
                f"{fixed} / {broke}", "elementwise over the same 150 rows",
                (fixed, broke) == (PUB["distill_vs_rules"]["fixed"], PUB["distill_vs_rules"]["broke"]))
    rep.verdict("5.5", "distill vs rules: exact McNemar p",
                f"{PUB['distill_vs_rules']['p']:.3f}", f"{p_rules:.4f}",
                "exact binomial", _close_raw(PUB["distill_vs_rules"]["p"], p_rules, 3))
    rep.verdict("5.6", "distill vs 3B: fixed / broken",
                f"{PUB['distill_vs_3b']['fixed']} / {PUB['distill_vs_3b']['broke']}",
                f"{fixed3} / {broke3}", "elementwise over the same 150 rows",
                (fixed3, broke3) == (PUB["distill_vs_3b"]["fixed"], PUB["distill_vs_3b"]["broke"]))
    rep.verdict("5.7", "distill vs 3B: exact McNemar p", f"{PUB['distill_vs_3b']['p']:.2f}",
                f"{p_3b:.4f}", "exact binomial", _close_raw(PUB["distill_vs_3b"]["p"], p_3b, 2))

    # the artefact's own per-class table, derived from my per-row outcomes
    per = {}
    for o, e in zip(c6, ds.examples):
        c, t = per.get(e.label, (0, 0))
        per[e.label] = (c + int(o), t + 1)
    stored = {k: (v["correct"], v["n"]) for k, v in art["per_class"]["C6-ridge-both"].items()}
    agree = sum(1 for k in per if per[k] == stored.get(k))
    rep.verdict("5.8", "artefact per_class{C6-ridge-both} matches the refit",
                f"{len(stored)} classes", f"{agree}/{len(per)} classes agree",
                "per-class correct/n", per == stored,
                "" if per == stored else f"differs: {per} vs {stored}")


# --------------------------------------------------------------------------- #
# claims 6 and 7 — the heldout_v3 set and its joins
# --------------------------------------------------------------------------- #

def claim_6(rep: Report, rows: list[dict]) -> None:
    art = json.loads(V3_FAMILIES.read_text())
    fam = art["families"]
    by_family: dict[str, set[str]] = {}
    for r in rows:
        by_family.setdefault(r["family"], set()).add(r["split"])
    crossing = sorted(f for f, s in by_family.items() if len(s) > 1)
    rep.verdict("6.1", "heldout_v3 rows", str(PUB["v3_rows"]), f"{len(rows)}",
                "len(heldout_v3.json)", len(rows) == PUB["v3_rows"])
    rep.verdict("6.2", "heldout_v3 families", str(PUB["v3_families"]), f"{len(by_family)}",
                "distinct family ids", len(by_family) == PUB["v3_families"])
    rep.verdict("6.3", "heldout_v3 families crossing train/test", str(PUB["v3_crossing"]),
                f"{len(crossing)}", "family → set(split)",
                len(crossing) == PUB["v3_crossing"],
                f"crossing: {crossing[:3]}" if crossing else "no family sits on both sides")
    rep.verdict("6.4", "artefact counts.rows / families.families agree",
                f"{art['counts']['rows']} / {fam['families']}",
                f"{len(rows)} / {len(by_family)}", "fields vs rows",
                art["counts"]["rows"] == len(rows) and fam["families"] == len(by_family))
    sides = {k: len({r["family"] for r in rows if r["split"] == k})
             for k in sorted({r["split"] for r in rows})}
    rep.verdict("6.5", "artefact families_per_side agrees",
                json.dumps(fam["families_per_side"], sort_keys=True),
                json.dumps(sides, sort_keys=True), "rows → families per side",
                fam["families_per_side"] == sides)


def claim_7(rep: Report, rows: list[dict]) -> None:
    art = json.loads(V3_FAMILIES.read_text())["distill_overlap"]
    v3_fams = {r["family"] for r in rows}

    def corpus_families(names: tuple[str, ...]) -> set[str]:
        fams: set[str] = set()
        for name in names:
            for line in (DISTILL_DIR / name).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    fams.add(json.loads(line)["family"])
        return fams

    for cid, names, pub_test, fam_of_art, label in (
            ("7.1", ("train.jsonl", "val.jsonl"), PUB["join_test_rows"], "join",
             "join vs distill (train.jsonl + val.jsonl)"),
            ("7.2", ("train.jsonl",), PUB["join_trainonly_test_rows"], "train_only_join",
             "join vs distill:train (train.jsonl only)")):
        fams = corpus_families(names)
        hit = [r for r in rows if r["family"] in fams]
        test = sum(1 for r in hit if r["split"] == "test")
        train = sum(1 for r in hit if r["split"] == "train")
        rep.verdict(f"{cid}a", f"{label}: test rows removed by the join", f"{pub_test}",
                    f"{test}", "family join over committed corpus", test == pub_test)
        rep.verdict(f"{cid}b", f"{label}: overlapping families",
                    f"{art[fam_of_art]['overlapping_families']}",
                    f"{len(v3_fams & fams)}", "set intersection",
                    len(v3_fams & fams) == art[fam_of_art]["overlapping_families"])
        rep.verdict(f"{cid}c", f"{label}: rows (train/test)", f"{art[fam_of_art]['rows']} "
                    f"({art[fam_of_art]['train_rows']}/{art[fam_of_art]['test_rows']})",
                    f"{len(hit)} ({train}/{test})", "rows whose family is in the corpus",
                    len(hit) == art[fam_of_art]["rows"]
                    and train == art[fam_of_art]["train_rows"])

    rep.verdict("7.3", "the superseded 293/160 pair is a row count, not an intersection",
                "293 families / 160 test rows",
                f"{len(v3_fams & corpus_families(('train.jsonl', 'val.jsonl')))} families "
                f"/ 160 test rows", "intersection vs corpus size",
                len(v3_fams & corpus_families(("train.jsonl", "val.jsonl"))) != 293,
                "the join's row counts are right; its family count was the corpus's own "
                "293, and the artefact records that under `superseded`")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=5,
                    help="seeds for the claim-2 re-derivation (published protocol: 5)")
    ap.add_argument("--skip-router", action="store_true",
                    help="skip claim 2 — the only claim that costs minutes")
    args = ap.parse_args(argv)

    import eval_harness as H  # noqa: PLC0415 — the committed dataset grouping

    rep = Report()
    ds = H.load_v2(grouping="boundary")
    rows_v3, base_v3, tgt_v3 = _v3()

    claim_1(rep, ds)
    if args.skip_router:
        rep.add("2", "the deployable router (79.3% vs oracle 84.5%, lr 67.1%, 3b 74.0%)",
                "—", "not run", "--skip-router", "SKIP")
    else:
        claim_2(rep, ds, tuple(range(args.seeds)))
    claim_3(rep, rows_v3, base_v3, tgt_v3)
    claim_4(rep, rows_v3, base_v3, tgt_v3)
    claim_5(rep, ds)
    claim_6(rep, rows_v3)
    claim_7(rep, rows_v3)

    print(rep.render())
    fails = [c for c in rep.checks if c.status == "FAIL"]
    caveats = [c for c in rep.checks if c.status == "CAVEAT"]
    skipped = [c for c in rep.checks if c.status == "SKIP"]
    passed = [c for c in rep.checks if c.status == "PASS"]
    print()
    print(f"{len(passed)} pass, {len(fails)} fail, {len(caveats)} caveat, "
          f"{len(skipped)} skip")
    if caveats:
        print("\ncaveats (arithmetic right, scope overstated — for a human to word):")
        for c in caveats:
            print(f"  {c.cid} {c.claim}: {c.computed}")
            if c.note:
                print(f"      {c.note}")
    if fails:
        print("\nfailures:")
        for c in fails:
            print(f"  {c.cid} {c.claim}: published {c.published}, computed {c.computed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
