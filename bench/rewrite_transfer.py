#!/usr/bin/env python3
"""rewrite_transfer — does the p1 description rewrite survive an unseen set?

The claim in ``bench/results/p1-diagnose/FINDINGS.md`` §3 is that changing two of
the eight option descriptions moves d1-3B from 74.0% to 82.0% on
``tests/data/utterances_v2.json``. Those two strings were written by reading the
errors of the same 150 rows they are scored on, so the number is in-sample.

``bench/data/heldout_v3.json`` did not exist when they were written: 415 rows,
133 authored families, split assigned before any utterance was rendered, and
independent of v2. Applying the *unchanged* rewrite to it is a genuine
out-of-sample test. This script runs that test.

Two arms, and the rewrite is imported rather than restated:

``baseline``  ``ti_bench.prompts.OP_DESCRIPTIONS``, the frozen instrument
``targeted``  ``routing_diagnose.TARGETED``, the two changed descriptions

The v2 arms and the v3 baseline arm are already recorded, so only the v3
``targeted`` arm costs server time. That arm is sampled in strata, because the
server is shared and 415 rows at ~15 s is hours: the plan below runs the two
classes the rewrite targets first, on both sides of v3's split, then a
deterministic hash-ordered sample of the six it does not. Every row that ran is
in the JSONL and every row that did not is named by its absence — the report
prints coverage explicitly rather than implying the full set was measured.

Usage:
    python3 bench/rewrite_transfer.py --arm targeted --expect-model d1-3B-Q4_K_M.gguf
    python3 bench/rewrite_transfer.py --arm targeted --limit 52 --strata targets-test
    python3 bench/rewrite_transfer.py --report

The rows append to ``bench/results/p3-transfer/rows.jsonl`` and a re-run skips
what is already there, so a sampled run can be extended without paying twice.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

from model_guard import ModelAssertionError, preflight  # noqa: E402
from ti_bench.prompts import INSTRUCTIONS, OP_DESCRIPTIONS  # noqa: E402
from ti_bench.stats import mcnemar_exact, rate, wilson_ci  # noqa: E402
from routing_diagnose import Server  # noqa: E402
from routing_diagnose import TARGETED  # noqa: E402

V3 = ROOT / "bench/data/heldout_v3.json"
V2_RUN = ROOT / "bench/results/p1-diagnose/baseline_live.json"
V2_TARGETED = ROOT / "bench/results/p1-diagnose/rewrite2.json"
V3_BASELINE = ROOT / "bench/results/p2-heldout/v3-d1-3B-Q4.json"
OUT_DIR = ROOT / "bench/results/p3-transfer"
ROWS = OUT_DIR / "rows.jsonl"

TARGETED_GREP = dict(OP_DESCRIPTIONS)
TARGETED_GREP["filesys_grep"] = TARGETED["filesys_grep"]

# E2: the same meaning in different words, at the same length. If this arm moves the
# needle as far as grep_only does, the effect was the rewording, not the clarification.
GREP_CONTROL = dict(OP_DESCRIPTIONS)
GREP_CONTROL["filesys_grep"] = (
    "Look through the contents of local files for a pattern or text; the result is "
    "which code or document holds it."
)

# E3: the same eight descriptions, presented in the opposite order. Pure instrument --
# no string changes, so any movement is position, not text.
ORDER_REV = dict(reversed(list(OP_DESCRIPTIONS.items())))

ARMS = {"baseline": OP_DESCRIPTIONS, "targeted": TARGETED,
        "floor_a": OP_DESCRIPTIONS, "floor_b": OP_DESCRIPTIONS,
        "grep_only": TARGETED_GREP, "grep_control": GREP_CONTROL,
        "order_rev": ORDER_REV}
# The two labels the p1 rewrite changed, and the six it left byte-identical.
CHANGED = [k for k in TARGETED if TARGETED[k] != OP_DESCRIPTIONS[k]]
TARGET_LABELS = ("decide_extract", "filesys_grep")
OTHER_LABELS = tuple(sorted(set(OP_DESCRIPTIONS) - set(TARGET_LABELS)))


def load_v3() -> list[dict]:
    rows = json.loads(V3.read_text(encoding="utf-8"))
    for i, r in enumerate(rows):
        r["idx"] = i
    return rows


def _order(rows: list[dict]) -> list[dict]:
    """Hash order, so a prefix is an arbitrary subset rather than the first N rows.

    The same trick ``gen_heldout_v3.py`` uses for frame selection: a sha256 of a
    fixed string, not an RNG, so the order cannot depend on Python's hash seed or
    on when the run started.
    """
    return sorted(rows, key=lambda r: hashlib.sha256(f"p3-transfer|{r['id']}".encode()).hexdigest())


def plan(rows: list[dict]) -> list[dict]:
    """Every row tagged with its stratum, in the order the budget should spend.

    ``targets-*`` first: those are the two classes the rewrite claims to repair,
    so they are the rows that answer the question. ``guard-*`` after: the six
    classes the rewrite is not supposed to touch, which is where a trade would
    show up if it made one.
    """
    strata = {
        "targets-test": [r for r in rows if r["label"] in TARGET_LABELS and r["split"] == "test"],
        "guard-test": [r for r in rows if r["label"] in OTHER_LABELS and r["split"] == "test"],
        "targets-train": [r for r in rows if r["label"] in TARGET_LABELS and r["split"] == "train"],
        "guard-train": [r for r in rows if r["label"] in OTHER_LABELS and r["split"] == "train"],
    }
    out = []
    for name, members in strata.items():
        for r in _order(members):
            out.append({**r, "stratum": name})
    return out


def done_keys() -> set[tuple[str, str]]:
    if not ROWS.exists():
        return set()
    keys = set()
    for line in ROWS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            keys.add((rec["arm"], rec["id"]))
    return keys


def v2_spot_check(arm: str, base_url: str, expect_model: str, n: int) -> None:
    """Re-ask ``n`` hash-chosen v2 rows live and compare with the archived arm.

    The archived v2 targeted run is the in-sample claim. Copying its arithmetic
    reproduces nothing, so this asks the same rows again against the live server:
    agreement is what licenses treating the rest of that recording as a
    measurement of *this* endpoint rather than of one that has since been
    rebuilt or re-quantised.
    """
    criteria = ARMS[arm]
    try:
        served, warning = preflight(served_props(base_url), expect_model)
    except ModelAssertionError as e:
        raise SystemExit(f"ABORT: {e}")
    if warning:
        print(warning, file=sys.stderr)

    v2rows = v2_by_utterance()
    archive_path = V2_TARGETED if arm == "targeted" else V2_RUN
    arch = archived(archive_path, [v2rows[k] for k in sorted(v2rows)])
    picked = _order(list(v2rows.values()))[:n]
    server = Server(base_url, timeout=180.0, retries=4)
    out = []
    for i, row in enumerate(picked, 1):
        t0 = time.perf_counter()
        answer = server.ask(row["utterance"], criteria)
        ms = (time.perf_counter() - t0) * 1000
        rec = {"arm": arm, "id": row["id"], "utterance": row["utterance"],
               "label": row["label"], "choice": answer["choice"],
               "correct": answer["choice"] == row["label"],
               "conf": round(answer["confidence"], 4),
               "probabilities": answer["probabilities"], "ms": round(ms, 1),
               "archived_choice": arch[row["id"]]["choice"],
               "archived_correct": arch[row["id"]]["correct"],
               "agrees": answer["choice"] == arch[row["id"]]["choice"]}
        out.append(rec)
        print(f"  {i}/{len(picked)} {ms / 1000:.1f}s {row['label']:<14} live={rec['choice']} "
              f"archived={rec['archived_choice']} {'AGREE' if rec['agrees'] else 'DISAGREE'}",
              file=sys.stderr, flush=True)
    agree = sum(1 for r in out if r["agrees"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"v2_spot_check_{arm}.json").write_text(
        json.dumps({"arm": arm, "n": len(out), "agree": agree,
                    "rows": out}, indent=1) + "\n", encoding="utf-8")
    print(f"agreement {agree}/{len(out)} with {archive_path.name}")


def served_props(base_url: str) -> dict:
    with urllib.request.urlopen(base_url.rstrip("/") + "/props", timeout=10) as resp:
        return json.loads(resp.read())


def run(arm: str, base_url: str, expect_model: str, limit: int, strata: list[str],
        ordinals: str = "") -> None:
    criteria = ARMS[arm]
    try:
        served, warning = preflight(served_props(base_url), expect_model)
    except ModelAssertionError as e:
        raise SystemExit(f"ABORT: {e}")
    if warning:
        print(warning, file=sys.stderr)

    rows = plan(load_v3())
    if strata:
        rows = [r for r in rows if r["stratum"] in strata]
    if ordinals:
        keep = {int(x) for x in ordinals.split(",")}
        rows = [r for r in rows if r["idx"] in keep]
    seen = done_keys()
    todo = [r for r in rows if (arm, r["id"]) not in seen]
    if limit:
        todo = todo[:limit]
    print(f"arm={arm} k={len(criteria)} changed={CHANGED if arm == 'targeted' else []} "
          f"planned={len(rows)} already={len(rows) - len(todo) - max(0, len(rows) - len(todo) - limit) if limit else len(rows) - len(todo)} "
          f"running={len(todo)}", file=sys.stderr)

    server = Server(base_url, timeout=180.0, retries=4)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    with ROWS.open("a", encoding="utf-8") as sink:
        for i, row in enumerate(todo, 1):
            t0 = time.perf_counter()
            answer = server.ask(row["utterance"], criteria)
            ms = (time.perf_counter() - t0) * 1000
            rec = {"arm": arm, "id": row["id"], "idx": row["idx"],
                   "utterance": row["utterance"], "label": row["label"],
                   "family": row["family"], "split": row["split"], "band": row["band"],
                   "stratum": row["stratum"], "k": len(criteria),
                   "choice": answer["choice"], "correct": answer["choice"] == row["label"],
                   "conf": round(answer["confidence"], 4),
                   "probabilities": answer["probabilities"],
                   "ms": round(ms, 1), "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
            if i % 10 == 0 or i == len(todo):
                elapsed = time.perf_counter() - t_start
                eta = elapsed / i * (len(todo) - i) / 60
                print(f"  {i}/{len(todo)} {elapsed / i:.1f}s/row eta {eta:.0f}m "
                      f"[{row['stratum']} {row['label']} {'ok' if rec['correct'] else 'miss'}]",
                      file=sys.stderr, flush=True)


# ------------------------------------------------------------------ report

def load_rows() -> list[dict]:
    if not ROWS.exists():
        return []
    return [json.loads(line) for line in ROWS.read_text(encoding="utf-8").splitlines() if line.strip()]


def archived(path: Path, set_rows: list[dict], label_key: str = "label") -> dict[str, dict]:
    """Recorded run -> {utterance: row}, refusing to align against a mismatch."""
    run = json.loads(path.read_text(encoding="utf-8"))
    per_row = run["per_row"]
    if len(per_row) != len(set_rows):
        raise SystemExit(f"{path.name}: {len(per_row)} recorded rows vs {len(set_rows)} in dataset")
    out = {}
    for rec, row in zip(per_row, set_rows):
        if rec["utterance"] != row["utterance"]:
            raise SystemExit(f"{path.name}: misaligned at {row.get('id')} — "
                             f"{rec['utterance']!r} vs {row['utterance']!r}")
        out[row["id"]] = {"id": row["id"], "utterance": rec["utterance"],
                          "label": row[label_key], "family": row.get("family"),
                          "split": row.get("split"), "band": row.get("band"),
                          "choice": rec["choice"], "correct": bool(rec["correct"]),
                          "conf": rec.get("conf")}
    return out


def v2_by_utterance() -> dict[str, dict]:
    raw = json.loads((ROOT / "tests/data/utterances_v2.json").read_text())
    out = {}
    for i, r in enumerate(raw):
        out[f"v2:{i:03d}"] = {"id": f"v2:{i:03d}", "utterance": r["utterance"],
                             "label": r["label"], "family": None, "split": "all",
                             "band": r.get("band")}
    return out


def paired(a: dict[str, dict], b: dict[str, dict], keep=None) -> dict:
    """Fixed/broken/kept over the ids both arms cover, plus exact McNemar."""
    ids = sorted(set(a) & set(b))
    if keep:
        ids = [i for i in ids if keep(a[i])]
    fixed = sum(1 for i in ids if not a[i]["correct"] and b[i]["correct"])
    broke = sum(1 for i in ids if a[i]["correct"] and not b[i]["correct"])
    return {"n": len(ids), "fixed": fixed, "broke": broke, "ids": ids,
            "a_correct": sum(a[i]["correct"] for i in ids),
            "b_correct": sum(b[i]["correct"] for i in ids),
            "p": mcnemar_exact(fixed, broke)}


def cluster_bootstrap(a: dict[str, dict], b: dict[str, dict], ids: list[str],
                      reps: int = 4000, seed: int = 20261104) -> dict:
    """95% interval for the paired delta, resampling *families* not rows.

    v3's rows are not independent: 356 of 415 sit in a multi-row family, so a row
    bootstrap would understate the interval by pretending paraphrases of one
    authored sentence are independent observations. Families are the sampling
    unit here for the same reason the split is drawn on them.
    """
    by_family: dict[str, list[str]] = {}
    for i in ids:
        by_family.setdefault(a[i].get("family") or i, []).append(i)
    fams = sorted(by_family)
    rng = random.Random(seed)
    deltas = []
    for _ in range(reps):
        pick = [rng.choice(fams) for _ in fams]
        sub = [i for f in pick for i in by_family[f]]
        deltas.append((sum(b[i]["correct"] for i in sub) - sum(a[i]["correct"] for i in sub))
                      / len(sub))
    deltas.sort()
    observed = (sum(b[i]["correct"] for i in ids) - sum(a[i]["correct"] for i in ids)) / len(ids)
    return {"delta": round(observed, 4),
            "lo": round(deltas[int(0.025 * reps)], 4),
            "hi": round(deltas[int(0.975 * reps)], 4),
            "units": len(fams), "reps": reps}


def pct_ci(correct: int, n: int) -> str:
    """An arm's rate with its Wilson interval; '—' when no row covers it."""
    if not n:
        return "—"
    r = rate(correct, n)
    return (f"{100 * r['accuracy']:5.1f}% "
            f"[{100 * r['wilson95']['low']:.1f},{100 * r['wilson95']['high']:.1f}]")


def table(title: str, a: dict[str, dict], b: dict[str, dict], ids: list[str],
          names: tuple[str, str] = ("baseline", "targeted")) -> dict:
    print(f"\n== {title} (n={len(ids)}, paired) ==")
    if not ids:
        print("  (no rows in this slice yet)")
        return {"n": 0, "a": 0, "b": 0, "p": 1.0, "fixed": 0, "broke": 0,
                "delta": {"delta": None, "lo": None, "hi": None, "units": 0}}
    pa = sum(a[i]["correct"] for i in ids)
    pb = sum(b[i]["correct"] for i in ids)
    print(f"  {names[0]:<9} {pct_ci(pa, len(ids))}   {names[1]:<9} {pct_ci(pb, len(ids))}")
    comp = paired({i: a[i] for i in ids}, {i: b[i] for i in ids})
    boot = cluster_bootstrap(a, b, ids)
    print(f"  fixed {comp['fixed']} / broke {comp['broke']}  exact McNemar p={comp['p']:.4f}")
    print(f"  paired delta {100 * boot['delta']:+.1f} pts "
          f"[{100 * boot['lo']:+.1f},{100 * boot['hi']:+.1f}] "
          f"(cluster bootstrap over {boot['units']} families, {boot['reps']} reps)")
    return {"n": len(ids), "a": pa, "b": pb, "p": comp["p"], "fixed": comp["fixed"],
            "broke": comp["broke"], "delta": boot}


def by_class(a: dict[str, dict], b: dict[str, dict], ids: list[str]) -> list[dict]:
    out = []
    for lab in sorted({a[i]["label"] for i in ids}):
        sub = [i for i in ids if a[i]["label"] == lab]
        la = sum(a[i]["correct"] for i in sub)
        lb = sum(b[i]["correct"] for i in sub)
        f = sum(1 for i in sub if not a[i]["correct"] and b[i]["correct"])
        br = sum(1 for i in sub if a[i]["correct"] and not b[i]["correct"])
        out.append({"label": lab, "n": len(sub), "a": la, "b": lb,
                    "fixed": f, "broke": br, "p": mcnemar_exact(f, br),
                    "targeted_by_rewrite": lab in TARGET_LABELS})
    return out


def print_classes(rows: list[dict]) -> None:
    print(f"  {'class':<16}{'n':>4}  {'baseline':<18}{'targeted':<18}{'fix/brk':>9}{'p':>9}")
    for r in rows:
        mark = "*" if r["targeted_by_rewrite"] else " "
        print(f" {mark}{r['label']:<16}{r['n']:>4}  {pct_ci(r['a'], r['n']):<18}"
              f"{pct_ci(r['b'], r['n']):<18}{r['fixed']:>4}/{r['broke']:<4}{r['p']:>9.4f}")


def report() -> dict:
    out = {}
    v3 = load_v3()
    by_id_v3 = {r["id"]: r for r in v3}
    v2rows = v2_by_utterance()
    v2_base = archived(V2_RUN, [v2rows[k] for k in sorted(v2rows)])
    v2_tgt = archived(V2_TARGETED, [v2rows[k] for k in sorted(v2rows)])
    v3_base = archived(V3_BASELINE, v3)

    # 1. the in-sample claim, re-derived from the archive
    ids = sorted(set(v2_base) & set(v2_tgt))
    print("== in-sample reproduction (v2, archived recordings) ==")
    out["v2"] = table("v2 150 rows, archived baseline vs archived targeted", v2_base, v2_tgt, ids)
    print()
    print_classes(by_class(v2_base, v2_tgt, ids))

    # 2. the live targeted arm on v3
    live = load_rows()
    if not live:
        print("\n(no live rows recorded yet — run without --report first)")
        return out
    v3_tgt = {}
    for rec in live:
        if rec["arm"] != "targeted":
            continue
        row = by_id_v3[rec["id"]]
        v3_tgt[rec["id"]] = {"id": rec["id"], "utterance": rec["utterance"],
                             "label": row["label"], "family": row["family"],
                             "split": row["split"], "band": row["band"],
                             "choice": rec["choice"], "correct": bool(rec["correct"]),
                             "conf": rec["conf"]}

    # coverage, stated as the set difference rather than as a count alone
    plan_rows = plan(v3)
    ran = set(v3_tgt)
    print(f"\n== coverage: {len(ran)}/{len(v3)} v3 rows ran the targeted arm ==")
    for name in ("targets-test", "guard-test", "targets-train", "guard-train"):
        members = [r for r in plan_rows if r["stratum"] == name]
        n_ran = sum(1 for r in members if r["id"] in ran)
        print(f"  {name:<14}{n_ran:>4}/{len(members):<4}")
    missing = [r for r in plan_rows if r["id"] not in ran]
    out["coverage"] = {"ran": len(ran), "planned": len(plan_rows), "skipped": len(missing)}

    def ids_for(keep):
        return sorted(i for i in ran if keep(v3_tgt[i]))

    out["v3_all"] = table("v3, all rows run", v3_base, v3_tgt, ids_for(lambda r: True))
    out["v3_test"] = table("v3, test side", v3_base, v3_tgt,
                           ids_for(lambda r: r["split"] == "test"))
    out["v3_train"] = table("v3, train side", v3_base, v3_tgt,
                            ids_for(lambda r: r["split"] == "train"))
    for split in ("test", "train", "all"):
        sub = ids_for(lambda r: split == "all" or r["split"] == split)
        if sub:
            print(f"\n  per class, v3 {split} side:")
            print_classes(by_class(v3_base, v3_tgt, sub))

    # 3. composition: what the v2 -> v3 difference would be at a common class mix
    common = sorted(set(v2_base) & set(v2_tgt))
    v2_classes = by_class(v2_base, v2_tgt, common)
    v3_classes = {r["label"]: r for r in by_class(v3_base, v3_tgt, ids_for(lambda r: True))}
    shared = [r for r in v2_classes if r["label"] in v3_classes]
    w2 = {r["label"]: r["n"] for r in shared}
    tot2 = sum(w2.values())
    reweighted_a = sum(v3_classes[l]["a"] / v3_classes[l]["n"] * w2[l] for l in w2) / tot2
    reweighted_b = sum(v3_classes[l]["b"] / v3_classes[l]["n"] * w2[l] for l in w2) / tot2
    print("\n== composition: v3 rates reweighted to v2's class mix ==")
    print(f"  v3 as sampled:            baseline {100 * sum(v3_classes[l]['a'] for l in w2) / sum(v3_classes[l]['n'] for l in w2):.1f}%"
          f"  targeted {100 * sum(v3_classes[l]['b'] for l in w2) / sum(v3_classes[l]['n'] for l in w2):.1f}%")
    print(f"  v3 at v2's class weights:  baseline {100 * reweighted_a:.1f}%  targeted {100 * reweighted_b:.1f}%")
    print(f"  v2 (in-sample):            baseline {100 * sum(r['a'] for r in shared) / tot2:.1f}%"
          f"  targeted {100 * sum(r['b'] for r in shared) / tot2:.1f}%")
    out["composition"] = {"v3_reweighted_baseline": round(reweighted_a, 4),
                          "v3_reweighted_targeted": round(reweighted_b, 4),
                          "weights": w2}

    # 4. did the delta differ by class, on the two it targets vs the six it does not
    tgt_ids = ids_for(lambda r: r["label"] in TARGET_LABELS)
    oth_ids = ids_for(lambda r: r["label"] not in TARGET_LABELS)
    out["v3_targets"] = table("v3, the two rewritten classes", v3_base, v3_tgt, tgt_ids)
    out["v3_others"] = table("v3, the six untouched classes", v3_base, v3_tgt, oth_ids)
    out["v2_targets"] = table("v2, the two rewritten classes", v2_base, v2_tgt,
                              [i for i in common if v2_base[i]["label"] in TARGET_LABELS])
    out["v2_others"] = table("v2, the six untouched classes", v2_base, v2_tgt,
                             [i for i in common if v2_base[i]["label"] not in TARGET_LABELS])

    # 5. band breakdown, the same read p1 used
    print("\n== v3 by band ==")
    for band in ("t1", "t2", "t3"):
        sub = ids_for(lambda r: r["band"] == band)
        if sub:
            pa, pb = sum(v3_base[i]["correct"] for i in sub), sum(v3_tgt[i]["correct"] for i in sub)
            f = sum(1 for i in sub if not v3_base[i]["correct"] and v3_tgt[i]["correct"])
            br = sum(1 for i in sub if v3_base[i]["correct"] and not v3_tgt[i]["correct"])
            print(f"  {band} n={len(sub):<4}{pct_ci(pa, len(sub))} -> {pct_ci(pb, len(sub))}"
                  f"  fix/brk {f}/{br}  p={mcnemar_exact(f, br):.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT_DIR / 'summary.json'}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=sorted(ARMS), default="targeted")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--expect-model", default="")
    ap.add_argument("--limit", type=int, default=0, help="run at most N rows")
    ap.add_argument("--strata", default="", help="comma-separated stratum names")
    ap.add_argument("--ordinals", default="", help="comma-separated v3 idx values (spot checks)")
    ap.add_argument("--spot-v2", type=int, default=0,
                    help="ask N hash-chosen v2 rows live and compare with the archive")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report()
        return
    if args.spot_v2:
        v2_spot_check(args.arm, args.base_url, args.expect_model, args.spot_v2)
        return
    run(args.arm, args.base_url, args.expect_model, args.limit,
        [s for s in args.strata.split(",") if s], args.ordinals)


if __name__ == "__main__":
    main()
