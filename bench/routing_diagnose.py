#!/usr/bin/env python3
"""routing_diagnose — why d1-3B loses filesys_grep and decide_extract.

The frozen instrument (``ti_bench.prompts``) asks every row the same question
with the same eight options and the utterance as the whole state, so a
per-class difference cannot come from a per-class request shape. What it *can*
come from is which options are on the table and how their text reads. This
script varies one of those at a time and measures the same 150 labelled rows.

Four probes, each answering a different question:

``orders``
    Same rows, same eight options, two conflicting option orders. If the
    argmax moves, the answer is partly a function of presentation.
``k3``
    The truth plus that class's two favourite wrong answers, nothing else
    (K=3 vs K=8). Recovery here means the model can already separate the
    truth from its nearest rival and the other options were crowding it out;
    no recovery means the boundary itself is unresolved.
``neutral6``
    K=8 minus two options that no row of the class ever answered, so K drops
    to 6 without removing a competitor. It is the control for ``k3``: any
    gain that comes from a shorter list at all shows up here.
``rewrite``
    All 150 rows against a rewritten option set (``OPERATION_FIRST``) whose
    text names the operation and refuses the object-noun reading. Measured
    over the whole set so a gain in one class cannot hide a loss in another.

Usage: python bench/routing_diagnose.py [--base-url URL] [--expect-model NAME]
       [--out-dir bench/results/p1-diagnose] [--only EXP[,EXP...]]

Nothing here writes to ``bench/ti_bench/prompts.py``: that file is the
instrument every historical lane was scored with, and
``tests/test_dataset_pipeline.py`` pins its text against the dataset's copy.
The variant sets below are this probe's own, the way ``bench/or_bench.py``
keeps ``COMPACT_OP_DESCRIPTIONS``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_guard import ModelAssertionError, preflight  # noqa: E402
from ti_bench.prompts import INSTRUCTIONS, OP_DESCRIPTIONS  # noqa: E402
from ti_bench.stats import mcnemar_exact, rate  # noqa: E402

LABELS = list(OP_DESCRIPTIONS)

# The two classes the v2 sweep loses, and the wrong answers they lose them to
# (bench/results/local_clf_v2.json, 3B column). Kept here so the probes name
# their competitor instead of rediscovering it from a live run: a probe that
# picks its own competitor from the answers it is measuring cannot falsify
# anything.
FAILING = {
    "filesys_grep": ["web_fetch", "filesys_glob"],
    "decide_extract": ["filesys_glob", "web_search"],
}

# Removed in the `neutral6` control: two options that are in the frozen set but
# are not a competitor of either failing class in the archived run, so K drops
# from 8 to 6 while every real competitor stays on the table.
NEUTRAL_DROP = ["reddit_digest", "decide_pick"]

# The rewrite candidate. Each line names the *operation the caller performs* and
# then refuses the reading that produced the v2 confusion: "list the file names"
# is a string the extractor emits, not a listing; "which functions call urllib"
# is a search of local contents, not a fetch of anything on the network.
OPERATION_FIRST = {
    "filesys_read": "Return the whole contents of one file the user already named. "
                    "One file, contents only.",
    "filesys_glob": "Return a list of file paths chosen by name or pattern. Paths only; "
                    "it never reads a file's text.",
    "filesys_grep": "After reading local files, return the lines that match a pattern. "
                    "Use whenever the answer is 'which code/text contains X'.",
    "decide_pick": "Return a selection from a list of items already in hand. Nothing "
                   "new is read, listed, or fetched.",
    "decide_extract": "Return literal strings found in the user's own message — paths, "
                      "names, keywords, queries. It emits the words; it does not "
                      "list files, read files, or search the web.",
    "web_search": "Return a list of results from the internet for a query. Requires "
                  "going to the web now.",
    "web_fetch": "Return the contents of one URL. Requires a URL and a network fetch now.",
    "reddit_digest": "Return a summary of Reddit posts or comments.",
}


# ---------------------------------------------------------------- transport

class Server:
    """One live /v1/systemone endpoint, no server management of any kind.

    The server is shared, so a transient failure is retried with a short bounded
    backoff rather than failing a row: a dropped row would otherwise read as a
    miss and inflate every confusion count below it.
    """

    def __init__(self, base_url: str, timeout: float = 120.0, retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.calls = 0

    def ask(self, state, criteria, instructions: str = INSTRUCTIONS) -> dict:
        body = json.dumps({"state": state, "questions": {"route": {
            "type": "choice", "instructions": instructions,
            "criteria": dict(criteria)}}}).encode()
        last = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(
                    self.base_url + "/v1/systemone", data=body,
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    out = json.loads(resp.read())
                self.calls += 1
                answer = (out.get("answers") or {}).get("route") or {}
                return {"choice": answer.get("choice"),
                        "confidence": float(answer.get("confidence") or 0.0),
                        "probabilities": {k: float(v) for k, v in
                                          (answer.get("probabilities") or {}).items()},
                        "input_tokens": (out.get("usage") or {}).get("input_tokens")}
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"/v1/systemone failed after {self.retries} tries: {last}")


# ---------------------------------------------------------------- probes

def run_rows(server: Server, rows: list[dict], criteria_for, *,
             shuffle_seed: int | None = None, label: str = "") -> list[dict]:
    """Ask every row once with the option set ``criteria_for`` returns for it."""
    out = []
    for i, row in enumerate(rows):
        crit = criteria_for(row)
        answer = server.ask(row["utterance"], crit)
        out.append({"idx": row["idx"], "utterance": row["utterance"], "label": row["label"],
                    "band": row["band"], "choice": answer["choice"],
                    "correct": answer["choice"] == row["label"],
                    "conf": round(answer["confidence"], 4),
                    "k": len(crit),
                    "order": list(crit), "probabilities": answer["probabilities"],
                    "input_tokens": answer["input_tokens"],
                    "shuffle_seed": shuffle_seed})
        if (i + 1) % 20 == 0:
            print(f"  [{label}] {i + 1}/{len(rows)}", file=sys.stderr, flush=True)
    return out


def probe_orders(server: Server, rows: list[dict], baseline: list[dict] | None) -> dict:
    """The same rows under reversed option order, against the canonical pass.

    The canonical arm is the baseline sweep that already ran with this same
    server, so only the reversed arm is measured here: the comparison is paired
    row by row, and the measurement costs half the server time.
    """
    reversed_criteria = dict(reversed(list(OP_DESCRIPTIONS.items())))
    flipped = run_rows(server, rows, lambda r: reversed_criteria,
                       shuffle_seed=-1, label="ord/rev")
    canonical = []
    if baseline:
        by_text = {r["utterance"]: r for r in baseline}
        canonical = [{"idx": r["idx"], "utterance": r["utterance"],
                      "label": by_text[r["utterance"]]["label"],
                      "choice": by_text[r["utterance"]]["choice"],
                      "correct": by_text[r["utterance"]]["correct"]}
                     for r in flipped if r["utterance"] in by_text]
        flipped = [r for r in flipped if r["utterance"] in by_text]
    moved = sum(1 for a, b in zip(canonical, flipped) if a["choice"] != b["choice"])
    return {"experiment": "orders", "n": len(flipped),
            "canonical": _summary(canonical), "reversed": _summary(flipped),
            "canonical_source": "baseline sweep" if baseline else "measured here",
            "argmax_moved": moved,
            "argmax_moved_frac": moved / len(flipped) if flipped else 0.0,
            "per_row": {"canonical": canonical, "reversed": flipped},
            "changed_rows": [{"label": a["label"], "utterance": a["utterance"],
                              "forward": a["choice"], "reversed": b["choice"]}
                             for a, b in zip(canonical, flipped)
                             if a["choice"] != b["choice"]]}


def probe_k3(server: Server, rows: list[dict]) -> dict:
    """Truth plus that class's two favourite wrong answers; K=3."""
    def criteria_for(row: dict) -> dict:
        return {k: OP_DESCRIPTIONS[k] for k in [row["label"]] + FAILING[row["label"]]}
    per_row = run_rows(server, rows, criteria_for, label="k3")
    return {"experiment": "k3", "n": len(rows), "k": 3,
            "summary": _summary(per_row), "per_row": per_row}


def probe_neutral6(server: Server, rows: list[dict]) -> dict:
    """K=8 minus two non-competitors: the shorter-list control for k3."""
    kept = {k: v for k, v in OP_DESCRIPTIONS.items() if k not in NEUTRAL_DROP}
    per_row = run_rows(server, rows, lambda r: kept, label="n6")
    return {"experiment": "neutral6", "n": len(rows), "k": len(kept),
            "dropped": NEUTRAL_DROP, "summary": _summary(per_row), "per_row": per_row}


# Minimal pairs: each entry is a real v2 row the 3B misses, paired with an edit
# that removes only the noun carrying the *other* option's domain and leaves the
# operation untouched. If the answer follows the noun, the row was decided by
# surface vocabulary rather than by the operation; if it stays on the wrong
# option with the domain noun gone, the operation itself is not represented.
MINIMAL_PAIRS = [
    # filesys_grep, answered web_fetch: the noun is network, the operation is not
    ("filesys_grep", "which functions call urllib in the test suite?",
     "which functions call helper_fn in the test suite?"),
    ("filesys_grep", "find any use of the requests module in src",
     "find any use of the numpy module in src"),
    ("filesys_grep", "where is the http client wired up in service/net?",
     "where is the parser wired up in service/net?"),
    ("filesys_grep", "locate each place we catch HTTPError",
     "locate each place we catch ValueError"),
    ("filesys_grep", "where is handle_stream defined?",
     "where is compute_offsets defined?"),
    ("filesys_grep", "does anything here still call the v1 endpoint?",
     "does anything here still call the legacy helper?"),
    # filesys_grep, answered filesys_glob: the noun names a file, the ask is content
    ("filesys_grep", "any references to wallet.json left in the tree?",
     "any references to wallet_config left in the tree?"),
    ("filesys_grep", "which files under src discuss the shard map?",
     "which code under src discusses the shard map?"),
    # decide_extract, answered filesys_glob / web_search: the object is a path or
    # a keyword; the operation is 'give me the literal strings'
    ("decide_extract", "extract the file paths mentioned in the last message",
     "extract the items mentioned in the last message"),
    ("decide_extract", "name the dirs the request refers to",
     "name the things the request refers to"),
    ("decide_extract", "what literal paths show up in that instruction?",
     "what literal values show up in that instruction?"),
    ("decide_extract", "what search terms fit this ask?",
     "what terms fit this ask?"),
    ("decide_extract", "give me keyword variants for the upcoming lookup",
     "give me variants for the upcoming attempt"),
    ("decide_extract", "write out the file names the request refers to",
     "write out the words the request refers to"),
]


def probe_minimal_pairs(server: Server) -> dict:
    """Same row, one noun swapped: does the answer follow the noun?"""
    out = []
    for label, original, edited in MINIMAL_PAIRS:
        showed, edited_ans = {}, {}
        for name, text, sink in (("original", original, showed), ("edited", edited, edited_ans)):
            answer = server.ask(text, OP_DESCRIPTIONS)
            sink.update({"choice": answer["choice"], "correct": answer["choice"] == label,
                         "truth_mass": round(answer["probabilities"].get(label, 0.0), 4),
                         "conf": round(answer["confidence"], 4)})
        out.append({"label": label, "original": original, "edited": edited,
                    "before": showed, "after": edited_ans,
                    "flipped": showed["choice"] != edited_ans["choice"],
                    "recovered": (not showed["correct"]) and edited_ans["correct"]})
    recovered = sum(1 for r in out if r["recovered"])
    return {"experiment": "minimal_pairs", "n": len(out), "recovered": recovered,
            "recovered_frac": recovered / len(out), "pairs": out}


def print_minimal_pairs(result: dict) -> None:
    print(f"\n== minimal_pairs (n={result['n']}) ==")
    print(f"  swapping the domain-leaking noun recovers {result['recovered']}/{result['n']} rows")
    for p in result["pairs"]:
        print(f"  {p['label']:<15} {p['before']['choice']:<14}"
              f"truth_mass {p['before']['truth_mass']:.3f}  ->  "
              f"{p['after']['choice']:<14} truth_mass {p['after']['truth_mass']:.3f}"
              f"  {'RECOVERED' if p['recovered'] else ''}")
        print(f"{'':<17}was: {p['original']}")
        print(f"{'':<17}now: {p['edited']}")


# The targeted candidate: exactly two lines differ from the frozen set, so the
# other six descriptions keep whatever grip they already had. The global rewrite
# above proved the descriptions compete for rows, so a fix that rewords all eight
# is measuring a new equilibrium rather than the two classes it is trying to
# repair. Here `filesys_grep` gains the "which code/text contains it" cue that
# its misses lack, and `decide_extract` names the objects (keywords, paths,
# names) that `web_search` and `filesys_glob` currently claim on its behalf.
TARGETED = dict(OP_DESCRIPTIONS)
TARGETED.update({
    "filesys_grep": "Search inside local file contents for a pattern or string; the "
                    "answer is which code or text contains it.",
    "decide_extract": "Pull the literal strings out of the user's own message — queries, "
                      "keywords, paths, names. Answers with words from the request.",
})


def probe_rewrite(server: Server, rows: list[dict]) -> dict:
    """Rewritten option text over every row, so a gain cannot hide a traded loss."""
    per_row = run_rows(server, rows, lambda r: OPERATION_FIRST, label="rewrite")
    return {"experiment": "rewrite", "n": len(rows), "k": len(OPERATION_FIRST),
            "summary": _summary(per_row), "per_row": per_row}


def probe_targeted(server: Server, rows: list[dict]) -> dict:
    """Two descriptions changed, six byte-identical: the isolated fix."""
    per_row = run_rows(server, rows, lambda r: TARGETED, label="targeted")
    return {"experiment": "rewrite2", "n": len(rows), "k": len(TARGETED),
            "changed": [k for k in TARGETED if TARGETED[k] != OP_DESCRIPTIONS[k]],
            "summary": _summary(per_row), "per_row": per_row}


def _summary(per_row: list[dict]) -> dict:
    """Overall rate plus per-label, in the shape the report stores."""
    by_label: dict[str, dict] = {}
    for r in per_row:
        by_label.setdefault(r["label"], {"n": 0, "correct": 0})
        by_label[r["label"]]["n"] += 1
        by_label[r["label"]]["correct"] += int(r["correct"])
    return {"overall": rate(sum(1 for r in per_row if r["correct"]), len(per_row)),
            "per_label": {lab: rate(s["correct"], s["n"]) for lab, s in sorted(by_label.items())},
            "confusion": _confusion(per_row)}


def _confusion(per_row: list[dict]) -> dict:
    out: dict[str, dict[str, int]] = {}
    for r in per_row:
        if r["correct"]:
            continue
        out.setdefault(r["label"], {})
        out[r["label"]][str(r["choice"])] = out[r["label"]].get(str(r["choice"]), 0) + 1
    return out


# ---------------------------------------------------------------- reporting

def print_orders(result: dict) -> None:
    print(f"\n== orders (n={result['n']}) ==")
    for name in ("canonical", "reversed"):
        r = result[name]["overall"]
        print(f"  {name:<10} {r['accuracy'] * 100:5.1f}%  "
              f"wilson [{r['wilson95']['low'] * 100:.0f},{r['wilson95']['high'] * 100:.0f}]  "
              f"n={r['n']}")
    print(f"  argmax moved on {result['argmax_moved']}/{result['n']} rows "
          f"({result['argmax_moved_frac'] * 100:.0f}%) purely by reversing the option order")
    for row in result["changed_rows"][:40]:
        print(f"     {row['label']:<15} fwd={str(row['forward']):<15} "
              f"rev={str(row['reversed']):<15} | {row['utterance'][:52]}")


def print_probe(result: dict) -> None:
    summary = result["summary"]
    print(f"\n== {result['experiment']} (n={result['n']}, K={result['k']}) ==")
    print(f"  overall {summary['overall']['accuracy'] * 100:5.1f}%")
    for lab, r in summary["per_label"].items():
        print(f"    {lab:<15} {r['accuracy'] * 100:5.1f}%  ({r['correct']}/{r['n']})")
    for lab, targets in summary["confusion"].items():
        print(f"    {lab:<15} misses {sum(targets.values())}: "
              + ", ".join(f"{k}x{n}" for k, n in
                          sorted(targets.items(), key=lambda kv: -kv[1])))


def print_rewrite_vs_baseline(rewrite: dict, baseline: dict) -> None:
    """Paired comparison on the same rows, with an exact McNemar p."""
    base = {r["idx"]: r for r in baseline}
    new = {r["idx"]: r for r in rewrite["per_row"]}
    common = sorted(set(base) & set(new))
    b = sum(1 for i in common if not base[i]["correct"] and new[i]["correct"])
    c = sum(1 for i in common if base[i]["correct"] and not new[i]["correct"])
    p = mcnemar_exact(b, c)
    print(f"\n== {rewrite['experiment']} vs baseline (paired, same {len(common)} rows) ==")
    print(f"  fixed {b} rows, broke {c} rows, exact McNemar p={p:.4f}")
    print(f"  baseline {sum(base[i]['correct'] for i in common)}/{len(common)} "
          f"-> rewrite {sum(new[i]['correct'] for i in common)}/{len(common)}")
    for lab in sorted({base[i]["label"] for i in common}):
        rows = [i for i in common if base[i]["label"] == lab]
        lo, hi = sum(base[i]["correct"] for i in rows), sum(new[i]["correct"] for i in rows)
        flag = "" if hi >= lo else "  <-- REGRESSION"
        print(f"    {lab:<15} {lo:>2}/{len(rows):<2} -> {hi:>2}/{len(rows):<2}{flag}")


def report(out_dir: Path) -> None:
    """Render every number in FINDINGS.md from the saved experiment artifacts.

    Nothing here talks to the server: the point of keeping the per-row files is
    that the diagnosis can be re-derived from them after the server is gone.
    """
    base = json.loads((out_dir / "baseline_live.json").read_text())
    rows = base["per_row"]
    print(f"baseline: n={base['n_rows']} acc={base['accuracy']:.4f} "
          f"model={base['model']['build_info']} p50={base['latency_ms']['p50']}ms")
    print(f"  argument shape: state=<utterance>, instructions=<one string>, "
          f"criteria=<{len(OP_DESCRIPTIONS)} options> — identical for all 8 labels")

    five, _targets = two_classes(rows)
    print("\n== per label (baseline) ==")
    print(f"{'label':<16}{'n':>4}{'correct':>9}{'acc':>8}{'truth rank 2':>14}{'truth mass':>12}")
    for lab in LABELS:
        sub = [r for r in rows if r["label"] == lab]
        rank2 = sum(1 for r in sub if _rank(r) == 2)
        mass = sum(r["probabilities"].get(lab, 0.0) for r in sub) / len(sub)
        c = sum(1 for r in sub if r["correct"])
        print(f"{lab:<16}{len(sub):>4}{c:>9}{c / len(sub):>8.3f}{rank2:>14}{mass:>12.3f}")

    print(f"\n== the {len(five)} misses in {' and '.join(FAILING)} ==")
    for lab in FAILING:
        bad = [r for r in rows if r["label"] == lab and not r["correct"]]
        counts: dict[str, int] = {}
        for r in bad:
            counts[r["choice"]] = counts.get(r["choice"], 0) + 1
        print(f"  {lab:<15} {len(bad)} misses -> "
              + ", ".join(f"{k}x{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))
        for r in bad:
            print(f"    {str(r['choice']):<15} conf={r['conf']:.2f} "
                  f"truth_mass={r['probabilities'].get(lab, 0.0):.3f} | {r['utterance']}")

    for name in ("k3", "neutral6", "rewrite", "rewrite2", "minimal_pairs"):
        path = out_dir / f"{name}.json"
        if not path.exists():
            print(f"\n== {name}: not run ==")
            continue
        _report_experiment(name, json.loads(path.read_text()), rows)

    path = out_dir / "orders.json"
    if path.exists():
        print_orders(json.loads(path.read_text()))

    _report_remap(rows)


def _rank(row: dict) -> int:
    p = {k: v for k, v in row["probabilities"].items() if k != "__none__"}
    return sorted(p, key=p.get, reverse=True).index(row["label"]) + 1


def two_classes(rows: list[dict]) -> tuple[list[dict], dict]:
    return ([r for r in rows if not r["correct"] and r["label"] in FAILING],
            {lab: sum(1 for r in rows if r["label"] == lab and not r["correct"])
             for lab in FAILING})


def _report_experiment(name: str, result: dict, baseline: list[dict]) -> None:
    print(f"\n== {name} ==")
    if name == "minimal_pairs":
        print_minimal_pairs(result)
        return
    print_probe(result)
    by_text = {r["utterance"]: r for r in baseline}
    pairs = [(by_text[r["utterance"]], r) for r in result["per_row"]
             if r["utterance"] in by_text]
    if not pairs:
        return
    b = sum(1 for old, new in pairs if not old["correct"] and new["correct"])
    c = sum(1 for old, new in pairs if old["correct"] and not new["correct"])
    before = sum(1 for old, _ in pairs if old["correct"])
    after = sum(1 for _, new in pairs if new["correct"])
    print(f"  paired vs baseline: {before}/{len(pairs)} -> {after}/{len(pairs)}  "
          f"fixed {b}, broke {c}, exact McNemar p={mcnemar_exact(b, c):.4f}")
    for lab in sorted({old["label"] for old, _ in pairs}):
        members = [(old, new) for old, new in pairs if old["label"] == lab]
        lo = sum(1 for old, _ in members if old["correct"])
        hi = sum(1 for _, new in members if new["correct"])
        flag = "  <-- REGRESSION" if hi < lo else ("  <-- gain" if hi > lo else "")
        print(f"    {lab:<15} {lo:>2}/{len(members):<2} -> {hi:>2}/{len(members):<2}{flag}")


def _report_remap(rows: list[dict]) -> None:
    """Could the answer alone be repaired? Every static remap, scored."""
    inverse: dict[str, dict[str, int]] = {}
    for r in rows:
        if not r["correct"]:
            inverse.setdefault(r["choice"], {})
            inverse[r["choice"]][r["label"]] = inverse[r["choice"]].get(r["label"], 0) + 1
    print("\n== remap: can the emitted label be repaired on its own? ==")
    for wrong, truths in sorted(inverse.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"  answered {wrong:<15} stands for {truths}")
    scored = []
    for wrong, truths in inverse.items():
        for truth, fixed in truths.items():
            broken = sum(1 for r in rows if r["correct"] and r["choice"] == wrong)
            scored.append((fixed - broken, wrong, truth, fixed, broken))
    scored.sort(reverse=True)
    print("  best static remaps by net rows:")
    for net, wrong, truth, fixed, broken in scored[:5]:
        print(f"    {wrong:<15} -> {truth:<15} +{fixed} fixed / -{broken} broken = {net:+d}")
    print(f"  best achievable net: {scored[0][0]:+d} rows — "
          + ("no static remap pays" if scored[0][0] <= 0 else "a remap exists"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--expect-model", default="d1-3B-Q4_K_M.gguf")
    ap.add_argument("--utterances", default=str(ROOT / "tests/data/utterances_v2.json"))
    ap.add_argument("--baseline", default=str(ROOT / "bench/results/p1-diagnose/baseline_live.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "bench/results/p1-diagnose"))
    ap.add_argument("--only", default="", help="comma-separated experiment names")
    ap.add_argument("--report", action="store_true",
                    help="re-derive the writeup's numbers from the saved artifacts "
                         "(no server traffic)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.report:
        report(out_dir)
        return
    rows = json.loads(Path(args.utterances).read_text())
    rows = [dict(r, idx=i) for i, r in enumerate(rows)]

    with urllib.request.urlopen(args.base_url.rstrip("/") + "/health", timeout=10) as resp:
        health = json.loads(resp.read())
    if health.get("status") != "ok":
        raise SystemExit(f"server unhealthy: {health}")
    with urllib.request.urlopen(args.base_url.rstrip("/") + "/props", timeout=10) as resp:
        props = json.loads(resp.read())
    try:
        served = preflight(props, args.expect_model, "", None, args.base_url + "/props")[0]
        print(f"model assertion: OK — serving {served.get('basename')}", file=sys.stderr)
    except ModelAssertionError as e:
        raise SystemExit(f"ABORT: {e}")

    # Copies: the failing-class probes renumber their rows, and the rewrite pass
    # below indexes the same list against the baseline by the file's own idx.
    failing = [dict(r, idx=i) for i, r in
               enumerate(r for r in rows if r["label"] in FAILING)]

    server = Server(args.base_url)
    wanted = {s.strip() for s in args.only.split(",") if s.strip()}

    def emit(result: dict) -> None:
        path = out_dir / f"{result['experiment']}.json"
        path.write_text(json.dumps(result, indent=2))
        # Merge, not replace: the probes are normally run as several jobs and a
        # summary that only knows about the last job's experiments reads as if
        # the others never ran.
        summary_path = out_dir / "summary.json"
        merged = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        merged.setdefault("experiments", {})[result["experiment"]] = {
            k: v for k, v in result.items() if k != "per_row"}
        merged["generated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        merged["base_url"] = args.base_url
        merged["served"] = served
        summary_path.write_text(json.dumps(merged, indent=2))
        print(f"  wrote {path.name}", file=sys.stderr, flush=True)

    t0 = time.perf_counter()
    if not wanted or "orders" in wanted:
        print(f"== orders: {len(failing)} rows, reversed arm only ==", file=sys.stderr, flush=True)
        base = None
        if Path(args.baseline).exists():
            base = json.loads(Path(args.baseline).read_text())["per_row"]
        result = probe_orders(server, failing, base)
        emit(result)
        print_orders(result)
    if not wanted or "k3" in wanted:
        print(f"== k3: {len(failing)} rows ==", file=sys.stderr, flush=True)
        result = probe_k3(server, failing)
        emit(result)
        print_probe(result)
    if not wanted or "neutral6" in wanted:
        print(f"== neutral6: {len(failing)} rows ==", file=sys.stderr, flush=True)
        result = probe_neutral6(server, failing)
        emit(result)
        print_probe(result)
    if not wanted or "minimal_pairs" in wanted:
        print("== minimal_pairs ==", file=sys.stderr, flush=True)
        result = probe_minimal_pairs(server)
        emit(result)
        print_minimal_pairs(result)
    if not wanted or "rewrite" in wanted:
        print(f"== rewrite: {len(rows)} rows ==", file=sys.stderr, flush=True)
        result = probe_rewrite(server, rows)
        result["instrument"] = "OPERATION_FIRST (this file); frozen OP_DESCRIPTIONS untouched"
        emit(result)
        print_probe(result)
        if Path(args.baseline).exists():
            print_rewrite_vs_baseline(result, json.loads(
                Path(args.baseline).read_text())["per_row"])

    if not wanted or "rewrite2" in wanted:
        print(f"== rewrite2 (targeted): {len(rows)} rows ==", file=sys.stderr, flush=True)
        result = probe_targeted(server, rows)
        result["instrument"] = "TARGETED (this file) — 2 of 8 descriptions changed"
        emit(result)
        print_probe(result)
        if Path(args.baseline).exists():
            print_rewrite_vs_baseline(result, json.loads(
                Path(args.baseline).read_text())["per_row"])

    print(f"\nserver calls: {server.calls}  wall {time.perf_counter() - t0:.0f}s",
          file=sys.stderr)


if __name__ == "__main__":
    main()
