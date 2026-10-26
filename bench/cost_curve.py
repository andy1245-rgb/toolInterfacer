#!/usr/bin/env python3
"""Call-reduction cost curve over a recorded or_bench run.

Answers one question: if a deterministic pre-filter absorbs the first N% of
intents, what does the *routing bill* become and what does accuracy do? The
model numbers come from a recorded run's per-row data, so nothing here is
sheet-price arithmetic — each row's measured cost is the provider's own
`usage.cost`, and each row's correctness is what the model actually answered.

Build: rules are ranked by measured precision on the same 150 rows (in-sample,
stated in the output), then absorbed in that order. Every prefix of the ranking
is a point on the curve; the 20/40/60% targets are read off as the prefix whose
coverage first reaches the target, and the achieved coverage is reported rather
than the target.

Usage:
  python bench/cost_curve.py bench/results/<run>.json [--config NAME] [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prefilter import classify, RULES  # noqa: E402

TARGETS = (0.20, 0.40, 0.60)


def load_config(summary: dict, wanted: str) -> tuple[str, dict]:
    models = summary["models"]
    if wanted:
        if wanted not in models:
            raise SystemExit(f"no config {wanted!r}; have {list(models)}")
        return wanted, models[wanted]
    best = max(models, key=lambda n: models[n]["accuracy"])
    return best, models[best]


def rule_stats(rows: list[dict]) -> dict:
    """Measured coverage and precision of each rule on the recorded rows."""
    stats: dict[str, dict] = {}
    for name, label, tier, _pattern in RULES:
        stats[name] = {"label": label, "tier": tier, "n": 0, "correct": 0}
    for row in rows:
        hit, rule = classify(row["utterance"])
        if not hit:
            continue
        slot = stats[rule]
        slot["n"] += 1
        slot["correct"] += int(hit == row["label"])
    for slot in stats.values():
        slot["precision"] = round(slot["correct"] / slot["n"], 4) if slot["n"] else None
        slot["coverage"] = round(slot["n"] / len(rows), 4) if rows else None
    return stats


def curve(rows: list[dict], stats: dict, spend_per_decision: float) -> list[dict]:
    """Prefix curve over rules ranked by measured precision, high to low."""
    ranked = sorted((s for s in stats.items() if s[1]["n"]),
                    key=lambda kv: (-(kv[1]["precision"] or 0), -(kv[1]["n"] or 0)))
    n_all = len(rows)
    points: list[dict] = []
    for k in range(len(ranked) + 1):
        taken = {name for name, _ in ranked[:k]}
        absorbed = [r for r in rows if classify(r["utterance"])[1] in taken]
        tail = [r for r in rows if classify(r["utterance"])[1] not in taken]
        tail_correct = sum(1 for r in tail if r["correct"])
        absorbed_correct = sum(1 for r in absorbed if classify(r["utterance"])[0] == r["label"])
        cost = spend_per_decision * len(tail)
        points.append({
            "rules": [name for name, _ in ranked[:k]],
            "absorbed_n": len(absorbed),
            "absorbed_share": round(len(absorbed) / n_all, 4),
            "prefilter_precision": round(absorbed_correct / len(absorbed), 4) if absorbed else None,
            "model_calls": len(tail),
            "model_call_share": round(len(tail) / n_all, 4),
            "model_accuracy_on_tail": round(tail_correct / len(tail), 4) if tail else None,
            "blended_accuracy": round((absorbed_correct + tail_correct) / n_all, 4),
            "usd_per_1k_decisions": round(cost / n_all * 1000, 4),
        })
    return points


def at_targets(points: list[dict]) -> list[dict]:
    out = []
    for target in TARGETS:
        pick = next((p for p in points if p["absorbed_share"] >= target), points[-1])
        out.append({"target": target, **pick})
    return out


def report(name: str, summary: dict, entry: dict, as_json: bool) -> dict:
    rows = entry["per_row"]
    n = len(rows)
    # per-decision cost of the accuracy pass only: one request per decision. Older
    # result files predate cost.per_decision_usd, so fall back to total/n.
    per_decision = entry["cost"].get("per_decision_usd")
    if per_decision is None:
        row_costs = [r["cost"] for r in rows if isinstance(r.get("cost"), (int, float))]
        per_decision = (sum(row_costs) / n if row_costs and n else
                        entry["cost"]["total_usd"] / n if n else 0.0)
    stats = rule_stats(rows)
    points = curve(rows, stats, per_decision)
    result = {
        "config": name,
        "source_n_rows": n,
        "measured_usd_total": entry["cost"]["total_usd"],
        "measured_usd_per_decision": round(per_decision, 8),
        "measured_usd_per_1k_decisions": round(per_decision * 1000, 4),
        "measured_accuracy": entry["accuracy"],
        "rules": {k: v for k, v in sorted(
            stats.items(), key=lambda kv: (-(kv[1]["precision"] or 0), kv[0])) if v["n"]},
        "curve": points,
        "at_targets": at_targets(points),
        "note": ("cost of a cascade point = measured per-decision cost x calls it still makes; "
                 "pre-filter precision is in-sample on these rows"),
    }
    if as_json:
        print(json.dumps(result, indent=2))
        return result

    print(f"config {name}: n={n} accuracy={entry['accuracy']:.1%} "
          f"${per_decision * 1000:.4f}/1k decisions measured")
    print(f"\n{'rule':<18}{'tier':>5}{'label':<16}{'n':>4}{'cover':>7}{'precision':>11}")
    for rule, s in result["rules"].items():
        print(f"{rule:<18}{s['tier']:>5}{s['label']:<16}{s['n']:>4}"
              f"{s['coverage'] * 100:>6.1f}%{s['precision'] * 100:>10.1f}%")
    print(f"\n{'absorbed':>9}{'rules':>7}{'pf prec':>9}{'calls':>7}"
          f"{'model acc':>10}{'blended':>9}{'$/1k':>8}")
    for p in points:
        pf = f"{p['prefilter_precision'] * 100:.1f}%" if p["prefilter_precision"] is not None else "-"
        ma = (f"{p['model_accuracy_on_tail'] * 100:.1f}%"
              if p["model_accuracy_on_tail"] is not None else "-")
        print(f"{p['absorbed_share'] * 100:>8.1f}%{len(p['rules']):>7}{pf:>9}"
              f"{p['model_calls']:>7}{ma:>10}{p['blended_accuracy'] * 100:>8.1f}%"
              f"{p['usd_per_1k_decisions']:>8.4f}")
    print("\nat the named targets (achieved coverage, not the target):")
    for p in result["at_targets"]:
        pf = f"{p['prefilter_precision'] * 100:.1f}%" if p["prefilter_precision"] is not None else "-"
        print(f"  {p['target'] * 100:>3.0f}% -> absorbed {p['absorbed_share'] * 100:5.1f}% "
              f"at {pf:>6} precision, model calls {p['model_calls']:>3}"
              f" = ${p['usd_per_1k_decisions']:.4f}/1k, "
              f"blended acc {p['blended_accuracy'] * 100:.1f}%")
    print(f"\n{result['note']}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result", help="bench/results/or-bench-*.json")
    ap.add_argument("--config", default="", help="config key inside the run (default: best accuracy)")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args()
    summary = json.loads(Path(args.result).read_text())
    name, entry = load_config(summary, args.config)
    report(name, summary, entry, args.json)


if __name__ == "__main__":
    main()
