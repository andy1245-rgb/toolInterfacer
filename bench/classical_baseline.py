#!/usr/bin/env python3
"""classical_baseline.py — measure the `classical` backend the way the sibling
router benchmark measured it, plus the checks that number needed.

The recorded result (jevResearchCompiled `exp_classic.py`, model
`tfidf-word+linearsvc`) is 93.19% / none-detection 0.60 / p50 0.61 ms on 382
utterances, 18-way. This script reproduces that protocol in-repo and adds the
four things it did not report:

  * provenance arithmetic — can 93.19% and 0.60 be expressed on 382/25 rows;
  * overlap — does any of the 382 appear verbatim in the training references;
  * near/far decomposition — is the score surface-matching or generalising;
  * template-grouped vs random CV — how much of the score is template reuse.

It also runs the backend on this repo's own frozen 150-row set
(`tests/data/utterances_v2.json`), 8-way, which is a different task: the 18-way
set has a `__none__` class and the v2 set has none.

Usage: python3 bench/classical_baseline.py [--out-dir bench/results]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from live_probe import INSTRUCTIONS, OP_DESCRIPTIONS  # noqa: E402
from ti.conf import load  # noqa: E402
from ti.decide.backends.classical import ClassicalBackend, load_corpus, resolve_path  # noqa: E402
from ti.decide.linear_text import TfidfVectorizer, fit  # noqa: E402

ROUTING_382 = ROOT / "tests/data/routing_382.json"
REGISTRY_382 = ROOT / "bench/registry_382.json"
UTTERANCES_V2 = ROOT / "tests/data/utterances_v2.json"
UTTERANCES_V1 = ROOT / "tests/data/utterances.json"
NONE = "__none__"


def percentile(vals: list[float], pct: float) -> float:
    if not vals:
        return 0.0
    ordered = sorted(vals)
    idx = (len(ordered) - 1) * pct
    lo, hi = int(idx), min(int(idx) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)


def evaluate(backend: ClassicalBackend, rows: list[dict], text_key: str,
             label_key: str, criteria: dict[str, str]) -> dict:
    """Run one `choice` call per row and score it. `None` means abstain."""
    per_row, latencies = [], []
    for row in rows:
        text, gold = row[text_key], row[label_key]
        t0 = time.perf_counter()
        answer = backend.choice(text, INSTRUCTIONS, criteria)
        latencies.append((time.perf_counter() - t0) * 1000)
        pick = answer.choice if answer.choice is not None else NONE
        per_row.append({"text": text, "gold": gold, "pick": pick,
                        "correct": pick == gold, "conf": answer.confidence,
                        "ms": round(latencies[-1], 3),
                        "pmax": max(answer.probabilities.values()) if answer.probabilities else 0.0})

    correct = sum(r["correct"] for r in per_row)
    none_rows = [r for r in per_row if r["gold"] == NONE]
    tool_rows = [r for r in per_row if r["gold"] != NONE]
    by_label: dict[str, dict] = {}
    for r in per_row:
        slot = by_label.setdefault(r["gold"], {"n": 0, "correct": 0})
        slot["n"] += 1
        slot["correct"] += int(r["correct"])
    return {
        "n": len(per_row),
        "accuracy": round(correct / len(per_row), 4),
        "correct": correct,
        "none_detection": (round(sum(r["pick"] == NONE for r in none_rows) / len(none_rows), 3)
                           if none_rows else None),
        "false_none_rate": (round(sum(r["pick"] == NONE for r in tool_rows) / len(tool_rows), 3)
                            if tool_rows else None),
        "latency_p50_ms": round(percentile(latencies, 0.5), 3),
        "latency_p95_ms": round(percentile(latencies, 0.95), 3),
        "confidence_p50": round(statistics.median(r["conf"] for r in per_row), 4),
        "pmax_p50": round(statistics.median(r["pmax"] for r in per_row), 4),
        "accuracy_by_label": {k: round(v["correct"] / v["n"], 3)
                              for k, v in sorted(by_label.items())},
        "per_row": per_row,
    }


def template_key(row: dict, templates: dict[str, list[str]]) -> str:
    """Group a generated row back to the template that produced it.

    gen_data.py builds each test row by substituting slots into a registry
    template, so a row belongs to the template whose literal prefix and suffix it
    still carries. Rows that match nothing get a per-label bucket, which can only
    make the grouped split more optimistic, never less.
    """
    for template in templates.get(row["tool"], []):
        head, _, tail = template.partition("{")
        suffix = tail.split("}")[-1] if "}" in tail else ""
        if row["text"].startswith(head) and (not suffix or row["text"].endswith(suffix)):
            return f"{row['tool']}::{template}"
    return f"{row['tool']}::unmatched"


def grouped_folds(keys: list[str], k: int, seed: int = 42) -> list[list[int]]:
    """Deterministic group split: whole groups land in one fold."""
    import random
    rng = random.Random(seed)
    uniq = sorted(set(keys))
    rng.shuffle(uniq)
    fold_of = {key: i % k for i, key in enumerate(uniq)}
    folds: list[list[int]] = [[] for _ in range(k)]
    for i, key in enumerate(keys):
        folds[fold_of[key]].append(i)
    return folds


def cross_validate(rows: list[dict], text_key: str, label_key: str, criteria: dict[str, str],
                   fold_keys: list[str], k: int, cfg: dict) -> dict:
    """5-fold CV over `rows` alone — a supervision ceiling, not a transfer number."""
    folds = grouped_folds(fold_keys, k)
    correct = total = 0
    for f in range(k):
        held = set(folds[f])
        train = [r for i, r in enumerate(rows) if i not in held]
        backend = ClassicalBackend(
            config=dict(cfg, corpus_path="(cv)"),
            corpus=[(r[text_key], r[label_key]) for r in train])
        for i in folds[f]:
            answer = backend.choice(rows[i][text_key], INSTRUCTIONS, criteria)
            pick = answer.choice if answer.choice is not None else NONE
            correct += pick == rows[i][label_key]
            total += 1
    return {"n": total, "accuracy": round(correct / total, 4), "correct": correct, "k": k}


def nearest_reference_similarity(corpus: list[tuple[str, str]],
                                 rows: list[dict], text_key: str) -> list[float]:
    """Max TF-IDF cosine from each row to any training reference."""
    texts = [t for t, _ in corpus]
    vec = TfidfVectorizer().fit(texts)
    refs = [({i: v for i, v in zip(idx, vals)})
            for idx, vals in vec.transform(texts)]
    out = []
    for row in rows:
        idx, vals = vec.vectorize(row[text_key])
        query = dict(zip(idx, vals))
        out.append(max((sum(w * query.get(i, 0.0) for i, w in ref.items()) for ref in refs),
                       default=0.0))
    return out


def bucket_report(per_row: list[dict], sims: list[float]) -> dict:
    """Accuracy split by how close the row sits to a training reference."""
    out: dict[str, dict] = {}
    for lo, name in [(0.0, "far   (<0.80)"), (0.8, "near  (>=0.80)")]:
        hi = 0.8 if lo == 0.0 else 1.01
        sel = [r for r, s in zip(per_row, sims) if lo <= s < hi]
        out[name] = {"n": len(sel),
                     "accuracy": round(sum(r["correct"] for r in sel) / len(sel), 4) if sel else None}
    return out


def band_breakdown(per_row: list[dict], rows: list[dict]) -> dict:
    """Per-band accuracy — the shape `bench/band_summary.py` reports for the LLM lanes."""
    out: dict[str, dict] = {}
    for row, scored in zip(rows, per_row):
        band = row.get("band")
        if band is None:
            continue
        slot = out.setdefault(band, {"n": 0, "correct": 0})
        slot["n"] += 1
        slot["correct"] += int(scored["correct"])
    return {b: {"n": s["n"], "accuracy": round(s["correct"] / s["n"], 4)}
            for b, s in sorted(out.items())}


def coverage_curve(per_row: list[dict], gate: float) -> dict:
    """Accuracy on the rows a confidence gate would commit to.

    The gate is `policy.gate_route`, expressed in p_max units, so this is the
    risk/coverage point the decision actually rides on: below the gate the hop
    escalates or asks back, above it the backend's answer is used.
    """
    curve = {}
    for threshold in [round(0.1 * i, 2) for i in range(1, 10)]:
        kept = [r for r in per_row if r["conf"] >= threshold]
        curve[f"{threshold:.2f}"] = {
            "coverage": round(len(kept) / len(per_row), 3),
            "precision": round(sum(r["correct"] for r in kept) / len(kept), 4) if kept else None,
        }
    kept = [r for r in per_row if r["conf"] >= gate]
    curve["shipped_gate"] = {
        "gate": gate,
        "coverage": round(len(kept) / len(per_row), 3),
        "precision": round(sum(r["correct"] for r in kept) / len(kept), 4) if kept else None,
    }
    return curve


def cascade_with_d1(d1_runs: list[dict], classical: dict) -> dict:
    """Classical as a front gate in front of d1-3B, on the same 150 rows.

    A row the classical backend commits to confidently never reaches the 3B; the
    rest fall through to d1's recorded answer for that row. Thresholds are swept
    rather than picked, because any single one would be chosen on this set.
    """
    by_threshold = {}
    for threshold in [0.0, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]:
        accs, hits, d1_ms, gate_ms = [], 0, [], []
        fixed = broke = 0
        for run in d1_runs:
            correct = 0
            for c, d in zip(classical["per_row"], run["per_row"]):
                gated = c["conf"] >= threshold and c["pick"] != NONE
                pick = c["pick"] if gated else d["choice"]
                correct += pick == c["gold"]
                # paired evidence, counted only on the first run to keep it one
                # row per utterance: rows the cascade fixes vs. rows it breaks
                if run is d1_runs[0]:
                    fixed += (pick == c["gold"]) and not d["correct"]
                    broke += (pick != c["gold"]) and d["correct"]
                if run is d1_runs[0]:
                    (gate_ms if gated else d1_ms).append(c["ms"] if gated else d["ms"])
            accs.append(correct / len(classical["per_row"]))
            hits = (sum(1 for c in classical["per_row"]
                        if c["conf"] >= threshold and c["pick"] != NONE))
        by_threshold[f"{threshold:.2f}"] = {
            "accuracy": round(sum(accs) / len(accs), 4),
            "accuracy_per_run": [round(a, 4) for a in accs],
            "gate_coverage": round(hits / len(classical["per_row"]), 3),
            "rows_fixed_vs_d1": fixed, "rows_broken_vs_d1": broke,
            "mean_ms": round((sum(d1_ms) + sum(gate_ms)) / (len(d1_ms) + len(gate_ms)), 1),
        }
    return by_threshold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "bench/results"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(load()["decide"]["classical"])
    rows_382 = json.loads(ROUTING_382.read_text())
    registry = json.loads(REGISTRY_382.read_text())
    corpus = load_corpus(REGISTRY_382)
    templates = {t["id"]: t["templates"] for t in registry["tools"]}

    # -- provenance --------------------------------------------------------
    recorded = {"accuracy": 0.9319, "none_detection": 0.60, "latency_p50_ms": 0.61}
    train_texts = {t for t, _ in corpus}
    none_rows = [r for r in rows_382 if r["tool"] == NONE]
    provenance = {
        "recorded": recorded,
        "recorded_arithmetic": {
            "356/382": round(356 / 382, 4),
            "15/25": round(15 / 25, 3),
            "matches": round(356 / 382, 4) == recorded["accuracy"] and 15 / 25 == recorded["none_detection"],
        },
        "train_rows": len(corpus),
        "train_classes": len({label for _, label in corpus}),
        "test_rows": len(rows_382),
        "test_classes": len({r["tool"] for r in rows_382}),
        "none_rows_in_test": len(none_rows),
        "verbatim_train_test_overlap": sum(r["text"] in train_texts for r in rows_382),
    }

    # -- reproduction: fit the 200 registry references, score the 382 ------
    criteria_382 = {label: label for _, label in corpus}
    backend = ClassicalBackend(config=cfg, corpus=corpus)
    t0 = time.perf_counter()
    model = backend.model()
    fit_seconds = round(time.perf_counter() - t0, 2)
    reproduction = evaluate(backend, rows_382, "text", "tool", criteria_382)
    reproduction["fit_seconds"] = fit_seconds
    reproduction["model_fingerprint"] = model.fingerprint
    reproduction["config"] = cfg

    sims = nearest_reference_similarity(corpus, rows_382, "text")
    reproduction["by_reference_distance"] = bucket_report(reproduction["per_row"], sims)
    reproduction["coverage_curve"] = coverage_curve(reproduction["per_row"],
                                                    load()["policy"]["gate_route"])
    reproduction["reference_distance"] = {
        "median": round(statistics.median(sims), 3),
        "min": round(min(sims), 3),
        "max": round(max(sims), 3),
        "at_or_above_0.95": sum(s >= 0.95 for s in sims),
    }

    # -- in-sample contrast: fit the 382 and score the same 382 ------------
    in_sample_backend = ClassicalBackend(
        config=dict(cfg, corpus_path="(in-sample)"),
        corpus=[(r["text"], r["tool"]) for r in rows_382])
    in_sample = evaluate(in_sample_backend, rows_382, "text", "tool", criteria_382)

    # -- template-grouped vs random CV over the 382 ------------------------
    group_keys = [template_key(r, templates) for r in rows_382]
    random_keys = [f"row::{i}" for i in range(len(rows_382))]
    r382 = {
        "reproduction": reproduction,
        "in_sample_on_382": {k: v for k, v in in_sample.items() if k != "per_row"},
        "cv_template_grouped": cross_validate(rows_382, "text", "tool", criteria_382,
                                              group_keys, 5, cfg),
        "cv_random_rows": cross_validate(rows_382, "text", "tool", criteria_382,
                                         random_keys, 5, cfg),
        "template_groups": len(set(group_keys)),
        "provenance": provenance,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_dir / "classical_382.json").write_text(json.dumps(r382, indent=1))

    # -- this repo's own frozen 150-row set -------------------------------
    rows_v2 = json.loads(UTTERANCES_V2.read_text())
    rows_v1 = json.loads(UTTERANCES_V1.read_text())
    criteria_v2 = dict(OP_DESCRIPTIONS)
    v2_backend = ClassicalBackend(
        config=dict(cfg, corpus_path="tests/data/utterances.json"),
        corpus=[(r["utterance"], r["label"]) for r in rows_v1])
    v2 = evaluate(v2_backend, rows_v2, "utterance", "label", criteria_v2)
    pair_keys = [("pair::" + "|".join(sorted((r["label"], r["tier_target"])))
                  if r.get("tier_target") else f"row::{i}")
                 for i, r in enumerate(rows_v2)]
    d1_runs = [json.loads(p.read_text()) for p in sorted(
        (ROOT / "bench/results/dell").glob("v2-d1-3B-Q4-systemone-props-asserted*.json"))]
    result_v2 = {
        "train": {"source": "tests/data/utterances.json", "n": len(rows_v1),
                  "classes": len({r["label"] for r in rows_v1})},
        "transfer_v1_to_v2": {k: v for k, v in v2.items() if k != "per_row"},
        "transfer_by_band": band_breakdown(v2["per_row"], rows_v2),
        "transfer_coverage_curve": coverage_curve(v2["per_row"], load()["policy"]["gate_route"]),
        "cv_v2_grouped": cross_validate(rows_v2, "utterance", "label", criteria_v2,
                                        pair_keys, 5, cfg),
        "cv_v2_random": cross_validate(rows_v2, "utterance", "label", criteria_v2,
                                       [f"row::{i}" for i in range(len(rows_v2))], 5, cfg),
        "incumbent_d1_3B_systemone": {"accuracy": 0.740, "n": 150,
                                      "bands": {"t1": 0.812, "t2": 0.710, "t3": 0.673},
                                      "source": "docs/model-sweep-consolidated.md"},
        "cascade_gate_then_d1": (cascade_with_d1(d1_runs, v2) if d1_runs else {}),
        "cascade_source_files": [p.name for p in sorted(
            (ROOT / "bench/results/dell").glob("v2-d1-3B-Q4-systemone-props-asserted*.json"))],
        "config": cfg,
        "model_fingerprint": v2_backend.model().fingerprint,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_dir / "classical_v2.json").write_text(json.dumps(result_v2, indent=1))

    # -- report ------------------------------------------------------------
    rec = provenance["recorded"]
    print(f"# provenance: {provenance['train_rows']} train refs, "
          f"{provenance['test_rows']} test rows, verbatim overlap "
          f"{provenance['verbatim_train_test_overlap']}, "
          f"356/382={provenance['recorded_arithmetic']['356/382']}")
    print(f"\n# 382-row reproduction (fit {fit_seconds}s, fp {model.fingerprint[:16]})")
    print(f"{'':<26}{'recorded':>10}{'in-repo':>10}")
    print(f"{'accuracy':<26}{rec['accuracy']:>10.4f}"
          f"{reproduction['accuracy']:>10.4f}   ({reproduction['correct']}/{reproduction['n']})")
    print(f"{'none-detection':<26}{rec['none_detection']:>10.2f}"
          f"{reproduction['none_detection']:>10.2f}")
    print(f"{'false-none rate':<26}{0.0:>10.2f}{reproduction['false_none_rate']:>10.2f}")
    print(f"{'p50 latency (ms)':<26}{rec['latency_p50_ms']:>10.2f}"
          f"{reproduction['latency_p50_ms']:>10.2f}")
    print(f"\n# reference distance: median {reproduction['reference_distance']['median']}, "
          f">=0.95 on {reproduction['reference_distance']['at_or_above_0.95']} rows")
    for name, slot in reproduction["by_reference_distance"].items():
        print(f"  {name}: n={slot['n']:<4} accuracy={slot['accuracy']}")
    gate = reproduction["coverage_curve"]["shipped_gate"]
    print(f"# gate_route={gate['gate']}: coverage {gate['coverage']:.1%}, "
          f"precision on committed rows {gate['precision']}")
    print(f"\n# in-sample on the 382:  {in_sample['accuracy']:.4f} "
          f"({in_sample['correct']}/{in_sample['n']})")
    print(f"# CV on 382, template-grouped ({r382['template_groups']} groups): "
          f"{r382['cv_template_grouped']['correct']}/{r382['cv_template_grouped']['n']}"
          f" = {r382['cv_template_grouped']['accuracy']:.4f}")
    print(f"# CV on 382, random rows:              "
          f"{r382['cv_random_rows']['correct']}/{r382['cv_random_rows']['n']}"
          f" = {r382['cv_random_rows']['accuracy']:.4f}")
    print(f"\n# v2 150-row set, trained on v1 ({len(rows_v1)} rows): "
          f"{v2['correct']}/{v2['n']} = {v2['accuracy']:.1%}"
          f"  (d1-3B systemone: 74.0%)")
    for band, slot in result_v2["transfer_by_band"].items():
        print(f"  {band}: n={slot['n']:<4} accuracy={slot['accuracy']:.1%}")
    print(f"# v2 CV, boundary-pair grouped: "
          f"{result_v2['cv_v2_grouped']['correct']}/{result_v2['cv_v2_grouped']['n']}"
          f" = {result_v2['cv_v2_grouped']['accuracy']:.1%}")
    print(f"# v2 CV, random rows:           "
          f"{result_v2['cv_v2_random']['correct']}/{result_v2['cv_v2_random']['n']}"
          f" = {result_v2['cv_v2_random']['accuracy']:.1%}")
    if result_v2["cascade_gate_then_d1"]:
        print("\n# classical gate in front of d1-3B, same 150 rows (threshold swept):")
        for threshold, slot in result_v2["cascade_gate_then_d1"].items():
            print(f"  gate>={threshold}: cascade {slot['accuracy']:.1%} "
                  f"(d1 alone 74.0%) at {slot['gate_coverage']:.0%} coverage, "
                  f"mean {slot['mean_ms']:.0f} ms")
    print(f"\nwrote {out_dir / 'classical_382.json'}\nwrote {out_dir / 'classical_v2.json'}")


if __name__ == "__main__":
    main()
