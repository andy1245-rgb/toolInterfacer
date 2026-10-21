#!/usr/bin/env python3
"""P2 live routing probe — one `choice` question per utterance against a live
llama.cpp /v1/systemone server, measured for latency + accuracy vs labels.

Usage: python bench/live_probe.py [--limit N] [--threads T] [--backend NAME]
       [--base-url URL] [--host NAME] [--utterances tests/data/utterances.json]
       [--expect-model d1-3B-Q4_K_M.gguf]
       [--out bench/results/live_probe_summary.json]

--expect-model is the served-model guard: /health answers "ok" for any
llama-server, so without it a stale server on the port is measured silently
(see docs/model-sweep-consolidated.md, CORRECTION). When given, /props is
asserted against the expected basename and the run aborts before any probe
traffic on a mismatch.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_guard import ModelAssertionError, preflight  # noqa: E402
from ti.decide.resolve import resolve  # noqa: E402
from ti.conf import load  # noqa: E402

INSTRUCTIONS = "Which single operation should handle the user command?"

OP_DESCRIPTIONS = {
    "filesys_read": "Read one file's contents from disk.",
    "filesys_glob": "List files matching a name pattern (paths only, no content search).",
    "filesys_grep": "Search inside file contents for a regular expression.",
    "decide_pick": "Pick the best item(s) from a candidate list.",
    "decide_extract": "Extract query strings or parameter values from the user command.",
    "web_search": "Search the web by keywords.",
    "web_fetch": "Fetch one URL's full contents.",
    "reddit_digest": "Summarize a Reddit subreddit or thread.",
}


def percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = (len(sorted_vals) - 1) * pct
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="run only the first N rows (0 = all)")
    ap.add_argument("--utterances", default=str(ROOT / "tests/data/utterances.json"))
    ap.add_argument("--out", default=str(ROOT / "bench/results/live_probe_summary.json"))
    ap.add_argument("--threads", type=int, default=8, help="server -t value, recorded in the summary")
    ap.add_argument("--backend", default="d1_systemone",
                    choices=["d1_systemone", "logit_read", "heuristic_micro"],
                    help="decide backend resolved through ti.decide.resolve")
    ap.add_argument("--base-url", default="",
                    help="override conf base_url for the backend (e.g. ssh-tunnel address)")
    ap.add_argument("--host", default="localhost", help="label recorded in the summary")
    ap.add_argument("--expect-model", default="",
                    help="gguf basename the served model must match; a mismatch aborts the "
                         "run before any probe traffic (no assertion is made if omitted)")
    ap.add_argument("--expect-sha", default="", help="optional served-model sha to assert")
    ap.add_argument("--expect-size", type=int, default=None,
                    help="optional served-model byte size to assert")
    return ap


def guard_served_model(props: dict, expect_model: str, expect_sha: str = "",
                       expect_size: int | None = None, source: str = "/props",
                       stream=None) -> dict:
    """Abort (SystemExit) on a /props mismatch; warn loudly when unasserted."""
    stream = stream if stream is not None else sys.stderr
    try:
        served, warning = preflight(props, expect_model, expect_sha, expect_size, source)
    except ModelAssertionError as e:
        raise SystemExit(f"ABORT: {e}")
    if warning:
        print(warning, file=stream)
    return served


def main() -> None:
    args = build_parser().parse_args()

    cfg = load()
    if args.base_url:
        cfg["decide"]["d1_systemone"]["base_url"] = args.base_url
        cfg["decide"].setdefault("logit_read", {})["base_url"] = args.base_url
    cfg["decide"]["backends_in_use"] = {"choice": args.backend, "noul": args.backend}
    base_url = args.base_url or cfg["decide"].get(args.backend, {}).get("base_url", "")
    backend = resolve("choice")
    rows = json.loads(Path(args.utterances).read_text())
    if args.limit:
        rows = rows[: args.limit]

    if args.backend == "heuristic_micro":
        props = {"model_path": "local/ti/decide/backends/heuristic_micro.py",
                 "ftype": "n/a", "build_info": "n/a"}
        served = {}
        if args.expect_model:
            print("note: heuristic_micro is in-process — --expect-model does not apply",
                  file=sys.stderr)
        else:
            guard_served_model(props, "")
    else:
        with urllib.request.urlopen(base_url.rstrip("/") + "/health", timeout=10) as resp:
            health = json.loads(resp.read())
        if health.get("status") != "ok":
            raise SystemExit(f"server unhealthy: {health}")
        with urllib.request.urlopen(base_url.rstrip("/") + "/props", timeout=10) as resp:
            props = json.loads(resp.read())
        served = guard_served_model(props, args.expect_model, args.expect_sha,
                                    args.expect_size, base_url.rstrip("/") + "/props")

    per_row, malformed, latencies, abstains = [], 0, [], 0
    for i, row in enumerate(rows):
        t0 = time.perf_counter()
        try:
            ans = backend.choice(row["utterance"], INSTRUCTIONS, OP_DESCRIPTIONS)
        except Exception as e:
            per_row.append({"idx": i, "utterance": row["utterance"], "label": row["label"],
                            "choice": None, "correct": False, "conf": 0.0, "probabilities": {},
                            "ms": (time.perf_counter() - t0) * 1000, "error": str(e)})
            malformed += 1
            continue
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)
        if ans is None:  # deliberate abstain (no opinion — caller must fall back)
            per_row.append({"idx": i, "utterance": row["utterance"], "label": row["label"],
                            "choice": None, "correct": False, "conf": 0.0, "probabilities": {},
                            "ms": ms, "error": "abstain (no opinion)"})
            malformed += 1
            abstains += 1
            continue
        conf = float(ans.confidence)
        probs = {k: float(v) for k, v in ans.probabilities.items()}
        choice = ans.choice
        correct = bool(choice == row["label"])
        per_row.append({"idx": i, "utterance": row["utterance"], "label": row["label"],
                        "choice": choice, "correct": correct, "conf": conf, "probabilities": probs,
                        "ms": ms, "error": None if choice else "missing choice"})
        if not choice:
            malformed += 1

    lat_sorted = sorted(latencies)
    correct_rows = [r for r in per_row if r["correct"]]
    by_op = {}
    for r in per_row:
        s = by_op.setdefault(r["label"], {"n": 0, "correct": 0})
        s["n"] += 1
        s["correct"] += int(r["correct"])
    conf_bands = []
    for lo in [i / 10 for i in range(10)]:
        in_band = [r for r in per_row if lo <= r["conf"] < lo + 0.1]
        bands_hit = [r for r in in_band if r["correct"]]
        conf_bands.append({"band": round(lo, 1), "n": len(in_band),
                           "accuracy": round(len(bands_hit) / len(in_band), 4) if in_band else None,
                           "mean_conf": round(statistics.fmean(r["conf"] for r in in_band), 4)
                           if in_band else None})

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "backend": args.backend,
        "host": args.host,
        "base_url": base_url,
        "server_threads": args.threads,
        "model": {"path": props.get("model_path"),
                  "ftype": props.get("model_ftype"),
                  "build_info": (props.get("build_info") or "")},
        "model_assertion": {
            "expect_model": args.expect_model,
            "checked": bool(args.expect_model) and args.backend != "heuristic_micro",
            "served_basename": served.get("basename", "")},
        "n_rows": len(rows),
        "malformed_responses": malformed - abstains,
        "abstains": abstains,
        "accuracy": round(len(correct_rows) / len(rows), 4),
        "latency_ms": {
            "p50": round(percentile(lat_sorted, 0.5), 1),
            "p95": round(percentile(lat_sorted, 0.95), 1),
            "mean": round(statistics.fmean(latencies), 1) if latencies else 0.0,
            "min": round(lat_sorted[0], 1) if lat_sorted else 0.0,
            "max": round(lat_sorted[-1], 1) if lat_sorted else 0.0,
        },
        "accuracy_by_op": {op: {"n": s["n"],
                                "accuracy": round(s["correct"] / s["n"], 4)}
                           for op, s in sorted(by_op.items())},
        "confidence_calibration": conf_bands,
        "conf_distribution": {"min": round(min(r["conf"] for r in per_row), 4),
                              "mean": round(statistics.fmean(r["conf"] for r in per_row), 4),
                              "max": round(max(r["conf"] for r in per_row), 4)},
        "per_row": per_row,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    if args.expect_model:
        print(f"model assertion: OK — serving {served.get('basename')} "
              f"(expected {args.expect_model})")
    print(f"n={summary['n_rows']} accuracy={summary['accuracy']:.1%} "
          f"malformed={malformed} p50={summary['latency_ms']['p50']}ms "
          f"p95={summary['latency_ms']['p95']}ms (threads={args.threads})")
    print(f"conf: min={summary['conf_distribution']['min']:.2f} "
          f"mean={summary['conf_distribution']['mean']:.2f} "
          f"max={summary['conf_distribution']['max']:.2f}")
    for r in per_row:
        mark = "ok " if r["correct"] else ("mal" if r.get("error") else "miss")
        print(f"  {r['idx']:>3} {r['ms']:>8.1f}ms conf={r['conf']:.3f} {mark} "
              f"{str(r['choice']):<16} <- {r['utterance'][:58]}")
    if len(correct_rows) < len(rows):
        print(f"\n== wrong ({len(rows) - len(correct_rows)}) ==")
        for r in per_row:
            if not r["correct"]:
                print(f"  {r['idx']:>3} want={r['label']:<16} got={r['choice']} "
                      f"conf={r['conf']:.2f} | {r['utterance']}")


if __name__ == "__main__":
    main()
