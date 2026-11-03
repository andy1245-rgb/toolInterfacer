#!/usr/bin/env python3
"""Interleaved A/B probe — two llama-server arms, one frozen utterance set.

`bench/live_probe.py` measures one arm per run, so two sequential runs let a
load or thermal swing on this box masquerade as a backend difference. This
probe interleaves instead: every utterance is asked of every arm inside the
same tight window, and the arm order rotates per block, so any monotonic drift
lands on both arms rather than on whichever ran second. The paired per-row
deltas it records (same row, both arms, seconds apart) are the actual result;
the per-arm percentiles are reported alongside them.

The probe bypasses `ti.decide.resolve`, exactly like `bench/results/
ab-interleaved-30-run2.json` did: the `choice` path there wraps the backend in
a regex pre-gate and a (state, instructions, criteria)-keyed decision cache, and
under interleaving a hit in the second arm would report cache latency as model
latency. Each arm is therefore a bare `d1_systemone` backend with its own
base_url — same prompt, same kwargs, no gate, no cache.

The served model is asserted on every arm via model_guard before any probe
traffic (`--expect-model`), because /health answers "ok" for any llama-server
and a stale server on a port would be measured silently.

Usage:
  python3 bench/probe_interleaved.py \
      --arm cpu=http://127.0.0.1:8080 --arm vulkan=http://127.0.0.1:8081 \
      --expect-model d1-3B-Q4_K_M.gguf \
      --out bench/results/igpu-vulkan-v1-$(date +%F).json
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

from live_probe import INSTRUCTIONS, OP_DESCRIPTIONS, guard_served_model, percentile  # noqa: E402
from ti.decide.primitives import make_backend  # noqa: E402

BACKEND = "d1_systemone"


def parse_arm(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--arm wants LABEL=URL, got {spec!r}")
    label, url = spec.split("=", 1)
    return label.strip(), url.strip().rstrip("/")


def fetch_json(url: str, payload: dict | None = None, timeout: int = 120) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def preflight_arm(label: str, url: str, expect_model: str, expect_sha: str,
                  expect_size: int | None) -> dict:
    """Assert the served model on one arm; abort the whole run on a mismatch."""
    health = fetch_json(url + "/health", timeout=10)
    if health.get("status") != "ok":
        raise SystemExit(f"arm {label}: server unhealthy: {health}")
    props = fetch_json(url + "/props", timeout=10)
    try:
        served = guard_served_model(props, expect_model, expect_sha, expect_size,
                                    url + "/props")
    except SystemExit as e:
        raise SystemExit(f"arm {label}: {e}")
    print(f"arm {label}: {url} -> {served.get('basename')} "
          f"(ftype={props.get('model_ftype')} build={(props.get('build_info') or '')[:60]})",
          file=sys.stderr)
    return {"label": label, "url": url, "model_path": props.get("model_path"),
            "ftype": props.get("model_ftype"), "build_info": props.get("build_info")}


def ask(backend, row: dict) -> tuple[float, object, bool]:
    t0 = time.perf_counter()
    ans = backend.choice(row["utterance"], INSTRUCTIONS, OP_DESCRIPTIONS)
    ms = (time.perf_counter() - t0) * 1000
    choice = None if ans is None else ans.choice
    conf = 0.0 if ans is None else float(ans.confidence)
    return ms, (choice, conf), bool(choice == row["label"])


def throughput(backend_url: str, prompt: str, n_predict: int, reps: int = 3) -> dict:
    """Prompt-processing vs generation rate from llama.cpp's own timings block."""
    prompt_ps, eval_ps, samples = [], [], []
    for _ in range(reps):
        out = fetch_json(backend_url + "/completion",
                         {"prompt": prompt, "n_predict": n_predict,
                          "temperature": 0.0, "cache_prompt": False})
        t = out.get("timings") or {}
        samples.append({k: t.get(k) for k in
                        ("prompt_n", "prompt_ms", "prompt_per_second",
                         "predicted_n", "predicted_ms", "predicted_per_second")})
        if t.get("prompt_per_second"):
            prompt_ps.append(float(t["prompt_per_second"]))
        if t.get("predicted_per_second"):
            eval_ps.append(float(t["predicted_per_second"]))
    return {"n_predict": n_predict,
            "prompt_tokens": samples[0]["prompt_n"] if samples else None,
            "prompt_tokens_per_second": round(statistics.median(prompt_ps), 1) if prompt_ps else None,
            "predicted_tokens_per_second": round(statistics.median(eval_ps), 1) if eval_ps else None,
            "samples": samples}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, metavar="LABEL=URL",
                    help="an arm to measure; pass twice (or more) for an A/B")
    ap.add_argument("--utterances", default=str(ROOT / "tests/data/utterances.json"))
    ap.add_argument("--limit", type=int, default=0, help="run only the first N rows (0 = all)")
    ap.add_argument("--out", default=str(ROOT / "bench/results/probe_interleaved.json"))
    ap.add_argument("--block", type=int, default=5,
                    help="rows per interleave block before the lead arm rotates (default 5)")
    ap.add_argument("--warmup", type=int, default=2,
                    help="calls discarded per arm before measurement (default 2)")
    ap.add_argument("--expect-model", default="",
                    help="gguf basename every arm must be serving (asserted via /props)")
    ap.add_argument("--expect-sha", default="")
    ap.add_argument("--expect-size", type=int, default=None)
    ap.add_argument("--throughput", action="store_true",
                    help="also measure prompt-processing and generation tokens/s per arm")
    ap.add_argument("--throughput-reps", type=int, default=3)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    arms = [parse_arm(s) for s in args.arm]
    if len(arms) < 2:
        raise SystemExit("need at least two --arm specs for an interleaved comparison")
    if not args.expect_model:
        print("!! --expect-model not set: no served-model assertion (see bench/model_guard.py)",
              file=sys.stderr)

    rows = json.loads(Path(args.utterances).read_text())
    if args.limit:
        rows = rows[: args.limit]

    served = {label: preflight_arm(label, url, args.expect_model, args.expect_sha,
                                   args.expect_size)
              for label, url in arms}

    backends = {}
    for label, url in arms:
        be = make_backend(BACKEND)
        be.base_url = url        # per-arm target; resolve() deliberately not used
        backends[label] = be

    order = [label for label, _ in arms]
    warmup_ms = {label: [] for label in order}
    for i in range(args.warmup):
        row = rows[i % len(rows)]
        for label in order:
            ms, _, _ = ask(backends[label], row)
            warmup_ms[label].append(round(ms, 1))

    per_row = []
    for start in range(0, len(rows), args.block):
        block = list(range(start, min(start + args.block, len(rows))))
        rotate = (start // args.block) % len(order)
        block_order = order[rotate:] + order[:rotate]
        for idx in block:
            row = rows[idx]
            for label in block_order:
                ms, (choice, conf), correct = ask(backends[label], row)
                per_row.append({"arm": label, "idx": idx, "utterance": row["utterance"],
                                "label": row["label"], "choice": choice, "conf": conf,
                                "correct": correct, "ms": round(ms, 1)})
        print(f"  rows {start + 1}-{block[-1] + 1}/{len(rows)} "
              f"(lead arm: {block_order[0]})", file=sys.stderr)

    per_arm = {}
    for label in order:
        got = [r for r in per_row if r["arm"] == label]
        lat = sorted(r["ms"] for r in got)
        per_arm[label] = {
            "n": len(got),
            "p50_ms": round(percentile(lat, 0.50), 1),
            "p90_ms": round(percentile(lat, 0.90), 1),
            "p99_ms": round(percentile(lat, 0.99), 1),
            "mean_ms": round(statistics.fmean(lat), 1),
            "min_ms": round(lat[0], 1),
            "max_ms": round(lat[-1], 1),
            "accuracy": round(sum(1 for r in got if r["correct"]) / len(got), 4),
            "malformed": sum(1 for r in got if not r["choice"]),
        }

    paired = {}
    if len(order) == 2:
        a, b = order
        deltas = []
        for idx in {r["idx"] for r in per_row}:
            ra = next(r for r in per_row if r["idx"] == idx and r["arm"] == a)
            rb = next(r for r in per_row if r["idx"] == idx and r["arm"] == b)
            deltas.append({"idx": idx, f"{b}_minus_{a}_ms": round(rb["ms"] - ra["ms"], 1),
                           "agree": ra["choice"] == rb["choice"]})
        d = [x[f"{b}_minus_{a}_ms"] for x in deltas]
        paired = {
            "arms": [a, b],
            "delta_definition": f"{b} minus {a}, same row, same block",
            "median_delta_ms": round(statistics.median(d), 1),
            f"rows_{b}_faster": sum(1 for v in d if v < 0),
            f"rows_{a}_faster": sum(1 for v in d if v > 0),
            "rows_tied": sum(1 for v in d if v == 0),
            "rows_disagreeing_choice": sum(1 for x in deltas if not x["agree"]),
            "per_row": deltas,
        }

    thr = {}
    if args.throughput:
        prompt = (INSTRUCTIONS + "\n\n" + "\n".join(f"- {k}: {v}" for k, v in OP_DESCRIPTIONS.items())
                  + "\n\n" + rows[0]["utterance"])
        for label, url in arms:
            thr[label] = {
                "prefill_dominated": throughput(url, prompt, 1, args.throughput_reps),
                "decode_dominated": throughput(url, prompt, 128, args.throughput_reps),
            }

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": ("bare d1_systemone backend per arm (resolve() bypassed: no decision "
                   "cache, no heuristic pre-gate); every row asked of every arm inside "
                   "one block, lead arm rotated per block; prompt imported verbatim from "
                   "bench/live_probe.py"),
        "utterances": str(Path(args.utterances).name),
        "block": args.block,
        "warmup_discarded_per_arm": args.warmup,
        "warmup_discarded_ms": warmup_ms,
        "model_assertion": {"expect_model": args.expect_model,
                            "checked": bool(args.expect_model),
                            "per_arm": served},
        "per_arm": per_arm,
        "paired": paired,
        "throughput": thr,
        "per_row": per_row,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print(f"\n{'arm':<10} {'n':>3} {'p50':>8} {'p90':>8} {'p99':>8} {'acc':>7}")
    for label in order:
        s = per_arm[label]
        print(f"{label:<10} {s['n']:>3} {s['p50_ms']:>8.1f} {s['p90_ms']:>8.1f} "
              f"{s['p99_ms']:>8.1f} {s['accuracy']:>6.1%}")
    if paired:
        print(f"paired median delta ({paired['arms'][1]} - {paired['arms'][0]}): "
              f"{paired['median_delta_ms']} ms")
    for label in order:
        if thr:
            p = thr[label]["prefill_dominated"]
            g = thr[label]["decode_dominated"]
            print(f"{label:<10} prefill {p['prompt_tokens_per_second']} tok/s  "
                  f"decode {g['predicted_tokens_per_second']} tok/s")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
