#!/usr/bin/env python3
"""OpenRouter model sweep — one `choice` question per utterance against a
hosted chat-completions endpoint, measured for accuracy, latency and cost
against the local d1-3B routing baseline.

The prompt material is imported verbatim from bench/live_probe.py
(INSTRUCTIONS + OP_DESCRIPTIONS) so these numbers are directly comparable to
the local probe runs under bench/results/.

Two passes per model:
  accuracy — all rows, bounded concurrency (default 8)
  latency  — latency-n rows, strictly sequential, one request at a time, so
             the p50/p95 is not inflated by concurrency

Usage:
  export OPENROUTER_API_KEY=...        # from your funBot .env
  python bench/or_bench.py [--models z-ai/glm-5.3-flash,...] [--limit N]
                           [--concurrency 8] [--latency-n 20]
                           [--model-max-tokens MODEL=N ...]
                           [--model-budget-usd 0.50] [--budget-usd 2.00]
                           [--out bench/results/or-bench-<date>.json]

The key is read only from the environment; it is never logged, written, or
placed in argv.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from live_probe import INSTRUCTIONS, OP_DESCRIPTIONS, percentile  # noqa: E402

API_URL = "https://openrouter.ai/api/v1/chat/completions"
KEY_ENV = "OPENROUTER_API_KEY"

# USD per 1M tokens, resolved against the live OpenRouter catalog. Multi-endpoint
# models are quoted at their worst case, so these are only a fallback for rows
# where the provider reported no cost — the reported `usage.cost` always wins.
MODELS = {
    "z-ai/glm-5.3-flash": {"in": 0.15, "out": 0.50},
    "deepseek/deepseek-v4-flash": {"in": 0.091, "out": 0.182},
    "deepseek/deepseek-v4.1-flash": {"in": 0.02, "out": 1.00},
    "qwen/qwen3.8-flash": {"in": 0.15, "out": 0.47},
    "qwen/qwen3.8-27b": {"in": 0.20, "out": 2.50},
    "openai/gpt-6-luna": {"in": 0.10, "out": 0.50},
    "meta/muse-spark-1.3": {"in": 1.25, "out": 4.25},
    "meta/muse-glimmer-30b": {"in": 0.30, "out": 1.20},
    "meta/muse-wonder": {"in": 4.00, "out": 20.00},
}

LABELS = list(OP_DESCRIPTIONS)
# Tolerate a bare op name ("read" for "filesys_read") without inventing labels:
# the mapping is derived from the label set, so it can never point at a label
# that is not in the task.
SUFFIX_TO_LABEL = {label.split("_", 1)[1]: label for label in LABELS}
JSON_HINT = re.compile(r"response_?format|json|structured", re.I)


class Ledger:
    """Provider-reported spend, so a run can stop before it overruns a budget.

    The per-model cap exists because a multi-dollar-per-Mtok model can burn a
    whole allowance inside one accuracy pass; the check happens before every
    request, so an over-budget model keeps the rows it paid for.
    """

    def __init__(self, model_cap: float, total_cap: float):
        self.model_cap = model_cap
        self.total_cap = total_cap
        self.model_spent = 0.0
        self.total_spent = 0.0
        self._lock = threading.Lock()

    def charge(self, cost: object) -> None:
        if not isinstance(cost, (int, float)):
            return
        with self._lock:
            self.model_spent += float(cost)
            self.total_spent += float(cost)

    def stop_reason(self) -> str | None:
        with self._lock:
            if self.model_cap and self.model_spent >= self.model_cap:
                return (f"per-model budget ${self.model_cap:.2f} reached "
                        f"(${self.model_spent:.4f} spent)")
            if self.total_cap and self.total_spent >= self.total_cap:
                return (f"run budget ${self.total_cap:.2f} reached "
                        f"(${self.total_spent:.4f} spent)")
        return None


def build_system_prompt() -> str:
    ops = "\n".join(f"- {name}: {desc}" for name, desc in OP_DESCRIPTIONS.items())
    return (
        f"{INSTRUCTIONS}\n\n"
        f"Operations:\n{ops}\n\n"
        "Answer with strict JSON only, no prose:\n"
        '{"choice": "<one operation label above>", "confidence": <number 0-1>}'
    )


def max_tokens_for(args, model: str) -> int:
    """Per-model override, else the run default. Reasoning models need headroom
    for the reasoning tokens before any content is emitted."""
    return args.max_tokens_by_model.get(model, args.max_tokens)


def build_payload(model: str, utterance: str, max_tokens: int, json_mode: bool) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": utterance},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "usage": {"include": True},  # ask OpenRouter to report real cost
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    return payload


def backoff(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            return min(float(retry_after), 30.0)
        except ValueError:
            pass
    return min(1.5 * (2 ** attempt), 15.0) * (0.75 + random.random() * 0.5)


def post(payload: dict, key: str, timeout: float, attempts: int = 4) -> tuple[dict | None, dict | None]:
    """POST one completion. Retries 429/5xx/network with backoff; returns
    (body, error) where error is {"status": int|None, "error": str}."""
    body = json.dumps(payload).encode("utf-8")
    last: dict = {"status": None, "error": "no attempt made"}
    for attempt in range(attempts):
        req = urllib.request.Request(
            API_URL,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "X-Title": "ti-operation-routing-bench",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8")), None
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8", "replace")[:400]
            last = {"status": e.code, "error": text}
            if e.code == 429 or e.code >= 500:
                time.sleep(backoff(attempt, e.headers.get("Retry-After") if e.headers else None))
                continue
            return None, last
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = {"status": None, "error": f"{type(e).__name__}: {e}"}
            time.sleep(backoff(attempt, None))
        except json.JSONDecodeError as e:
            last = {"status": None, "error": f"undecodable response body: {e}"}
            time.sleep(backoff(attempt, None))
    return None, last


def message_text(body: dict) -> str:
    choices = body.get("choices") or []
    if not choices:
        return ""
    content = (choices[0].get("message") or {}).get("content")
    if isinstance(content, list):  # some providers return content parts
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or ""


def finish_reason(body: dict) -> str:
    choices = body.get("choices") or []
    return (choices[0].get("finish_reason") or "") if choices else ""


def parse_reply(text: str) -> tuple[object, object, str]:
    """Pull (choice, confidence) out of a reply; tolerant of prose wrappers."""
    if not text or not text.strip():
        return None, None, "empty"
    stripped = text.strip()
    candidates = [stripped]
    block = re.search(r"\{.*\}", stripped, re.S)
    if block:
        candidates.append(block.group(0))
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj.get("choice"), obj.get("confidence"), "json"
    return None, None, "no-json"


def resolve_label(raw: object) -> tuple[str | None, str]:
    """Map a reply's choice onto one of the 8 labels. Returns (label, how)."""
    if not isinstance(raw, str) or not raw.strip():
        return None, "none"
    text = raw.strip()
    if text in LABELS:
        return text, "exact"
    lowered = text.lower().replace("-", "_").replace(" ", "_")
    if lowered in LABELS:
        return lowered, "case"
    if lowered in SUFFIX_TO_LABEL:
        return SUFFIX_TO_LABEL[lowered], "alias"
    return None, "none"


def ask(model: str, utterance: str, key: str, args, state: dict) -> dict:
    """One decision. Never raises; a bad reply comes back as ok=False."""
    cap = max_tokens_for(args, model)
    t0 = time.perf_counter()
    body, err = post(build_payload(model, utterance, cap, state["json_mode"]),
                     key, args.timeout)
    if err and state["json_mode"] and err.get("status") == 400 and JSON_HINT.search(err["error"]):
        state["json_mode"] = False  # provider refused response_format — fall back to text
        t0 = time.perf_counter()
        body, err = post(build_payload(model, utterance, cap, False), key, args.timeout)
    latency_ms = (time.perf_counter() - t0) * 1000
    if err:
        return {"ok": False, "latency_ms": latency_ms, "error": err,
                "choice": None, "choice_raw": None, "confidence": 0.0,
                "prompt_tokens": 0, "completion_tokens": 0, "cost": None,
                "provider": None, "reasoning_tokens": 0, "finish_reason": ""}

    usage = body.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    state["ledger"].charge(usage.get("cost"))
    text = message_text(body)
    raw_choice, raw_conf, how = parse_reply(text)
    label, resolution = resolve_label(raw_choice)
    try:
        confidence = min(max(float(raw_conf), 0.0), 1.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if raw_conf is not None and not isinstance(raw_conf, (int, float)):
        resolution = f"{resolution}/conf-unparsed"
    return {
        "ok": bool(label),
        "latency_ms": latency_ms,
        "error": None if label else {"status": 200, "error": f"unusable reply ({how})"},
        "choice": label,
        "choice_raw": raw_choice if isinstance(raw_choice, str) else None,
        "resolution": resolution,
        "confidence": confidence,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "cost": usage.get("cost"),
        "provider": body.get("provider") or "unknown",
        "reasoning_tokens": int(details.get("reasoning_tokens") or 0),
        "model_returned": body.get("model"),
        "finish_reason": finish_reason(body),
    }


def run_row(model: str, idx: int, row: dict, key: str, args, state: dict) -> dict:
    stop = state["ledger"].stop_reason()
    if stop:
        return {
            "idx": idx,
            "utterance": row["utterance"],
            "label": row["label"],
            "band": row.get("band"),
            "tier_target": row.get("tier_target"),
            "choice": None, "choice_raw": None, "resolution": None,
            "correct": False, "confidence": 0.0, "latency_ms": 0.0,
            "prompt_tokens": 0, "completion_tokens": 0, "cost": None,
            "ok": False, "finish_reason": None,
            "error": {"status": None, "error": f"budget stop: {stop}"},
            "skipped": True,
        }
    ans = ask(model, row["utterance"], key, args, state)
    return {
        "idx": idx,
        "utterance": row["utterance"],
        "label": row["label"],
        "band": row.get("band"),
        "tier_target": row.get("tier_target"),
        "choice": ans["choice"],
        "choice_raw": ans.get("choice_raw"),
        "resolution": ans.get("resolution"),
        "correct": bool(ans["choice"] == row["label"]),
        "confidence": ans["confidence"],
        "latency_ms": round(ans["latency_ms"], 1),
        "prompt_tokens": ans["prompt_tokens"],
        "completion_tokens": ans["completion_tokens"],
        "reasoning_tokens": ans.get("reasoning_tokens"),
        "provider": ans.get("provider"),
        "cost": ans.get("cost"),
        "ok": ans["ok"],
        "finish_reason": ans.get("finish_reason"),
        "error": ans["error"],
    }


def run_concurrent(model: str, rows: list[dict], key: str, args, state: dict) -> list[dict]:
    out: list[dict | None] = [None] * len(rows)

    def work(i: int) -> None:
        out[i] = run_row(model, i, rows[i], key, args, state)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(work, range(len(rows))))
    return [r for r in out if r is not None]


def run_sequential(model: str, picks: list[tuple[int, dict]], key: str, args,
                   state: dict) -> list[dict]:
    return [run_row(model, idx, row, key, args, state) for idx, row in picks]


def cost_of(rows: list[dict], pricing: dict) -> dict:
    """Measured provider cost when present, price-sheet fallback otherwise."""
    measured = sum(float(r["cost"]) for r in rows if isinstance(r.get("cost"), (int, float)))
    prompted = sum(r["prompt_tokens"] for r in rows)
    completed = sum(r["completion_tokens"] for r in rows)
    computed = (prompted * pricing["in"] + completed * pricing["out"]) / 1_000_000
    return {"measured_usd": round(measured, 6), "computed_usd": round(computed, 6),
            "prompt_tokens": prompted, "completion_tokens": completed,
            "source": "provider usage.cost" if measured > 0 else "price sheet x tokens"}


def aggregate(rows: list[dict], latency_rows: list[dict], pricing: dict) -> dict:
    attempted = [r for r in rows if not r.get("skipped")]
    n = len(attempted)
    correct = [r for r in attempted if r["correct"]]
    by_band: dict[str, dict] = {}
    by_label: dict[str, dict] = {}
    for r in attempted:
        for bucket, key in ((by_band, r["band"]), (by_label, r["label"])):
            slot = bucket.setdefault(key, {"n": 0, "correct": 0})
            slot["n"] += 1
            slot["correct"] += int(r["correct"])
    providers: dict[str, dict] = {}
    for r in rows:
        if r.get("skipped"):
            continue
        name = r.get("provider") or "unknown"
        slot = providers.setdefault(name, {"n": 0, "cost_usd": 0.0})
        slot["n"] += 1
        slot["cost_usd"] += float(r.get("cost") or 0)
    total_reported = sum(v["cost_usd"] for v in providers.values())
    lat = sorted(r["latency_ms"] for r in latency_rows if r["ok"])
    cost = cost_of(rows + latency_rows, pricing)  # every request we paid for counts
    spend = cost["measured_usd"] or cost["computed_usd"]
    n_correct = len(correct)
    return {
        "n_rows": len(rows),
        "n_attempted": n,
        "skipped": len(rows) - n,
        "accuracy": round(n_correct / n, 4) if n else 0.0,
        "malformed": sum(1 for r in attempted if not r["ok"]),
        "truncated": sum(1 for r in attempted if r.get("finish_reason") == "length"),
        "failures": sum(1 for r in attempted if r["error"] and r["error"].get("status") not in (200,)),
        "alias_resolutions": sum(1 for r in attempted if r.get("resolution") == "alias"),
        "reasoning_tokens_median": (
            round(statistics.median([r.get("reasoning_tokens") or 0 for r in attempted]), 1)
            if attempted else None),
        "providers": {name: {"n": v["n"],
                             "share": round(v["n"] / n, 4) if n else 0.0,
                             "cost_usd": round(v["cost_usd"], 6),
                             "cost_share": round(v["cost_usd"] / total_reported, 4)
                             if total_reported else 0.0}
                      for name, v in sorted(providers.items(), key=lambda kv: -kv[1]["n"])},
        "accuracy_by_band": {k: {"n": v["n"], "accuracy": round(v["correct"] / v["n"], 4)}
                             for k, v in sorted(by_band.items())},
        "accuracy_by_label": {k: {"n": v["n"], "accuracy": round(v["correct"] / v["n"], 4)}
                              for k, v in sorted(by_label.items())},
        "latency_ms": {
            "mode": "sequential",
            "n": len(lat),
            "p50": round(percentile(lat, 0.5), 1) if lat else None,
            "p95": round(percentile(lat, 0.95), 1) if lat else None,
            "mean": round(statistics.fmean(lat), 1) if lat else None,
            "min": round(lat[0], 1) if lat else None,
            "max": round(lat[-1], 1) if lat else None,
        },
        "cost": {
            **cost,
            "total_usd": round(spend, 6),
            "per_1k_decisions_usd": round(spend / n * 1000, 4) if n else None,
            "per_1k_correct_usd": round(spend / n_correct * 1000, 4) if n_correct else None,
        },
        "per_row": rows,
        "latency_rows": latency_rows,
    }


def frontier(models: dict, metric: str) -> dict:
    """Models not strictly dominated on (accuracy max, metric min)."""
    def value(name: str) -> float:
        if metric == "cost_per_1k_correct":
            v = models[name]["cost"]["per_1k_correct_usd"]
        else:
            v = models[name]["latency_ms"]["p50"]
        return float("inf") if v is None else v

    def dominates(a: str, b: str) -> bool:
        acc_a, acc_b, va, vb = (models[a]["accuracy"], models[b]["accuracy"], value(a), value(b))
        return acc_a >= acc_b and va <= vb and (acc_a > acc_b or va < vb)

    names = list(models)
    dominated = [b for b in names if any(dominates(a, b) for a in names if a != b)]
    return {"metric": metric,
            "on_frontier": [n for n in names if n not in dominated],
            "dominated": dominated}


def rank(models: dict) -> dict:
    priced = {n: {"accuracy": m["accuracy"],
                  "cost_per_1k_correct": m["cost"]["per_1k_correct_usd"] or float("inf"),
                  "p50_ms": m["latency_ms"]["p50"] or float("inf")}
              for n, m in models.items()}
    by_dollar = sorted(priced, key=lambda n: (-priced[n]["accuracy"], priced[n]["cost_per_1k_correct"]))
    by_latency = sorted(priced, key=lambda n: (-priced[n]["accuracy"], priced[n]["p50_ms"]))
    return {
        "by_accuracy_then_cost": by_dollar,
        "by_accuracy_then_latency": by_latency,
        "accuracy_per_dollar": {n: round(priced[n]["accuracy"] / priced[n]["cost_per_1k_correct"], 4)
                                if priced[n]["cost_per_1k_correct"] != float("inf") else None
                                for n in priced},
        "pareto_vs_cost": frontier(models, "cost_per_1k_correct"),
        "pareto_vs_latency": frontier(models, "p50_ms"),
    }


def print_report(summary: dict) -> None:
    models = summary["models"]
    print(f"\n{'model':<30}{'acc':>7}{'t1':>7}{'t2':>7}{'t3':>7}"
          f"{'p50':>8}{'p95':>8}{'$total':>9}{'$/1k':>8}{'$/1k-ok':>9}{'bad':>5}{'fail':>5}")
    for name, m in models.items():
        b = m["accuracy_by_band"]
        f = lambda k: f"{b.get(k, {}).get('accuracy', 0):.1%}" if k in b else "-"  # noqa: E731
        p50 = m["latency_ms"]["p50"]
        p95 = m["latency_ms"]["p95"]
        lat = f"{p50:>7.0f}m{p95:>7.0f}m" if p50 is not None else f"{'-':>7}{'-':>7}"
        acc = f"{m['accuracy']:>6.1%}" if m["n_attempted"] == m["n_rows"] else \
            f"{m['accuracy']:>5.1%}*"
        print(f"{name:<30}{acc}{f('t1'):>7}{f('t2'):>7}{f('t3'):>7}"
              f"{lat}"
              f"{m['cost']['total_usd']:>9.5f}{m['cost']['per_1k_decisions_usd'] or 0:>8.4f}"
              f"{m['cost']['per_1k_correct_usd'] or 0:>9.4f}{m['malformed']:>5}{m['failures']:>5}"
              f"   ({m['n_attempted']}/{m['n_rows']} rows)")
    base = summary["baselines"]["local_d1_3B_Q4"]
    print(f"\nbaseline {base['model']}: {base['accuracy']:.1%} acc, p50 {base['p50_ms']} ms, $0 (local)")

    print("\nper-label accuracy:")
    header = [n.split("/")[-1] for n in models]
    print(f"  {'label':<16}" + "".join(f"{h[:14]:>16}" for h in header))
    for label in LABELS:
        cells = "".join(f"{models[n]['accuracy_by_label'].get(label, {}).get('accuracy', 0):>15.0%} "
                        for n in models)
        print(f"  {label:<16}{cells}")

    r = summary["ranking"]
    print(f"\nranked by accuracy, then $-per-1k-correct: {r['by_accuracy_then_cost']}")
    print(f"ranked by accuracy, then p50 ms:            {r['by_accuracy_then_latency']}")
    print(f"Pareto frontier vs $-per-1k-correct: {r['pareto_vs_cost']['on_frontier']}")
    print(f"  strictly dominated:                {r['pareto_vs_cost']['dominated']}")
    print(f"Pareto frontier vs p50 latency:      {r['pareto_vs_latency']['on_frontier']}")
    print(f"  strictly dominated:                {r['pareto_vs_latency']['dominated']}")
    print("\nserving provider(s) per model:")
    for name, m in models.items():
        dist = ", ".join(f"{p} {v['n']} ({v['share']:.0%}, ${v['cost_usd']:.5f})"
                         for p, v in m["providers"].items()) or "-"
        extra = f"  [budget stop: {m['budget_stop']}]" if m.get("budget_stop") else ""
        skip = f"  [skipped {m['skipped']} rows]" if m.get("skipped") else ""
        trunc = f"  [truncated {m['truncated']} rows]" if m.get("truncated") else ""
        print(f"  {name:<30} {dist}{extra}{skip}{trunc}")
    print(f"\nwall time: {summary['wall_time_s']}s   "
          f"total cost incurred: ${summary['total_cost_usd']:.5f}   "
          f"(accuracy pass x{summary['concurrency']} concurrent; "
          f"latency pass sequential, n={summary['latency_run']['n']})")


def pick_output(out: Path) -> Path:
    """Never overwrite a recorded result file."""
    if not out.exists():
        return out
    stem, suffix = out.stem, out.suffix
    for run in range(2, 100):
        candidate = out.with_name(f"{stem}-run{run}{suffix}")
        if not candidate.exists():
            return candidate
    raise SystemExit(f"refusing to overwrite {out}: too many existing runs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="comma-separated model ids (default: all known)")
    ap.add_argument("--utterances", default=str(ROOT / "tests/data/utterances_v2.json"))
    ap.add_argument("--limit", type=int, default=0, help="run only the first N rows")
    ap.add_argument("--concurrency", type=int, default=8, help="max requests in flight (accuracy pass)")
    ap.add_argument("--latency-n", type=int, default=20, help="sequential latency rows per model")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--model-max-tokens", action="append", default=[], metavar="MODEL=N",
                    help="per-model max_tokens override, repeatable (reasoning models "
                         "truncate their JSON answer at a cap tuned for non-reasoning ones)")
    ap.add_argument("--model-budget-usd", type=float, default=0.50,
                    help="stop a model once provider-reported spend reaches this (0 = no cap)")
    ap.add_argument("--budget-usd", type=float, default=2.00,
                    help="stop the whole run once total spend reaches this (0 = no cap)")
    ap.add_argument("--timeout", type=float, default=45.0)
    ap.add_argument("--out", default="", help="result file (default bench/results/or-bench-<date>.json)")
    ap.add_argument("--no-latency-run", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the prompt, send nothing")
    ap.add_argument("--key-env", default=KEY_ENV, help="environment variable holding the key")
    args = ap.parse_args()
    args.max_tokens_by_model = {}
    for spec in args.model_max_tokens:
        name, _, cap = spec.partition("=")
        if not name or not cap.isdigit():
            raise SystemExit(f"--model-max-tokens wants MODEL=N, got {spec!r}")
        args.max_tokens_by_model[name.strip()] = int(cap)

    rows = json.loads(Path(args.utterances).read_text())
    if args.limit:
        rows = rows[: args.limit]
    wanted = [m.strip() for m in args.models.split(",") if m.strip()] or list(MODELS)
    unknown = [m for m in wanted if m not in MODELS]
    if unknown:
        raise SystemExit(f"unknown model id(s): {unknown}; known: {list(MODELS)}")

    if args.dry_run:
        print(build_system_prompt())
        print(f"\nuser: {rows[0]['utterance']}")
        print(f"\nmodels: {wanted}\nrows: {len(rows)}")
        return

    key = os.environ.get(args.key_env, "")
    if not key:
        raise SystemExit(f"{args.key_env} is not set — export it from your funBot .env first")

    # sequential latency slice, deterministically spread across the dataset
    stride = max(1, len(rows) // max(args.latency_n, 1))
    latency_picks = [(i, rows[i]) for i in range(0, len(rows), stride)][: args.latency_n]

    out = Path(args.out) if args.out else Path(
        ROOT / f"bench/results/or-bench-{time.strftime('%Y-%m-%d')}.json")
    started = time.perf_counter()
    summary: dict = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "harness": "bench/or_bench.py",
        "endpoint": API_URL,
        "dataset": str(Path(args.utterances).relative_to(ROOT)),
        "n_rows": len(rows),
        "concurrency": args.concurrency,
        "latency_run": {"n": 0 if args.no_latency_run else len(latency_picks),
                        "mode": "sequential",
                        "row_indices": [] if args.no_latency_run else [i for i, _ in latency_picks]},
        "prompt_source": "bench/live_probe.py INSTRUCTIONS + OP_DESCRIPTIONS (verbatim import)",
        "request": {"temperature": 0, "max_tokens": args.max_tokens,
                    "max_tokens_by_model": args.max_tokens_by_model,
                    "response_format": '{"type":"json_object"} with plain-text JSON fallback',
                    "usage_include": True},
        "budget": {"per_model_usd": args.model_budget_usd, "total_usd": args.budget_usd},
        "baselines": {
            "local_d1_3B_Q4": {"model": "d1-3B-Q4_K_M (local llama.cpp)",
                               "accuracy": 0.74, "p50_ms": 147.9,
                               "source": "bench/results/dell/v2-d1-3B-Q4-run2.json"},
            "char_4gram_1nn": {"model": "char 4-gram 1-NN (leave-one-out)",
                               "accuracy": 0.54, "p50_ms": 0.5,
                               "source": "docs/model-sweep-consolidated"},
        },
        "models": {},
    }

    for model in wanted:
        pricing = MODELS[model]
        ledger = Ledger(args.model_budget_usd, args.budget_usd)
        state = {"json_mode": True, "ledger": ledger}
        print(f"[{model}] accuracy pass: {len(rows)} rows, {args.concurrency} in flight, "
              f"max_tokens={max_tokens_for(args, model)}", flush=True)
        t_model = time.perf_counter()
        acc_rows = run_concurrent(model, rows, key, args, state)
        latency_rows: list[dict] = []
        if not args.no_latency_run:
            print(f"[{model}] latency pass: {len(latency_picks)} rows, strictly sequential",
                  flush=True)
            latency_rows = run_sequential(model, latency_picks, key, args, state)
        entry = aggregate(acc_rows, latency_rows, pricing)
        entry["pricing_usd_per_mtok"] = pricing
        entry["max_tokens"] = max_tokens_for(args, model)
        entry["json_mode"] = "accepted" if state["json_mode"] else "rejected, text parse used"
        entry["budget_stop"] = ledger.stop_reason()
        entry["wall_time_s"] = round(time.perf_counter() - t_model, 1)
        summary["models"][model] = entry
        print(f"[{model}] acc={entry['accuracy']:.1%} "
              f"p50={entry['latency_ms']['p50']}ms malformed={entry['malformed']} "
              f"truncated={entry['truncated']} skipped={entry['skipped']} "
              f"failures={entry['failures']} cost=${entry['cost']['total_usd']:.5f} "
              f"providers={ {k: v['n'] for k, v in entry['providers'].items()} }", flush=True)

    summary["wall_time_s"] = round(time.perf_counter() - started, 1)
    summary["total_cost_usd"] = round(
        sum(m["cost"]["total_usd"] for m in summary["models"].values()), 6)
    summary["total_decisions"] = sum(len(m["per_row"]) + len(m["latency_rows"])
                                     for m in summary["models"].values())
    summary["ranking"] = rank(summary["models"])

    out.parent.mkdir(parents=True, exist_ok=True)
    out = pick_output(out)
    out.write_text(json.dumps(summary, indent=2))
    print_report(summary)
    shown = out.relative_to(ROOT) if ROOT in out.parents else out
    print(f"\nwrote {shown}")


if __name__ == "__main__":
    main()
