#!/usr/bin/env python3
"""OpenRouter model sweep — one `choice` question per utterance against a
hosted chat-completions endpoint, measured for accuracy, latency and cost
against the local d1-3B routing baseline.

The prompt material is imported verbatim from bench/ti_bench/prompts.py
(INSTRUCTIONS + OP_DESCRIPTIONS, the same frozen probe bench/live_probe.py uses)
so these numbers are directly comparable to the local probe runs under
bench/results/.

Two passes per model:
  accuracy — all rows, bounded concurrency (default 8)
  latency  — latency-n rows, strictly sequential, one request at a time, so
             the p50/p95 is not inflated by concurrency

Usage:
  export OPENROUTER_API_KEY=...        # from your funBot .env
  python bench/or_bench.py [--models z-ai/glm-5.3-flash,...] [--limit N]
                           [--providers Relace|order:Relace+Wafer|sort:price]
                           [--variants full,compact] [--answers json,letter]
                           [--reasoning minimal] [--model-max-tokens M=N]
                           [--concurrency 8] [--latency-n 20]
                           [--out bench/results/or-bench-<date>.json]

The key is read only from the environment; it is never logged, written, or
placed in argv. Both spend caps default to 0 (off): a cap that silently drops
rows changes the accuracy denominator, so it is opt-in and always recorded.
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

from ti_bench.prompts import INSTRUCTIONS, OP_DESCRIPTIONS  # noqa: E402
from ti_bench.stats import percentile  # noqa: E402

API_URL = "https://openrouter.ai/api/v1/chat/completions"
KEY_ENV = "OPENROUTER_API_KEY"

# USD per 1M tokens, catalogue `pricing` block. For z-ai/glm-5.3-flash this is
# the *sheet*, not the bill: 38 endpoints serve the model and none bills this
# (see PROVIDERS). Used only as the computed-cost fallback when the provider
# returns no usage.cost.
MODELS = {
    "z-ai/glm-5.3-flash": {"in": 0.15, "out": 0.50, "cache": 0.03},
    "z-ai/glm-5.3-flash:batch": {"in": 0.05, "out": 0.18, "cache": 0.01},
    "deepseek/deepseek-v4-flash": {"in": 0.091, "out": 0.182},
    "deepseek/deepseek-v4.1-flash": {"in": 0.02, "out": 1.00},
    "qwen/qwen3.8-flash": {"in": 0.15, "out": 0.47},
    "qwen/qwen3.8-27b": {"in": 0.20, "out": 2.50},
}

# Per-endpoint prices for z-ai/glm-5.3-flash, USD per 1M tokens, read from
# /api/v1/models/z-ai/glm-5.3-flash/endpoints. One model id => many endpoints,
# each with its own quantization and its own price; pinning is the lever.
PROVIDERS = {
    "Relace": {"quant": "unknown", "in": 0.0413, "out": 0.0758, "cache": 0.0300},
    "StreamLake": {"quant": "fp8", "in": 0.0435, "out": 0.1450, "cache": 0.0087},
    "Wafer": {"quant": "nvfp4", "in": 0.0540, "out": 0.2000, "cache": 0.0530},
    "Darkbloom": {"quant": "fp4", "in": 0.0650, "out": 0.1800, "cache": 0.0325},
    "GMICloud": {"quant": "fp8", "in": 0.0750, "out": 0.2500, "cache": 0.0150},
    "OpenInference": {"quant": "fp4", "in": 0.0767, "out": 0.0100, "cache": 0.0198},
    "Near AI": {"quant": "fp8", "in": 0.0800, "out": 0.5000, "cache": 0.0120},
}

# accuracy guardrail: a configuration below this may be reported as a trade-off
# only, never recommended.
GUARDRAIL_ACCURACY = 0.933

LETTERS = "ABCDEFGH"

# Output contract variants. "json" reproduces the baseline prompt byte for byte.
ANSWER_FORMATS = {
    "json": {
        "parse": "json",
        "tail": ('Answer with strict JSON only, no prose:\n'
                 '{"choice": "<one operation label above>", "confidence": <number 0-1>}'),
    },
    "json-noconf": {
        "parse": "json",
        "tail": 'Answer with strict JSON only, no prose:\n{"choice": "<one operation label above>"}',
    },
    "label": {
        "parse": "label",
        "tail": "Reply with one operation label from the list above and nothing else.",
    },
    "letter": {
        "parse": "letter",
        "tail": "Reply with the single letter of the best operation and nothing else.",
    },
}

# OpenRouter's `reasoning` request field, one mode per key.
REASONING_MODES = {
    "default": None,
    "off": {"enabled": False},
    "minimal": {"effort": "minimal"},
    "exclude": {"exclude": True},
}

# Compaction of the 8 descriptions: same discriminative axis, fewer tokens.
COMPACT_OP_DESCRIPTIONS = {
    "filesys_read": "Read one file.",
    "filesys_glob": "List paths matching a name pattern (no content search).",
    "filesys_grep": "Regex search file contents.",
    "decide_pick": "Pick the best item(s) from candidates.",
    "decide_extract": "Extract query strings or parameters from the command.",
    "web_search": "Web search by keywords.",
    "web_fetch": "Fetch one URL.",
    "reddit_digest": "Digest a subreddit or thread.",
}

PROMPT_VARIANTS = {"full": OP_DESCRIPTIONS, "compact": COMPACT_OP_DESCRIPTIONS}


LABELS = list(OP_DESCRIPTIONS)
# Tolerate a bare op name ("read" for "filesys_read") without inventing labels:
# the mapping is derived from the label set, so it can never point at a label
# that is not in the task.
SUFFIX_TO_LABEL = {label.split("_", 1)[1]: label for label in LABELS}
JSON_HINT = re.compile(r"response_?format|json|structured", re.I)


def build_system_prompt(variant: str = "full", answer: str = "json") -> str:
    """Static block. It goes FIRST and the utterance LAST on purpose: an
    identical leading prefix is what a prefix cache can serve."""
    descriptions = PROMPT_VARIANTS[variant]
    if answer == "letter":
        ops = "\n".join(f"{LETTERS[i]}. {name}: {desc}"
                        for i, (name, desc) in enumerate(descriptions.items()))
    else:
        ops = "\n".join(f"- {name}: {desc}" for name, desc in descriptions.items())
    return f"{INSTRUCTIONS}\n\nOperations:\n{ops}\n\n{ANSWER_FORMATS[answer]['tail']}"


class Ledger:
    """Provider-reported spend, so a run can stop before it overruns a budget.

    One ledger spans the whole run: `model_spent` resets per model via
    `begin_model`, `total_spent` does not, so a run-wide cap is a run-wide cap.
    The check happens before every request, so an over-budget model keeps the
    rows it already paid for. A single request can carry spend past the cap.
    """

    def __init__(self, model_cap: float, total_cap: float):
        self.model_cap = model_cap
        self.total_cap = total_cap
        self.model_spent = 0.0
        self.total_spent = 0.0
        self._lock = threading.Lock()

    def begin_model(self) -> None:
        with self._lock:
            self.model_spent = 0.0

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


def max_tokens_for(args, model: str) -> int:
    """Per-model override, else the run default. Reasoning models need headroom
    for the reasoning tokens before any content is emitted."""
    return args.max_tokens_by_model.get(model, args.max_tokens)


def build_payload(cfg: dict, utterance: str, max_tokens: int, json_mode: bool) -> dict:
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": build_system_prompt(cfg["variant"], cfg["answer"])},
            {"role": "user", "content": utterance},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "usage": {"include": True},  # ask OpenRouter to report real cost
    }
    if cfg.get("provider"):
        payload["provider"] = cfg["provider"]
    if cfg.get("reasoning") is not None:
        payload["reasoning"] = cfg["reasoning"]
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    return payload


# Request-level counters, so a run can report rate limits and retries rather
# than silently absorbing them. Guarded: the accuracy pass is threaded.
_METRICS: dict = {"rate_limited": 0, "retries": 0, "http_errors": {}}
_METRICS_LOCK = threading.Lock()


def note(key: str, bump: int = 1, sub: str | None = None) -> None:
    with _METRICS_LOCK:
        if sub is None:
            _METRICS[key] = _METRICS.get(key, 0) + bump
        else:
            bucket = _METRICS.setdefault(key, {})
            bucket[sub] = bucket.get(sub, 0) + bump


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
            note("http_errors", sub=str(e.code))
            if e.code == 429 or e.code >= 500:
                if e.code == 429:
                    note("rate_limited")
                note("retries")
                time.sleep(backoff(attempt, e.headers.get("Retry-After") if e.headers else None))
                continue
            return None, last
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = {"status": None, "error": f"{type(e).__name__}: {e}"}
            note("retries")
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


def parse_answer(text: str, answer: str) -> tuple[object, object, str]:
    """(choice, confidence, how) under the config's answer format."""
    if ANSWER_FORMATS[answer]["parse"] == "json":
        return parse_reply(text)
    stripped = (text or "").strip()
    if not stripped:
        return None, None, "empty"
    if answer == "letter":
        hit = re.search(r"\b([A-H])\b", stripped.upper())
        if hit:
            idx = ord(hit.group(1)) - ord("A")
            if idx < len(LABELS):
                return LABELS[idx], None, "letter"
        return None, None, "no-letter"
    for candidate in (stripped, stripped.strip("\"'`.,;:*").splitlines()[0],
                      stripped.split()[0] if stripped.split() else ""):
        if candidate and resolve_label(candidate)[0]:
            return candidate, None, "label"
    for token in re.findall(r"[A-Za-z_]+(?:\.[A-Za-z_]+)?", stripped):
        if resolve_label(token)[0]:
            return token, None, "label-in-prose"
    return stripped[:40], None, "label-unresolved"


def ask(cfg: dict, utterance: str, key: str, args, state: dict) -> dict:
    """One decision. Never raises; a bad reply comes back as ok=False."""
    cap = max_tokens_for(args, cfg["model"])
    t0 = time.perf_counter()
    body, err = post(build_payload(cfg, utterance, cap, state["json_mode"]),
                     key, args.timeout)
    if err and state["json_mode"] and err.get("status") == 400 and JSON_HINT.search(err["error"]):
        state["json_mode"] = False  # provider refused response_format — fall back to text
        t0 = time.perf_counter()
        body, err = post(build_payload(cfg, utterance, cap, False), key, args.timeout)
    latency_ms = (time.perf_counter() - t0) * 1000
    if err:
        return {"ok": False, "latency_ms": latency_ms, "error": err,
                "choice": None, "choice_raw": None, "confidence": 0.0,
                "prompt_tokens": 0, "completion_tokens": 0, "cost": None,
                "provider": None}

    usage = body.get("usage") or {}
    state["ledger"].charge(usage.get("cost"))
    pdet = usage.get("prompt_tokens_details") or {}
    cdet = usage.get("completion_tokens_details") or {}
    cdet_cost = usage.get("cost_details") or {}
    text = message_text(body)
    raw_choice, raw_conf, how = parse_answer(text, cfg["answer"])
    label, resolution = resolve_label(raw_choice)
    try:
        confidence = min(max(float(raw_conf), 0.0), 1.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if raw_conf is not None and not isinstance(raw_conf, (int, float)):
        resolution = f"{resolution}/conf-unparsed"
    reasoning = ((body.get("choices") or [{}])[0].get("message") or {}).get("reasoning")
    if reasoning is None and (body.get("choices") or [{}])[0].get("message"):
        reasoning = body["choices"][0]["message"].get("reasoning_content")
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
        "cached_tokens": int(pdet.get("cached_tokens") or 0),
        "reasoning_tokens": int(cdet.get("reasoning_tokens") or 0),
        "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else 0,
        "cost_prompt": cdet_cost.get("upstream_inference_prompt_cost"),
        "cost_completion": cdet_cost.get("upstream_inference_completions_cost"),
        "cost": usage.get("cost"),
        "provider": body.get("provider"),
        "provider_model": body.get("model"),
        "finish_reason": finish_reason(body),
    }


def served_by_pin(cfg: dict, served: str | None) -> bool:
    """True when a requested endpoint list was not honoured.

    `only` is a hard pin, `order` a preference; both count as a fallback when a
    different endpoint answers, because either way the measured cost belongs to
    an endpoint the configuration did not ask for.
    """
    block = cfg.get("provider") or {}
    requested = (block.get("only") or []) + (block.get("order") or [])
    return bool(requested) and served not in requested


def run_row(cfg: dict, idx: int, row: dict, key: str, args, state: dict) -> dict:
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
            "provider": None, "fallback": False,
            "ok": False, "finish_reason": None,
            "error": {"status": None, "error": f"budget stop: {stop}"},
            "skipped": True,
        }
    ans = ask(cfg, row["utterance"], key, args, state)
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
        "cached_tokens": ans.get("cached_tokens", 0),
        "reasoning_tokens": ans.get("reasoning_tokens", 0),
        "reasoning_chars": ans.get("reasoning_chars", 0),
        "cost_prompt": ans.get("cost_prompt"),
        "cost_completion": ans.get("cost_completion"),
        "cost": ans.get("cost"),
        "provider": ans.get("provider"),
        "fallback": served_by_pin(cfg, ans.get("provider")),
        "ok": ans["ok"],
        "finish_reason": ans.get("finish_reason"),
        "error": ans["error"],
    }


def run_concurrent(cfg: dict, rows: list[dict], key: str, args, state: dict) -> list[dict]:
    out: list[dict | None] = [None] * len(rows)

    def work(i: int) -> None:
        out[i] = run_row(cfg, i, rows[i], key, args, state)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(work, range(len(rows))))
    return [r for r in out if r is not None]


def run_sequential(cfg: dict, picks: list[tuple[int, dict]], key: str, args,
                   state: dict) -> list[dict]:
    return [run_row(cfg, idx, row, key, args, state) for idx, row in picks]


def cost_of(rows: list[dict], pricing: dict) -> dict:
    """Measured provider cost when present, price-sheet fallback otherwise."""
    measured = sum(float(r["cost"]) for r in rows if isinstance(r.get("cost"), (int, float)))
    prompted = sum(r["prompt_tokens"] for r in rows)
    completed = sum(r["completion_tokens"] for r in rows)
    cached = sum(r.get("cached_tokens") or 0 for r in rows)
    fresh = max(prompted - cached, 0)
    computed = (fresh * pricing["in"] + cached * pricing.get("cache", pricing["in"])
                + completed * pricing["out"]) / 1_000_000
    return {"measured_usd": round(measured, 6), "computed_usd": round(computed, 6),
            "prompt_tokens": prompted, "completion_tokens": completed,
            "source": "provider usage.cost" if measured > 0 else "price sheet x tokens"}


def token_detail(rows: list[dict]) -> dict:
    """Per-request token shape, and the rate actually charged per token.

    The effective rates come from usage.cost_details, which splits the upstream
    bill into a prompt part and a completion part — dividing each by its own
    token count recovers the per-M rate the endpoint billed, which is how a
    pinned endpoint's real price is confirmed rather than assumed.
    """
    n = len(rows) or 1
    prompted = sum(r["prompt_tokens"] for r in rows)
    completed = sum(r["completion_tokens"] for r in rows)
    cached = sum(r.get("cached_tokens") or 0 for r in rows)
    reasoning = sum(r.get("reasoning_tokens") or 0 for r in rows)
    cost_prompt = sum(r.get("cost_prompt") or 0 for r in rows)
    cost_completion = sum(r.get("cost_completion") or 0 for r in rows)
    return {
        "prompt_per_req": round(prompted / n, 1),
        "completion_per_req": round(completed / n, 1),
        "reasoning_per_req": round(reasoning / n, 1),
        "reasoning_share_of_completion": round(reasoning / completed, 4) if completed else None,
        "cached_per_req": round(cached / n, 1),
        "cached_share_of_prompt": round(cached / prompted, 4) if prompted else None,
        "max_completion": max((r["completion_tokens"] for r in rows), default=0),
        "effective_in_usd_per_mtok": round(cost_prompt / prompted * 1e6, 4)
        if prompted and cost_prompt else None,
        "effective_out_usd_per_mtok": round(cost_completion / completed * 1e6, 4)
        if completed and cost_completion else None,
    }


def aggregate(rows: list[dict], latency_rows: list[dict], pricing: dict) -> dict:
    # A row the spend ledger never paid for is not a wrong answer: it is excluded
    # from every denominator and counted in `skipped`, so a capped run cannot
    # read as a bad one.
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
    lat = sorted(r["latency_ms"] for r in latency_rows if r["ok"])
    cost = cost_of(rows + latency_rows, pricing)  # every request we paid for counts
    acc_only = cost_of(rows, pricing)  # the accuracy pass alone: one request per decision
    spend = cost["measured_usd"] or cost["computed_usd"]
    per_decision = (acc_only["measured_usd"] or acc_only["computed_usd"]) / n if n else 0.0
    n_correct = len(correct)
    served: dict[str, int] = {}
    for r in attempted:
        served[r.get("provider") or "unknown"] = served.get(r.get("provider") or "unknown", 0) + 1
    accuracy = round(n_correct / n, 4) if n else 0.0
    return {
        "n_rows": len(rows),
        "n_attempted": n,
        "skipped": len(rows) - n,
        "accuracy": accuracy,
        "meets_guardrail": accuracy >= GUARDRAIL_ACCURACY,
        "malformed": sum(1 for r in attempted if not r["ok"]),
        "truncated": sum(1 for r in attempted if r.get("finish_reason") == "length"),
        "failures": sum(1 for r in attempted
                        if r["error"] and r["error"].get("status") not in (200,)),
        "alias_resolutions": sum(1 for r in attempted if r.get("resolution") == "alias"),
        "providers_served": dict(sorted(served.items())),
        "fallbacks": sum(1 for r in attempted if r.get("fallback")),
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
            "per_decision_usd": round(per_decision, 8),
            "per_1k_decisions_request_only_usd": round(per_decision * 1000, 4),
            "per_1k_correct_request_only_usd": round(per_decision / (n_correct / n) * 1000, 4)
            if n_correct else None,
        },
        "tokens": token_detail(rows + latency_rows),
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
    print(f"\n{'config':<38}{'acc':>7}{'gate':>5}{'t1':>7}{'t2':>7}{'t3':>7}"
          f"{'p50':>8}{'p95':>8}{'$total':>9}{'$/1k':>8}{'$/1k-ok':>9}"
          f"{'bad':>4}{'fail':>5}{'fb':>4}{'skip':>5}")
    for name, m in models.items():
        b = m["accuracy_by_band"]
        f = lambda k: f"{b.get(k, {}).get('accuracy', 0):.1%}" if k in b else "-"  # noqa: E731
        p50 = m["latency_ms"]["p50"]
        p95 = m["latency_ms"]["p95"]
        lat = f"{p50:>7.0f}m{p95:>7.0f}m" if p50 is not None else f"{'-':>7}{'-':>7}"
        print(f"{name:<38}{m['accuracy']:>6.1%}{'ok' if m['meets_guardrail'] else 'LOW':>5}"
              f"{f('t1'):>7}{f('t2'):>7}{f('t3'):>7}{lat}"
              f"{m['cost']['total_usd']:>9.5f}{m['cost']['per_1k_decisions_usd'] or 0:>8.4f}"
              f"{m['cost']['per_1k_correct_usd'] or 0:>9.4f}{m['malformed']:>4}"
              f"{m['failures']:>5}{m.get('fallbacks', 0):>4}{m.get('skipped', 0):>5}")
    base = summary["baselines"]["local_d1_3B_Q4"]
    print(f"\nbaseline {base['model']}: {base['accuracy']:.1%} acc, p50 {base['p50_ms']} ms, $0 (local)")
    print(f"accuracy guardrail: >= {GUARDRAIL_ACCURACY:.1%} (gate column 'LOW' = trade-off only)")
    capped = {n: m for n, m in models.items() if m.get("skipped")}
    if capped:
        print("a non-zero skip column means a spend cap stopped the pass: those rows are "
              "out of every denominator, and the bands they covered are not represented")
        for n, m in capped.items():
            print(f"  {n:<38} {m['n_attempted']}/{m['n_rows']} attempted "
                  f"({m['budget_stop']}), bands {sorted(m['accuracy_by_band'])}")

    print("\ntoken shape and the rate actually charged (usage.cost_details split):")
    print(f"  {'config':<38}{'in/req':>7}{'out/req':>8}{'reas/req':>9}{'cached%':>8}"
          f"{'$in/M':>8}{'$out/M':>8}  served")
    for name, m in models.items():
        t = m["tokens"]
        cached = t["cached_share_of_prompt"]
        print(f"  {name:<38}{t['prompt_per_req']:>7.1f}{t['completion_per_req']:>8.1f}"
              f"{t['reasoning_per_req']:>9.1f}"
              f"{cached * 100 if cached is not None else 0:>7.1f}%"
              f"{t['effective_in_usd_per_mtok'] or 0:>8.4f}"
              f"{t['effective_out_usd_per_mtok'] or 0:>8.4f}  {m['providers_served']}")

    print("\nper-label accuracy:")
    header = [n.split("|")[-2] if "|" in n else n.split("/")[-1] for n in models]
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
    metrics = summary.get("request_metrics") or {}
    print(f"\nrequests: rate_limited={metrics.get('rate_limited', 0)} "
          f"retries={metrics.get('retries', 0)} http_errors={metrics.get('http_errors', {})} "
          f"fallbacks={sum(m.get('fallbacks', 0) for m in models.values())}")
    print(f"wall time: {summary['wall_time_s']}s   "
          f"total cost incurred: ${summary['total_cost_usd']:.5f}   "
          f"(accuracy pass x{summary['concurrency']} concurrent; "
          f"latency pass sequential, n={summary['latency_run']['n']})")


def provider_block(pin: str, allow_fallbacks: bool) -> dict | None:
    """Build the OpenRouter `provider` request field for one pin.

    A named endpoint is pinned with `only` and, by default, with fallbacks
    disallowed — a silent fallback would measure a different endpoint than the
    one under test. `sort:*` deliberately allows fallbacks: it is a policy, not
    a pin.
    """
    if not pin or pin == "auto":
        return None
    if pin.startswith("sort:"):
        return {"sort": pin.split(":", 1)[1], "allow_fallbacks": True}
    if pin.startswith("order:"):
        return {"order": [p for p in pin.split(":", 1)[1].split("+") if p],
                "allow_fallbacks": True}
    return {"only": [pin], "allow_fallbacks": allow_fallbacks}


def pricing_for(cfg: dict) -> dict:
    """Price sheet for the computed-cost fallback: the pinned endpoint's when
    known, else the catalogue block."""
    return PROVIDERS.get(cfg.get("pin") or "auto", MODELS[cfg["model"]])


def config_label(cfg: dict) -> str:
    """Deterministic name for a run configuration; the results-file key."""
    pin = cfg.get("pin") or "auto"
    extra = [x for x in (cfg["variant"] if cfg["variant"] != "full" else "",
                         cfg["answer"] if cfg["answer"] != "json" else "",
                         "r:" + cfg["reasoning_mode"]
                         if cfg["reasoning_mode"] != "default" else "") if x]
    if pin == "auto" and not extra:
        return cfg["model"]
    return "|".join([cfg["model"].split("/")[-1]]
                     + ([pin] if pin != "auto" else []) + extra)


def build_configs(models: list[str], args) -> list[tuple[str, dict]]:
    """Cartesian product of the sweep dimensions, one labelled config each."""
    picks = lambda raw, table, what: ([v.strip() for v in raw.split(",") if v.strip()]  # noqa: E731
                                      or list(table))
    pins = [p.strip() for p in args.providers.split(",") if p.strip()] or ["auto"]
    variants = picks(args.variants, PROMPT_VARIANTS, "variant") if args.variants else ["full"]
    answers = picks(args.answers, ANSWER_FORMATS, "answer") if args.answers else ["json"]
    reasons = picks(args.reasoning, REASONING_MODES, "reasoning") \
        if args.reasoning else ["default"]
    bad = ([f"provider {p}" for p in pins
            if p != "auto" and not p.startswith(("sort:", "order:")) and p not in PROVIDERS]
           + [f"variant {v}" for v in variants if v not in PROMPT_VARIANTS]
           + [f"answer {a}" for a in answers if a not in ANSWER_FORMATS]
           + [f"reasoning {r}" for r in reasons if r not in REASONING_MODES])
    if bad:
        raise SystemExit(f"unknown {', '.join(bad)}; known providers {list(PROVIDERS)}, "
                         f"variants {list(PROMPT_VARIANTS)}, answers {list(ANSWER_FORMATS)}, "
                         f"reasoning {list(REASONING_MODES)}")
    configs = []
    for model in models:
        for pin in pins:
            for variant in variants:
                for answer in answers:
                    for reason in reasons:
                        cfg = {"model": model, "pin": pin,
                               "provider": provider_block(pin, args.allow_fallbacks),
                               "variant": variant, "answer": answer,
                               "reasoning_mode": reason,
                               "reasoning": REASONING_MODES[reason]}
                        configs.append((config_label(cfg), cfg))
    return configs


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
    ap.add_argument("--models", default="", help="comma-separated model ids (default: all known in MODELS)")
    ap.add_argument("--providers", default="",
                    help=f"endpoint pin(s): a name from {list(PROVIDERS)}, sort:price, "
                         "sort:throughput, order:Relace+Wafer (tries in order, "
                         "falls back), or auto (default). Cartesian with the other axes.")
    ap.add_argument("--allow-fallbacks", action="store_true",
                    help="allow a pinned endpoint to fall back (default: strict pin; a "
                         "fallback is still counted per row when it happens)")
    ap.add_argument("--variants", default="", help="prompt variants: full,compact")
    ap.add_argument("--answers", default="",
                    help="answer formats: json,json-noconf,label,letter")
    ap.add_argument("--reasoning", default="",
                    help=f"reasoning modes: {list(REASONING_MODES)}")
    ap.add_argument("--utterances", default=str(ROOT / "tests/data/utterances_v2.json"))
    ap.add_argument("--limit", type=int, default=0, help="run only the first N rows")
    ap.add_argument("--sample", type=int, default=0,
                    help="deterministic stride sample of N rows across the whole set "
                         "(unlike --limit, which is the first N and is band-skewed)")
    ap.add_argument("--concurrency", type=int, default=8, help="max requests in flight (accuracy pass)")
    ap.add_argument("--latency-n", type=int, default=20, help="sequential latency rows per model")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--model-max-tokens", action="append", default=[], metavar="MODEL=N",
                    help="per-model max_tokens override, repeatable (a reasoning model "
                         "truncates its JSON answer at a cap tuned for non-reasoning ones)")
    ap.add_argument("--model-budget-usd", type=float, default=0.0,
                    help="stop a config once provider-reported spend reaches this "
                         "(0 = no cap, the default)")
    ap.add_argument("--budget-usd", type=float, default=0.0,
                    help="stop the whole run once total spend reaches this "
                         "(0 = no cap, the default)")
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
    if args.sample and 0 < args.sample < len(rows):
        step = len(rows) / args.sample
        rows = [rows[int(i * step)] for i in range(args.sample)]
    wanted = [m.strip() for m in args.models.split(",") if m.strip()] or list(MODELS)
    unknown = [m for m in wanted if m not in MODELS]
    if unknown:
        raise SystemExit(f"unknown model id(s): {unknown}; known: {list(MODELS)}")

    if args.dry_run:
        print(build_system_prompt())
        print(f"\nuser: {rows[0]['utterance']}")
        print(f"\nmodels: {wanted}\nrows: {len(rows)}")
        for label, cfg in build_configs(wanted, args):
            payload = build_payload(cfg, rows[0]["utterance"], args.max_tokens, True)
            print(f"\n[{label}] prompt_chars={len(payload['messages'][0]['content'])} "
                  f"provider={payload.get('provider')} reasoning={payload.get('reasoning')}")
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
        "sample": args.sample or None,
        "bands": {b: sum(1 for r in rows if r.get("band") == b)
                  for b in sorted({r.get("band") for r in rows if r.get("band")})},
        "concurrency": args.concurrency,
        "latency_run": {"n": 0 if args.no_latency_run else len(latency_picks),
                        "mode": "sequential",
                        "row_indices": [] if args.no_latency_run else [i for i, _ in latency_picks]},
        "prompt_source": "bench/ti_bench/prompts.py INSTRUCTIONS + OP_DESCRIPTIONS (verbatim import)",
        "request": {"temperature": 0, "max_tokens": args.max_tokens,
                    "max_tokens_by_model": args.max_tokens_by_model,
                    "response_format": '{"type":"json_object"} on json answers only',
                    "usage_include": True,
                    "allow_fallbacks": args.allow_fallbacks},
        "budget": {"per_model_usd": args.model_budget_usd, "total_usd": args.budget_usd},
        "sweep": {"providers": args.providers or "auto",
                  "variants": args.variants or "full",
                  "answers": args.answers or "json",
                  "reasoning": args.reasoning or "default"},
        "baselines": {
            "local_d1_3B_Q4": {"model": "d1-3B-Q4_K_M (local llama.cpp)",
                               "accuracy": 0.74, "p50_ms": 147.9,
                               "source": "bench/results/dell/v2-d1-3B-Q4-run2.json"},
            "char_4gram_1nn": {"model": "char 4-gram 1-NN (leave-one-out)",
                               "accuracy": 0.54, "p50_ms": 0.5,
                               "source": "recorded with the sweep; not restated in any doc"},
        },
        "models": {},
    }

    ledger = Ledger(args.model_budget_usd, args.budget_usd)
    for label, cfg in build_configs(wanted, args):
        pricing = pricing_for(cfg)
        ledger.begin_model()
        state = {"json_mode": cfg["answer"].startswith("json"), "ledger": ledger}
        print(f"[{label}] accuracy pass: {len(rows)} rows, {args.concurrency} in flight, "
              f"max_tokens={max_tokens_for(args, cfg['model'])}", flush=True)
        t_model = time.perf_counter()
        acc_rows = run_concurrent(cfg, rows, key, args, state)
        latency_rows: list[dict] = []
        if not args.no_latency_run:
            print(f"[{label}] latency pass: {len(latency_picks)} rows, strictly sequential",
                  flush=True)
            latency_rows = run_sequential(cfg, latency_picks, key, args, state)
        entry = aggregate(acc_rows, latency_rows, pricing)
        entry["pricing_usd_per_mtok"] = pricing
        entry["config"] = {k: v for k, v in cfg.items()}
        entry["max_tokens"] = max_tokens_for(args, cfg["model"])
        entry["budget_stop"] = ledger.stop_reason()
        entry["json_mode"] = "accepted" if state["json_mode"] else "rejected, text parse used"
        entry["wall_time_s"] = round(time.perf_counter() - t_model, 1)
        summary["models"][label] = entry
        print(f"[{label}] acc={entry['accuracy']:.1%} "
              f"p50={entry['latency_ms']['p50']}ms malformed={entry['malformed']} "
              f"truncated={entry['truncated']} skipped={entry['skipped']} "
              f"failures={entry['failures']} fallbacks={entry['fallbacks']} "
              f"served={entry['providers_served']} cost=${entry['cost']['total_usd']:.5f}",
              flush=True)

    summary["wall_time_s"] = round(time.perf_counter() - started, 1)
    summary["total_cost_usd"] = round(
        sum(m["cost"]["total_usd"] for m in summary["models"].values()), 6)
    summary["total_decisions"] = sum(len(m["per_row"]) + len(m["latency_rows"])
                                     for m in summary["models"].values())
    summary["request_metrics"] = dict(_METRICS)
    summary["ranking"] = rank(summary["models"])

    out.parent.mkdir(parents=True, exist_ok=True)
    out = pick_output(out)
    out.write_text(json.dumps(summary, indent=2))
    print_report(summary)
    shown = out.relative_to(ROOT) if ROOT in out.parents else out
    print(f"\nwrote {shown}")


if __name__ == "__main__":
    main()
