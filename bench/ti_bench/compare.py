"""compare — two reports, one McNemar number and one latency delta.

Rows are paired on ``uid`` (a hash of the utterance text), never on position: a
reordered or re-limited set would otherwise produce a paired statistic over
unrelated rows that looks perfectly reasonable. A set mismatch is an error, not a
silent intersection — the whole point of a frozen set is that both arms answered
the same questions.

The accuracy test is the *exact* McNemar (binomial) test, not chi-square: with
discordant counts in the 0-10 range that a 150-row set actually produces, the
approximation's p-value is not trustworthy.
"""
from __future__ import annotations

import statistics

from . import stats
from .scoring import ERROR


class CompareError(ValueError):
    """The two reports cannot be compared as a paired measurement."""


def compare(a: dict, b: dict) -> dict:
    """McNemar exact + paired latency delta for report ``a`` vs report ``b``."""
    set_a, set_b = a["set"]["name"], b["set"]["name"]
    if set_a != set_b:
        raise CompareError(f"different frozen sets: {set_a!r} vs {set_b!r}")
    rows_a = {r["uid"]: r for r in a["per_row"]}
    rows_b = {r["uid"]: r for r in b["per_row"]}
    shared = [uid for uid in rows_a if uid in rows_b]
    if len(shared) != len(rows_a) or len(shared) != len(rows_b):
        only_a = len(rows_a) - len(shared)
        only_b = len(rows_b) - len(shared)
        raise CompareError(
            f"set {set_a!r} does not line up: {only_a} row(s) only in {a['arm']['name']}, "
            f"{only_b} only in {b['arm']['name']} — compare needs the same rows")
    shared.sort(key=lambda uid: rows_a[uid]["idx"])

    both_correct = both_wrong = a_only = b_only = 0
    deltas, a_faster, b_faster, equal = [], 0, 0, 0
    for uid in shared:
        ra, rb = rows_a[uid], rows_b[uid]
        if ra["correct"] and rb["correct"]:
            both_correct += 1
        elif ra["correct"]:
            a_only += 1
        elif rb["correct"]:
            b_only += 1
        else:
            both_wrong += 1
        if (ra["latency_ms"] is not None and rb["latency_ms"] is not None
                and ra["source"] != ERROR and rb["source"] != ERROR):
            delta = rb["latency_ms"] - ra["latency_ms"]
            deltas.append(delta)
            if delta < 0:
                b_faster += 1
            elif delta > 0:
                a_faster += 1
            else:
                equal += 1

    lat_a, lat_b = a["latency"], b["latency"]
    return {
        "set": set_a,
        "n": len(shared),
        "arms": {"a": _arm_of(a), "b": _arm_of(b)},
        "accuracy": {"a": a["accuracy"], "b": b["accuracy"]},
        "mcnemar": {"both_correct": both_correct, "both_wrong": both_wrong,
                    "a_only": a_only, "b_only": b_only,
                    "discordant": a_only + b_only,
                    "p_value": stats.mcnemar_exact(a_only, b_only)},
        "latency": {
            "n_pairs": len(deltas),
            "a": {"p50": lat_a["p50"], "p90": lat_a["p90"], "p99": lat_a["p99"]},
            "b": {"p50": lat_b["p50"], "p90": lat_b["p90"], "p99": lat_b["p99"]},
            "delta_percentiles": {"p50": round(lat_b["p50"] - lat_a["p50"], 3),
                                  "p90": round(lat_b["p90"] - lat_a["p90"], 3),
                                  "p99": round(lat_b["p99"] - lat_a["p99"], 3)},
            "median_paired_delta": round(statistics.median(deltas), 3) if deltas else None,
            "mean_paired_delta": round(statistics.fmean(deltas), 3) if deltas else None,
            "b_faster": b_faster, "a_faster": a_faster, "equal": equal,
        },
    }


def _arm_of(report: dict) -> dict:
    arm = report["arm"]
    return {"name": arm["name"], "backend": arm["backend"], "offline": arm.get("offline")}


def render(data: dict) -> list[str]:
    """The human-readable comparison, as lines."""
    a, b = data["arms"]["a"], data["arms"]["b"]
    acc, mc, lat = data["accuracy"], data["mcnemar"], data["latency"]
    out = [f"# compare  A={a['name']} ({a['backend']}, {_where(a)})  "
           f"B={b['name']} ({b['backend']}, {_where(b)})",
           f"# set={data['set']} n={data['n']} paired on row uid", "",
           "## accuracy — McNemar exact (two-sided)",
           f"  both correct    {mc['both_correct']:>5}",
           f"  both wrong      {mc['both_wrong']:>5}",
           f"  A only correct  {mc['a_only']:>5}",
           f"  B only correct  {mc['b_only']:>5}",
           f"  discordant      {mc['discordant']:>5}"
           f"   (b={mc['a_only']} c={mc['b_only']})   exact p = {_p(mc['p_value'])}"]
    for side, block in (("A", acc["a"]), ("B", acc["b"])):
        out.append(f"  {side}  {block['correct']}/{block['n']} = {_pct(block['accuracy'])}"
                   f"  wilson95 [{_pct(block['wilson95']['low'])}, "
                   f"{_pct(block['wilson95']['high'])}]")
    out += ["", f"## latency ms (paired on {lat['n_pairs']} rows, error rows excluded)",
            f"  {'arm':<6}{'p50':>10}{'p90':>10}{'p99':>10}"]
    for side in ("a", "b"):
        row = lat[side]
        out.append(f"  {side.upper():<6}{row['p50']:>10.3f}{row['p90']:>10.3f}"
                   f"{row['p99']:>10.3f}")
    d = lat["delta_percentiles"]
    if lat["n_pairs"]:
        out.append(f"  {'B-A':<6}{d['p50']:>+10.3f}{d['p90']:>+10.3f}{d['p99']:>+10.3f}")
        out.append(f"  median paired delta (B-A) {lat['median_paired_delta']:+.3f} ms · "
                   f"mean {lat['mean_paired_delta']:+.3f} ms · "
                   f"B faster on {lat['b_faster']}, slower on {lat['a_faster']}, "
                   f"equal on {lat['equal']}")
    else:
        out.append("  no rows carry a latency on both sides")
    return out


def _p(value: float) -> str:
    return f"{value:.3e}" if 0 < value < 1e-4 else f"{value:.6f}"


def _where(arm: dict) -> str:
    return "offline" if arm.get("offline") else "endpoint"


def _pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"
