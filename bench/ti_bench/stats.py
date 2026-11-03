"""stats — the numbers the harness reports, defined once.

Two of these are easy to get subtly wrong, so they live here and every bench
script imports them rather than carrying a copy:

* ``percentile`` — linear interpolation on the sorted sample, the definition
  ``bench/live_probe.py`` has always used (a bench that interpolates differently
  produces a p50 that does not match the historical lanes).
* ``wilson_ci`` / ``mcnemar_exact`` — the interval and the paired test. Stdlib
  only: ``math.comb`` is enough for the exact binomial, and adding scipy to a
  dependency-free repo to call one function is not a trade worth making.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

# two-sided 95% normal quantile; the value Python's statistics docs use
Z95 = 1.959963984540054


def percentile(vals: Sequence[float], pct: float) -> float:
    """Linearly interpolated percentile of ``vals`` (unsorted input is fine)."""
    if not vals:
        return 0.0
    vals = sorted(vals)
    idx = (len(vals) - 1) * pct
    lo, hi = int(idx), min(int(idx) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (idx - lo)


def wilson_ci(correct: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion; (0, 1) when n == 0.

    Preferred over the normal approximation at the sample sizes a lane actually
    produces (n=30 to n=150), where the normal interval can run outside [0, 1]
    and claims precision near 100% that the data does not support.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = correct / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p for ``b``/``c`` discordant pairs.

    Under H0 the discordant pairs split Binomial(b+c, 1/2). The exact two-sided
    p is ``2 * P(X <= min(b, c))`` capped at 1 — the binomial test, not the
    chi-square approximation, because discordant counts of 0-10 are the normal
    case here and the approximation is poor exactly there.
    """
    n = b + c
    if n <= 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2.0 * tail)


def rate(correct: int, n: int, z: float = Z95) -> dict:
    """A proportion with its Wilson interval, in the shape a report stores."""
    lo, hi = wilson_ci(correct, n, z)
    return {"correct": correct, "n": n,
            "accuracy": round(correct / n, 4) if n else None,
            "wilson95": {"low": round(lo, 4), "high": round(hi, 4)}}


def calibration(pairs: Iterable[tuple[float, bool]], width: float = 0.1) -> list[dict]:
    """Confidence bands -> observed accuracy (the reliability table).

    ``pairs`` is ``(confidence, correct)`` for rows that produced a *numeric*
    answer confidence; an abstain or an error has no confidence to calibrate and
    must not be binned as "conf 0.0, wrong" (that would fake a bad low band).
    """
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(round(1 / width))]
    for conf, correct in pairs:
        idx = min(max(int(conf / width), 0), len(bins) - 1)  # 1.0 lands in the top bin
        bins[idx].append((float(conf), bool(correct)))
    out = []
    for i, members in enumerate(bins):
        if not members:
            out.append({"lo": round(i * width, 2), "hi": round((i + 1) * width, 2),
                        "n": 0, "correct": 0, "accuracy": None, "mean_conf": None})
            continue
        hits = sum(1 for _, ok in members if ok)
        out.append({"lo": round(i * width, 2), "hi": round((i + 1) * width, 2),
                    "n": len(members), "correct": hits,
                    "accuracy": round(hits / len(members), 4),
                    "mean_conf": round(sum(c for c, _ in members) / len(members), 4)})
    return out
