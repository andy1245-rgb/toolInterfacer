"""conf_scale — put every backend's confidence on one comparable scale.

Two backends read the same decision wall but report confidence differently:

  * d1_systemone (llama.cpp /v1/systemone) returns
    ``confidence_choice = (p_max - 1/K) / (1 - 1/K)`` — the TypeSafe linear
    formula the server applies to LFM2_D1 (server-decision.cpp
    ``decision_confidence_choice``).
  * logit_read returns ``1 - H / ln K`` — a reparameterized entropy.

They are not comparable: at K=8 a d1 conf of 0.7 is p_max ~0.74, whereas a
logit_read conf of 0.7 is p_max ~0.87, so one gate threshold selects different
peaks per backend (round1/A-model-performance.md, finding 7).

This module is pure and stdlib-only. It defines both formulas, and a dispatcher
that normalizes either backend's confidence onto the common peak-probability
(p_max) scale. Nothing here is wired into a default path yet — it is a helper
callers opt into.
"""
from __future__ import annotations

import math
from typing import Sequence

_BACKENDS = ("d1_systemone", "logit_read")


def pmax_from_probs(probs: Sequence[float]) -> float:
    """Peak probability of a (normalized) distribution."""
    return max(probs)


def d1_conf(pmax: float, k: int) -> float:
    """d1 / TypeSafe linear confidence, as the server computes it.

    Mirrors ``decision_confidence_choice`` (server-decision.cpp:1121-1128):
    degenerate (<2 options) returns 1.0 and the result is floored at 0.
    """
    if k < 2:
        return 1.0
    uniform = 1.0 / k
    return max(0.0, (pmax - uniform) / (1.0 - uniform))


def entropy_conf(probs: Sequence[float]) -> float:
    """``1 - H(p) / ln K`` — logit_read's reparameterized entropy confidence."""
    h = -sum(p * math.log(p) for p in probs if p > 0)
    return float(1.0 - h / math.log(len(probs)))


def to_common_scale(name: str, probs: Sequence[float]) -> float:
    """Normalize a backend's confidence onto the common p_max scale.

    d1_systemone's confidence is an exact affine function of p_max, so it
    inverts to p_max losslessly. logit_read's ``1 - H/lnK`` is not a function of
    p_max alone; like the research note's worked example, we assume the
    non-peak mass is spread uniformly and recover the p_max that reproduces the
    same entropy.
    """
    k = len(probs)
    if name == "d1_systemone":
        return _pmax_from_d1_conf(d1_conf(pmax_from_probs(probs), k), k)
    if name == "logit_read":
        return _pmax_from_entropy_conf(entropy_conf(probs), k)
    raise ValueError(f"conf_scale: unknown backend {name!r}; known: {_BACKENDS}")


def _pmax_from_d1_conf(conf: float, k: int) -> float:
    if k < 2:
        return 1.0
    return conf * (1.0 - 1.0 / k) + 1.0 / k


def _pmax_from_entropy_conf(conf: float, k: int) -> float:
    if k < 2:
        return 1.0
    conf = min(1.0, max(0.0, conf))
    if conf <= 0.0:
        return 1.0 / k
    if conf >= 1.0:
        return 1.0
    h_target = (1.0 - conf) * math.log(k)
    lo, hi = 1.0 / k, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _uniform_remainder_entropy(mid, k) > h_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _uniform_remainder_entropy(p: float, k: int) -> float:
    rest = (1.0 - p) / (k - 1)
    h = 0.0
    if p > 0.0:
        h -= p * math.log(p)
    if rest > 0.0:
        h -= (1.0 - p) * math.log(rest)
    return h
