"""Confidence helpers — settle gate arithmetic here, not inside steps."""
from __future__ import annotations

from ..conf import load


def gate_route() -> float:
    """Route gate, in p_max units (the common peak-probability axis).

    Every decide backend reports confidence on its own scale (see
    decide/conf_scale.py), so a route gate is only meaningful on the one axis
    they share: p_max = max(probabilities) over the K options. The cascade
    backend returns its answers on that axis — path-calibrated precision on a
    commit, the readout's own p_max on a defer — so 0.70 here means "a peak
    probability of 0.70" for whichever backend the cascade returns, and a
    deferral (p_max < decide.cascade.tau2 in the shipped config) sits below it.
    With decide.cascade enabled this is the same number as
    decide.cascade.tau_hi, i.e. the stage-1 commit line.
    """
    return float(load()["policy"]["gate_route"])


def gate_slot() -> float:
    return float(load()["policy"]["gate_slot"])


def gate_noul() -> float:
    return float(load()["policy"]["gate_noul"])
