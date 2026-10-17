"""Confidence helpers — settle gate arithmetic here, not inside steps."""
from __future__ import annotations

from ..conf import load


def gate_route() -> float:
    return float(load()["policy"]["gate_route"])


def gate_slot() -> float:
    return float(load()["policy"]["gate_slot"])


def gate_noul() -> float:
    return float(load()["policy"]["gate_noul"])
