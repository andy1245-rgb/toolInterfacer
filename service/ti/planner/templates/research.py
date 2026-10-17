"""research — search → pick top-k → fetch+verify loop → digest.

Steps are declarative; every specific (query counts, k, page cap) resolves
from conf.research at apply()-time. P1 ships the step graph in fixture mode;
P4 wires real CLIs.
"""
from __future__ import annotations

from ...core.gates import Gate
from ...core.runner import Step
from ...conf import load


def apply(intent: str, digest: dict) -> list[Step]:
    cfg = load()["research"]
    k = int(cfg["pick_top_k"])
    steps: list[Step] = [
        Step(id="queries", op="__decide_extract__",
             params={"instructions": "Extract 3-6 search queries for the intent."}),
        Step(id="hits", op="web_search", inject={"queries": "queries"},
             params={}),
        Step(id="picked", op="__decide_pick__",
             params={"source_artifact": "hits", "instructions": "Pick the top relevant URLs.",
                     "top_k": k}),
        Step(id="pages", op="web_fetch", inject={"urls": "picked"},
             gate=Gate(pre=[{"artifact": "picked", "kind": "conf_gte", "min": 0.7}]),
             params={}),
    ]
    return steps
