"""disk_lookup — the L1 disk slot-fill, in two shapes.

A listing intent terminates on the candidate list: there is no single file to
read, so asking decide_pick which one "matches best" is a question with no
answer (live, it returned conf 0.548 and the slot gate turned the run into a
degraded best guess). A singular intent keeps the pick → read chain unchanged.

The template owns its terminal: build() returns the steps and the artifact key
the run must end on, so no call site has to guess which shape it got.
"""
from __future__ import annotations

from ...conf import load
from ...core.gates import Gate
from ...core.runner import Step
from ..shape import ENUMERATIVE, classify


def build(intent: str, digest: dict) -> tuple[list[Step], str]:
    candidates = Step(id="candidates", op="filesys_glob",
                      params={"pattern": "**/*", "cwd": str(digest.get("cwd", "."))})
    if classify(intent) == ENUMERATIVE:
        return [candidates], "candidates"
    cfg = load()["policy"]
    return [
        candidates,
        Step(id="picked_file", op="decide_pick",
             params={"source_artifact": "candidates",
                     "instructions": "Which candidate file matches the intent best?",
                     "top_k": 1}),
        Step(id="content", op="filesys_read", inject={"path": "picked_file"},
             gate=Gate(pre=[{"artifact": "picked_file", "kind": "conf_gte",
                             "min": float(cfg["gate_slot"])}]),
             params={}),
    ], "content"
