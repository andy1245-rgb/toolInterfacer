"""disk_lookup — one-hop file resolution + read (L1/L2 bridge template)."""
from __future__ import annotations

from ...core.gates import Gate
from ...core.runner import Step


def apply(intent: str, digest: dict) -> list[Step]:
    return [
        Step(id="candidates", op="filesys_glob",
             params={"pattern": "**/*", "cwd": str(digest.get("cwd", "."))}),
        Step(id="picked_file", op="__decide_pick__",
             params={"source_artifact": "candidates",
                     "instructions": "Which candidate file matches the intent best?",
                     "top_k": 1}),
        Step(id="content", op="filesys_read", inject={"path": "picked_file"},
             gate=Gate(pre=[{"artifact": "picked_file", "kind": "conf_gte", "min": 0.7}]),
             params={}),
    ]
