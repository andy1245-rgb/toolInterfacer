from ...core.gates import Gate
from ...core.runner import Step
from ...conf import load


def apply(intent: str, digest: dict) -> list[Step]:
    cfg = load()["policy"]
    return [
        Step(id="candidates", op="filesys_glob",
             params={"pattern": "**/*", "cwd": str(digest.get("cwd", "."))}),
        Step(id="picked_file", op="decide_pick",
             params={"source_artifact": "candidates",
                     "instructions": "Which candidate file matches the intent best?",
                     "top_k": 1}),
        Step(id="content", op="filesys_read", inject={"path": "picked_file"},
             gate=Gate(pre=[{"artifact": "picked_file", "kind": "conf_gte",
                             "min": float(cfg["gate_slot"])}]),
             params={}),
    ]
