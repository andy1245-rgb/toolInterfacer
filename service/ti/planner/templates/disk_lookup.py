"""disk_lookup — the L1 disk slot-fill, in two shapes.

A listing intent terminates on the candidate list: there is no single file to
read, so asking decide_pick which one "matches best" is a question with no
answer (live, it returned conf 0.548 and the slot gate turned the run into a
degraded best guess). A singular intent keeps the pick → read chain unchanged.

A listing is also only an answer if it covers the directory the caller meant, so
the listing shape roots its candidates glob at that directory — the supplied
params.path, else a directory named in the intent — and reports a named-but-
missing directory as an assumption rather than passing off the cwd's entries.

The template owns its terminal: build() returns the steps, the artifact key the
run must end on, and any assumption the plan itself establishes, so no call site
has to guess which shape it got.
"""
from __future__ import annotations

from typing import Any

from ...conf import load
from ...core.gates import Gate
from ...core.runner import Step
from ..shape import ENUMERATIVE, classify
from ..scope import resolve


def build(intent: str, digest: dict,
          params: dict[str, Any] | None = None) -> tuple[list[Step], str, list[str]]:
    if classify(intent) == ENUMERATIVE:
        step, assumptions = _listing(intent, params or {}, digest)
        return [step], "candidates", assumptions
    candidates = Step(id="candidates", op="filesys_glob",
                      params={"pattern": "**/*", "cwd": str(digest.get("cwd", "."))})
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
    ], "content", []


def _listing(intent: str, params: dict[str, Any],
             digest: dict) -> tuple[Step, list[str]]:
    """Root the glob at the named directory; report one that does not resolve.

    `base` keeps the hits in the caller's frame (the digest cwd), so a scoped
    listing reads as "bench/lane.sh" and not as a bare "lane.sh" whose
    directory the caller has to guess.
    """
    scope = resolve(intent, params, digest)
    if scope.root:
        return Step(id="candidates", op="filesys_glob",
                    params={"pattern": "**/*", "cwd": scope.root,
                            "base": scope.frame}), []
    step = Step(id="candidates", op="filesys_glob",
                params={"pattern": "**/*", "cwd": str(digest.get("cwd", "."))})
    if scope.named is None:
        return step, []
    return step, [f"listing: no directory {scope.named!r} here, so the listing "
                  f"covers the cwd ({scope.frame}) instead"]
