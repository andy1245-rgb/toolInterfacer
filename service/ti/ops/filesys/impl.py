"""filesys.read / glob / grep — the pi tools' service-side mirrors.

Same step shapes pi's registration exposes, so results are plausibly the same
in bench and in-session. P1 implements these (pure stdlib, no CLI needed).

Every path these ops touch goes through `confine`: it resolves the path and
refuses one outside the run root, so `params.path`, `params.cwd` and `digest.cwd`
cannot name `/etc` (or climb to it, or symlink to it). The walk is bounded by
`walk`'s depth/file/deadline ceilings and prunes the configured deny roots, and
`pattern` is compiled only if `regexguard` accepts its shape. `conf` reports
whether the answer is complete: it used to be 1.0 unconditionally, which made a
truncated listing look like a whole one.
"""
from __future__ import annotations

import fnmatch
import os
import time
from dataclasses import dataclass
from typing import Any, Iterator

from ...conf import load
from ..op_api import op
from . import confine, regexguard
from .walk import DEADLINE, MAX_HITS, Limits, WalkState, walk


@dataclass
class FileHit:
    path: str
    excerpt: str


def _conf() -> dict[str, Any]:
    return load()["ops"]["filesys"]


def _walk(root: str, limits: Limits | None = None,
          state: WalkState | None = None) -> Iterator[tuple[str, list[str]]]:
    """The ops' walk: `walk.walk` under the `ops.filesys` ceilings."""
    for dirpath, files, _depth in walk(root, limits, state):
        yield dirpath, files


def _confidence(state: WalkState) -> float:
    """How much of the tree this answer covers.

    1.0 is a walk that saw everything it was allowed to and refused nothing.
    0.7 is a cap or a deny-list prune: a real, complete-over-what-it-saw listing
    of part of the tree. 0.5 is the deadline, which is the one that says "there
    was more here than I could look at".
    """
    if state.complete and not state.refused:
        return 1.0
    if state.stopped == DEADLINE:
        return 0.5
    return 0.7


def _matched(rel: str, pattern: str) -> bool:
    """Glob semantics for a listing pattern, over a walk-relative path.

    `**/` is a depth prefix that also matches zero directories, so a top-level
    entry is a candidate for `**/bench_*`. Because fnmatch's `*` crosses `/`,
    the prefix is stripped and the remainder matched against the entry's own
    name: matching the whole relative path instead let a directory component
    satisfy the wildcard and returned `bench_x/lane.py` for `**/bench_*`.
    Matching is case-insensitive: a caller writing `dockerfile` means Dockerfile.
    """
    name = rel.replace(os.sep, "/").lower()
    pat = pattern.replace(os.sep, "/").lower()
    if not pat.startswith("**/"):
        return fnmatch.fnmatchcase(name, pat)
    # `**/` is a depth prefix, not a path component: strip every one of them and
    # match the rest against the entry's own name. Comparing the stripped form
    # against the whole path instead let a directory satisfy the wildcard.
    rest = pat
    while rest.startswith("**/"):
        rest = rest[3:]
    return fnmatch.fnmatchcase(name.rsplit("/", 1)[-1], rest)


@op("filesys_glob", kind="path_list", required=["pattern"], optional=["cwd", "base"])
def run(params: dict[str, Any], store: Any) -> tuple[str, Any, float]:
    cfg = _conf()
    limits = Limits.from_conf()
    root = confine.resolve(params.get("cwd") or ".")
    # `base` only changes the frame hits are reported in: a scoped listing walks
    # the directory it was given but still speaks the caller's cwd. The walk root
    # and the pattern both stay where they are, and `base` is not confined because
    # it is not a path this op acts on — it is the coordinate system the answer is
    # printed in, so naming `/` there reveals nothing the walk did not already
    # return and stats nothing.
    frame = os.path.expanduser(str(params["base"])) if params.get("base") else root
    pattern = str(params["pattern"])
    max_hits = int(cfg["glob"]["max_hits"])
    state = WalkState()
    hits: list[str] = []
    for dirpath, files in _walk(root, limits, state):
        for name in files:
            full = os.path.join(dirpath, name)
            if _matched(os.path.relpath(full, root), pattern):
                hits.append(os.path.relpath(full, frame))
                if len(hits) >= max_hits:
                    state.stop(MAX_HITS)
                    break
        if state.stopped is not None:
            break
    _ = store
    return "path_list", hits, _confidence(state)


@op("filesys_read", kind="file_content", required=["path"], optional=["max_lines"])
def run(params: dict[str, Any], store: Any) -> tuple[str, Any, float]:
    cfg = _conf()["read"]
    path = confine.resolve(params["path"])
    limit = min(int(params.get("max_lines", cfg["max_lines"])), int(cfg["max_lines"]))
    byte_cap = int(cfg["max_bytes"])
    chunks: list[str] = []
    size = 0
    truncated = False
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for index, line in enumerate(fh):
            if index >= limit:
                truncated = True
                break
            room = byte_cap - size
            if room <= 0:
                truncated = True
                break
            if len(line) > room:          # one long line: the answer is its head
                chunks.append(line[:room])
                truncated = True
                break
            chunks.append(line)
            size += len(line)
    _ = store
    return "file_content", "".join(chunks), 0.7 if truncated else 1.0


@op("filesys_grep", kind="grep_hits", required=["pattern", "cwd"], optional=["glob"])
def run(params: dict[str, Any], store: Any) -> tuple[str, Any, float]:
    cfg = _conf()
    gcfg = cfg["grep"]
    max_chars = int(gcfg["pattern_max_chars"])
    pattern = regexguard.compile_guarded(str(params["pattern"]), max_chars=max_chars)
    name_filter = (regexguard.compile_glob(str(params["glob"]), max_chars=max_chars)
                   if params.get("glob") else None)
    root = confine.resolve(params["cwd"])
    limits = Limits.from_conf()
    max_hits = int(gcfg["max_hits"])
    line_cap = int(gcfg["line_max_chars"])
    per_dir = int(gcfg["files_per_dir"])
    deadline = time.monotonic() + limits.deadline_s
    state = WalkState()
    hits: list[FileHit] = []
    for dirpath, files in _walk(root, limits, state):
        for name in files[:per_dir]:
            if len(hits) >= max_hits:
                state.stop(MAX_HITS)
                break
            if name_filter is not None and not name_filter.match(name):
                continue
            if time.monotonic() > deadline:
                state.stop(DEADLINE)
                break
            path = os.path.join(dirpath, name)
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        if time.monotonic() > deadline:
                            state.stop(DEADLINE)
                            break
                        # The match runs on windows of the clipped line: a
                        # bound on the subject is the only budget `re` honours,
                        # and a window keeps a minified one-line file searchable.
                        # A match straddling a window boundary is not found.
                        if any(pattern.search(line[start:start + line_cap])
                               for start in range(0, max(len(line), 1), line_cap)):
                            hits.append(FileHit(os.path.relpath(path, root), line[:200]))
                            if len(hits) >= max_hits:
                                state.stop(MAX_HITS)
                                break
            except OSError:
                continue
        if state.stopped is not None:
            break
    _ = store
    return "grep_hits", [hit.__dict__ for hit in hits], _confidence(state)
