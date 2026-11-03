"""filesys.read / glob / grep — the pi tools' service-side mirrors.

Same step shapes pi's registration exposes, so results are plausibly the same
in bench and in-session. P1 implements these (pure stdlib, no CLI needed).
"""
from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass
from typing import Any

from ..op_api import op


@dataclass
class FileHit:
    path: str
    excerpt: str


# Vendored, generated, and binary trees: never a sensible answer to "read the
# file we discussed", and they dominate the candidate list (extern/llama.cpp/build
# alone contributes thousands of .d files, which then cost decision-model prefill).
SKIP_DIRS = frozenset({"build", "dist", "extern", "models", "node_modules",
                       "venv", ".venv", "__pycache__", ".git", ".mypy_cache",
                       ".pytest_cache", ".ruff_cache"})


def _walk(root: str):
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in SKIP_DIRS]
        yield dirpath, files


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
    cwd = params.get("cwd") or os.getcwd()
    root = os.path.expanduser(cwd)
    # `base` only changes the frame hits are reported in: a scoped listing walks
    # the directory it was given but still speaks the caller's cwd. The walk root
    # and the pattern both stay where they are.
    frame = os.path.expanduser(params["base"]) if params.get("base") else root
    hits: list[str] = []
    for dirpath, files in _walk(root):
        for f in files:
            if f.startswith("."):
                continue
            full = os.path.join(dirpath, f)
            if _matched(os.path.relpath(full, root), params["pattern"]):
                hits.append(os.path.relpath(full, frame))
    _ = store
    return "path_list", hits, 1.0


@op("filesys_read", kind="file_content", required=["path"], optional=["max_lines", "cwd"])
def run(params: dict[str, Any], store: Any) -> tuple[str, Any, float]:
    """Read one file. A relative `path` resolves against `cwd`, never the process's.

    The service is spawned with its own cwd (lifecycle.ts pins it to the repo root)
    while every path in a request is in the *caller's* frame: filesys_glob reports
    hits relative to the digest cwd, so a picked name read against the service's
    own cwd raised ENOENT for every session whose cwd was not the service's
    (observed live: `[Errno 2] No such file or directory: 'notes.md'`).
    """
    path = os.path.expanduser(str(params["path"]))
    cwd = params.get("cwd")
    if cwd and not os.path.isabs(path):
        path = os.path.join(os.path.expanduser(str(cwd)), path)
    limit = int(params.get("max_lines", 2000))
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = [next_l for idx, next_l in enumerate(fh) if idx < limit]
    _ = store
    return "file_content", "".join(lines), 1.0


@op("filesys_grep", kind="grep_hits", required=["pattern", "cwd"], optional=["glob"])
def run(params: dict[str, Any], store: Any) -> tuple[str, Any, float]:
    """Naive first cut (P1): single-file or re.search-per-file line hits."""
    pat = re.compile(params["pattern"])
    cwd = os.path.expanduser(params["cwd"])
    hits: list[FileHit] = []
    for dirpath, _dirs, files in os.walk(cwd):
        for f in files[:50]:
            fpath = os.path.join(dirpath, f)
            if params.get("glob") and not re.match(params["glob"].replace("**", ".*"), f):
                continue
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        if pat.search(line) and len(hits) < 200:
                            hits.append(FileHit(os.path.relpath(fpath, cwd), line[:200]))
            except OSError:
                continue
    _ = store
    return "grep_hits", [h.__dict__ for h in hits], 1.0
