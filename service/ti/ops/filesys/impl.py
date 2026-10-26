"""filesys.read / glob / grep — the pi tools' service-side mirrors.

Same step shapes pi's registration exposes, so results are plausibly the same
in bench and in-session. P1 implements these (pure stdlib, no CLI needed).
"""
from __future__ import annotations

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


@op("filesys_glob", kind="path_list", required=["pattern"], optional=["cwd"])
def run(params: dict[str, Any], store: Any) -> tuple[str, Any, float]:
    cwd = params.get("cwd") or os.getcwd()
    root = os.path.expanduser(cwd)
    hits: list[str] = []
    for dirpath, files in _walk(root):
        for f in files:
            if f.startswith("."):
                continue
            rel = os.path.relpath(os.path.join(dirpath, f), root)
            if re.match(params["pattern"].replace("**", ".*"), rel):
                hits.append(rel)
    _ = store
    return "path_list", hits, 1.0


@op("filesys_read", kind="file_content", required=["path"], optional=["max_lines"])
def run(params: dict[str, Any], store: Any) -> tuple[str, Any, float]:
    path = os.path.expanduser(params["path"])
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
