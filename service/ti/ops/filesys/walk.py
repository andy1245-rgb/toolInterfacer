"""walk — `os.walk` with a ceiling on every dimension a caller can inflate.

Four caps, all checked *between* entries so our own work cannot starve them:

  max_depth     how far down the walk goes. Nothing legitimate in a workspace
                sits 40 directories deep, and a caller can build that tree.
  max_files     how many files the walk may hand back (``WalkState.files``).
                This is what bounds a grep over a large tree, and a glob over a
                tree of dirs whose names no pattern matches.
  deadline_s    a wall-clock ceiling read with ``time.monotonic()`` before each
                directory and each entry.
  deny list     a directory at or under a configured deny root is not descended
                into at all (see ``confine``). This is the cap that matters on
                this box: an ``os.walk`` into a dead FUSE mount blocks in
                uninterruptible sleep, and no deadline in this process can reach
                a thread already inside it. The prune is a string test on the
                directory path, done before the descent, so no syscall ever
                enters the mount.

One skip policy for both ops: hidden directories and the vendored/generated names
in SKIP_DIRS are not walked. A ``.git`` or ``node_modules`` tree is never a
sensible answer to "read the file we discussed" either, and walking one is the
second largest source of "this should have returned instantly".
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Iterator

from ...conf import load
from . import confine

# Vendored, generated, and binary trees: never a sensible answer to "read the
# file we discussed", and they dominate the candidate list (extern/llama.cpp/build
# alone contributes thousands of .d files, which then cost decision-model prefill).
SKIP_DIRS = frozenset({"build", "dist", "extern", "models", "node_modules",
                       "venv", ".venv", "__pycache__", ".git", ".mypy_cache",
                       ".pytest_cache", ".ruff_cache"})

COMPLETE = "complete"
MAX_DEPTH = "max_depth"
MAX_FILES = "max_files"
MAX_HITS = "max_hits"
DEADLINE = "deadline"


@dataclass(frozen=True)
class Limits:
    """The walk's ceilings, from `ops.filesys` in conf."""
    max_depth: int
    max_files: int
    deadline_s: float

    @classmethod
    def from_conf(cls) -> "Limits":
        cfg = load()["ops"]["filesys"]
        return cls(max_depth=int(cfg["max_depth"]),
                   max_files=int(cfg["max_files"]),
                   deadline_s=float(cfg["deadline_s"]))


@dataclass
class WalkState:
    """What the walk did, for the op to report rather than to swallow.

    `stopped` is None for a walk that saw everything it was allowed to; a cap
    names itself otherwise. `refused` records deny-list prunes, so a listing that
    skipped a mount says so in its confidence instead of looking complete.
    """
    stopped: str | None = None
    files: int = 0
    dirs: int = 0
    refused: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.stopped is None

    def stop(self, reason: str) -> None:
        if self.stopped is None:
            self.stopped = reason


def walk(root: str, limits: Limits | None = None,
         state: WalkState | None = None) -> Iterator[tuple[str, list[str], int]]:
    """Yield ``(dirpath, filenames, depth)`` depth-first under `root`.

    `root` is expected to be a confined real path (``confine.resolve``); the deny
    check is applied to every directory *before* it is descended into, so it holds
    for subdirectories the caller never named. The iteration order matches
    ``os.walk``: a directory's files, then its subdirectories in readdir order.
    """
    limits = limits if limits is not None else Limits.from_conf()
    state = state if state is not None else WalkState()
    deadline = time.monotonic() + limits.deadline_s
    stack: list[tuple[str, int]] = [(root, 0)]
    while stack:
        if state.stopped is not None:
            return
        if time.monotonic() > deadline:
            state.stop(DEADLINE)
            return
        dirpath, depth = stack.pop()
        try:
            with os.scandir(dirpath) as listing:
                entries = list(listing)
        except OSError:
            continue
        files: list[str] = []
        subdirs: list[tuple[str, int]] = []
        for entry in entries:
            if time.monotonic() > deadline:
                state.stop(DEADLINE)
                return
            name = entry.name
            if name.startswith("."):
                continue
            child = os.path.join(dirpath, name)
            # The deny check comes before anything that could stat: `is_dir` is a
            # readdir d_type on most filesystems and a stat on the rest, and a
            # stat on the wrong directory is the hang this list exists to avoid.
            if confine.is_denied(child):
                state.refused.append(child)
                continue
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                continue
            if not is_dir:
                state.files += 1
                files.append(name)
                if state.files >= limits.max_files:
                    state.stop(MAX_FILES)
                continue
            if name in SKIP_DIRS:
                continue
            if depth + 1 > limits.max_depth:
                state.stop(MAX_DEPTH)
                continue
            subdirs.append((child, depth + 1))
        state.dirs += 1
        yield dirpath, files, depth
        if state.stopped is not None:
            return
        stack.extend(reversed(subdirs))
