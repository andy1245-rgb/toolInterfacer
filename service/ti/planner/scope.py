"""scope — which directory a listing intent names, and whether it exists.

A listing is only an answer if it covers the directory the caller meant, so the
plan needs that directory before it runs. Two sources, in order: the path the
caller supplied, then a directory named lexically in the intent ("the bench
directory", "in bench/", "under docs", "inside src").

Resolution never guesses. A name that does not resolve comes back as `named`
with no root, so the run can say which name it could not honour instead of
quietly returning the cwd's entries as if they were the answer.

Lexical only: no model call, no network. Used by the listing shape alone — the
singular read path's candidate set must not move underneath decide_pick.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

# "<name> directory" and "in|under|inside|within <name>" — the two phrasings a
# caller uses to name a target. Matched against the lowercased intent.
_NAME_BEFORE_NOUN = re.compile(
    r"(?:^|\s)(?:the\s+)?([\w.~][\w.~/\\-]*)"
    r"\s+(?:dir|dirs|directory|directories|folder|folders|tree)\b")
_NAME_AFTER_PREP = re.compile(
    r"\b(?:in|under|inside|within)\s+(?:the\s+)?([\w.~][\w.~/\\-]*)")

# Filler that can occupy the name slot without naming a directory we could
# resolve. "the repo root" is a relation to the cwd, not a directory called root.
_NOT_A_DIRECTORY = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "here", "there", "it",
    "me", "my", "our", "your", "their", "current", "cwd", "workdir", "worktree",
    "repo", "repository", "project", "codebase", "root", "top", "all", "any",
    "some", "each", "everything", "which", "what", "file", "files", "entry",
    "entries", "content", "contents", "listing", "directory", "directories",
    "folder", "folders", "tree",
})


@dataclass(frozen=True)
class Scope:
    """frame: the caller's absolute cwd — the coordinate system result paths are
    reported in. root: the existing directory to walk, or None. named: a
    directory that was asked for but is not one here, kept for the report."""
    frame: str
    root: str | None = None
    named: str | None = None


def resolve(intent: str, params: dict[str, Any], digest: dict[str, Any]) -> Scope:
    frame = _absolute(str(digest.get("cwd") or "."))
    supplied = str(params.get("path") or "").strip()
    if supplied:
        return _scoped(supplied, frame)
    named = extract(intent)
    if named is None:
        return Scope(frame)
    return _scoped(named, frame)


def extract(intent: str) -> str | None:
    """The first directory name the intent states, or None if it states none."""
    text = (intent or "").lower()
    for rule in (_NAME_BEFORE_NOUN, _NAME_AFTER_PREP):
        for match in rule.finditer(text):
            name = _clean(match.group(1))
            if name and name not in _NOT_A_DIRECTORY:
                return name
    return None


def _scoped(name: str, frame: str) -> Scope:
    root = _existing_dir(name, frame)
    return Scope(frame, root, named=None if root else name)


def _existing_dir(name: str, frame: str) -> str | None:
    """The directory `name` denotes, or None when it is not one we may look at.

    `os.path.isdir` is a `stat`, and those are exactly what the file policy exists
    to gate: the deny list covers trees whose session is dead, where one stat
    blocks in uninterruptible sleep and nothing in this process can stop it. So a
    name that resolves outside the run root, or at or under a denied root, is not
    "a directory here" — which is already the shape of the answer this function
    gives for a path that does not exist (found by docs/review2-2026-11-03.md NB2:
    `params.path` reached this stat without passing the file policy at all).
    """
    from ..ops.filesys import confine

    try:
        path = confine.confined_root(name, base=frame)
    except confine.PathRefused:
        return None
    return path if os.path.isdir(path) else None


def _absolute(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _clean(token: str) -> str:
    return token.strip("`'\"").rstrip("/").strip(".,;:!?()[]<>")
