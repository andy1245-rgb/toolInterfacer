"""Hop log — one JSONL record per executed hop. Opt-in, owner-only, content-free.

This is a *trace*, and `tools/dataset/ingest.py` is the only reader in the tree: it
takes `op` and `soft.intent_context` and nothing else. So the record carries the
execution shape (op, kind, conf, ms, arg *names*) and the utterance — never an
argument value, never a byte of whatever the hop read. `artifact_summary` used to
be the first 200 characters of a read file, for a field no consumer read
(docs/logging-2026-11-04.md).

`hops_log.enabled` is false by default: a decision-routing tool should not ship a
recorder of the intents it is handed. When it is on, the path is `hops_log.path`
(default under the user's state dir — never the working directory) and the file is
the owner's: the directory is created 0700 and the file 0600, and a file that
group or other can already read is narrowed rather than refused (see
`_private_dir` for why that differs from `http/auth.py`'s refusal).

Nothing here may fail a tool call. A write that cannot happen (read-only tree, a
directory that cannot be made private, a symlink at the log path) sets `reason`
and stops writing for the process; the run continues.
"""
from __future__ import annotations

import json
import os
import stat
import sys
import time
from typing import Any

_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


class HopLog:
    def __init__(self) -> None:
        from ..conf import load
        cfg = load()["hops_log"]
        self.enabled = bool(cfg["enabled"])
        self.path = os.path.expanduser(str(cfg["path"]))
        self.reason: str | None = None     # set when enabled but not writable
        self._warned = False

    def append(self, record: dict[str, Any]) -> None:
        if not self.enabled or self.reason is not None:
            return
        try:
            self._write({"ts": time.time(), **record})
        except OSError as exc:
            self.reason = f"hop log off for this process: {exc}"
            if not self._warned:
                self._warned = True
                print(f"ti: {self.reason} ({self.path})", file=sys.stderr)

    def _write(self, rec: dict[str, Any]) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            _private_dir(directory)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | _NOFOLLOW, 0o600)
        try:
            if stat.S_IMODE(os.fstat(fd).st_mode) & 0o077:
                os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fd = -1
                fh.write(json.dumps(rec, default=str) + "\n")
        finally:
            if fd >= 0:
                os.close(fd)


def _private_dir(directory: str) -> None:
    """Create the log directory 0700, and narrow one that others can enter.

    Narrowing rather than refusing is the deliberate difference from
    `http/auth.py:121`, and it follows from what a log is. A token whose mode let
    group read it is already compromised — chmod cannot un-read it, which is why
    that file is refused. A log is not a credential: nobody else needs it, nobody
    has been harmed by the mode alone, and closing the mode stops the exposure
    instead of silently ending the collection the user asked for.

    A directory we cannot narrow stays a refusal: `os.chmod` raises, `append`
    records why and writes nothing. Writing into a directory someone else can
    modify lets them swap in a symlink and choose where the record goes.
    """
    os.makedirs(directory, mode=0o700, exist_ok=True)
    info = os.stat(directory)
    if not stat.S_ISDIR(info.st_mode):
        raise NotADirectoryError(f"{directory!r} is not a directory")
    if stat.S_IMODE(info.st_mode) & 0o077:
        os.chmod(directory, 0o700)
