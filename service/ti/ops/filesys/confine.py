"""confine — the one place a filesys path becomes an absolute path.

Every filesys op starts here: `resolve(path)` returns the real, absolute path the
op is allowed to act on, or raises `PathRefused`. Two policies decide the answer.

  the run root
        the directory the run is about. The HTTP layer sets it once per request
        from the body's digest cwd, after checking that cwd against the
        configured policy roots (`ops.filesys.roots`). A direct caller — a test,
        the CLI, `use_tool` from another module — gets the process cwd, which is
        what "here" already means to it. Relative paths resolve against it, the
        way `planner/scope.py` resolves a named directory against the digest cwd.

  the deny list
        roots that must not be stat'ed, let alone walked (`ops.filesys.deny_roots`,
        default `~/mnt`). They exist because a path can resolve onto a network
        filesystem whose session is dead: one `os.stat` inside it blocks in
        uninterruptible sleep, and neither a deadline nor a thread kill reaches
        it. Measured on this box, four separate times, for 22 minutes to 9.8
        hours. Nothing in this process can recover from that, so the refusal has
        to happen before the first syscall.

Why this is a resolver rather than `os.path.realpath`: realpath stats every
component of the path it is given, so asking "is this path inside the deny list?"
with it would itself enter the denied directory — the check would be the hang.
The resolver below walks the path one component at a time, checks the deny list
*before* each lstat, follows a symlink by rewriting the pending component list
from the link's target, and refuses any target that leaves the run root. Link
hops are capped, so a symlink cycle terminates instead of spinning.
"""
from __future__ import annotations

import os
import stat
from contextvars import ContextVar
from typing import Any

from ...conf import load

# A chain this long is a mistake or an attack, not a working tree.
MAX_SYMLINK_HOPS = 32


class PathRefused(ValueError):
    """A path an op was asked to act on was refused.

    ValueError, so a caller that already treats a bad argument as a bad argument
    keeps working; the HTTP layer maps it to a 403 of its own.
    """


class OutsideRoot(PathRefused):
    """The path resolves outside the run root."""


class DeniedPath(PathRefused):
    """The path is at or under a configured deny root."""


# The run root, set per request by the HTTP layer and per test by the suite.
# Unset (the default) means "the process cwd", resolved once.
_ROOT: ContextVar[str | None] = ContextVar("ti_filesys_root", default=None)
_PROCESS_ROOT: str | None = None


def filesys_conf() -> dict[str, Any]:
    return load()["ops"]["filesys"]


def deny_roots() -> list[str]:
    """Configured deny roots, absolute and normalised. `~` is expanded, so the
    shipped default is portable; a relative entry is taken from the process cwd."""
    out: list[str] = []
    for entry in filesys_conf().get("deny_roots") or []:
        text = os.path.expanduser(str(entry))
        if not text.strip():
            continue
        if not os.path.isabs(text):
            text = os.path.join(os.getcwd(), text)
        path = os.path.normpath(text)
        if path not in out:
            out.append(path)
    return out


def denying_root(path: str) -> str | None:
    """The deny root that covers `path`, or None. Pure string work: no syscall."""
    norm = os.path.normpath(path)
    for denied in deny_roots():
        if norm == denied or norm.startswith(denied.rstrip(os.sep) + os.sep):
            return denied
    return None


def is_denied(path: str) -> bool:
    return denying_root(path) is not None


def set_root(path: str | None) -> Any:
    """Point the run root at `path` for the current context; returns a reset token."""
    if path is None:
        return _ROOT.set(None)
    return _ROOT.set(real_root(path))


def reset_root(token: Any) -> None:
    _ROOT.reset(token)


def run_root() -> str:
    """The directory paths are resolved and confined against."""
    current = _ROOT.get()
    if current is not None:
        return current
    global _PROCESS_ROOT
    if _PROCESS_ROOT is None:
        _PROCESS_ROOT = real_root(os.getcwd())
    return _PROCESS_ROOT


def policy_roots() -> list[str]:
    """The configured roots a run's cwd must live inside (`ops.filesys.roots`).

    The shipped default is `["~", ".", "/tmp"]`:

    - `~` covers the ordinary case — the service is spawned with its own repo as
      the cwd (`pi-extension/src/lifecycle.ts`) while a run is about the
      session's cwd, which is some project under the operator's home.
    - `.` is the directory the service was *started* in, and `/tmp` is scratch
      space. Both are what make a clone work wherever it lives and a session
      started in a temp directory work at all. Without them, `git clone` into
      `/tmp` and the README's first example answers `403 scope_refused`, because
      the run root is the clone and the only permitted root was `~` (measured by
      `docs/quickstart-stranger-2026-11-03.md` §1 and `docs/review2-2026-11-03.md`
      NB4).

    Both are directories the operator chose by starting the service where they
    did. Neither widens what a *request* may name: the run root is still resolved
    inside one of these roots, so "list what is in /etc" is refused unless the
    service was started in `/etc`.
    """
    out: list[str] = []
    for entry in filesys_conf().get("roots") or ["~", "."]:
        text = str(entry)
        if not text.strip():
            continue
        path = real_root(text)
        if path not in out:
            out.append(path)
    return out


def confined_root(cwd: Any = None, *, base: str | None = None) -> str:
    """The run root for a request: `cwd` (default: the process cwd), resolved
    inside one of the configured policy roots (`ops.filesys.roots`).

    This is the check that keeps a request from choosing the whole filesystem as
    its frame: "list what is in /etc" does not get to walk `/etc` because it said
    so. What the run may then touch is bounded again by `resolve`, one path at a
    time, against the root returned here.

    `base` is the frame a *relative* `cwd` resolves against (default: the process
    cwd). `planner.scope` passes the run frame, because a directory named in the
    intent is relative to the session, not to wherever the service was started.
    """
    roots = policy_roots()
    candidate = str(cwd) if cwd not in (None, "") else "."
    frame = base if base is not None else os.getcwd()
    denied: DeniedPath | None = None
    last: PathRefused | None = None
    for policy in roots:
        try:
            return resolve(candidate, base=frame, root=policy)
        except DeniedPath as exc:
            denied = exc
        except PathRefused as exc:
            last = exc
    if denied is not None:
        raise denied
    listing = ", ".join(repr(root) for root in roots) or "(none configured)"
    raise OutsideRoot(f"run root {candidate!r} is outside the permitted roots ({listing}); "
                      f"ops.filesys.roots is what widens this") from last


def real_root(path: str) -> str:
    """The real absolute form of a *trusted* root (the cwd, a configured root).

    Trusted means "not chosen by a request": policy roots and the process cwd.
    The deny check still applies — a root configured inside a denied tree is
    refused here, so a run rooted at a dead mount never starts.
    """
    text = os.path.expanduser(str(path))
    if not text.strip():
        raise PathRefused("empty root")
    if "\x00" in text:
        raise PathRefused("root contains a NUL byte, so it does not name a directory")
    if not os.path.isabs(text):
        text = os.path.join(os.getcwd(), text)
    lexical = os.path.normpath(text)
    _refuse_denied(lexical)
    real = _resolve_hops(lexical, os.sep)
    _refuse_denied(real)
    return real


def resolve(path: Any, *, base: str | None = None, root: str | None = None) -> str:
    """The real path `path` names, confined to `root` (default: the run root).

    `path` may be relative — it resolves against `base` (default: `root`) — and
    may be written with `..`: the traversal is collapsed lexically before any
    syscall, so "escape by climbing" is refused rather than walked out of, and
    "escape by symlink" is caught by the hop resolver below.
    """
    root_abs = root if root is not None else run_root()
    base_abs = base if base is not None else root_abs
    text = os.path.expanduser(str(path))
    if not text.strip():
        raise PathRefused(f"empty path (the run root is {root_abs!r})")
    if "\x00" in text:
        raise PathRefused("path contains a NUL byte, so it does not name a file")
    if not os.path.isabs(text):
        text = os.path.join(base_abs, text)
    lexical = os.path.normpath(text)
    _refuse_denied(lexical)
    if not _inside(lexical, root_abs):
        raise OutsideRoot(f"{str(path)!r} resolves outside the permitted root "
                          f"{root_abs!r}")
    real = _resolve_hops(lexical, root_abs)
    _refuse_denied(real)
    if not _inside(real, root_abs):
        raise OutsideRoot(f"{str(path)!r} resolves outside the permitted root "
                          f"{root_abs!r} (real path {real!r})")
    return real


def _inside(path: str, root: str) -> bool:
    if path == root:
        return True
    return path.startswith(root.rstrip(os.sep) + os.sep)


def _refuse_denied(path: str) -> None:
    denied = denying_root(path)
    if denied is not None:
        raise DeniedPath(f"{path!r} is at or under a denied root ({denied!r}): this "
                         f"tree is refused before it is touched")


def _resolve_hops(path: str, root: str) -> str:
    """Resolve `path`'s symlinks component by component, never stat'ing a deny root.

    Absolute input only. Each hop is lstat'ed after the deny check; a symlink
    replaces the pending component list with its target, and a target outside
    `root` is refused rather than followed. A component that does not exist ends
    the syscall work: the rest of the path is kept lexically, which is what a
    caller asking about a file that is not there yet gets from realpath too.
    """
    parts = [part for part in path.split(os.sep) if part]
    out: list[str] = []
    hops = 0
    index = 0
    while index < len(parts):
        part = parts[index]
        index += 1
        if part == ".":
            continue
        if part == "..":
            if out:
                out.pop()
            continue
        current = os.sep + os.sep.join(out + [part])
        _refuse_denied(current)
        try:
            info = os.lstat(current)
        except OSError:
            out.append(part)
            out.extend(parts[index:])
            break
        if stat.S_ISLNK(info.st_mode):
            hops += 1
            if hops > MAX_SYMLINK_HOPS:
                raise OutsideRoot(f"{path!r} follows more than {MAX_SYMLINK_HOPS} "
                                  f"symlinks (a loop, or a chain that hides where "
                                  f"it lands)")
            target = os.readlink(current)
            if not os.path.isabs(target):
                target = os.path.join(os.sep + os.sep.join(out), target)
            target = os.path.normpath(target)
            _refuse_denied(target)
            if not _inside(target, root):
                raise OutsideRoot(f"symlink {current!r} points outside the permitted "
                                  f"root {root!r} ({target!r})")
            parts = [piece for piece in target.split(os.sep) if piece] + parts[index:]
            out = []
            index = 0
            continue
        out.append(part)
    return os.sep + os.sep.join(out) if out else os.sep
