"""auth — the shared token every POST must carry.

The service is a local HTTP endpoint whose whole purpose is to read files and to
run the decision model on behalf of whatever asks it. Nothing about a loopback
socket says "this request came from the extension": any process on the box — a
page doing a fetch, a stray script, a compromised tool — can POST to 127.0.0.1
and get the same authority. So a request is served only if it carries the token
from `http.token_env` (default `TI_HTTP_TOKEN`) or from the runtime file at
`http.token_file`.

Sources, in order:

1. the environment. The extension spawns the service with `{...process.env}`
   (`pi-extension/src/lifecycle.ts`), so a token exported in the session's shell
   reaches both ends without a file.
2. the runtime file, if it exists — created 0600, and refused if its mode lets
   anyone but its owner read it: a readable token is not a token.
3. generated: 32 bytes of `secrets.token_urlsafe`, written 0600, so a box that
   has never had a token gets one instead of an unauthenticated service.

There is no "no token" mode. `require_token()` raises rather than letting a
server start that would serve unauthenticated requests, and `api.serve` calls it
before it binds.

The client half is `pi-extension/src/token.ts`, which resolves the token the same
way — `TI_HTTP_TOKEN` first, then the runtime file — and `bridge.ts` sends it in
`X-TI-Token` on every POST. `/health` stays open on purpose: `ensureService()`
polls it to decide whether to spawn, so an authenticated health check would make
the extension spawn a second service every session.
"""
from __future__ import annotations

import hmac
import os
import secrets
import stat
from typing import Any

from ..conf import load

TOKEN_HEADER = "X-TI-Token"
TOKEN_FILE_ENV = "TI_HTTP_TOKEN_FILE"
DEFAULT_TOKEN_FILE = "~/.local/state/ti/http.token"

_TOKEN: str | None = None


class TokenUnavailable(RuntimeError):
    """No usable token: it could not be read, was refused, or could not be written."""


def conf() -> dict[str, Any]:
    return load()["http"]


def token_path() -> str:
    """Where the runtime token lives (`TI_HTTP_TOKEN_FILE` overrides the conf)."""
    raw = os.environ.get(TOKEN_FILE_ENV) or conf().get("token_file") or DEFAULT_TOKEN_FILE
    return os.path.expanduser(str(raw))


def require_token() -> str:
    """The token requests must carry. Raises instead of yielding a service with none."""
    return token()


def token() -> str:
    global _TOKEN
    if _TOKEN is None:
        _TOKEN = _establish()
    return _TOKEN


def reset() -> None:
    """Forget the cached token (tests, a reloaded conf, a rotated file)."""
    global _TOKEN
    _TOKEN = None


def authorized(headers: Any) -> bool:
    """True when `headers` carries the token, compared in constant time."""
    supplied = _supplied(headers)
    if not supplied:
        return False
    try:
        expected = token()
    except TokenUnavailable:
        return False
    return hmac.compare_digest(supplied.encode(), expected.encode())


def _supplied(headers: Any) -> str:
    value = str(headers.get(TOKEN_HEADER) or "").strip()
    if value:
        return value
    scheme = str(headers.get("Authorization") or "").strip()
    if scheme.lower().startswith("bearer "):
        return scheme[7:].strip()
    return ""


def _establish() -> str:
    name = str(conf().get("token_env") or "TI_HTTP_TOKEN")
    from_env = os.environ.get(name, "").strip()
    if from_env:
        return from_env
    path = token_path()
    if os.path.exists(path):
        return _read(path)
    return _create(path)


def _read(path: str) -> str:
    try:
        info = os.stat(path)
    except OSError as exc:
        raise TokenUnavailable(f"cannot read the token file {path!r}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise TokenUnavailable(f"{path!r} is not a regular file; refusing to use it as a token")
    mode = stat.S_IMODE(info.st_mode)
    if mode & 0o077:
        raise TokenUnavailable(
            f"{path!r} has mode {mode:03o}: group and other can read it, so it is not a "
            f"token any more. `chmod 600 {path}` (or delete it to have a fresh one made)")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            value = fh.read().strip()
    except OSError as exc:
        raise TokenUnavailable(f"cannot read the token file {path!r}: {exc}") from exc
    if not value:
        raise TokenUnavailable(f"{path!r} is empty; delete it to have a fresh token made")
    return value


def _create(path: str) -> str:
    directory = os.path.dirname(path) or "."
    try:
        os.makedirs(directory, mode=0o700, exist_ok=True)
    except OSError as exc:
        raise TokenUnavailable(f"cannot create the token directory {directory!r}: {exc}") from exc
    value = secrets.token_urlsafe(32)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return _read(path)      # another process made one first: theirs is the token
    except OSError as exc:
        raise TokenUnavailable(f"cannot write the token file {path!r}: {exc}") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(value + "\n")
    except OSError as exc:
        raise TokenUnavailable(f"cannot write the token file {path!r}: {exc}") from exc
    return value
