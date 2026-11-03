"""api.py — POST /v1/useTool, POST /v1/plan (mode:plan), GET /health.

Boot imports all op modules (registry fires), resolves the decide backend map
from conf, and serves. P1 wires useTool → inspector → runner end to end.

Ingress order — the order a request meets the guards, and why:

  token        401. A POST must carry `X-TI-Token` (or a bearer token) equal to
               the runtime token; see auth.py. Anything on the box can open a
               loopback socket, so the socket is not the identity. `/health`
               stays open: the extension polls it before spawning a service.
  body         411/413/400. A Content-Length is required (a chunked body has no
               bound we can refuse before reading it), capped at
               `http.max_body_bytes`, and must be a JSON object — a bad body is
               an answer, not a dropped connection.
  route        404. `/v1/useTool`, `/v1/plan`.
  op allowlist 400. An L0 request (`params.tool` + `params.args`, the documented
               passthrough) may only name an op in `http.allowed_ops` ∩ the table
               below. The filesys mirrors are the ones whose every path is
               confined and whose every walk and match is bounded; the CLI ops
               and the decision primitives are not a capability this endpoint
               hands out.
  scope root   403. The run root is the body's digest cwd, and it must resolve
               inside one of `ops.filesys.roots` (`confine.confined_root`). It
               becomes the confinement root for the whole run — every path an op
               touches is resolved against it — so a `..` climb, a symlink out,
               a path at or under a denied root, and a cwd outside the policy all
               stop here, before `use_tool` sees the body.

What is not here: a handler thread is allowed to block on a local file, so the
walk's deadline and the deny list do the bounding (see ops/filesys/walk.py); the
thread pool is bounded so a flood refuses instead of exhausting the process.
"""
from __future__ import annotations

import ipaddress
import json
from http.server import BaseHTTPRequestHandler
from typing import Any

from ..conf import load
from ..ops.filesys import confine
from ..ops.op_api import all_ops, import_all
from . import auth
from .pool import BoundedThreadingHTTPServer

# The ops an L0 body may name, with the arguments that are paths. One line each
# on why it is allowed:
#   filesys_read  the product call ("read the file we discussed"): confined to
#                 the run root, line- and byte-capped
#   filesys_glob  a listing: confined, depth/hit/deadline-bounded, deny-pruned
#   filesys_grep  a search: confined, pattern-guarded, deadline-bounded
# Deliberately absent: web_fetch / web_search / reddit_digest spawn external CLIs
# with caller-chosen argv, and decide_pick / decide_extract hand caller-chosen
# text straight to the decision model as the prompt. A legitimate intent still
# reaches all of them through the plan layer, which chooses the arguments itself.
L0_OPS: dict[str, tuple[str, ...]] = {
    "filesys_read": ("path",),
    "filesys_glob": ("cwd",),
    "filesys_grep": ("cwd",),
}

ROUTES = {"/v1/useTool": False, "/v1/plan": True}


class _Refused(Exception):
    """A request refused at the boundary, with the status and reason to send."""

    def __init__(self, code: int, error: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.error = error
        self.detail = detail


class Handler(BaseHTTPRequestHandler):
    server_version = "ti"
    sys_version = ""

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json({"status": "ok", "ops": all_ops(), "conf": bool(load())})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if not auth.authorized(self.headers):
            self.close_connection = True
            self._refuse(401, "unauthenticated",
                         f"POST needs the service token in {auth.TOKEN_HEADER} "
                         f"(or Authorization: Bearer); see service/ti/http/auth.py")
            return
        dry = ROUTES.get(self.path)
        if dry is None:
            self._json({"error": "not found"}, 404)
            return
        try:
            body = self._body()
            payload, code = self._use_tool(body, dry=dry)
        except _Refused as refused:
            self._refuse(refused.code, refused.error, refused.detail)
            return
        except Exception as exc:  # noqa: BLE001 — an answer beats a dropped connection
            self._json({"status": "error", "error": str(exc), "hops": []}, 500)
            return
        self._json(payload, code)

    def _body(self) -> dict[str, Any]:
        encoding = str(self.headers.get("Transfer-Encoding") or "").lower()
        if encoding and encoding != "identity":
            self.close_connection = True
            raise _Refused(411, "length_required",
                           "a chunked body has no length to bound; send Content-Length")
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self.close_connection = True
            raise _Refused(411, "length_required", "Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError:
            self.close_connection = True
            raise _Refused(400, "bad_length",
                           f"Content-Length {raw_length!r} is not a number") from None
        if length < 0:
            self.close_connection = True
            raise _Refused(400, "bad_length", f"Content-Length {length} is negative")
        cap = int(getattr(self.server, "max_body_bytes", 262144))
        if length > cap:
            self.close_connection = True
            raise _Refused(413, "body_too_large",
                           f"body is {length} bytes; the limit is {cap} "
                           f"(http.max_body_bytes)")
        raw = self.rfile.read(length)
        if len(raw) != length:
            self.close_connection = True
            raise _Refused(400, "short_body", f"read {len(raw)} of {length} bytes")
        try:
            body = json.loads(raw or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise _Refused(400, "invalid_json", str(exc)) from None
        if not isinstance(body, dict):
            raise _Refused(400, "invalid_body", "body must be a JSON object")
        return body

    def _use_tool(self, body: dict[str, Any], dry: bool = False) -> tuple[dict[str, Any], int]:
        from ..service_entry import use_tool
        params = body.get("params")
        params = params if isinstance(params, dict) else {}
        tool = params.get("tool")
        if tool is not None:
            self._check_op(str(tool))
        digest = body.get("digest")
        cwd = digest.get("cwd") if isinstance(digest, dict) else None
        try:
            root = confine.confined_root(cwd)
        except confine.PathRefused as exc:
            raise _Refused(403, "scope_refused", str(exc)) from None
        if tool is not None:
            self._check_path_args(str(tool), params.get("args"), root)
        return use_tool(body, dry=dry), 200

    def _check_op(self, tool: str) -> None:
        allowed_conf = load()["http"].get("allowed_ops")
        # conf narrows the code's table and never widens it; an explicit empty
        # list is a decision ("no L0 at all"), not an unset key
        allowed = set(L0_OPS)
        if isinstance(allowed_conf, list):
            allowed &= set(allowed_conf)
        if tool in allowed:
            return
        raise _Refused(400, "op_not_allowed",
                       f"L0 may call {sorted(allowed)}; {tool!r} is not one of them. "
                       f"Ask for the intent instead and let the plan layer choose "
                       f"the op and its arguments.")

    def _check_path_args(self, tool: str, args: Any, root: str) -> None:
        """Resolve the path arguments now, so an escape is a 403 that names the
        argument instead of a 200 carrying an error string.

        The ops resolve them again (`confine.resolve`); this is the boundary
        saying which argument was refused and why.
        """
        if args is None:
            return
        if not isinstance(args, dict):
            raise _Refused(400, "bad_args", "params.args must be an object")
        for name in L0_OPS.get(tool, ()):
            value = args.get(name)
            if value is None:
                continue
            if not isinstance(value, str):
                raise _Refused(400, "bad_args", f"params.args.{name} must be a string")
            try:
                confine.resolve(value, root=root)
            except confine.PathRefused as exc:
                raise _Refused(403, "path_refused",
                               f"params.args.{name}: {exc}") from None

    def _refuse(self, code: int, error: str, detail: str) -> None:
        self.close_connection = True
        self._json({"error": error, "detail": detail}, code)

    def _json(self, obj: dict[str, Any], code: int = 200) -> None:
        blob = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)


def make_server(host: str | None = None, port: int | None = None) -> BoundedThreadingHTTPServer:
    """Bind the service socket.

    Raises instead of serving without a token, and refuses a non-loopback bind:
    the token is the identity, loopback is the network boundary, and widening the
    second one is a decision an operator makes in a config edit, not by a stray
    `--host`.
    """
    cfg = load()["http"]
    auth.require_token()
    import_all()
    address = host or str(cfg.get("host", "127.0.0.1"))
    if not _is_loopback(address):
        raise ValueError(f"refusing to bind {address!r}: the service is a loopback "
                         f"endpoint (http.host in service/ti/conf/defaults.json)")
    bound = int(port if port is not None else cfg.get("port", 1113))
    return BoundedThreadingHTTPServer(
        (address, bound), Handler,
        max_workers=int(cfg.get("max_workers", 16)),
        max_body_bytes=int(cfg.get("max_body_bytes", 262144)))


def _is_loopback(host: str) -> bool:
    if host in ("", "localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def serve(host: str | None = None, port: int | None = None) -> None:
    make_server(host, port).serve_forever()


def main() -> None:
    try:
        serve()
    except (auth.TokenUnavailable, OSError, ValueError) as exc:
        raise SystemExit(f"ti: {exc}") from exc


if __name__ == "__main__":
    main()
