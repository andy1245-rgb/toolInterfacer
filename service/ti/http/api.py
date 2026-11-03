"""api.py — POST /v1/useTool, POST /v1/plan (mode:plan), GET /health.

Boot imports all op modules (registry fires), resolves the decide backend map
from conf, and serves. P1 wires useTool → inspector → runner end to end.
"""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..ops.op_api import all_ops, import_all
from ..conf import load


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/health":
            # The decide map is on the wire because a live probe has to be able to
            # say which backend answered, not just that *a* service did: "done"
            # from the replay fixture and "done" from d1-3B are indistinguishable
            # in a useTool body, and the fixture one proves nothing about the model.
            decide = load().get("decide") or {}
            self._json({"status": "ok", "ops": all_ops(),
                        "conf": bool(load()),
                        "decide": dict(decide.get("backends_in_use") or {}),
                        "cascade": bool((decide.get("cascade") or {}).get("enabled", False))})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/v1/useTool":
            self._json(self._use_tool(body))
        elif self.path == "/v1/plan":
            self._json(self._use_tool(body, dry=True))
        else:
            self._json({"error": "not found"}, 404)

    def _use_tool(self, body: dict, dry: bool = False) -> dict:
        from ..service_entry import use_tool
        try:
            # ask_via_response: this transport's only channel to the caller is the
            # response body, so a gate the run cannot settle comes back as
            # need_input + question rather than as a silent degraded guess.
            return use_tool(body, dry=dry, ask_via_response=not dry)
        except Exception as exc:  # noqa: BLE001 — surface as structured error, not a 500 hang
            return {"status": "error", "error": str(exc), "hops": []}

    def _json(self, obj: dict, code: int = 200) -> None:
        blob = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)


def serve(host: str | None = None, port: int | None = None) -> None:
    cfg = load()["http"]
    _ = host
    import_all()
    addr = host or str(cfg.get("host", "127.0.0.1")), int(port or cfg.get("port", 1113))
    ThreadingHTTPServer(addr, Handler).serve_forever()


def main() -> None:
    serve()


if __name__ == "__main__":
    main()
