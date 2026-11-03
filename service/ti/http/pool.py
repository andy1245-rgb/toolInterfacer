"""pool — a ThreadingHTTPServer with a hard ceiling on live handler threads.

`ThreadingHTTPServer` accepts a connection and spawns a thread for it, with no
upper bound and no timeout: N connections cost N threads, each holding a stack
and whatever the handler is doing. A request that parses waits, so a caller that
opens a hundred sockets holds a hundred threads, and the process runs out of
memory rather than rejecting anything.

This server acquires a slot before spawning and releases it when the handler
returns. Past the ceiling a connection is answered 503 immediately and closed:
under a flood the service degrades into fast refusals, and the threads it does
have keep serving.
"""
from __future__ import annotations

import threading
from http.server import ThreadingHTTPServer
from typing import Any

_BUSY_BODY = b'{"error": "busy", "detail": "the service is at its concurrency ceiling"}'
_BUSY_HEAD = (b"HTTP/1.0 503 Service Unavailable\r\n"
              b"Content-Type: application/json\r\n"
              b"Content-Length: " + str(len(_BUSY_BODY)).encode() + b"\r\n"
              b"Retry-After: 1\r\n"
              b"Connection: close\r\n\r\n")


class BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """A threaded HTTP server that refuses rather than spawning past `max_workers`."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], handler: type,
                 *, max_workers: int = 16, max_body_bytes: int = 262144,
                 request_queue_size: int = 16) -> None:
        self.max_workers = max(1, int(max_workers))
        self.max_body_bytes = int(max_body_bytes)
        self.request_queue_size = int(request_queue_size)
        self._slots = threading.BoundedSemaphore(self.max_workers)
        super().__init__(address, handler)

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._slots.acquire(blocking=False):
            self._refuse(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()

    def _refuse(self, request: Any) -> None:
        try:
            request.sendall(_BUSY_HEAD + _BUSY_BODY)
        except OSError:
            pass
        finally:
            self.shutdown_request(request)
