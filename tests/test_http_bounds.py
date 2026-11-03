"""The HTTP surface over a real socket: authenticated, allowlisted, confined, capped.

Every case here is one of the review of 2026-11-03's executed proofs (B4) turned
into an assertion. Against the service as it shipped, each of these returned 200
or hung:

    POST with no token at all          -> 200, an L0 filesys_read of /etc/hostname
    POST with params.tool=web_fetch    -> 200, an external CLI driven by the body
    POST args.path=../../../../etc/... -> 200, a read outside the run
    POST digest.cwd=/etc               -> 200, a listing of /etc
    POST a 300 KB body                 -> read into memory whole
    a connection flood                 -> one thread each, forever
    filesys_grep "(a+)+$"              -> still running at a 25s client kill

The last case is the one that matters most: the extension's own handshake —
spawn exactly what `pi-extension/src/lifecycle.ts` spawns, poll `/health`, then
POST with the token from the environment — still returns a result. A security fix
that breaks the product is not a fix.
"""
import copy
import http.client
import json
import os
import pathlib
import socket
import subprocess
import sys
import shutil
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))

from ti.conf import load  # noqa: E402
from ti.http import api, auth, pool  # noqa: E402

TOKEN = "test-token-not-the-real-one"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _status(payload: bytes) -> int:
    head = payload.split(b"\r\n", 1)[0]
    if not head.startswith(b"HTTP/"):
        return 0
    try:
        return int(head.split(b" ")[1])
    except (IndexError, ValueError):
        return 0


class ServiceCase(unittest.TestCase):
    """A real server on an ephemeral port, a tmp run root, a known token."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = pathlib.Path(self._tmp.name)
        (self.root / "hello.txt").write_text("hello from the run root\n", encoding="utf-8")

        self._http = copy.deepcopy(load()["http"])
        self._filesys = copy.deepcopy(load()["ops"]["filesys"])
        self.addCleanup(self._restore_conf)
        load()["ops"]["filesys"]["roots"] = [str(self.root)]
        load()["ops"]["filesys"]["deny_roots"] = [str(self.root / "mnt")]
        load()["http"]["max_body_bytes"] = 4096

        self._environ = dict(os.environ)
        self.addCleanup(self._restore_env)
        os.environ["TI_HTTP_TOKEN"] = TOKEN
        auth.reset()
        self.addCleanup(auth.reset)

        self.server = api.make_server(port=0)
        self.port = int(self.server.server_address[1])
        self._thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self._thread.start()
        self.addCleanup(self._stop)

    def _restore_conf(self) -> None:
        load()["http"] = self._http
        load()["ops"]["filesys"] = self._filesys

    def _restore_env(self) -> None:
        os.environ.clear()
        os.environ.update(self._environ)

    def _stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self._thread.join(5)

    def post(self, body: object = None, *, path: str = "/v1/useTool",
             token: str | None = TOKEN) -> tuple[int, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        try:
            payload = json.dumps(body if body is not None else {}).encode()
            headers = {"Content-Type": "application/json"}
            if token is not None:
                headers["X-TI-Token"] = token
            conn.request("POST", path, body=payload, headers=headers)
            response = conn.getresponse()
            raw = response.read()
            return response.status, (json.loads(raw) if raw else {})
        finally:
            conn.close()

    def get_health(self) -> tuple[int, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        try:
            conn.request("GET", "/health")
            response = conn.getresponse()
            raw = response.read()
            return response.status, (json.loads(raw) if raw else {})
        finally:
            conn.close()

    def send_raw(self, request: bytes) -> tuple[int, bytes]:
        with socket.create_connection(("127.0.0.1", self.port), timeout=15) as sock:
            try:
                sock.sendall(request)
            except OSError:
                pass
            data = b""
            try:
                while True:
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    data += chunk
            except OSError:
                pass
        return _status(data), data

    def l0(self, tool: str, args: dict, *, cwd: str | None = None) -> dict:
        body: dict = {"intent": "use the tool", "params": {"tool": tool, "args": args}}
        if cwd is not None:
            body["digest"] = {"cwd": cwd}
        return body


class AuthTest(ServiceCase):
    """/health stays open for the lifecycle poll; everything else needs the token."""

    def test_health_needs_no_token(self):
        status, body = self.get_health()
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")
        self.assertIn("filesys_read", body["ops"])

    def test_a_request_without_a_token_is_refused(self):
        status, body = self.post(self.l0("filesys_read", {"path": "hello.txt"},
                                         cwd=str(self.root)), token=None)
        self.assertEqual(status, 401)
        self.assertEqual(body["error"], "unauthenticated")

    def test_a_request_with_the_wrong_token_is_refused(self):
        status, body = self.post(self.l0("filesys_read", {"path": "hello.txt"},
                                         cwd=str(self.root)), token="not-the-token")
        self.assertEqual(status, 401)

    def test_a_bearer_token_is_accepted_too(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        try:
            conn.request("POST", "/v1/useTool",
                         body=json.dumps(self.l0("filesys_read", {"path": "hello.txt"},
                                                 cwd=str(self.root))).encode(),
                         headers={"Content-Type": "application/json",
                                  "Authorization": f"Bearer {TOKEN}"})
            self.assertEqual(conn.getresponse().status, 200)
        finally:
            conn.close()

    def test_an_authenticated_call_reaches_the_product_path(self):
        status, body = self.post(self.l0("filesys_read", {"path": "hello.txt"},
                                         cwd=str(self.root)))
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "done", body)
        self.assertIn("hello from the run root", str(body["result"]))
        self.assertEqual(body["hops"][0]["op"], "filesys_read")


class AllowlistTest(ServiceCase):
    """L0 names an op; only the confined, bounded ones may be named."""

    def test_an_op_outside_the_allowlist_is_refused(self):
        status, body = self.post(self.l0("web_fetch", {"urls": ["http://example.invalid/"]},
                                         cwd=str(self.root)))
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "op_not_allowed")
        self.assertIn("filesys_read", body["detail"])

    def test_a_decision_primitive_is_refused(self):
        status, body = self.post(self.l0("decide_pick", {"instructions": "x"},
                                         cwd=str(self.root)))
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "op_not_allowed")

    def test_the_allowlist_is_narrowed_by_conf(self):
        load()["http"]["allowed_ops"] = []
        status, body = self.post(self.l0("filesys_read", {"path": "hello.txt"},
                                         cwd=str(self.root)))
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "op_not_allowed")

    def test_args_must_be_an_object(self):
        status, body = self.post({"intent": "x", "params": {"tool": "filesys_read",
                                                            "args": ["hello.txt"]},
                                  "digest": {"cwd": str(self.root)}})
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "bad_args")


class ConfinementTest(ServiceCase):
    """No request chooses the filesystem it runs against."""

    def test_a_dotdot_escape_is_refused(self):
        status, body = self.post(self.l0("filesys_read",
                                         {"path": "../../../../etc/hostname"},
                                         cwd=str(self.root)))
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "path_refused")
        self.assertIn("params.args.path", body["detail"])

    def test_an_absolute_path_outside_the_root_is_refused(self):
        status, body = self.post(self.l0("filesys_read", {"path": "/etc/hostname"},
                                         cwd=str(self.root)))
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "path_refused")

    def test_a_symlink_escape_is_refused(self):
        os.symlink("/etc", self.root / "link")
        status, body = self.post(self.l0("filesys_read", {"path": "link/hostname"},
                                         cwd=str(self.root)))
        self.assertEqual(status, 403)
        self.assertIn("symlink", body["detail"])

    def test_a_cwd_outside_the_policy_roots_is_refused(self):
        status, body = self.post(self.l0("filesys_glob", {"pattern": "**/*"}, cwd="/etc"))
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "scope_refused")

    def test_a_cwd_under_a_denied_root_is_refused(self):
        (self.root / "mnt").mkdir()
        status, body = self.post(self.l0("filesys_glob", {"pattern": "**/*"},
                                         cwd=str(self.root / "mnt")))
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "scope_refused")
        self.assertIn("denied root", body["detail"])

    def test_a_glob_cwd_escape_is_refused(self):
        status, body = self.post(self.l0("filesys_glob", {"pattern": "**/*", "cwd": "/etc"},
                                         cwd=str(self.root)))
        self.assertEqual(status, 403)
        self.assertIn("cwd", body["detail"])


class BodyTest(ServiceCase):
    """The body is bounded and parsed, and a bad one is answered."""

    def test_an_oversize_body_is_refused(self):
        head = (f"POST /v1/useTool HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                f"X-TI-Token: {TOKEN}\r\nContent-Type: application/json\r\n"
                f"Content-Length: 20000\r\n\r\n").encode()
        status, raw = self.send_raw(head + b"{" + b" " * 20000)
        self.assertEqual(status, 413)
        self.assertIn(b"body_too_large", raw)

    def test_a_chunked_body_is_refused(self):
        head = (f"POST /v1/useTool HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                f"X-TI-Token: {TOKEN}\r\nTransfer-Encoding: chunked\r\n\r\n").encode()
        status, raw = self.send_raw(head)
        self.assertEqual(status, 411)
        self.assertIn(b"length_required", raw)

    def test_invalid_json_gets_an_answer(self):
        head = (f"POST /v1/useTool HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                f"X-TI-Token: {TOKEN}\r\nContent-Type: application/json\r\n"
                f"Content-Length: 3\r\n\r\n").encode()
        status, raw = self.send_raw(head + b"{o}")
        self.assertEqual(status, 400)
        self.assertIn(b"invalid_json", raw)

    def test_a_body_that_is_not_an_object_is_refused(self):
        status, body = self.post([1, 2, 3])
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_body")

    def test_an_unknown_route_is_404(self):
        status, _body = self.post({"intent": "x"}, path="/v1/nope")
        self.assertEqual(status, 404)


class PoolTest(unittest.TestCase):
    """The thread ceiling: past max_workers a connection is refused, not queued."""

    def setUp(self) -> None:
        class Slow(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                time.sleep(0.6)
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, *args: object) -> None:
                return

        self.server = pool.BoundedThreadingHTTPServer(
            ("127.0.0.1", 0), Slow, max_workers=1)
        self.port = int(self.server.server_address[1])
        self._thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self._thread.start()
        self.addCleanup(self._stop)

    def _stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self._thread.join(5)

    def test_a_connection_past_the_ceiling_is_refused(self):
        first = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        self.addCleanup(first.close)
        first.request("GET", "/")
        time.sleep(0.15)                      # the one worker is inside its sleep
        second = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        self.addCleanup(second.close)
        second.request("GET", "/")
        response = second.getresponse()
        self.assertEqual(response.status, 503)
        self.assertIn(b"busy", response.read())
        self.assertEqual(first.getresponse().status, 200)


class TokenFileTest(unittest.TestCase):
    """A runtime token file: generated 0600, and refused if it is readable."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = pathlib.Path(self._tmp.name, "state", "token")
        self._environ = dict(os.environ)
        self.addCleanup(self._restore)
        os.environ.pop("TI_HTTP_TOKEN", None)
        os.environ["TI_HTTP_TOKEN_FILE"] = str(self.path)
        auth.reset()
        self.addCleanup(auth.reset)

    def _restore(self) -> None:
        os.environ.clear()
        os.environ.update(self._environ)

    def test_a_token_is_generated_owner_only_and_accepted(self):
        value = auth.require_token()
        self.assertTrue(value)
        self.assertEqual(self.path.read_text(encoding="utf-8").strip(), value)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertTrue(auth.authorized({auth.TOKEN_HEADER: value}))
        self.assertFalse(auth.authorized({auth.TOKEN_HEADER: value + "x"}))
        self.assertFalse(auth.authorized({}))

    def test_a_world_readable_token_file_is_refused(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("token\n", encoding="utf-8")
        self.path.chmod(0o644)
        with self.assertRaises(auth.TokenUnavailable) as caught:
            auth.require_token()
        self.assertIn("chmod 600", str(caught.exception))


class LifecycleHandshakeTest(unittest.TestCase):
    """What the extension does — spawn, poll /health, call with the token — works.

    `pi-extension/src/lifecycle.ts` spawns `python3 -m ti.http.api` detached with
    cwd at the repo root and `PYTHONPATH=service`, then polls `/health` until it
    answers. The client half of the token is the one line this branch documents;
    the cases below prove the service half is a handshake the extension can
    complete, not a wall it 401s against forever.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = pathlib.Path(self._tmp.name)
        (self.root / "hello.txt").write_text("hello from the run root\n", encoding="utf-8")
        self.port = _free_port()
        overlay = self.root / "overlay.json"
        overlay.write_text(json.dumps({
            "http": {"port": self.port},
            "ops": {"filesys": {"roots": [str(self.root)], "deny_roots": []}},
        }), encoding="utf-8")
        self.log = self.root / "service.log"
        env = {**os.environ, "TI_HTTP_TOKEN": TOKEN,
               "TI_HTTP_TOKEN_FILE": str(self.root / "token"),
               "TI_CONF_OVERLAY": str(overlay),
               "PYTHONPATH": "service"}
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "ti.http.api"], cwd=str(ROOT), env=env,
            stdout=self.log.open("wb"), stderr=subprocess.STDOUT)
        self.addCleanup(self._stop)

    def _stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(5)

    def _wait_for_health(self, timeout: float = 25.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                self.fail(f"the service exited before serving /health:\n"
                          f"{self.log.read_text(errors='replace')[-2000:]}")
            try:
                conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=1)
                conn.request("GET", "/health")
                if conn.getresponse().status == 200:
                    conn.close()
                    return
                conn.close()
            except OSError:
                pass
            time.sleep(0.1)
        self.fail(f"/health never answered; service log:\n"
                  f"{self.log.read_text(errors='replace')[-2000:]}")

    def _post(self, body: dict, token: str | None) -> tuple[int, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        try:
            headers = {"Content-Type": "application/json"}
            if token is not None:
                headers["X-TI-Token"] = token
            conn.request("POST", "/v1/useTool", body=json.dumps(body).encode(),
                         headers=headers)
            response = conn.getresponse()
            raw = response.read()
            return response.status, (json.loads(raw) if raw else {})
        finally:
            conn.close()

    def test_the_spawned_service_serves_the_call_it_was_spawned_for(self):
        self._wait_for_health()
        body = {"intent": "read the file we discussed",
                "params": {"tool": "filesys_read", "args": {"path": "hello.txt"}},
                "digest": {"cwd": str(self.root)}}
        status, payload = self._post(body, TOKEN)
        self.assertEqual(status, 200, payload)
        self.assertEqual(payload["status"], "done", payload)
        self.assertIn("hello from the run root", str(payload["result"]))

    def test_the_spawned_service_refuses_an_unauthenticated_call(self):
        self._wait_for_health()
        body = {"intent": "read the file we discussed",
                "params": {"tool": "filesys_read", "args": {"path": "hello.txt"}},
                "digest": {"cwd": str(self.root)}}
        status, payload = self._post(body, None)
        self.assertEqual(status, 401, payload)


class SoftPathTest(ServiceCase):
    """`params.path` is a scope hint, and it is checked before any syscall.

    It is not an op argument, so `_check_path_args` never saw it: the value went
    to `planner.scope._existing_dir`, which called `os.path.isdir` on it. One
    `stat`, and a name inside the sshfs mount blocks that thread for hours
    (docs/review2-2026-11-03.md NB2).
    """

    def test_a_denied_root_named_in_params_path_is_refused_at_the_door(self):
        dead = self.root / "mnt"
        dead.mkdir()
        (dead / "secret.txt").write_text("x", encoding="utf-8")
        status, body = self.post({"intent": "list the files in the directory",
                                  "params": {"path": str(dead)},
                                  "digest": {"cwd": str(self.root)}})
        self.assertEqual(status, 403, body)
        self.assertEqual(body["error"], "path_refused")
        self.assertIn("params.path", body["detail"])

    def test_a_path_outside_every_policy_root_is_refused(self):
        outside = tempfile.mkdtemp(prefix="ti-outside-")
        self.addCleanup(shutil.rmtree, outside, True)
        status, body = self.post({"intent": "list the files in the directory",
                                  "params": {"path": outside},
                                  "digest": {"cwd": str(self.root)}})
        self.assertEqual(status, 403, body)
        self.assertIn("params.path", body["detail"])

    def test_a_permitted_directory_still_scopes_the_listing(self):
        sub = self.root / "docs"
        sub.mkdir()
        (sub / "a.md").write_text("x", encoding="utf-8")
        status, body = self.post({"intent": "list the files in the directory",
                                  "params": {"path": str(sub)},
                                  "digest": {"cwd": str(self.root)}})
        self.assertEqual(status, 200, body)
        self.assertEqual(body["status"], "done", body)


if __name__ == "__main__":
    unittest.main()
