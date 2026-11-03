"""Probe guard tests — served-model assertion logic, the shared token, and the
artifact loader. Stub dicts and a stub socket, no model and no service."""
import contextlib
import io
import json
import os
import pathlib
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT / "service"))

import live_e2e as probe  # noqa: E402
import live_probe  # noqa: E402
import model_guard  # noqa: E402
from ti import conf  # noqa: E402
from ti.http import auth  # noqa: E402

D1_PATH = r"C:\models\d1bench\models\d1-3B-Q4_K_M.gguf"
QWEN_PATH = r"D:\models\Qwen3.5-4B-Q4_K_M.gguf"
D1 = "d1-3B-Q4_K_M.gguf"
QWEN = "Qwen3.5-4B-Q4_K_M.gguf"


def props(path, **extra):
    d = {"model_path": path, "model_ftype": "Q4_K - Medium", "build_info": "b11512-a11f57ba9"}
    d.update(extra)
    return d


class AssertModelMatchTest(unittest.TestCase):
    def test_matching_basename_passes(self):
        served = model_guard.assert_model_match(props(D1_PATH), D1)
        self.assertEqual(served["basename"], D1)
        self.assertEqual(served["ftype"], "Q4_K - Medium")

    def test_matching_basename_passes_with_posix_path(self):
        served = model_guard.assert_model_match({"model_path": "/models/" + D1}, D1)
        self.assertEqual(served["basename"], D1)

    def test_basename_handles_both_separators(self):
        self.assertEqual(model_guard.basename(D1_PATH), D1)
        self.assertEqual(model_guard.basename("/a/b/" + D1), D1)
        self.assertEqual(model_guard.basename(""), "")

    def test_mismatched_path_aborts(self):
        with self.assertRaises(model_guard.ModelAssertionError) as ctx:
            model_guard.assert_model_match(props(QWEN_PATH), D1)
        self.assertIn(D1, str(ctx.exception))
        self.assertIn(QWEN, str(ctx.exception))

    def test_phantom_case_detected(self):
        # the real regression: a "Qwen3.5-4B" lane answered by d1-3B-Q4
        with self.assertRaises(model_guard.ModelAssertionError):
            model_guard.assert_model_match(props(D1_PATH), QWEN)

    def test_empty_props_aborts(self):
        for bad in ({}, None):
            with self.assertRaises(model_guard.ModelAssertionError):
                model_guard.assert_model_match(bad, D1)

    def test_props_without_model_path_aborts(self):
        with self.assertRaises(model_guard.ModelAssertionError):
            model_guard.assert_model_match({"model_ftype": "Q4_K - Medium"}, D1)
        with self.assertRaises(model_guard.ModelAssertionError):
            model_guard.assert_model_match({"model_path": ""}, D1)

    def test_optional_sha_and_size(self):
        p = props(D1_PATH, model_sha="abc123", model_size=2_000_000_000)
        self.assertEqual(model_guard.assert_model_match(p, D1, "abc123", 2_000_000_000)["sha"],
                         "abc123")
        with self.assertRaises(model_guard.ModelAssertionError):
            model_guard.assert_model_match(p, D1, "deadbeef")
        with self.assertRaises(model_guard.ModelAssertionError):
            model_guard.assert_model_match(p, D1, "", 123)
        with self.assertRaises(model_guard.ModelAssertionError):  # sha requested but absent
            model_guard.assert_model_match(props(D1_PATH), D1, "abc123")


class PreflightTest(unittest.TestCase):
    def test_no_expect_returns_warning(self):
        served, warning = model_guard.preflight(props(D1_PATH), "")
        self.assertEqual(served["basename"], D1)
        self.assertIsNotNone(warning)
        self.assertIn("no served-model assertion", warning.lower())

    def test_with_expect_no_warning(self):
        served, warning = model_guard.preflight(props(QWEN_PATH), QWEN)
        self.assertEqual(served["basename"], QWEN)
        self.assertIsNone(warning)

    def test_with_expect_mismatch_raises(self):
        with self.assertRaises(model_guard.ModelAssertionError):
            model_guard.preflight(props(D1_PATH), QWEN)


class LiveProbeGuardTest(unittest.TestCase):
    def test_guard_returns_served_on_match(self):
        self.assertEqual(live_probe.guard_served_model(props(D1_PATH), D1)["basename"], D1)

    def test_guard_aborts_on_mismatch_before_probing(self):
        with self.assertRaises(SystemExit) as ctx:
            live_probe.guard_served_model(props(D1_PATH), QWEN)
        self.assertIn("ABORT", str(ctx.exception))

    def test_guard_warns_when_expect_model_absent(self):
        err = io.StringIO()
        live_probe.guard_served_model(props(D1_PATH), "", stream=err)
        self.assertIn("no served-model assertion", err.getvalue().lower())

    def test_guard_aborts_on_empty_props(self):
        with self.assertRaises(SystemExit):
            live_probe.guard_served_model({}, D1)

    def test_expect_model_flag_is_wired(self):
        ap = live_probe.build_parser()
        self.assertEqual(ap.parse_args([]).expect_model, "")  # absent -> warn path
        self.assertEqual(ap.parse_args(["--expect-model", D1]).expect_model, D1)
        self.assertEqual(ap.parse_args(["--expect-model", D1, "--expect-sha", "x"]).expect_sha, "x")


class GuardCliTest(unittest.TestCase):
    def _props_file(self, text_or_obj):
        fh = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        if isinstance(text_or_obj, str):
            fh.write(text_or_obj)
        else:
            json.dump(text_or_obj, fh)
        fh.close()
        self.addCleanup(pathlib.Path(fh.name).unlink)
        return fh.name

    def test_cli_ok(self):
        path = self._props_file(props(QWEN_PATH))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            rc = model_guard.main(["--expect-model", QWEN, "--props-file", path])
        self.assertEqual(rc, 0)
        self.assertIn("props OK", out.getvalue())

    def test_cli_mismatch_exits_nonzero(self):
        path = self._props_file(props(D1_PATH))
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = model_guard.main(["--expect-model", QWEN, "--props-file", path])
        self.assertEqual(rc, 1)
        self.assertIn("ABORT", err.getvalue())

    def test_cli_empty_props_nonzero(self):
        path = self._props_file({})
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(model_guard.main(["--expect-model", D1, "--props-file", path]), 1)

    def test_cli_bad_json_nonzero(self):
        path = self._props_file("{not json")
        with contextlib.redirect_stderr(io.StringIO()):
            rc = model_guard.main(["--expect-model", D1, "--props-file", path])
        self.assertEqual(rc, 2)


class ServeLaneArgvTest(unittest.TestCase):
    """The argv builder is the fix for the shell-quoting root cause."""

    def _cfg(self):
        return json.loads((ROOT / "bench" / "lanes.json").read_text())

    def test_chat_template_kwargs_serialized_in_process(self):
        import serve_lane
        argv = serve_lane.build_argv(self._cfg(), "qwen", "/models")
        pair = argv[argv.index("--chat-template-kwargs") + 1]
        self.assertEqual(json.loads(pair), {"enable_thinking": False})
        self.assertEqual(pair, '{"enable_thinking": false}')

    def test_model_resolved_against_models_dir(self):
        import serve_lane
        argv = serve_lane.build_argv(self._cfg(), "d1", "C:/models")
        self.assertEqual(argv[0], "-m")
        self.assertTrue(argv[1].endswith("d1-3B-Q4_K_M.gguf"))
        self.assertIn("C:/models", argv[1])

    def test_absolute_model_not_rewritten(self):
        import serve_lane
        cfg = self._cfg()
        cfg["lanes"]["d1"]["model"] = "D:/abs/d1-3B-Q4_K_M.gguf"
        self.assertEqual(serve_lane.build_argv(cfg, "d1", "C:/models")[1], "D:/abs/d1-3B-Q4_K_M.gguf")


class _TokenSocket(BaseHTTPRequestHandler):
    """A service that keeps what it was sent and can refuse like the real one."""

    def do_POST(self) -> None:
        seen = str(self.headers.get(probe.TOKEN_HEADER) or "")
        if self.path == "/refuse":
            self._reply(401, {"error": "unauthenticated"})
            return
        self._reply(200, {"seen": seen})

    def _reply(self, code: int, payload: dict) -> None:
        blob = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def log_message(self, *args) -> None:  # a test socket has nothing to log
        pass


class TokenResolutionTest(unittest.TestCase):
    """The probe must resolve the token in `service/ti/http/auth.py`'s order.

    Post-auth, `bench/live_e2e.py` sent no token at all and every POST came back
    401, which the run reported as `0/4 cases passed` — a worse number than the
    code deserved. These cases pin the order (environment, then the runtime
    file), the header, and the loud 401.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = pathlib.Path(self.tmp.name, "state", "token")
        self._environ = dict(os.environ)
        self.addCleanup(self._restore)
        for name in ("TI_HTTP_TOKEN", "TI_HTTP_TOKEN_FILE", "TI_CONF_OVERLAY"):
            os.environ.pop(name, None)
        os.environ["TI_HTTP_TOKEN_FILE"] = str(self.path)
        auth.reset()

    def _restore(self):
        os.environ.clear()
        os.environ.update(self._environ)
        conf.reload()
        auth.reset()

    def _write_file(self, value: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(value + "\n", encoding="utf-8")
        self.path.chmod(0o600)
        auth.reset()

    def test_the_environment_wins_over_the_runtime_file(self):
        self._write_file("file-token")
        os.environ["TI_HTTP_TOKEN"] = "env-token"
        resolved = probe.resolve_token()
        self.assertEqual(resolved.value, "env-token")
        self.assertEqual(resolved.source, "$TI_HTTP_TOKEN")
        # the service's own resolution has to pick the same one
        self.assertEqual(resolved.value, auth.token())

    def test_the_runtime_file_answers_when_the_environment_is_empty(self):
        self._write_file("file-token")
        resolved = probe.resolve_token()
        self.assertEqual(resolved.value, "file-token")
        self.assertEqual(resolved.source, str(self.path))
        self.assertEqual(resolved.value, auth.token())

    def test_the_variable_name_is_the_one_the_conf_declares(self):
        overlay = pathlib.Path(self.tmp.name, "overlay.json")
        overlay.write_text(json.dumps({"http": {"token_env": "TI_BENCH_TOKEN"}}),
                           encoding="utf-8")
        os.environ["TI_CONF_OVERLAY"] = str(overlay)
        conf.reload()
        os.environ["TI_BENCH_TOKEN"] = "conf-token"
        resolved = probe.resolve_token()
        self.assertEqual(probe.token_env_name(), "TI_BENCH_TOKEN")
        self.assertEqual(resolved.value, "conf-token")
        self.assertEqual(resolved.value, auth.token())

    def test_an_explicit_token_comes_first(self):
        self._write_file("file-token")
        os.environ["TI_HTTP_TOKEN"] = "env-token"
        self.assertEqual(probe.resolve_token("cli-token"), probe.Token("cli-token", "--token"))

    def test_the_header_cannot_drift_from_the_service_or_the_extension(self):
        self.assertEqual(probe.TOKEN_HEADER, auth.TOKEN_HEADER)
        self.assertEqual(probe.TOKEN_HEADER, "X-TI-Token")
        # the client half names the constant it mirrors
        source = (ROOT / "pi-extension" / "src" / "token.ts").read_text(encoding="utf-8")
        self.assertIn(f'TOKEN_HEADER = "{probe.TOKEN_HEADER}"', source)

    def test_a_refused_token_is_an_error_that_names_the_cause(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _TokenSocket)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self._stop, server, thread)
        base = f"http://127.0.0.1:{server.server_address[1]}"
        token = probe.Token("abc", "$TI_HTTP_TOKEN")
        self.assertEqual(probe.http_json(f"{base}/echo", {"intent": "x"}, 10, token),
                         {"seen": "abc"})
        with self.assertRaises(probe.Unauthenticated) as caught:
            probe.http_json(f"{base}/refuse", {"intent": "x"}, 10, token)
        message = str(caught.exception)
        self.assertIn("401", message)
        self.assertIn("$TI_HTTP_TOKEN", message)      # what was sent, and where from
        self.assertIn("--token", message)            # what to do about it
        self.assertIn("auth.py", message)

    def test_a_refusal_is_not_scored_as_a_failed_case(self):
        def transport(body, timeout):
            raise probe.Unauthenticated("the service refused this probe's credentials")
        with self.assertRaises(probe.Unauthenticated):
            probe.run_cases([probe.cases(False, 60.0)[0]], probe.Ctx(), [], transport)

    def test_a_degraded_run_says_so_in_its_summary(self):
        rows = [probe.Row(case=case) for case in probe.cases(False, 60.0)[:2]]
        rows[0].body = {"degraded": True}
        rows[1].body = {"degraded": False}
        notes = probe.summary_notes(rows, "d1-3B-Q4_K_M.gguf", False)
        self.assertEqual(len(notes), 1)
        self.assertIn(rows[0].case.name, notes[0])
        self.assertIn("1 of 2", notes[0])
        # no served-model assertion is its own disclaimer, even with no degraded case
        self.assertTrue(probe.summary_notes(rows[1:], "", False))
        self.assertEqual(probe.summary_notes(rows[1:], "d1-3B-Q4_K_M.gguf", False), [])

    def _stop(self, server, thread):
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


class ClassicalLoaderTest(unittest.TestCase):
    """A malformed recorded artifact is named and skipped — it never ends the run."""

    def setUp(self):
        import classical_baseline
        self.baseline = classical_baseline
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)

    def _write(self, name: str, text: str) -> pathlib.Path:
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_a_malformed_artifact_is_reported_by_name_and_skipped(self):
        good = self._write("good.json", json.dumps({"per_row": [
            {"choice": "filesys_read", "ms": 1.0, "correct": True},
            {"choice": "filesys_glob", "ms": 2.0, "correct": False}]}))
        # the shape `e523336` left behind: a Windows path with unescaped backslashes
        bad = self._write("bad.json", r'{"model": {"path": "C:\models\d1.gguf"}}')
        runs, problems = self.baseline.load_runs([bad, good], 2)
        self.assertEqual(len(runs), 1)                       # the good one survived
        self.assertEqual(len(problems), 1)
        self.assertEqual(problems[0]["file"], bad.name)
        self.assertIn("JSONDecodeError", problems[0]["problem"])
        self.assertEqual(len(runs[0]["per_row"]), 2)
        self.assertNotIn(good.name, [p["file"] for p in problems])

    def test_a_misaligned_artifact_is_skipped_rather_than_paired(self):
        # pairing rows positionally against a short file would mislabel every row
        # and report a worse accuracy, which is not the same as a missing file
        path = self._write("short.json", json.dumps({"per_row": [
            {"choice": "filesys_read", "ms": 1.0, "correct": True}]}))
        runs, problems = self.baseline.load_runs([path], 2)
        self.assertEqual(runs, [])
        self.assertIn("1 per_row entries, expected 2", problems[0]["problem"])

    def test_an_artifact_missing_a_field_the_sweep_reads_is_skipped(self):
        path = self._write("nofield.json", json.dumps({"per_row": [
            {"choice": "filesys_read", "ms": 1.0, "correct": True},
            {"choice": "filesys_glob", "ms": 1.0}]}))
        runs, problems = self.baseline.load_runs([path], 2)
        self.assertEqual(runs, [])
        self.assertIn("per_row[1]", problems[0]["problem"])

    def test_a_directory_of_only_bad_artifacts_loads_nothing_and_says_why(self):
        path = self._write("bad.json", "{not json")
        runs, problems = self.baseline.load_runs([path], 150)
        self.assertEqual(runs, [])
        self.assertEqual(len(problems), 1)


if __name__ == "__main__":
    unittest.main()
