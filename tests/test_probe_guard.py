"""Probe guard tests — served-model assertion logic. Stub /props dicts, no network."""
import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT / "service"))

import live_probe  # noqa: E402
import model_guard  # noqa: E402

D1_PATH = r"C:\Users\andre\Documents\PersonalCodingProjects\personal\Research\jevResearch\lab\d1bench\models\d1-3B-Q4_K_M.gguf"
QWEN_PATH = r"C:\Users\andre\Documents\PersonalCodingProjects\personal\toolInterfacerLab\models\Qwen3.5-4B-Q4_K_M.gguf"
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


if __name__ == "__main__":
    unittest.main()
