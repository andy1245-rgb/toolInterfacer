"""Bench harness tests — the statistics, the no-clobber rule, and the selftest path.

Two numbers in this harness are load-bearing enough to be pinned against a hand
computation rather than against a golden file this module wrote itself: the Wilson
interval (it bounds every accuracy claim) and the McNemar exact p (it decides
whether an A/B run showed anything). Both are recomputed here from their closed
form, so a refactor that changes them fails on the arithmetic, not on a diff.

The live-arm path is covered without a model: a stub HTTP server answers
``/health`` and ``/props``, so the guard's accept and abort branches, and the
"no probe traffic on mismatch" rule, are exercised for real.
"""
import io
import json
import math
import pathlib
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT / "service"))

import run  # noqa: E402  (bench/run.py)
from ti.conf import load as load_conf  # noqa: E402
from ti_bench import arms, compare, guard, report, scoring, selftest, sets, stats  # noqa: E402

D1 = "d1-3B-Q4_K_M.gguf"
QWEN = "Qwen3.5-4B-Q4_K_M.gguf"


class WilsonCiTest(unittest.TestCase):
    def test_hand_computed_27_of_30(self):
        # p=0.9, z=1.96: centre = 0.9+0.06403 / 1.12805 = 0.85469, half = 0.11081
        lo, hi = stats.wilson_ci(27, 30)
        self.assertAlmostEqual(lo, 0.7438, places=4)
        self.assertAlmostEqual(hi, 0.9654, places=4)

    def test_hand_computed_zero_of_ten(self):
        # the textbook 0/10 interval: [0, 0.2775] — the normal approximation
        # would claim a zero-width interval here, which is why Wilson is used
        lo, hi = stats.wilson_ci(0, 10)
        self.assertAlmostEqual(lo, 0.0, places=12)
        self.assertAlmostEqual(hi, 0.2775, places=4)

    def test_matches_the_closed_form(self):
        z = 1.959963984540054
        for correct, n in ((5, 5), (111, 150), (1, 1), (0, 10), (64, 64)):
            with self.subTest(correct=correct, n=n):
                p = correct / n
                centre = (p + z * z / (2 * n)) / (1 + z * z / n)
                half = (z / (1 + z * z / n)) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
                lo, hi = stats.wilson_ci(correct, n)
                self.assertAlmostEqual(lo, centre - half, places=10)
                self.assertAlmostEqual(hi, centre + half, places=10)

    def test_brackets_the_point_estimate_and_stays_inside_the_unit_interval(self):
        for correct, n in ((0, 40), (1, 40), (20, 40), (40, 40), (149, 150)):
            with self.subTest(correct=correct, n=n):
                block = stats.rate(correct, n)
                lo = block["wilson95"]["low"]
                hi = block["wilson95"]["high"]
                self.assertTrue(0.0 <= lo <= block["accuracy"] <= hi <= 1.0)

    def test_empty_sample_is_uninformative_not_a_crash(self):
        self.assertEqual(stats.wilson_ci(0, 0), (0.0, 1.0))
        self.assertIsNone(stats.rate(0, 0)["accuracy"])

    def test_more_rows_tighten_the_interval(self):
        def width(n):
            lo, hi = stats.wilson_ci(n // 2, n)
            return hi - lo
        self.assertGreater(width(30), width(150))
        self.assertGreater(width(150), width(600))


class McNemarExactTest(unittest.TestCase):
    def test_hand_computed_b0_c5(self):
        # P(X <= 0) = C(5,0)/2^5 = 1/32; two-sided p = 2/32 = 0.0625
        self.assertAlmostEqual(stats.mcnemar_exact(0, 5), 0.0625, places=12)

    def test_hand_computed_b1_c9(self):
        # P(X <= 1) = (C(10,0)+C(10,1))/2^10 = 11/1024 = 0.0107421875
        # two-sided p = 0.021484375
        self.assertAlmostEqual(stats.mcnemar_exact(1, 9), 0.021484375, places=12)

    def test_matches_the_binomial_tail_sum(self):
        for b, c in ((2, 7), (4, 4), (0, 12), (9, 3)):
            with self.subTest(b=b, c=c):
                n, k = b + c, min(b, c)
                expected = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
                self.assertAlmostEqual(stats.mcnemar_exact(b, c), expected, places=12)

    def test_a_single_discordant_pair_proves_nothing(self):
        # the measured case in docs/decision-pair-vs-hosted.md: 0 pair-only,
        # 1 hosted-only, exact p = 1.000
        self.assertEqual(stats.mcnemar_exact(0, 1), 1.0)
        self.assertEqual(stats.mcnemar_exact(1, 0), 1.0)

    def test_symmetric_in_its_arguments(self):
        for b, c in ((0, 0), (1, 2), (3, 8), (12, 0)):
            with self.subTest(b=b, c=c):
                self.assertAlmostEqual(stats.mcnemar_exact(b, c), stats.mcnemar_exact(c, b),
                                       places=12)

    def test_no_discordant_rows_is_p_one(self):
        self.assertEqual(stats.mcnemar_exact(0, 0), 1.0)

    def test_never_exceeds_one(self):
        for b in range(0, 8):
            for c in range(0, 8):
                self.assertLessEqual(stats.mcnemar_exact(b, c), 1.0)


class _StubProps(BaseHTTPRequestHandler):
    """Serves /health and /props from class attrs, and records every path asked."""

    model_path = ""
    requested: list[str] = []

    def do_GET(self):  # noqa: N802 (http.server's naming)
        type(self).requested.append(self.path)
        if self.path == "/health":
            body = {"status": "ok"}
        elif self.path == "/props":
            body = {"model_path": type(self).model_path, "model_ftype": "Q4_K - Medium",
                    "build_info": "b11512-a11f57ba9"}
        else:
            self.send_error(404)
            return
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


class GuardTest(unittest.TestCase):
    """The /props gate, against a stub server: no model, no network beyond loopback."""

    def setUp(self):
        self.handler = type("Handler", (_StubProps,), {"requested": [], "model_path": D1})
        self.server = HTTPServer(("127.0.0.1", 0), self.handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self._tmp = tempfile.TemporaryDirectory(prefix="ti-bench-test-")
        self.addCleanup(self._tmp.cleanup)
        self.tmp = pathlib.Path(self._tmp.name)
        spec = {"arms": {"stub": {"backend": "fixture", "requires": ["llama-server"],
                                  "endpoint": "decide.d1_systemone.base_url",
                                  "expect_model": D1}}}
        self.spec_path = self.tmp / "arms.json"
        self.spec_path.write_text(json.dumps(spec))
        cfg = load_conf()
        original = cfg["decide"]["d1_systemone"]["base_url"]
        self.addCleanup(cfg["decide"]["d1_systemone"].__setitem__, "base_url", original)

    def test_matching_props_is_asserted_and_recorded(self):
        rep = scoring.score(arms.get_arm("stub", self.spec_path), sets.load("v2")[:4],
                            warmup=0, base_url=self.url)
        self.assertTrue(rep["guard"]["checked"])
        self.assertEqual(rep["guard"]["served_basename"], D1)
        self.assertEqual(rep["guard"]["expect_model"], D1)
        self.assertEqual(rep["guard"]["source"], self.url + "/props")

    def test_mismatch_aborts_non_zero_before_any_probe_traffic(self):
        self.handler.model_path = QWEN  # a stale server answering under another name
        out = self.tmp / "report.json"
        code = run.main(["run", "--arm", "stub", "--arms", str(self.spec_path),
                         "--out", str(out), "--base-url", self.url])
        self.assertEqual(code, 3)
        self.assertFalse(out.exists(), "a refused run wrote a report")
        self.assertEqual(sorted(self.handler.requested), ["/health", "/props"])

    def test_an_unasserted_arm_is_refused_without_the_flag(self):
        spec = {"arms": {"stub": {"backend": "fixture", "requires": ["llama-server"],
                                  "endpoint": "decide.d1_systemone.base_url"}}}
        self.spec_path.write_text(json.dumps(spec))
        arm = arms.get_arm("stub", self.spec_path)
        with self.assertRaises(guard.GuardError) as ctx:
            _preflight_at(arm, self.url)
        self.assertIn("expect-model", str(ctx.exception))

    def test_the_flag_lets_an_unasserted_run_through_with_a_warning(self):
        spec = {"arms": {"stub": {"backend": "fixture", "requires": ["llama-server"],
                                  "endpoint": "decide.d1_systemone.base_url"}}}
        self.spec_path.write_text(json.dumps(spec))
        err = io.StringIO()
        block = _preflight_at(arms.get_arm("stub", self.spec_path), self.url,
                              allow_unasserted=True, stream=err)
        self.assertFalse(block["checked"])
        self.assertEqual(block["served_basename"], D1)
        self.assertIn("no served-model assertion", err.getvalue().lower())

    def test_missing_hosted_key_refuses_a_hosted_arm(self):
        spec = {"arms": {"hosted": {"backend": "fixture", "requires": ["hosted"]}}}
        self.spec_path.write_text(json.dumps(spec))
        import os
        saved = os.environ.pop("TI_OPENROUTER_API_KEY", None)
        if saved is not None:
            self.addCleanup(os.environ.__setitem__, "TI_OPENROUTER_API_KEY", saved)
        with self.assertRaises(guard.GuardError):
            guard.preflight(arms.get_arm("hosted", self.spec_path))

    def test_unreachable_endpoint_is_a_guard_error_not_a_traceback(self):
        arm = arms.get_arm("stub", self.spec_path)
        with self.assertRaises(guard.GuardError):
            _preflight_at(arm, "http://127.0.0.1:1")


def _preflight_at(arm: arms.Arm, url: str, **kwargs) -> dict:
    """Run the guard against a specific endpoint (the arm's conf path repointed)."""
    saved = arms.conf_value(arm.endpoint)
    arms.set_conf(arm.endpoint, url)
    try:
        return guard.preflight(arm, **kwargs)
    finally:
        arms.set_conf(arm.endpoint, saved)


class NoClobberTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="ti-bench-clobber-")
        self.addCleanup(self._tmp.cleanup)
        self.tmp = pathlib.Path(self._tmp.name)
        self.report = scoring.score(arms.get_arm("fixture"), sets.load("v2")[:6], warmup=0)

    def test_a_second_write_is_refused(self):
        path = report.write_report(self.tmp / "r.json", self.report)
        before = path.read_text()
        with self.assertRaises(report.ReportExistsError):
            report.write_report(path, self.report)
        self.assertEqual(path.read_text(), before, "the refused write touched the file")

    def test_force_is_the_only_way_past_it(self):
        path = report.write_report(self.tmp / "r.json", self.report)
        changed = dict(self.report, generated="overwritten")
        report.write_report(path, changed, force=True)
        self.assertEqual(json.loads(path.read_text())["generated"], "overwritten")

    def test_the_cli_refuses_an_existing_out(self):
        out = self.tmp / "cli.json"
        self.assertEqual(run.main(["run", "--arm", "fixture", "--out", str(out)]), 0)
        before = out.read_text()
        code = run.main(["run", "--arm", "fixture", "--out", str(out)])
        self.assertEqual(code, 1)
        self.assertEqual(out.read_text(), before)

    def test_the_cli_writes_with_force(self):
        out = self.tmp / "cli.json"
        self.assertEqual(run.main(["run", "--arm", "fixture", "--out", str(out)]), 0)
        self.assertEqual(run.main(["run", "--arm", "fixture", "--out", str(out), "--force"]), 0)


class ReportShapeTest(unittest.TestCase):
    def setUp(self):
        self.report = scoring.score(arms.get_arm("fixture"), sets.load("v2")[:20], warmup=0)

    def test_a_real_run_satisfies_the_contract(self):
        self.assertEqual(report.validate(self.report), [])

    def test_every_row_carries_the_fields_a_reader_needs(self):
        for row in self.report["per_row"]:
            self.assertTrue({"id", "band", "expected", "chosen", "confidence", "latency_ms",
                             "real_call", "source"} <= set(row))
            self.assertTrue(row["id"].startswith("v2:"))
            self.assertIn(row["band"], ("t1", "t2", "t3"))

    def test_an_offline_arm_reports_no_real_calls(self):
        self.assertEqual(self.report["counts"]["real_calls"], 0)
        self.assertEqual({r["source"] for r in self.report["per_row"]}, {"in-process"})

    def test_validate_catches_broken_counts(self):
        broken = json.loads(json.dumps(self.report))
        broken["counts"]["correct"] += 1
        self.assertTrue(report.validate(broken))

    def test_validate_catches_an_impossible_percentile_order(self):
        broken = json.loads(json.dumps(self.report))
        broken["latency"]["p50"], broken["latency"]["p99"] = (
            broken["latency"]["p99"], broken["latency"]["p50"])
        self.assertTrue(report.validate(broken))

    def test_validate_catches_a_row_that_claims_a_real_call_while_abstaining(self):
        broken = json.loads(json.dumps(self.report))
        broken["per_row"][0].update(source="abstain", real_call=True, confidence=None,
                                    correct=False, chosen=None)
        self.assertTrue(report.validate(broken))

    def test_read_report_refuses_a_foreign_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump({"per_row": []}, fh)
        self.addCleanup(pathlib.Path(fh.name).unlink)
        with self.assertRaises(ValueError):
            report.read_report(fh.name)


class CompareTest(unittest.TestCase):
    def setUp(self):
        rows = sets.load("v2")[:24]
        self.a = scoring.score(arms.get_arm("fixture"), rows, warmup=0)
        self.b = scoring.score(arms.get_arm("heuristic"), rows, warmup=0)
        self.data = compare.compare(self.a, self.b)

    def test_cells_reconcile_with_the_rows(self):
        mc = self.data["mcnemar"]
        self.assertEqual(mc["both_correct"] + mc["both_wrong"] + mc["discordant"], self.data["n"])
        self.assertEqual(mc["discordant"], mc["a_only"] + mc["b_only"])
        by_uid = {r["uid"]: r for r in self.a["per_row"]}
        a_only = sum(1 for r in self.b["per_row"]
                     if not r["correct"] and by_uid[r["uid"]]["correct"])
        self.assertEqual(mc["a_only"], a_only)

    def test_p_value_is_the_exact_test_of_the_discordant_counts(self):
        mc = self.data["mcnemar"]
        self.assertAlmostEqual(mc["p_value"],
                               stats.mcnemar_exact(mc["a_only"], mc["b_only"]), places=12)

    def test_latency_delta_is_paired_on_the_same_rows(self):
        lat = self.data["latency"]
        self.assertEqual(lat["n_pairs"], self.data["n"])
        self.assertEqual(lat["b_faster"] + lat["a_faster"] + lat["equal"], lat["n_pairs"])

    def test_render_prints_the_p_value_and_the_discordant_counts(self):
        text = "\n".join(compare.render(self.data))
        self.assertIn("exact p", text)
        self.assertIn("discordant", text)

    def test_two_sets_cannot_be_compared(self):
        other = scoring.score(arms.get_arm("fixture"), sets.load("v1")[:24], warmup=0)
        with self.assertRaises(compare.CompareError):
            compare.compare(self.a, other)

    def test_a_missing_row_is_an_error_not_a_silent_intersection(self):
        trimmed = json.loads(json.dumps(self.b))
        trimmed["per_row"].pop()
        trimmed["counts"]["n"] -= 1
        trimmed["accuracy"]["n"] -= 1
        with self.assertRaises(compare.CompareError):
            compare.compare(self.a, trimmed)


class ArmSpecTest(unittest.TestCase):
    def test_the_shipped_spec_loads_and_keeps_offline_arms_offline(self):
        spec = arms.load_arms()
        self.assertIn("fixture", spec)
        self.assertTrue(spec["fixture"].offline)
        self.assertTrue(spec["heuristic"].offline)
        self.assertTrue(spec["d1"].needs_llama)
        self.assertFalse(spec["d1"].offline)
        self.assertTrue(spec["pair"].needs_hosted)

    def test_a_typo_in_a_requirement_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "arms.json"
            path.write_text(json.dumps({"arms": {"x": {"backend": "fixture",
                                                       "requires": ["llama_server"]}}}))
            with self.assertRaises(arms.ArmSpecError):
                arms.load_arms(path)

    def test_a_server_arm_must_name_its_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "arms.json"
            path.write_text(json.dumps({"arms": {"x": {"backend": "d1_systemone",
                                                       "requires": ["llama-server"]}}}))
            with self.assertRaises(arms.ArmSpecError):
                arms.load_arms(path)

    def test_an_unknown_arm_lists_the_known_ones(self):
        with self.assertRaises(SystemExit) as ctx:
            arms.get_arm("not_an_arm")
        self.assertIn("fixture", str(ctx.exception))

    def test_list_names_the_offline_arms_and_the_guarded_ones(self):
        buf = io.StringIO()
        stdout, sys.stdout = sys.stdout, buf
        try:
            self.assertEqual(run.main(["list"]), 0)
        finally:
            sys.stdout = stdout
        text = buf.getvalue()
        self.assertIn("offline", text)
        self.assertIn("/props " + D1, text)

    def test_override_applies_a_conf_variant(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "arms.json"
            path.write_text(json.dumps({"arms": {"variant": {
                "backend": "fixture", "override": {"decide.fixture.default_choice_conf": 0.42}}}}))
            cfg = arms.apply(arms.get_arm("variant", path))
            self.assertEqual(cfg["decide"]["fixture"]["default_choice_conf"], 0.42)


class SelftestPathTest(unittest.TestCase):
    def test_selftest_passes_offline(self):
        buf = io.StringIO()
        stdout, sys.stdout = sys.stdout, buf
        try:
            code = run.main(["selftest"])
        finally:
            sys.stdout = stdout
        self.assertEqual(code, 0, buf.getvalue())
        self.assertIn("PASSED", buf.getvalue())

    def test_selftest_reports_each_step(self):
        buf = io.StringIO()
        stdout, sys.stdout = sys.stdout, buf
        try:
            selftest.main()
        finally:
            sys.stdout = stdout
        text = buf.getvalue()
        for step in ("spec:", "set v2:", "scored", "report write", "compare"):
            self.assertIn(step, text)

    def test_selftest_leaves_no_report_in_the_repo(self):
        before = sorted(p.name for p in (ROOT / "bench/results").glob("*.json"))
        with io.StringIO() as buf:
            stdout, sys.stdout = sys.stdout, buf
            try:
                selftest.main()
            finally:
                sys.stdout = stdout
        after = sorted(p.name for p in (ROOT / "bench/results").glob("*.json"))
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
