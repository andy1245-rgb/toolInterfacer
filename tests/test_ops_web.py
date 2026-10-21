"""Ops web/reddit adapters: CLI contract in, artifacts out. No network, no live CLIs.

Every test points ``ops.*.script`` at ``tests/data/fake_cli.py`` through
``TI_CONF_OVERLAY`` and asserts on what the op returns, so the CLI contract
(footer / path-only / JSON-on-stdout, exit codes, timeouts) is exercised for
real — as a subprocess — without a skill registry or a socket.

Two tests replay the recorded devskills fixtures when that tree is present
(``TI_DEVSKILLS_FIXTURES``, default ``~/toolInterfacerExtras/devskills/*/fixtures``)
so the parsers are also checked against a payload the live CLIs actually wrote.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))

from ti.conf import reload  # noqa: E402
from ti.ops import cli as op_cli  # noqa: E402
from ti.ops.reddit import digest as reddit_digest  # noqa: E402
from ti.ops.web import fetch as web_fetch  # noqa: E402
from ti.ops.web import search as web_search  # noqa: E402

FAKE = ROOT / "tests" / "data" / "fake_cli.py"
DEVSKILLS = pathlib.Path(os.environ.get(
    "TI_DEVSKILLS_FIXTURES", str(ROOT / "tests" / "data" / "devskills")))
WS_FIXTURE = (DEVSKILLS / "web-search" / "fixtures" /
              "what-is-the-current-python-stable-version" / "combined.json")
WF_FIXTURE = DEVSKILLS / "web-fetch" / "fixtures" / "https-example-com" / "combined.json"

_ENV_KEYS = ("TI_CONF", "TI_CONF_OVERLAY", "FAKE_CLI_OUT")


class OpsCase(unittest.TestCase):
    """Temp artifact dir + conf overlay pointing every op at the fake CLI."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ti-ops-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self._saved_env = {k: os.environ.get(k) for k in _ENV_KEYS}
        self.addCleanup(self._restore_env)
        os.environ.pop("TI_CONF", None)
        os.environ["FAKE_CLI_OUT"] = self.tmp

    def _restore_env(self):
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        reload()

    def configure(self, ops: dict) -> None:
        """Write ``ops`` as a conf overlay and make it the live conf."""
        path = pathlib.Path(self.tmp) / "overlay.json"
        path.write_text(json.dumps({"ops": ops}), encoding="utf-8")
        os.environ["TI_CONF_OVERLAY"] = str(path)
        reload()

    def standard_ops(self, **extra):
        ops = {
            "web_search": {"script": str(FAKE), "backend": "free-search", "brave": "off"},
            "web_fetch": {"script": str(FAKE), "max_chars": 500},
            "reddit": {"cwd": str(pathlib.Path(self.tmp) / "no-such-project"),
                       "cwd_candidates": [str(pathlib.Path(self.tmp) / "also-missing")],
                       "cli": ["fake"]},
            "cli_timeout_s": 10,
        }
        ops.update(extra)
        return ops

    def argv_of(self, artifact_path):
        """The argv the fake CLI actually received."""
        doc = json.loads(pathlib.Path(artifact_path).read_text(encoding="utf-8"))
        return doc["_argv"]


class WebSearchTest(OpsCase):
    def test_happy_path_merges_and_dedupes_two_queries(self):
        self.configure(self.standard_ops())
        kind, data, conf = web_search.run({"queries": ["alpha", "beta"]}, store=None)

        self.assertEqual(kind, "url_list")
        self.assertEqual(conf, 1.0)
        urls = [item["url"] for item in data]
        # https://example.test/a#frag and https://EXAMPLE.test/a are one URL
        self.assertEqual(len(urls), 3, urls)
        self.assertEqual(len(set(urls)), 3)
        self.assertIn("https://other.test/alpha", urls)
        self.assertIn("https://other.test/beta", urls)
        for item in data:
            self.assertTrue({"title", "url", "snippet", "sources",
                             "source_count", "queries"} <= set(item))
        merged = next(i for i in data if i["url"].startswith("https://example.test/a"))
        self.assertEqual(merged["queries"], ["alpha", "beta"])
        self.assertEqual(sorted(merged["sources"]), ["s1", "s2"])
        self.assertEqual(merged["source_count"], 2)

    def test_single_query_as_scalar_string(self):
        """The runner hands a 1-element picked_items artifact over unwrapped."""
        self.configure(self.standard_ops())
        kind, data, conf = web_search.run({"queries": "alpha"}, store=None)
        self.assertEqual(kind, "url_list")
        self.assertEqual(len(data), 2)
        self.assertEqual(conf, 1.0)

    def test_argv_is_a_list_without_a_shell(self):
        self.configure(self.standard_ops())
        query = "what is 2 > 1; rm -rf /tmp/nope"
        web_search.run({"queries": [query]}, store=None)
        argv = self.argv_of(pathlib.Path(self.tmp) / "combined.json")
        self.assertIn(query, argv)               # passed through as one argv item
        self.assertEqual(argv[-2:], ["--", query][-2:])
        self.assertNotIn("; rm -rf /tmp/nope", " ".join(argv[:-1]))
        self.assertIn("--no-brave", argv)        # ops.web_search.brave = off
        self.assertIn("--no-preview", argv)

    def test_partial_failure_reports_coverage_conf(self):
        self.configure(self.standard_ops())
        kind, data, conf = web_search.run({"queries": ["alpha", "boom"]}, store=None)
        self.assertEqual(kind, "url_list")
        self.assertEqual(conf, 0.5)
        self.assertTrue(all(item["queries"] == ["alpha"] for item in data))

    def test_empty_result_set_is_not_an_error(self):
        self.configure(self.standard_ops())
        kind, data, conf = web_search.run({"queries": ["empty"]}, store=None)
        self.assertEqual((kind, data, conf), ("url_list", [], 0.0))

    def test_every_query_failing_raises(self):
        self.configure(self.standard_ops())
        with self.assertRaises(op_cli.CliFailed) as ctx:
            web_search.run({"queries": ["boom", "boom"]}, store=None)
        self.assertIn("boom", str(ctx.exception))

    def test_web_search_backend_reads_manifests(self):
        ops = self.standard_ops()
        ops["web_search_provider"] = {"script": str(FAKE), "provider": "duckduckgo"}
        self.configure(ops)
        kind, data, conf = web_search.run({"queries": ["alpha"], "backend": "web-search"},
                                          store=None)
        self.assertEqual(kind, "url_list")
        self.assertEqual(conf, 1.0)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["sources"], ["duckduckgo"])
        argv = self.argv_of(pathlib.Path(self.tmp) / "combined.json")
        self.assertIn("--provider", argv)
        self.assertIn("duckduckgo", argv)

    def test_missing_script_is_a_config_error(self):
        ops = self.standard_ops()
        ops["web_search"] = {"script": str(pathlib.Path(self.tmp) / "nope"),
                             "backend": "free-search"}
        self.configure(ops)
        with self.assertRaises(op_cli.CliMissingBinary):
            web_search.run({"queries": ["alpha"]}, store=None)


class WebFetchTest(OpsCase):
    def test_happy_path_keeps_input_order(self):
        self.configure(self.standard_ops())
        kind, pages, conf = web_fetch.run(
            {"urls": ["https://a.test/one", "https://b.test/two"]}, store=None)
        self.assertEqual(kind, "pages")
        self.assertEqual(conf, 1.0)
        self.assertEqual([p["url"] for p in pages],
                         ["https://a.test/one", "https://b.test/two"])
        for page in pages:
            self.assertEqual(page["method"], "curl")
            self.assertEqual(page["content"], "content:" + page["url"])
            self.assertIsNone(page["error"])

    def test_scalar_url_from_single_pick(self):
        self.configure(self.standard_ops())
        kind, pages, conf = web_fetch.run({"urls": "https://a.test/one"}, store=None)
        self.assertEqual(kind, "pages")
        self.assertEqual(len(pages), 1)
        self.assertEqual(conf, 1.0)

    def test_failed_page_is_reported_not_dropped(self):
        self.configure(self.standard_ops())
        kind, pages, conf = web_fetch.run(
            {"urls": ["https://a.test/one", "https://boom.test/x"]}, store=None)
        self.assertEqual(kind, "pages")
        self.assertEqual(len(pages), 2)
        self.assertEqual(pages[0]["content"], "content:https://a.test/one")
        failed = pages[1]
        self.assertEqual(failed["url"], "https://boom.test/x")
        self.assertEqual(failed["content"], "")
        self.assertIsNone(failed["method"])
        self.assertIn("exit 3", failed["error"])
        self.assertAlmostEqual(conf, 0.5)

    def test_page_without_text_is_a_failure_not_an_error_entry(self):
        self.configure(self.standard_ops())
        kind, pages, conf = web_fetch.run({"urls": ["https://notext.test/x"]}, store=None)
        self.assertEqual(kind, "pages")
        self.assertEqual(pages[0]["content"], "")
        self.assertEqual(pages[0]["error"], "All fallbacks failed")
        self.assertEqual(conf, 0.0)

    def test_degraded_extraction_scores_half(self):
        self.configure(self.standard_ops())
        _kind, pages, conf = web_fetch.run({"urls": ["https://degraded.test/x"]}, store=None)
        self.assertEqual(pages[0]["method"], "python-stdlib")
        self.assertEqual(conf, 0.5)

    def test_max_chars_comes_from_conf(self):
        self.configure(self.standard_ops())
        web_fetch.run({"urls": ["https://a.test/one"]}, store=None)
        argv = self.argv_of(pathlib.Path(self.tmp) / "combined.json")
        self.assertEqual(argv[argv.index("--max-chars") + 1], "500")
        self.assertIn("https://a.test/one", argv)

    def test_empty_input_is_an_empty_artifact(self):
        self.configure(self.standard_ops())
        self.assertEqual(web_fetch.run({"urls": []}, store=None), ("pages", [], 0.0))


class RedditDigestTest(OpsCase):
    def test_unavailable_cli_raises_with_the_paths_it_tried(self):
        self.configure(self.standard_ops())
        with self.assertRaises(op_cli.CliMissingBinary) as ctx:
            reddit_digest.run({"query": "switch lube"}, store=None)
        message = str(ctx.exception)
        self.assertIn("no-such-project", message)
        self.assertIn("also-missing", message)
        self.assertIn("reddit-research CLI unavailable", message)

    def test_happy_path_reads_raw_json_and_caps_threads(self):
        project = pathlib.Path(self.tmp) / "reddit-project"
        project.mkdir()
        shutil.copy(FAKE, project / "cli.py")
        ops = self.standard_ops()
        ops["reddit"] = {"cwd": str(project), "cwd_candidates": [],
                         "cli": ["cli.py"]}
        self.configure(ops)

        kind, data, conf = reddit_digest.run(
            {"query": "switch lube", "subreddit": "MechanicalKeyboards",
             "thread_cap": 3}, store=None)
        self.assertEqual(kind, "thread_digest")
        self.assertEqual(conf, 1.0)
        self.assertEqual(data["query"], "switch lube")
        self.assertEqual(data["subreddit"], "MechanicalKeyboards")
        self.assertEqual(len(data["posts"]), 3)
        post = data["posts"][0]
        self.assertEqual(post["id"], "p0")
        self.assertNotIn("unknown_field", post)
        self.assertEqual(set(post), set(reddit_digest.POST_FIELDS))

    def test_missing_venv_entry_point(self):
        project = pathlib.Path(self.tmp) / "reddit-project"
        project.mkdir()
        ops = self.standard_ops()
        ops["reddit"] = {"cwd": str(project), "cwd_candidates": [],
                         "cli": [".venv/bin/python", "-m", "reddit_research.cli"]}
        self.configure(ops)
        with self.assertRaises(op_cli.CliMissingBinary) as ctx:
            reddit_digest.run({"query": "switch lube"}, store=None)
        self.assertIn("uv sync", str(ctx.exception))


class CliAdapterTest(OpsCase):
    """The adapter itself, one failure mode per test."""

    def run_fake(self, target, scenario="fetch", timeout_s=10.0, tolerate=()):
        self.configure(self.standard_ops(cli_timeout_s=timeout_s))
        argv = [sys.executable, str(FAKE)]
        argv += ["--max-chars", "500"] if scenario == "fetch" else []
        argv += ["--", target]
        return op_cli.run_result(argv, timeout_s, tolerate=tolerate)

    def test_footer_result_file(self):
        run, doc = self.run_fake("https://a.test/one")
        self.assertEqual(run.exit_code, 0)
        self.assertEqual(doc["text"], "content:https://a.test/one")
        found = op_cli.result_path(run)
        self.assertEqual(found.name, "combined.json")
        self.assertTrue(str(found).startswith(self.tmp))

    def test_print_path_only(self):
        _run, doc = self.run_fake("https://pathonly.test/x")
        self.assertEqual(doc["text"], "content:https://pathonly.test/x")

    def test_json_on_stdout(self):
        _run, doc = self.run_fake("https://jsonout.test/x")
        self.assertEqual(doc["text"], "content:https://jsonout.test/x")

    def test_nonzero_exit_raises_with_the_run_attached(self):
        with self.assertRaises(op_cli.CliFailed) as ctx:
            self.run_fake("https://boom.test/x")
        self.assertEqual(ctx.exception.run.exit_code, 3)
        self.assertIn("fetch failed", str(ctx.exception))

    def test_nonzero_exit_can_be_tolerated(self):
        run, doc = self.run_fake("https://notext.test/x", tolerate=(1,))
        self.assertEqual(run.exit_code, 1)
        self.assertNotIn("text", doc)

    def test_timeout_raises(self):
        with self.assertRaises(op_cli.CliTimeout) as ctx:
            self.run_fake("https://timeout.test/x", timeout_s=0.4)
        self.assertIn("timed out", str(ctx.exception))

    def test_missing_result_file_raises(self):
        with self.assertRaises(op_cli.CliMissingResult) as ctx:
            self.run_fake("https://nofile.test/x")
        self.assertIn("never-written.json", str(ctx.exception))

    def test_unparseable_result_raises(self):
        with self.assertRaises(op_cli.CliUnparseable):
            self.run_fake("https://badjson.test/x")

    def test_missing_binary_raises(self):
        with self.assertRaises(op_cli.CliMissingBinary):
            op_cli.launcher(str(pathlib.Path(self.tmp) / "absent"))

    def test_executable_script_with_lf_shebang_runs_itself(self):
        script = pathlib.Path(self.tmp) / "plain-cli"
        script.write_bytes(b"#!/usr/bin/env python3\nprint('ok')\n")
        script.chmod(0o755)
        self.assertEqual(op_cli.launcher(str(script)), [str(script)])

    def test_crlf_shebang_falls_back_to_this_interpreter(self):
        """The live free-search script is stored CRLF; direct exec exits 127."""
        script = pathlib.Path(self.tmp) / "crlf-cli"
        script.write_bytes(b"#!/usr/bin/env python3\r\nprint('ok')\n")
        script.chmod(0o755)
        argv = op_cli.launcher(str(script))
        self.assertEqual(argv, [sys.executable, str(script)])
        run = op_cli.run_cli(argv, timeout_s=10)
        self.assertEqual((run.exit_code, run.stdout.strip()), (0, "ok"))

    def test_non_python_non_executable_script_raises(self):
        script = pathlib.Path(self.tmp) / "shell-cli"
        script.write_bytes(b"echo hi\n")
        with self.assertRaises(op_cli.CliMissingBinary):
            op_cli.launcher(str(script))

    def test_dev_mode_is_appended_when_conf_sets_it(self):
        self.configure(self.standard_ops(dev_mode="replay"))
        argv = op_cli.cli_argv(str(FAKE), ["--no-preview", "--", "q"])
        self.assertEqual(argv[-5:], ["--dev-mode", "replay", "--no-preview", "--", "q"])
        web_fetch.run({"urls": ["https://a.test/one"]}, store=None)
        self.assertIn("--dev-mode", self.argv_of(pathlib.Path(self.tmp) / "combined.json"))

    def test_normalize_url(self):
        self.assertEqual(web_search.normalize_url("https://Example.TEST/a?b=1#frag"),
                         "https://example.test/a?b=1")
        self.assertEqual(web_search.normalize_url("not a url"), "not a url")


class DevSkillsFixtureTest(OpsCase):
    """Parse a payload the live CLIs actually wrote (read-only fixture replay)."""

    @unittest.skipUnless(WS_FIXTURE.is_file(), f"no recorded fixture at {WS_FIXTURE}")
    def test_recorded_free_search_payload(self):
        os.environ["FAKE_CLI_REPLAY"] = str(WS_FIXTURE)
        self.addCleanup(os.environ.pop, "FAKE_CLI_REPLAY", None)
        self.configure(self.standard_ops())
        kind, data, conf = web_search.run(
            {"queries": ["what is the current python stable version"]}, store=None)
        self.assertEqual(kind, "url_list")
        self.assertEqual(conf, 1.0)
        self.assertEqual(len(data), 3)
        self.assertTrue(all(item["url"].startswith("https://news.google.com/")
                            for item in data))
        self.assertEqual(data[0]["sources"], ["google_news"])

    @unittest.skipUnless(WF_FIXTURE.is_file(), f"no recorded fixture at {WF_FIXTURE}")
    def test_recorded_web_fetch_payload(self):
        doc = json.loads(WF_FIXTURE.read_text(encoding="utf-8"))
        page, score = web_fetch._page("https://example.com", doc)
        self.assertEqual(page["method"], "python-stdlib")
        self.assertIn("Example Domain", page["content"])
        self.assertIsNone(page["error"])
        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
