"""P0 skeleton test: every module imports; registries work; config parses."""
import importlib
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))

MODULES = [
    "ti", "ti.conf", "ti.core.artifacts", "ti.core.budget", "ti.core.gates",
    "ti.core.runner", "ti.decide", "ti.decide.primitives", "ti.decide.resolve",
    "ti.decide.backends.fixture", "ti.decide.backends.heuristic_micro",
    "ti.decide.backends.d1_systemone",
    "ti.ops", "ti.ops.op_api", "ti.ops.decisions", "ti.ops.web.search", "ti.ops.web.fetch",
    "ti.ops.reddit.digest", "ti.ops.filesys.impl",
    "ti.planner", "ti.planner.inspector", "ti.planner.free",
    "ti.planner.templates.research", "ti.planner.templates.reddit_digest",
    "ti.planner.templates.disk_lookup",
    "ti.context.packer", "ti.context.session_digest",
    "ti.policy.askback", "ti.policy.confidence",
    "ti.hops.log", "ti.http.api", "ti.service_entry",
]


class SkeletonTest(unittest.TestCase):
    def test_every_module_imports(self):
        for name in MODULES:
            importlib.import_module(name)

    def test_conf_parses_and_has_every_section(self):
        from ti.conf import load
        conf = load()
        for section in ("decide", "policy", "budget", "context", "research",
                        "reddit", "ops", "hops_log", "http"):
            self.assertIn(section, conf)

    def test_decide_registry_resolves(self):
        from ti.decide.primitives import make_backend
        fb = make_backend("fixture")
        ans = fb.choice("turn on kitchen lights", "Which tool?",
                        {"light_set": "toggle lights", "__none__": "nothing"})
        self.assertIn(ans.choice, {"light_set", "__none__"})

    def test_heuristic_micro_routes_url(self):
        from ti.decide.primitives import make_backend
        hm = make_backend("heuristic_micro")
        ans = hm.choice("fetch https://example.com docs", "route",
                        {"web_fetch": "fetch url", "web_search": "search"})
        self.assertIsNotNone(ans)
        self.assertEqual(ans.choice, "web_fetch")

    def test_ops_registry_and_filesys_ops_run(self):
        from ti.ops.op_api import all_ops, get, validate_params
        self.assertIn("filesys_read", all_ops())
        args, soft = validate_params("filesys_read", {"path": "README.md", "extra_hint": "x"})
        self.assertEqual(args["path"], "README.md")
        self.assertEqual(soft, {"extra_hint": "x"})
        kind, data, conf = get("filesys_read").run(args, store=None)
        self.assertEqual(kind, "file_content")
        self.assertIn("toolInterfacer", data)
        self.assertEqual(conf, 1.0)

    def test_planner_inspector_levels(self):
        from ti.planner.inspector import inspect
        self.assertEqual(inspect("write a file", {"tool": "write"}, {})["level"], "L0")
        meta = inspect("research the options for cron jobs", {}, {})
        self.assertEqual(meta["level"], "L2")
        self.assertEqual(meta["template"], "research")

    def test_packer_trims(self):
        from ti.context.packer import pack
        state = pack("x", {"k": "v" * 20_000}, {}, {})
        self.assertLessEqual(len(str(state)) // 4, 2050)

    def test_fixture_backend_replay_shape(self):
        from ti.decide.primitives import make_backend
        fb = make_backend("fixture")
        p = fb.noul("s", "is this relevant?")
        self.assertTrue(0.0 <= p <= 1.0)

    def test_defaults_json_is_stable_json(self):
        path = ROOT / "service" / "ti" / "conf" / "defaults.json"
        json.loads(path.read_text())


if __name__ == "__main__":
    unittest.main()
