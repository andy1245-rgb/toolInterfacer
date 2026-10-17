"""P1 basic-loop tests: full engine, fixture decide backend, zero network."""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))

from ti.service_entry import use_tool  # noqa: E402


class LoopTest(unittest.TestCase):
    def test_plan_mode_returns_steps(self):
        out = use_tool({"intent": "read the file we discussed",
                        "mode": "plan",
                        "digest": {"cwd": str(ROOT)}})
        self.assertEqual(out["status"], "plan")
        ops = [s["op"] for s in out["steps"]]
        self.assertEqual(ops, ["filesys_glob", "decide_pick", "filesys_read"])

    def test_disk_lookup_auto_runs_end_to_end(self):
        out = use_tool({"intent": "read the file we discussed",
                        "digest": {"cwd": str(ROOT)}})
        self.assertEqual(out["status"], "done", out)
        self.assertFalse(out["degraded"])
        steps = [h["step"] for h in out["hops"]]
        self.assertEqual(steps, ["candidates", "picked_file", "content"])
        content = str(out["result"])
        self.assertIn("toolInterfacer", content)
        self.assertLess(out["wall_ms"], 5000)

    def test_conf_gate_blocks_then_ask_answer_resolves(self):
        # force the pick conf below the slot gate via an explicit fixture map
        from ti.conf import reload
        reload()
        import ti.decide.primitives as prim
        hashed = None  # default fixture path; we instead monkeypatch the backend
        calls: list = []

        class LowPick:
            name = "low_pick"

            def choice(self, state, instructions, criteria):
                calls.append(criteria)
                return prim.ChoiceAnswer("README.md", dict.fromkeys(criteria, 0.3), 0.3)

            def noul(self, state, question):
                return 0.9

        prim._REGISTRY["low_pick"] = LowPick
        from ti.conf import load
        backends = list(load()["decide"]["backends_in_use"])
        conf_backup = dict(load()["decide"]["backends_in_use"])
        load()["decide"]["backends_in_use"]["choice"] = "low_pick"
        try:
            answers = iter([{"choice": "README.md"}])

            def on_ask(q):
                self.assertEqual(q.kind, "choice" if q.options else "free")
                return next(answers)

            out = use_tool({"intent": "read the file we discussed",
                            "digest": {"cwd": str(ROOT)}}, on_ask=on_ask)
            self.assertEqual(out["status"], "done", out)
            self.assertIn("toolInterfacer", str(out["result"]))
            self.assertTrue(calls)
        finally:
            prim._REGISTRY.pop("low_pick", None)
            load()["decide"]["backends_in_use"] = conf_backup

    def test_runresult_schema(self):
        out = use_tool({"intent": "write a file", "params": {"tool": "filesys_read",
                                                             "args": {"path": "README.md"}}})
        self.assertEqual(out["status"], "done")
        self.assertEqual(out["hops"][0]["op"], "filesys_read")
        self.assertEqual(out["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
