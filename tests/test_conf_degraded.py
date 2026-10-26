"""Degraded-run confidence: a gate block must not erase the confidence reached.

The record these cases pin is a live one (intent "what files are in the bench
directory", served over HTTP where use_tool is called without on_ask):

    filesys_glob 1.0 -> decide_pick 0.548 -> askback 0.0 -> filesys_read 1.0

which reported confidence 0.0. The `askback` hop is written before the ask is
attempted, so its 0.0 is a placeholder, and _route_conf's min picked it up on
every gate block. On the HTTP path that is every gate block, not an occasional
one: with no on_ask the ask raises immediately, so the record lost the decide
signal exactly on the cases worth training on.

The map is pinned at a fixture choice backend (the shipped one is the live d1
model) and the decision cache is process-global, so each case starts from an
empty table.
"""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))

from ti.conf import load  # noqa: E402
from ti.decide.cache import clear as clear_decisions  # noqa: E402
from ti.decide.primitives import ChoiceAnswer  # noqa: E402
from ti.policy.askback import MaxAsksReached  # noqa: E402
from ti.service_entry import use_tool  # noqa: E402

# decide_pick conf in the reproduced record; below conf.policy.gate_slot (0.7),
# so the read step's pre-gate blocks.
PICK_CONF = 0.548
ASKBACK_CONF = 0.9          # what a resolved choice answer re-writes the artifact with


class _LowPick:
    """A choice backend that picks README.md and is sure of it only to `conf`."""

    name = "conf_degraded_pick"

    def __init__(self, conf: float = PICK_CONF) -> None:
        self._conf = conf

    def choice(self, state, instructions, criteria):
        probs = dict.fromkeys(criteria, self._conf) or {"README.md": self._conf}
        return ChoiceAnswer("README.md", probs, self._conf)

    def noul(self, state, question):
        return 0.9


class ConfDegradedTest(unittest.TestCase):
    def setUp(self):
        import ti.decide.primitives as prim
        self._prim = prim
        self._backends = load()["decide"]["backends_in_use"]
        self._previous = self._backends["choice"]
        prim._REGISTRY[_LowPick.name] = _LowPick
        self._backends["choice"] = _LowPick.name
        clear_decisions()
        self.addCleanup(self._restore)

    def _restore(self):
        self._prim._REGISTRY.pop(_LowPick.name, None)
        self._backends["choice"] = self._previous
        clear_decisions()

    def _run(self, on_ask=None):
        return use_tool({"intent": "read the file we discussed",
                         "digest": {"cwd": str(ROOT)}}, on_ask=on_ask)

    def _hops(self, out):
        return [(h["op"], h["conf"]) for h in out["hops"]]

    def test_gate_block_without_ask_back_reports_the_decide_confidence(self):
        out = self._run()
        self.assertEqual(out["status"], "done", out)
        self.assertTrue(out["degraded"])
        # The live hop list, verbatim: the placeholder is still logged, it just
        # stops standing in for the run's confidence.
        self.assertEqual(self._hops(out),
                         [("filesys_glob", 1.0), ("decide_pick", PICK_CONF),
                          ("askback", 0.0), ("filesys_read", 1.0)])
        self.assertEqual(out["confidence"], PICK_CONF)
        ask_hop = [h for h in out["hops"] if h["op"] == "askback"][0]
        self.assertFalse(ask_hop["ask_available"], "no on_ask was passed, so nothing was asked")

    def test_resolved_ask_back_sets_the_reported_confidence(self):
        asked: list = []

        def on_ask(q):
            asked.append(q)
            return {"choice": "README.md"}

        out = self._run(on_ask=on_ask)
        self.assertEqual(out["status"], "done", out)
        self.assertFalse(out["degraded"], out)
        self.assertEqual([op for op, _conf in self._hops(out)],
                         ["filesys_glob", "decide_pick", "askback",
                          "askback.resolved", "filesys_read"])
        self.assertEqual(len(asked), 1)
        self.assertEqual(asked[0].kind, "choice")
        resolved = [h for h in out["hops"] if h["op"] == "askback.resolved"][0]
        self.assertEqual(resolved["conf"], ASKBACK_CONF)
        self.assertEqual(resolved["resolves"], "picked_file")
        # The answered pick superseded the node the gate rejected, so the min
        # over the signals that produced the result is the ask-back's.
        self.assertEqual(out["confidence"], ASKBACK_CONF)

    def test_ask_back_solicited_but_unanswered_reports_the_decide_confidence(self):
        def on_ask(q):
            raise MaxAsksReached()

        out = self._run(on_ask=on_ask)
        self.assertTrue(out["degraded"], out)
        self.assertEqual([op for op, _conf in self._hops(out)],
                         ["filesys_glob", "decide_pick", "askback", "filesys_read"])
        # No answer came back, so the ask-back measured nothing: the run reports
        # the real decide signal and says the gate went unresolved in assumptions.
        self.assertEqual(out["confidence"], PICK_CONF)
        self.assertIn("gate unresolved", " ".join(out["assumptions"]))
        ask_hop = [h for h in out["hops"] if h["op"] == "askback"][0]
        self.assertFalse(ask_hop["ask_available"])

    def test_a_low_pick_is_reported_low(self):
        # Dropping the placeholder excludes non-signal; it does not raise a floor.
        self._prim._REGISTRY[_LowPick.name] = lambda: _LowPick(0.2)
        out = self._run()
        self.assertTrue(out["degraded"], out)
        self.assertEqual(out["confidence"], 0.2)


if __name__ == "__main__":
    unittest.main()
