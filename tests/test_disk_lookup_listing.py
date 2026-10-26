"""disk_lookup has two plan shapes: a listing ends on the candidate list, a
singular read keeps pick → read.

The listing shape exists because the template used to end every plan in a read of
one file: decide_pick was asked which single file best matches a request to
enumerate files, a question with no answer — live it honestly returned conf 0.548,
gate_slot 0.70 blocked the read, and the run degraded to a best guess. No
confidence can fix that; the plan shape was wrong.

These cases pin the split, the exactness of the listing, the singular fallback,
and that shape detection costs no decision call.

The decide backend is a counting fake pinned in the registry (the pattern
test_loop and test_fast_path already use), so no case can reach the network and
the call counts themselves are the evidence.
"""
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))

import ti.decide.primitives as prim  # noqa: E402
from ti.conf import load  # noqa: E402
from ti.decide.cache import clear as clear_decisions  # noqa: E402
from ti.decide.primitives import ChoiceAnswer, DecideBackend  # noqa: E402
from ti.planner import shape  # noqa: E402
from ti.service_entry import use_tool  # noqa: E402

READ_STEPS = ["candidates", "picked_file", "content"]
READ_OPS = ["filesys_glob", "decide_pick", "filesys_read"]


class _Counter(DecideBackend):
    """Picks README.md (readable from ROOT) and counts how often it was asked."""

    name = "shape_counter"

    def __init__(self) -> None:
        self.calls = 0

    def choice(self, state, instructions, criteria):
        self.calls += 1
        pick = "README.md" if "README.md" in criteria else next(iter(criteria))
        rest = max(len(criteria) - 1, 1)
        probs = {k: (0.95 if k == pick else 0.05 / rest) for k in criteria}
        return ChoiceAnswer(pick, probs, 0.95)

    def noul(self, state, question):
        return 0.9


class ShapeCase(unittest.TestCase):
    """Points decide.backends_in_use.choice at the counter; restores after."""

    def setUp(self):
        self.decide = load()["decide"]
        self._before = self.decide["backends_in_use"]["choice"]
        self.counter = _Counter()
        prim._REGISTRY["shape_counter"] = lambda: self.counter
        self.decide["backends_in_use"]["choice"] = "shape_counter"
        clear_decisions()
        self.addCleanup(self._restore)

    def _restore(self):
        self.decide["backends_in_use"]["choice"] = self._before
        prim._REGISTRY.pop("shape_counter", None)
        clear_decisions()


class ListingShapeTest(ShapeCase):
    def test_a_listing_terminates_on_the_candidate_list(self):
        out = use_tool({"intent": "what files are in the bench directory",
                        "digest": {"cwd": str(ROOT)}})
        self.assertEqual(out["status"], "done", out)
        self.assertFalse(out["degraded"], out)
        self.assertEqual([h["step"] for h in out["hops"]], ["candidates"])
        self.assertNotIn("filesys_read", [h["op"] for h in out["hops"]])
        self.assertIsInstance(out["result"], list)
        self.assertIn("service/ti/service_entry.py", out["result"])
        # strictly fewer hops than the read path, and no decision call at all
        self.assertLess(len(out["hops"]), len(READ_OPS))
        self.assertEqual(self.counter.calls, 0)
        self.assertEqual(out["confidence"], 1.0)

    def test_a_listing_returns_exactly_the_directory_entries(self):
        entries = ("alpha.txt", "beta.txt", "notes.md")
        with tempfile.TemporaryDirectory() as tmp:
            for name in entries:
                pathlib.Path(tmp, name).write_text("x", encoding="utf-8")
            out = use_tool({"intent": "list the files", "digest": {"cwd": tmp}})
        self.assertEqual(out["status"], "done", out)
        self.assertEqual(sorted(out["result"]), sorted(entries))
        self.assertEqual([h["op"] for h in out["hops"]], ["filesys_glob"])
        self.assertEqual(self.counter.calls, 0)

    def test_every_listing_phrasing_reaches_the_listing_shape(self):
        for intent in ("what files are in the bench directory", "list the files", "ls",
                       "what's in this directory", "show me the files in the bench dir"):
            with self.subTest(intent=intent):
                self.assertEqual(shape.classify(intent), shape.ENUMERATIVE)
                out = use_tool({"intent": intent, "mode": "plan",
                                "digest": {"cwd": str(ROOT)}})
                self.assertEqual(out["final_key"], "candidates", out)
                self.assertEqual([s["op"] for s in out["steps"]], ["filesys_glob"])
        self.assertEqual(self.counter.calls, 0)


class SingularShapeTest(ShapeCase):
    def test_a_read_intent_keeps_the_three_step_path(self):
        out = use_tool({"intent": "read the file we discussed",
                        "digest": {"cwd": str(ROOT)}})
        self.assertEqual(out["status"], "done", out)
        self.assertEqual([h["step"] for h in out["hops"]], READ_STEPS)
        self.assertEqual([h["op"] for h in out["hops"]], READ_OPS)
        self.assertIn("toolInterfacer", str(out["result"]))
        self.assertEqual(self.counter.calls, 1)

    def test_read_and_show_phrasings_stay_singular(self):
        for intent in ("read README.md", "open README.md", "show me README.md",
                       "show me the file we discussed", "the file we discussed"):
            with self.subTest(intent=intent):
                self.assertEqual(shape.classify(intent), shape.SINGULAR)


class AmbiguousIntentTest(ShapeCase):
    def test_unrecognised_intents_default_to_singular(self):
        for intent in ("the thing we discussed", "handle that", "open it",
                       "show me that file", "what's in README.md", ""):
            with self.subTest(intent=intent):
                self.assertEqual(shape.classify(intent), shape.SINGULAR)

    def test_an_ambiguous_intent_builds_the_read_plan(self):
        out = use_tool({"intent": "the thing we discussed", "mode": "plan",
                        "digest": {"cwd": str(ROOT)}})
        self.assertEqual(out["status"], "plan", out)
        self.assertEqual(out["final_key"], "content")
        self.assertEqual([s["op"] for s in out["steps"]], READ_OPS)


class NoDecisionCallTest(ShapeCase):
    def test_shape_detection_makes_no_decision_call(self):
        listing = use_tool({"intent": "ls", "digest": {"cwd": str(ROOT)}})
        self.assertEqual([h["op"] for h in listing["hops"]], ["filesys_glob"])
        self.assertEqual(self.counter.calls, 0)
        # control: the counter is wired into the loop the read path uses
        use_tool({"intent": "read the file we discussed", "digest": {"cwd": str(ROOT)}})
        self.assertEqual(self.counter.calls, 1)


if __name__ == "__main__":
    unittest.main()
