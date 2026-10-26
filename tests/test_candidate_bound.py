"""decide_pick must not hand the model an unbounded candidate list.

The decision model's prefill dominates latency on a CPU-only box (~19 ms/token
measured on the Lenovo: 4 candidates = ~2 s, 64 = ~31 s). `_rank` keeps the most
relevant candidates so `decide.max_candidates` can stay small without hiding the
right answer.
"""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))

from ti.core.artifacts import ArtifactStore  # noqa: E402
from ti.decide.primitives import ChoiceAnswer  # noqa: E402
from ti.ops import decisions  # noqa: E402


class RankTest(unittest.TestCase):
    def test_rank_keeps_the_relevant_file_first(self):
        items = [f"src/module_{i}.py" for i in range(50)] + ["service/ti/ops/scheduler.py"]
        ranked = decisions._rank(items, "open the scheduler implementation")
        self.assertIn("service/ti/ops/scheduler.py", ranked[:5])

    def test_rank_is_stable_without_intent_words(self):
        items = ["a.py", "b.py", "c.py"]
        self.assertEqual(decisions._rank(items, "   "), items)
        self.assertEqual(decisions._rank(items, None), items)

    def test_rank_does_not_drop_anything(self):
        items = [f"f{i}.py" for i in range(20)]
        self.assertEqual(sorted(decisions._rank(items, "f7")), sorted(items))


class _Recording:
    name = "recording"

    def __init__(self) -> None:
        self.seen: dict[str, object] = {}

    def choice(self, state, instructions, criteria):
        self.seen["n"] = len(criteria)
        first = next(iter(criteria))
        return ChoiceAnswer(first, {k: 1.0 / len(criteria) for k in criteria}, 0.9)


class PickBoundTest(unittest.TestCase):
    def _run(self, items, intent):
        store = ArtifactStore()
        store.put("cand", "filesys_glob", "path_list", items, 1.0)
        rec = _Recording()
        original = decisions.resolve_backend
        decisions.resolve_backend = lambda kind: rec
        try:
            decisions.run_pick({"source_artifact": "cand", "instructions": "pick one",
                                "intent_context": intent, "top_k": 1}, store)
        finally:
            decisions.resolve_backend = original
        return rec.seen["n"]

    def test_candidate_list_is_capped(self):
        n = self._run([f"a/file_{i}.py" for i in range(200)], "file_123")
        self.assertLessEqual(n, decisions._max_candidates())
        self.assertGreaterEqual(n, 1)

    def test_cap_is_small_enough_for_a_cpu_box(self):
        # 6 candidates measured at ~2-3 s of prefill; 16 was ~8 s, 64 was ~31 s.
        self.assertLessEqual(decisions._max_candidates(), 8)

    def test_small_lists_are_untouched(self):
        self.assertEqual(self._run(["a.py", "b.py"], "a"), 2)


if __name__ == "__main__":
    unittest.main()
