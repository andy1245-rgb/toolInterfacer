"""decide_pick must not hand the model an unbounded candidate list.

The decision model's prefill dominates latency on a CPU-only box (~19 ms/token
measured on the Lenovo: 4 candidates = ~2 s, 64 = ~31 s). `_rank` keeps the most
relevant candidates so `decide.max_candidates` can stay small without hiding the
right answer.

There are two bounds, and neither alone is enough: `max_candidates` bounds *how
many* candidates reach the model, `max_criteria_chars` bounds how much text each
one carries. Until the second existed, `guard_state` bounded the state and the
option set travelled unbounded beside it -- docs/review-2026-11-03.md B5 measured
12,000,144 characters (~3,000,036 tokens at 4 chars/token) reaching a backend
configured for 2000 tokens.
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


class _CriteriaRecording(_Recording):
    """Keeps the criteria dict itself, not just its length."""

    def choice(self, state, instructions, criteria):
        self.seen["criteria"] = criteria
        return super().choice(state, instructions, criteria)


class CriteriaBoundTest(unittest.TestCase):
    def _criteria(self, candidates):
        store = ArtifactStore()
        store.put("cand", "web_search", "candidate_list", candidates, 1.0)
        rec = _CriteriaRecording()
        original = decisions.resolve_backend
        decisions.resolve_backend = lambda kind: rec
        try:
            decisions.run_pick({"source_artifact": "cand", "instructions": "pick one",
                                "top_k": 1}, store)
        finally:
            decisions.resolve_backend = original
        return rec.seen["criteria"]

    def test_a_huge_snippet_is_clipped(self):
        cap = decisions._max_desc_chars()
        criteria = self._criteria(
            [{"title": f"f{i}.py", "snippet": "x" * 2_000_000} for i in range(6)])
        self.assertTrue(criteria)
        for desc in criteria.values():
            self.assertLessEqual(len(desc), cap)

    def test_the_measured_12MB_case_is_bounded(self):
        criteria = self._criteria(
            [{"title": "t", "snippet": "y" * 2_000_000} for _ in range(6)])
        total = sum(len(v) for v in criteria.values())
        self.assertLess(total, 10_000, f"criteria carried {total} chars")

    def test_a_normal_description_is_untouched(self):
        criteria = self._criteria([{"title": "a.py", "snippet": "the config loader"}])
        self.assertEqual(criteria, {"a.py": "a.py :: the config loader"})

    def test_ids_are_not_clipped_because_the_pick_maps_back_through_them(self):
        # An id is the label the model answers with and run_pick maps that answer
        # back to the source item; truncating a key would silently drop the pick.
        long_id = "d" * (decisions._max_desc_chars() + 50)
        criteria = self._criteria([{"id": long_id, "title": "t"}])
        self.assertIn(long_id, criteria)

    def test_the_criteria_cap_is_a_latency_knob_not_a_wall(self):
        # 200 chars x 6 candidates ~= 300 tokens, inside the 2000-token state budget.
        self.assertLessEqual(decisions._max_desc_chars() * decisions._max_candidates(),
                             2000)

    def test_a_broken_conf_key_falls_back_to_the_default(self):
        from ti import conf
        original = conf.load
        try:
            conf.load = lambda: {"decide": {"max_criteria_chars": "nonsense"}}
            self.assertEqual(decisions._max_desc_chars(),
                             decisions.DEFAULT_MAX_DESC_CHARS)
            conf.load = lambda: {"decide": {}}
            self.assertEqual(decisions._max_desc_chars(),
                             decisions.DEFAULT_MAX_DESC_CHARS)
        finally:
            conf.load = original


if __name__ == "__main__":
    unittest.main()
