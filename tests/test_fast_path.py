"""Fast-path tests — the regex pre-gate and the decision cache in front of `choice`.

On the CPU-only box the live model costs ~1.9 s per /v1/systemone call, so the two
cheap wins resolve.py documents are pinned here: a hop heuristic_micro can settle
costs no backend call at all, and a question already answered costs no backend
call either.

Both are driven through resolve(), so the precedence under test is the shipped one
— the primary backend is a counting fake registered in the primitive registry (the
pattern test_cascade's loop tests already use). The cache is cleared around every
case because it is process-global, and no case touches the network.
"""
import pathlib
import sys
import threading
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))

import ti.decide.primitives as prim  # noqa: E402
from ti.conf import load  # noqa: E402
from ti.decide import cache as cache_mod  # noqa: E402
from ti.decide.cascade import CascadeBackend  # noqa: E402
from ti.decide.primitives import ChoiceAnswer, DecideBackend, make_backend  # noqa: E402
from ti.decide.resolve import FastPathBackend, resolve  # noqa: E402

ROUTE = "route the hop"
# No URL, no reddit scheme, no path-like token: the gate has nothing to say here.
FILES = {"intent": "read the file we discussed"}
CRITERIA = {"README.md": "read me", "docs/architecture.md": "notes"}
URL_STATE = {"intent": "fetch https://example.com/spec"}
WEB_CRITERIA = {"web_fetch": "fetch one url", "web_search": "search by keywords"}


class _Counter(DecideBackend):
    """Answers one fixed ChoiceAnswer and counts how often it was asked."""

    name = "counter"

    def __init__(self, choice: str = "README.md", confidence: float = 0.95) -> None:
        self.pick = choice
        self.confidence = confidence
        self.calls = 0

    def choice(self, state, instructions, criteria):
        self.calls += 1
        probs = {k: (self.confidence if k == self.pick else 1.0 - self.confidence)
                 for k in criteria}
        return ChoiceAnswer(self.pick, probs, self.confidence)

    def noul(self, state, question):
        return 0.5


class FastPathCase(unittest.TestCase):
    """Points decide.backends_in_use.choice at a counting fake; restores after."""

    def setUp(self):
        self.decide = load()["decide"]
        self._before = {"choice": self.decide["backends_in_use"]["choice"],
                        "routing": self.decide.get("routing"),
                        "cache": dict(self.decide.get("cache") or {}),
                        "cascade": self.decide["cascade"]["enabled"]}
        self.decide["cascade"]["enabled"] = False
        self.decide["backends_in_use"]["choice"] = "counter"
        self.decide["cache"] = {"enabled": True, "max_entries": 256}
        self.counter = _Counter()
        prim._REGISTRY["counter"] = lambda: self.counter
        cache_mod.clear()
        self.addCleanup(self._restore)

    def _restore(self):
        self.decide["backends_in_use"]["choice"] = self._before["choice"]
        self.decide["routing"] = self._before["routing"]
        self.decide["cache"] = self._before["cache"]
        self.decide["cascade"]["enabled"] = self._before["cascade"]
        prim._REGISTRY.pop("counter", None)
        cache_mod.clear()


class PreGateTest(FastPathCase):
    """heuristic_micro in front: it commits the two hops it knows, and abstains."""

    def test_a_gate_hit_makes_no_backend_call(self):
        backend = resolve("choice")
        self.assertEqual(backend.gate.name, "heuristic_micro")
        answer = backend.choice(URL_STATE, ROUTE, dict(WEB_CRITERIA))
        self.assertEqual(answer.choice, "web_fetch")
        self.assertEqual(self.counter.calls, 0)
        self.assertEqual(backend.last_source, "pre_gate")

    def test_a_gate_abstain_falls_through_to_the_backend(self):
        backend = resolve("choice")
        answer = backend.choice(dict(FILES), ROUTE, dict(CRITERIA))
        self.assertEqual(self.counter.calls, 1)
        self.assertEqual(answer.choice, "README.md")
        self.assertEqual(backend.last_source, "primary")

    def test_a_gate_hit_never_touches_the_cache(self):
        cache = cache_mod.DecisionCache(8)
        facade = FastPathBackend(self.counter, gate=make_backend("heuristic_micro"),
                                 cache=cache)
        answer = facade.choice({"intent": "summarize r/python today"}, ROUTE,
                               {"reddit_digest": "summarize a subreddit or thread"})
        self.assertEqual(answer.choice, "reddit_digest")
        self.assertEqual(facade.last_source, "pre_gate")
        self.assertEqual(cache.stats(), {"entries": 0, "max_entries": 8, "hits": 0,
                                         "misses": 0, "evictions": 0, "hit_rate": 0.0})

    def test_the_shipped_routing_value_selects_the_gate(self):
        self.assertEqual(resolve("choice").gate.name, "heuristic_micro")

    def test_an_unrecognised_routing_value_leaves_the_gate_off(self):
        self.decide["routing"] = "direct"
        backend = resolve("choice")
        self.assertIsNone(backend.gate)
        backend.choice(URL_STATE, ROUTE, dict(WEB_CRITERIA))
        self.assertEqual(self.counter.calls, 1, "no gate means the model answers")

    def test_the_gate_also_sits_in_front_of_the_cascade(self):
        self.decide["cascade"]["enabled"] = True
        backend = resolve("choice")
        self.assertIsInstance(backend.primary, CascadeBackend)
        self.assertEqual(backend.name, "cascade")
        self.assertEqual(backend.gate.name, "heuristic_micro")


class CacheTest(FastPathCase):
    """A repeat question costs no backend call and answers identically."""

    def test_a_cache_hit_returns_an_equal_answer_with_no_backend_call(self):
        first = resolve("choice").choice(dict(FILES), ROUTE, dict(CRITERIA))
        backend = resolve("choice")  # a fresh facade: the table is process-wide
        second = backend.choice(dict(FILES), ROUTE, dict(CRITERIA))
        self.assertEqual(second, first)
        self.assertIsNot(second, first, "a hit hands back a copy, not the stored object")
        self.assertEqual(self.counter.calls, 1)
        self.assertEqual(backend.last_source, "cache")

    def test_a_different_state_misses(self):
        backend = resolve("choice")
        backend.choice(dict(FILES), ROUTE, dict(CRITERIA))
        backend.choice({"intent": "list the python files"}, ROUTE, dict(CRITERIA))
        self.assertEqual(self.counter.calls, 2)
        self.assertEqual(backend.last_source, "primary")

    def test_different_instructions_miss(self):
        backend = resolve("choice")
        backend.choice(dict(FILES), ROUTE, dict(CRITERIA))
        backend.choice(dict(FILES), "pick the file that explains the architecture",
                       dict(CRITERIA))
        self.assertEqual(self.counter.calls, 2)

    def test_a_disabled_cache_neither_stores_nor_serves(self):
        self.decide["cache"] = {"enabled": False, "max_entries": 256}
        backend = resolve("choice")
        self.assertIsNone(backend.cache)
        backend.choice(dict(FILES), ROUTE, dict(CRITERIA))
        again = backend.choice(dict(FILES), ROUTE, dict(CRITERIA))
        self.assertEqual(self.counter.calls, 2)
        self.assertEqual(again.choice, "README.md")
        self.assertEqual(cache_mod.shared().stats()["entries"], 0)

    def test_the_other_primitives_are_not_cached_or_gated(self):
        backend = resolve("choice")
        self.assertEqual(backend.noul({"kind": "file_content"}, "relevant?"), 0.5)
        self.assertTrue(backend.health())


class DecisionCacheUnitTest(unittest.TestCase):
    """The table itself: bounded, LRU, copy-safe, and clean under threads."""

    def _answer(self, choice: str) -> ChoiceAnswer:
        return ChoiceAnswer(choice, {choice: 1.0}, 0.9)

    def test_lru_evicts_the_least_recently_used_entry(self):
        cache = cache_mod.DecisionCache(3)
        for key in ("a", "b", "c"):
            cache.put(key, self._answer(key))
        cache.get("a")                       # "a" is now the most recent entry
        cache.put("d", self._answer("d"))    # so "b" is the one to go
        self.assertIsNone(cache.get("b"))
        self.assertEqual([cache.get(k).choice for k in ("a", "c", "d")], ["a", "c", "d"])
        self.assertEqual(cache.stats()["evictions"], 1)
        self.assertEqual(cache.stats()["entries"], 3)

    def test_entries_never_exceed_max_entries(self):
        cache = cache_mod.DecisionCache(4)
        for i in range(10):
            cache.put(f"k{i}", self._answer(f"k{i}"))
        self.assertEqual(cache.stats()["entries"], 4)
        self.assertEqual(cache.stats()["evictions"], 6)

    def test_get_returns_a_copy_so_a_caller_cannot_poison_the_entry(self):
        cache = cache_mod.DecisionCache(2)
        cache.put("k", self._answer("opt0"))
        cache.get("k").probabilities["opt0"] = 0.0
        self.assertEqual(cache.get("k").probabilities, {"opt0": 1.0})

    def test_the_key_ignores_dict_order_but_not_content(self):
        nested = {"intent": "x", "candidates": {"a": "1", "b": "2"}}
        reordered = {"candidates": {"b": "2", "a": "1"}, "intent": "x"}
        self.assertEqual(cache_mod.choice_key(nested, ROUTE, dict(CRITERIA)),
                         cache_mod.choice_key(reordered, ROUTE, dict(CRITERIA)))
        self.assertNotEqual(cache_mod.choice_key(nested, ROUTE, dict(CRITERIA)),
                            cache_mod.choice_key({"intent": "y"}, ROUTE, dict(CRITERIA)))
        self.assertNotEqual(cache_mod.choice_key(nested, ROUTE, dict(CRITERIA)),
                            cache_mod.choice_key(nested, ROUTE, {"README.md": "read me"}))

    def test_stats_report_hits_misses_and_clear_resets_them(self):
        cache = cache_mod.DecisionCache(2)
        cache.put("k", self._answer("opt0"))
        cache.get("k")
        cache.get("nope")
        stats = cache.stats()
        self.assertEqual((stats["hits"], stats["misses"]), (1, 1))
        self.assertEqual(stats["hit_rate"], 0.5)
        cache.clear()
        self.assertEqual(cache.stats(), {"entries": 0, "max_entries": 2, "hits": 0,
                                         "misses": 0, "evictions": 0, "hit_rate": 0.0})

    def test_concurrent_use_stays_inside_the_bound(self):
        cache = cache_mod.DecisionCache(16)
        errors: list[BaseException] = []

        def worker(n: int) -> None:
            try:
                for i in range(50):
                    cache.put(f"k{(n * 7 + i) % 40}", self._answer("opt0"))
                    cache.get(f"k{i % 40}")
                    cache.stats()
            except Exception as exc:  # noqa: BLE001 — surfaced by the assertion below
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(cache.stats()["entries"], 16)


if __name__ == "__main__":
    unittest.main()
