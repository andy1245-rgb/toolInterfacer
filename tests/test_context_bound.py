"""Bounded context — the decide state cannot outgrow conf.context.state_max_tokens.

Two lines make the guarantee: the pi extension sends <= 8 turns x 240 chars, and
the service sanitizes at ingress plus guard_state's the decide state to
state_max_tokens. The first half of this file is the regression guard for the
second half being a no-op: the prompt the decision model's validated numbers were
measured on has to reach the backend byte-identical for a normal-sized state.
The rest drives the trim cases (10k candidates, a 5MB digest) and the conf knob.
"""
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))

import ti.decide.primitives as prim  # noqa: E402
from ti.conf import load  # noqa: E402
from ti.context import packer  # noqa: E402
from ti.context.session_digest import TURN_CAP, TURN_CHARS, sanitize  # noqa: E402
from ti.core.artifacts import ArtifactStore  # noqa: E402
from ti.decide.cache import clear as clear_decisions  # noqa: E402
from ti.ops import decisions  # noqa: E402
from ti.service_entry import use_tool  # noqa: E402
import ti.service_entry as service_entry  # noqa: E402

NORMAL_STATE = {
    "intent_context": "read the file we discussed",
    "candidates": {"README.md": "toolInterfacer — decision service :: read me",
                   "docs/architecture.md": "Architecture notes :: the two plugin walls"},
    "source_kind": "path_list",
}


def budget_chars() -> int:
    """conf-driven char cap the guard works against (~4 chars/token)."""
    return int(load()["context"]["state_max_tokens"]) * 4


class OfflineDecide(unittest.TestCase):
    """Auto-mode cases run the loop, whose shipped choice backend is the live d1
    model: pin the fixture backend and start from an empty decision cache."""

    def setUp(self):
        decide = load()["decide"]
        self._choice = decide["backends_in_use"]["choice"]
        decide["backends_in_use"]["choice"] = "fixture"
        clear_decisions()
        self.addCleanup(self._restore)

    def _restore(self):
        load()["decide"]["backends_in_use"]["choice"] = self._choice
        clear_decisions()


class ContextBoundTest(OfflineDecide):
    # (a) the regression guard: normal inputs must be untouched
    def test_normal_state_passes_through_identically(self):
        state = json.loads(json.dumps(NORMAL_STATE))
        before = json.dumps(state, sort_keys=True)
        guarded = packer.guard_state(state)
        self.assertIs(guarded, state)  # same object, not a re-shaped copy
        self.assertEqual(json.dumps(guarded, sort_keys=True), before)

    def test_normal_state_at_the_boundary_passes_through_identically(self):
        filler = "x" * (budget_chars() - len(str(NORMAL_STATE)) - 20)
        state = {**json.loads(json.dumps(NORMAL_STATE)), "intent_context": filler}
        self.assertLessEqual(len(str(state)), budget_chars())
        before = json.dumps(state, sort_keys=True)
        self.assertEqual(json.dumps(packer.guard_state(state), sort_keys=True), before)

    # (b) candidate lists: trimmed under budget, every key and type kept
    def test_ten_thousand_candidates_are_trimmed_to_budget(self):
        state = {"intent_context": "pick the right file",
                 "candidates": {f"cand_{i:05d}": f"candidate {i} :: " + "d" * 400
                                for i in range(10_000)},
                 "source_kind": "path_list"}
        before = len(str(state))
        out = packer.guard_state(state)
        after = len(str(out))
        self.assertLessEqual(after, budget_chars(), (before, after))
        self.assertLess(after, before)
        self.assertEqual(set(out), set(state))                  # no key lost
        self.assertIsInstance(out["candidates"], dict)          # key keeps its type
        self.assertEqual(out["source_kind"], "path_list")
        self.assertEqual(list(out["candidates"])[:1], ["cand_00000"])  # first N kept
        self.assertLessEqual(len(out["candidates"]), packer.CANDIDATE_CAP)
        self.assertLessEqual(max(len(v) for v in out["candidates"].values()),
                             packer.DESC_CHARS + 3)

    def test_candidate_list_shape_is_kept(self):
        state = {"intent_context": "pick",
                 "candidates": [{"title": f"f{i}", "snippet": "s" * 500}
                                for i in range(200)],
                 "source_kind": "path_list"}
        out = packer.guard_state(state)
        self.assertLessEqual(len(str(out)), budget_chars())
        self.assertIsInstance(out["candidates"], list)
        for entry in out["candidates"]:
            self.assertIsInstance(entry, dict)
            self.assertEqual(set(entry), {"title", "snippet"})

    # (c) a single 5MB digest value cannot ride through
    def test_five_megabyte_digest_is_bounded(self):
        state = {"intent_context": "continue where we left off",
                 "digest": {"turns": [{"role": "user", "summary": "x" * 5_000_000}],
                            "antecedents": []},
                 "candidates": {"a.py": "a :: file"},
                 "source_kind": "path_list"}
        before = len(str(state))
        out = packer.guard_state(state)
        after = len(str(out))
        self.assertGreater(before, 5_000_000)
        self.assertLessEqual(after, budget_chars(), (before, after))
        self.assertEqual(set(out), set(state))
        self.assertEqual(out["digest"], packer.STUB_DIGEST)

    # (d) ingress bounds turns/antecedents whatever the caller sent
    def test_sanitize_caps_turns_and_antecedents_at_eight(self):
        digest = {"turns": [{"role": "user", "summary": f"t{i}"} for i in range(50)],
                  "antecedents": [{"phrase": f"p{i}"} for i in range(50)],
                  "cwd": "/tmp", "notes": "x" * 100_000}
        out = sanitize(digest)
        self.assertEqual(len(out["turns"]), TURN_CAP)
        self.assertEqual(len(out["antecedents"]), TURN_CAP)
        # recent-weighted: the last 8 survive, not the first 8
        self.assertEqual([t["summary"] for t in out["turns"]],
                         [f"t{i}" for i in range(42, 50)])
        self.assertEqual([a["phrase"] for a in out["antecedents"]],
                         [f"p{i}" for i in range(42, 50)])
        self.assertEqual(out["cwd"], "/tmp")       # the L1 bridge resolves paths from it
        self.assertNotIn("notes", out)             # unbudgeted keys are dropped

    def test_sanitize_is_identity_for_an_extension_shaped_digest(self):
        digest = {"turns": [{"role": "user",
                             "summary": "read docs/architecture.md",
                             "mentions": ["docs/architecture.md"]},
                            {"role": "assistant", "summary": "s" * TURN_CHARS}],
                  "antecedents": [{"phrase": "docs/architecture.md",
                                   "resolved": str(ROOT / "docs" / "architecture.md")}],
                  "cwd": str(ROOT)}
        self.assertEqual(sanitize(digest), digest)

    def test_sanitize_bounds_a_huge_summary(self):
        out = sanitize({"turns": [{"role": "user", "summary": "x" * 5_000_000}],
                        "antecedents": []})
        self.assertEqual(len(out["turns"][0]["summary"]), TURN_CHARS)
        self.assertLessEqual(len(str(out)), TURN_CAP * TURN_CHARS)

    def test_sanitize_rejects_a_non_dict(self):
        self.assertEqual(sanitize("a pasted transcript"), {"turns": [], "antecedents": []})
        self.assertEqual(sanitize(None), {"turns": [], "antecedents": []})

    # (e) the cap is conf-driven, not a constant in the code
    def test_cap_follows_conf_state_max_tokens(self):
        state = {"intent_context": "y" * 3000,
                 "candidates": {"a.py": "a :: file"},
                 "source_kind": "path_list"}
        size = len(str(state))
        self.assertIs(packer.guard_state(state), state)  # default 2000 tok = 8000 chars
        conf = load()["context"]
        backup = conf["state_max_tokens"]
        try:
            conf["state_max_tokens"] = size // 4 - 1     # just under the payload
            out = packer.guard_state(state)
            self.assertLessEqual(len(str(out)), conf["state_max_tokens"] * 4)
            self.assertEqual(set(out), set(state))
            conf["state_max_tokens"] = size // 4 + 1     # just over it
            self.assertIs(packer.guard_state(state), state)
        finally:
            conf["state_max_tokens"] = backup

    def test_explicit_max_tokens_overrides_conf(self):
        state = {"intent_context": "z" * 500, "candidates": {}, "source_kind": "url_list"}
        self.assertIs(packer.guard_state(state, max_tokens=1000), state)
        trimmed = packer.guard_state(state, max_tokens=100)
        self.assertLess(len(str(trimmed)), len(str(state)))
        self.assertLessEqual(len(str(trimmed)), 400)
        self.assertEqual(set(trimmed), set(state))


class _Recorder:
    """Choice backend that records exactly what state reached it."""

    name = "recorder"

    def __init__(self) -> None:
        self.states = []

    def choice(self, state, instructions, criteria):
        self.states.append(state)
        return prim.ChoiceAnswer(next(iter(criteria)),
                                 {k: 1.0 / len(criteria) for k in criteria}, 0.9)

    def noul(self, state, question):
        return 0.9


class DecideOpBoundTest(unittest.TestCase):
    """The guard sits where the state is handed to the backend, so a fake backend
    stands in for every one of them (cascade forwards the state it is given)."""

    def setUp(self):
        self._conf_backup = load()["decide"]["backends_in_use"]["choice"]

    def tearDown(self):
        load()["decide"]["backends_in_use"]["choice"] = self._conf_backup
        prim._REGISTRY.pop("recorder", None)

    def test_decide_pick_hands_the_backend_a_bounded_state(self):
        store = ArtifactStore()
        store.put("candidates", "candidates", "path_list",
                  [{"title": f"file {i}", "snippet": "s" * 400} for i in range(500)])
        rec = _Recorder()
        prim._REGISTRY["recorder"] = lambda: rec
        load()["decide"]["backends_in_use"]["choice"] = "recorder"
        kind, picks, conf = decisions.run_pick(
            {"source_artifact": "candidates", "instructions": "which file?",
             "top_k": 1}, store)
        self.assertEqual(kind, "picked_items")
        self.assertTrue(picks)
        self.assertEqual(len(rec.states), 1)
        state = rec.states[0]
        self.assertLessEqual(len(str(state)), budget_chars())
        self.assertEqual(set(state), {"intent_context", "candidates", "source_kind"})
        self.assertLessEqual(len(state["candidates"]), packer.CANDIDATE_CAP)


class IngressBoundTest(OfflineDecide):
    def test_ingress_sanitizes_the_body_digest(self):
        seen = []
        real = service_entry.sanitize

        def spy(digest):
            out = real(digest)
            seen.append((digest, out))
            return out

        service_entry.sanitize = spy
        try:
            out = use_tool({"intent": "read the file we discussed", "mode": "plan",
                            "digest": {"cwd": "/tmp",
                                       "turns": [{"role": "user", "summary": "q" * 5000}
                                                 for _ in range(40)],
                                       "antecedents": [{"phrase": f"p{i}"}
                                                       for i in range(40)]}})
        finally:
            service_entry.sanitize = real
        self.assertEqual(out["status"], "plan", out)
        self.assertEqual(len(seen), 1)
        raw, bounded = seen[0]
        self.assertEqual(len(raw["turns"]), 40)             # what the caller sent
        self.assertEqual(len(bounded["turns"]), TURN_CAP)   # what the service used
        self.assertEqual(len(bounded["antecedents"]), TURN_CAP)
        self.assertLessEqual(len(bounded["turns"][0]["summary"]), TURN_CHARS)
        self.assertEqual(bounded["cwd"], "/tmp")
        self.assertEqual(out["steps"][0]["params"]["cwd"], "/tmp")  # L1 bridge still resolves

    def test_oversized_digest_runs_to_done(self):
        out = use_tool({"intent": "read the file we discussed",
                        "digest": {"cwd": str(ROOT),
                                   "turns": [{"role": "user", "summary": "q" * 100_000}
                                             for _ in range(40)]}})
        self.assertEqual(out["status"], "done", out)
        self.assertIn("toolInterfacer", str(out["result"]))


if __name__ == "__main__":
    unittest.main()
