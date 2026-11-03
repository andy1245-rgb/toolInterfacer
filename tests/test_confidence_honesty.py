"""Confidence honesty: report a number when something measured one, else nothing.

Three fabrications are pinned shut here (review 2026-11-03 B2/B3, audit finding
14):

  * `_route_conf` returned 1.0 when no hop carried a measurement, so an error run
    with no hops — and every L1 listing — reported maximum certainty;
  * `_passthrough` hardcoded 1.0 for L0, a decision the service never made;
  * the decision server's K<2 -> p_max 1.0 sentinel was imported into the pair and
    cascade gates, where it turned "the stage expressed no distribution" into
    "maximum certainty", so tau could never fire and the cascade handed the fitted
    commit precision to a 0.02-confidence answer.

The rule these assertions encode, in one sentence: a decision confidence is
reported when — and only when — a hop measured one, and the field is absent
otherwise (`confidence` is Optional in the contract), because "no measurement" and
"maximum certainty" must not be the same number.
"""
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))

from ti.conf import load  # noqa: E402
from ti.core.artifacts import ArtifactStore  # noqa: E402
from ti.core.gates import Gate, GateBlocked, check_pre  # noqa: E402
from ti.core.runner import Step  # noqa: E402
from ti.decide import conf_scale  # noqa: E402
from ti.decide.cache import clear as clear_decisions  # noqa: E402
from ti.decide.cascade import DEFAULT_PATH_CONF, CascadeBackend  # noqa: E402
from ti.decide.pair import STAGE1_ERROR, PairBackend  # noqa: E402
from ti.decide.primitives import ChoiceAnswer  # noqa: E402
from ti.ops.op_api import get as get_op, import_all  # noqa: E402
from ti.service_entry import _route_conf, _run_steps, use_tool  # noqa: E402

CRITERIA = {"opt0": "the first candidate", "opt1": "the second candidate"}
STATE = {"intent_context": "pick one"}
INSTRUCTION = "which candidate fits the request?"
TAU = 0.90


class _Stage:
    """A stub stage: counts calls, returns `answer`, or raises `error`."""

    def __init__(self, name, *, answer=None, error=None):
        self.name = name
        self.answer = answer
        self.error = error
        self.calls = 0

    def choice(self, state, instructions, criteria):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.answer

    def noul(self, state, question):
        return 0.5

    def health(self):
        return True


def _answer(pick="opt0", pmax=0.95, k=8):
    rest = (1.0 - pmax) / (k - 1)
    probs = {f"opt{i}": rest for i in range(k)}
    probs[pick] = pmax
    return ChoiceAnswer(pick, probs, pmax)


def _pair(stage1, stage2):
    return PairBackend([stage1, stage2], cfg={"tau": TAU, "escalate_on_error": True})


def _cascade(stages, **cfg):
    return CascadeBackend(stages, cfg={"enabled": True, "tau_hi": 0.7, "tau_lo": 0.4,
                                       "tau2": 0.5, **cfg})


class NoMeasurementTestCase(unittest.TestCase):
    """The fixture backend is the only one these cases may reach."""

    def setUp(self):
        self._decide = load()["decide"]
        self._choice = self._decide["backends_in_use"]["choice"]
        self._decide["backends_in_use"]["choice"] = "fixture"
        import_all()
        clear_decisions()
        self.addCleanup(self._restore)

    def _restore(self):
        self._decide["backends_in_use"]["choice"] = self._choice
        clear_decisions()


class MissingConfidenceTest(NoMeasurementTestCase):
    """(a)(b) An error run, an empty-hop run and L0 carry no confidence field."""

    def test_an_error_run_reports_no_confidence(self):
        # The run did measure a pick before the read failed. The measurement is on
        # the hop, where it is attributed to the step that made it; the envelope
        # describes a result that does not exist, so it claims nothing.
        with tempfile.TemporaryDirectory() as tmp:
            pathlib.Path(tmp, "present_but_not_picked.txt").write_text("x", encoding="utf-8")
            out = use_tool({"intent": "read the file we discussed", "digest": {"cwd": tmp}})
        self.assertEqual(out["status"], "error", out)
        self.assertTrue(out["hops"])
        self.assertNotIn("confidence", out)

    def test_an_empty_hop_error_reports_no_confidence(self):
        # The exact shape the review quoted: {"status": "error", "hops": []}.
        # Before the fix this was {"status": "error", "confidence": 1.0, "hops": []}.
        out = _run_steps([Step(id="first", op="no_such_op_registered")], {}, "first",
                         "anything", {}, on_ask=None, log_hops=False)
        self.assertEqual(out["status"], "error", out)
        self.assertEqual(out["hops"], [])
        self.assertNotIn("confidence", out)

    def test_an_l0_error_reports_no_confidence(self):
        out = use_tool({"intent": "read it", "params": {
            "tool": "filesys_read", "args": {"path": "no-such-file-xyz"}}})
        self.assertEqual(out["status"], "error", out)
        self.assertEqual(out["hops"], [])
        self.assertNotIn("confidence", out)

    def test_l0_passthrough_reports_no_route_confidence(self):
        # L0 is the caller's op and the caller's args: the service made no routing
        # decision, so there is no route confidence to state. The op's own conf
        # still travels on the hop with the op's name next to it.
        out = use_tool({"intent": "write a file", "params": {
            "tool": "filesys_read", "args": {"path": "README.md"}}})
        self.assertEqual(out["status"], "done", out)
        self.assertNotIn("confidence", out)
        op_conf = get_op("filesys_read").run({"path": str(ROOT / "README.md")}, None)[2]
        self.assertEqual(out["hops"][0]["conf"], op_conf)


class MeasuredConfidenceTest(NoMeasurementTestCase):
    """The other half of the rule: a measured run still reports its measurement."""

    def test_a_measured_run_reports_the_measurement(self):
        out = use_tool({"intent": "read the file we discussed",
                        "digest": {"cwd": str(ROOT)}})
        self.assertEqual(out["status"], "done", out)
        decided = [h for h in out["hops"] if h["op"].startswith("decide")]
        self.assertTrue(decided, out["hops"])
        self.assertIn("confidence", out)
        self.assertEqual(out["confidence"], min(h["conf"] for h in decided))

    def test_an_unmeasured_hop_contributes_nothing(self):
        self.assertIsNone(_route_conf([]))
        self.assertIsNone(_route_conf([{"step": "queries", "op": "decide_extract",
                                        "conf": None}]))
        self.assertEqual(_route_conf([{"step": "queries", "op": "decide_extract", "conf": None},
                                      {"step": "picked", "op": "decide_pick", "conf": 0.42}]),
                         0.42)

    def test_non_decide_hops_carry_no_run_confidence(self):
        # A coverage score from an op is not a decision measurement.
        self.assertIsNone(_route_conf([{"step": "candidates", "op": "filesys_glob",
                                        "conf": 1.0}]))


class StageDistributionTest(unittest.TestCase):
    """(c) A one-entry vector expresses no distribution, so it cannot clear tau."""

    def test_a_one_entry_stage1_does_not_clear_tau(self):
        stage1 = _Stage("stage1", answer=ChoiceAnswer("only", {"only": 0.02}, 0.02))
        stage2 = _Stage("stage2", answer=_answer("opt1"))
        pair = _pair(stage1, stage2)
        got = pair.choice(STATE, INSTRUCTION, {"only": "the only option"})
        self.assertEqual(got.choice, "opt1", "the stage2 answer, not the 0.02 one")
        self.assertEqual(stage2.calls, 1, "tau did not fire, so stage2 was asked")
        self.assertEqual(pair.last_reason, STAGE1_ERROR)
        self.assertEqual(pair.last_stage1_conf, 0.0, "stage1 measured nothing")

    def test_a_one_entry_stage1_is_not_reported_as_the_answer(self):
        # No stage2 answer to fall back on: there is no decision to return, and
        # rule 5 (degrade to stage1) must not resurrect the degenerate one.
        stage1 = _Stage("stage1", answer=ChoiceAnswer("only", {"only": 0.02}, 0.02))
        stage2 = _Stage("stage2", error=RuntimeError("no key"))
        with self.assertRaises(RuntimeError):
            _pair(stage1, stage2).choice(STATE, INSTRUCTION, {"only": "the only option"})

    def test_the_cascade_never_hands_a_one_entry_stage_the_commit_precision(self):
        commit = DEFAULT_PATH_CONF["stage1_confident"]
        one = ChoiceAnswer("only", {"only": 0.02}, 0.02)
        for stages in ([_Stage("stage1", answer=one), _Stage("stage2", answer=_answer("opt1"))],
                       [_Stage("stage1", answer=one)]):
            with self.subTest(stages=len(stages)):
                cascade = _cascade(stages)
                with self.assertRaises(ValueError) as raised:
                    cascade.choice(STATE, INSTRUCTION, {"only": "the only option"})
                self.assertIn("distribution", str(raised.exception))
                self.assertNotEqual(cascade.last_stage, "stage1")
        self.assertGreater(commit, 0.7, "the commit precision would have cleared gate_slot")

    def test_a_two_option_stage1_still_commits_on_its_measured_peak(self):
        # The fix must not turn the band gate off: a real distribution above tau_hi
        # still commits, on the measured p_max, and stage2 is not called.
        stage1 = _Stage("stage1", answer=_answer("opt0", 0.95))
        stage2 = _Stage("stage2", answer=_answer("opt1"))
        cascade = _cascade([stage1, stage2])
        got = cascade.choice(STATE, INSTRUCTION, CRITERIA)
        self.assertEqual((got.choice, cascade.last_stage), ("opt0", "stage1"))
        self.assertEqual(stage2.calls, 0)


class FormulaPropertyTest(unittest.TestCase):
    """(d) The arithmetic: meaning, direction, and bounds."""

    def _uniform_remainder(self, pmax, k):
        rest = (1.0 - pmax) / (k - 1)
        return [pmax] + [rest] * (k - 1)

    def test_pmax_is_the_peak_and_is_bounded(self):
        for k in (2, 3, 8):
            uniform = conf_scale.common_conf([1.0 / k] * k)
            self.assertAlmostEqual(uniform, 1.0 / k, msg="uniform scores the minimum")
            peak = conf_scale.common_conf([1.0] + [0.0] * (k - 1))
            self.assertEqual(peak, 1.0, "a point mass scores the maximum")
            for step in range(1, 10):
                pmax = 1.0 / k + (1.0 - 1.0 / k) * step / 10
                got = conf_scale.common_conf(self._uniform_remainder(pmax, k))
                self.assertAlmostEqual(got, pmax)
                self.assertLessEqual(got, 1.0)
                self.assertGreaterEqual(got, 1.0 / k)

    def test_pmax_is_monotone_in_the_peak(self):
        previous = 0.0
        for step in range(0, 11):
            pmax = 1.0 / 8 + (1.0 - 1.0 / 8) * step / 10
            got = conf_scale.common_conf(self._uniform_remainder(pmax, 8))
            self.assertGreaterEqual(got, previous, "a higher peak cannot score lower")
            previous = got

    def test_d1_conf_is_monotone_increasing_bounded_and_zero_at_uniform(self):
        k = 8
        self.assertEqual(conf_scale.d1_conf(1.0 / k, k), 0.0)
        self.assertEqual(conf_scale.d1_conf(1.0, k), 1.0)
        previous = -1.0
        for step in range(0, 21):
            pmax = 1.0 / k + (1.0 - 1.0 / k) * step / 20
            got = conf_scale.d1_conf(pmax, k)
            self.assertGreaterEqual(got, previous, "d1_conf must rise with p_max")
            self.assertGreaterEqual(got, 0.0)
            self.assertLessEqual(got, 1.0)
            previous = got
        self.assertEqual(conf_scale.d1_conf(0.01, k), 0.0, "floored, never negative")

    def test_entropy_conf_is_monotone_decreasing_in_entropy(self):
        for k in (2, 3, 8):
            self.assertAlmostEqual(conf_scale.entropy_conf([1.0 / k] * k), 0.0,
                                   msg="uniform is the lowest score, not a negative one")
            self.assertAlmostEqual(conf_scale.entropy_conf([1.0] + [0.0] * (k - 1)), 1.0)
            previous = -1.0
            for step in range(0, 21):
                pmax = 1.0 / k + (1.0 - 1.0 / k) * step / 20
                got = conf_scale.entropy_conf(self._uniform_remainder(pmax, k))
                self.assertGreaterEqual(got, previous, "less entropy cannot score lower")
                self.assertGreaterEqual(got, 0.0)
                self.assertLessEqual(got, 1.0)
                previous = got

    def test_the_two_formulas_agree_on_both_endpoints(self):
        for k in (2, 5, 8):
            self.assertEqual(conf_scale.d1_conf(1.0 / k, k),
                             conf_scale.entropy_conf([1.0 / k] * k))
            self.assertAlmostEqual(conf_scale.d1_conf(1.0, k),
                                   conf_scale.entropy_conf([1.0] + [0.0] * (k - 1)))

    def test_entropy_conf_inverts_to_a_peak_in_the_right_range(self):
        for pmax in (0.2, 0.5, 0.87, 0.99):
            vec = self._uniform_remainder(pmax, 8)
            recovered = conf_scale.to_common_scale("logit_read", vec)
            self.assertAlmostEqual(recovered, pmax, places=3)
            self.assertLessEqual(recovered, 1.0)

    def test_the_gate_side_entry_points_refuse_a_single_option(self):
        for call in (lambda: conf_scale.common_conf([0.9]),
                     lambda: conf_scale.entropy_conf([0.9]),
                     lambda: conf_scale.to_common_scale("d1_systemone", [0.9]),
                     lambda: conf_scale.to_common_scale("logit_read", [0.9]),
                     lambda: conf_scale.pmax_from_probs([])):
            with self.assertRaises(ValueError):
                call()
        # ...while the server-fidelity sentinel stays exactly as the server computes it
        self.assertEqual(conf_scale.d1_conf(0.9, 1), 1.0)


class GateDirectionTest(unittest.TestCase):
    """A confidence gate must not admit more as it tightens, nor admit silence."""

    THRESHOLDS = [0.0, 0.3, 0.5, 0.7, 0.9]

    def _passes(self, conf, minimum):
        store = ArtifactStore()
        store.put("picked", "step", "picked_items", ["x"], conf)
        gate = Gate(pre=[{"artifact": "picked", "kind": "conf_gte", "min": minimum}])
        try:
            check_pre(gate, "next", store)
        except GateBlocked:
            return False
        return True

    def test_raising_the_threshold_never_admits_more_artifacts(self):
        for conf in (0.0, 0.2, 0.55, 0.7, 0.85, 1.0):
            passing = [m for m in self.THRESHOLDS if self._passes(conf, m)]
            self.assertEqual(passing, [m for m in self.THRESHOLDS if m <= conf],
                             f"conf {conf} must pass exactly the thresholds at or below it")

    def test_an_unmeasured_artifact_cannot_clear_a_confidence_gate(self):
        # None is not "above 0.0": an absent measurement blocks and goes to
        # ask-back, which is more asking, not less.
        self.assertFalse(self._passes(None, 0.0))

    def test_the_default_artifact_confidence_is_absent_not_certain(self):
        self.assertIsNone(ArtifactStore().put("k", "step", "kind", "data").conf)


if __name__ == "__main__":
    unittest.main()
