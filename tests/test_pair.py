"""pair backend tests — local stage1 with a hosted stage2 behind it.

Every case injects both stages as stubs, so no case touches 127.0.0.1:8080, the
service on :1113, or OpenRouter: the module is about the *policy* — when stage2 is
called at all, and what comes back when it fails — and the policy is what is
pinned here. Stubs are cheap enough to count calls exactly, which is how "stage2
was never contacted" is asserted rather than inferred.

The five policies the module docstring states, one class each: confident stage1
(no escalation), low confidence (escalate), stage1 error with the fallback on
(hard fallback), stage1 error with it off (propagate), and stage1 below tau with
stage2 failing (degrade to stage1, never raise). Plus the tau boundary, the
observability record on every path, noul (no threshold, error only), and the
shipped config selecting the pair.
"""
import pathlib
import subprocess
import sys
import unittest
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
SERVICE = ROOT / "service"
sys.path.insert(0, str(SERVICE))

from ti.conf import load  # noqa: E402
from ti.decide import cache as cache_mod  # noqa: E402
from ti.decide.pair import (CONFIDENT, LOW_CONFIDENCE, STAGE1,  # noqa: E402
                            STAGE1_ERROR, STAGE2, STAGE2_ERROR, PairBackend)
from ti.decide.primitives import ChoiceAnswer, DecideBackend, make_backend  # noqa: E402
from ti.decide.resolve import FastPathBackend, resolve  # noqa: E402

CRITERIA = {"opt0": "the first candidate", "opt1": "the second candidate"}
STATE = {"intent_context": "pick one"}
INSTRUCTION = "which candidate fits the request?"
TAU = 0.90


class _Boom(RuntimeError):
    """A stage that fails the way a transport does: it raises, it returns nothing."""


def _answer(pick: str = "opt0", pmax: float = 0.95, conf: float | None = None,
            k: int = 8) -> ChoiceAnswer:
    """A k-option vector whose peak is exactly `pmax`, so p_max is unambiguous.

    The non-peak mass is spread uniformly, which keeps the peak the max for every
    pmax these cases use (>= 0.125 at k=8) instead of flipping over at 0.5 the way a
    two-option vector would.
    """
    rest = (1.0 - pmax) / (k - 1)
    probs = {f"opt{i}": rest for i in range(k)}
    probs[pick] = pmax
    return ChoiceAnswer(pick, probs, pmax if conf is None else conf)


class _Stage(DecideBackend):
    """A stub stage: counts its calls, returns `answer`, or raises `error`."""

    def __init__(self, name: str, *, answer: ChoiceAnswer | None = None,
                 error: Exception | None = None, noul_value: float = 0.5,
                 noul_error: Exception | None = None) -> None:
        self.name = name
        self.answer = answer
        self.error = error
        self.noul_value = noul_value
        self.noul_error = noul_error
        self.calls = 0
        self.noul_calls = 0

    def choice(self, state: Any, instructions: str,
               criteria: dict[str, str]) -> ChoiceAnswer | None:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.answer  # may be None: "this stage produced nothing"

    def noul(self, state: Any, question: str) -> float:
        self.noul_calls += 1
        if self.noul_error is not None:
            raise self.noul_error
        return self.noul_value


def _pair(stage1: _Stage, stage2: _Stage, **cfg: Any) -> PairBackend:
    return PairBackend([stage1, stage2], cfg={"tau": TAU, **cfg})


class ConfidentStage1Test(unittest.TestCase):
    """Rule 1: stage1 above tau answers, and stage2 is never contacted."""

    def test_confident_stage1_returns_its_answer_unchanged(self):
        first = _answer("opt0", 0.95)
        stage1, stage2 = _Stage("stage1", answer=first), _Stage("stage2", answer=_answer("opt1"))
        got = _pair(stage1, stage2).choice(STATE, INSTRUCTION, CRITERIA)
        self.assertIs(got, first, "no copy, no confidence rewrite: the answer is stage1's")
        self.assertEqual(stage2.calls, 0, "a confident stage1 must not contact stage2")

    def test_stage1_at_exactly_tau_commits(self):
        stage1 = _Stage("stage1", answer=_answer("opt0", TAU))
        stage2 = _Stage("stage2", answer=_answer("opt1"))
        got = _pair(stage1, stage2).choice(STATE, INSTRUCTION, CRITERIA)
        self.assertEqual(got.choice, "opt0")
        self.assertEqual(stage2.calls, 0, "p_max == tau is a commit (>=), not a tie to break")

    def test_stage1_below_tau_escalates(self):
        stage1 = _Stage("stage1", answer=_answer("opt0", TAU - 1e-9))
        stage2 = _Stage("stage2", answer=_answer("opt1"))
        got = _pair(stage1, stage2).choice(STATE, INSTRUCTION, CRITERIA)
        self.assertEqual(got.choice, "opt1")
        self.assertEqual(stage2.calls, 1, "epsilon under tau is a miss")

    def test_tau_reads_p_max_not_the_confidence_the_stage_reported(self):
        # A flat vector asserting high confidence: the common axis is the peak, and
        # that is what tau is fit on, not the number the backend put in `confidence`.
        flat = ChoiceAnswer("opt0", {"opt0": 0.5, "opt1": 0.5}, 0.99)
        stage1, stage2 = _Stage("stage1", answer=flat), _Stage("stage2", answer=_answer("opt1"))
        self.assertEqual(_pair(stage1, stage2).choice(STATE, INSTRUCTION, CRITERIA).choice,
                         "opt1")
        self.assertEqual(stage2.calls, 1)

    def test_a_single_option_vector_is_not_a_distribution_to_gate_on(self):
        # A one-entry vector always sums to 1, so every value of it is a "peak":
        # reading it as p_max 1.0 turned "the stage expressed no distribution" into
        # "maximum certainty" and cleared tau at any confidence (a 0.02 answer
        # committed). It is a stage error, and stage1 errors escalate.
        stage1 = _Stage("stage1", answer=ChoiceAnswer("opt0", {"opt0": 0.02}, 0.02))
        stage2 = _Stage("stage2", answer=_answer("opt1"))
        pair = _pair(stage1, stage2)
        got = pair.choice(STATE, INSTRUCTION, CRITERIA)
        self.assertEqual(got.choice, "opt1")
        self.assertEqual(stage2.calls, 1, "no distribution on stage1 is a reason to ask stage2")
        self.assertEqual(pair.last_reason, STAGE1_ERROR)
        self.assertTrue(pair.last_escalated)
        self.assertEqual(pair.last_stage1_conf, 0.0, "nothing was measured on stage1")

    def test_a_single_option_stage1_with_no_fallback_raises(self):
        # The degenerate answer must not be handed back as a decision either: with
        # stage2 unavailable there is nothing to return, and rule 5 has no stage1
        # answer to degrade to because stage1 never produced a decision.
        stage1 = _Stage("stage1", answer=ChoiceAnswer("only", {"only": 0.02}, 0.02))
        stage2 = _Stage("stage2", error=_Boom("no key"))
        with self.assertRaises(RuntimeError):
            _pair(stage1, stage2).choice(STATE, INSTRUCTION, CRITERIA)


class EscalationTest(unittest.TestCase):
    """Rule 2: below tau, stage2's answer is the answer."""

    def test_low_confidence_stage1_escalates_and_stage2_answers(self):
        first = _answer("opt0", 0.40)
        second = _answer("opt1", 0.80)
        stage1, stage2 = _Stage("stage1", answer=first), _Stage("stage2", answer=second)
        got = _pair(stage1, stage2).choice(STATE, INSTRUCTION, CRITERIA)
        self.assertIs(got, second)
        self.assertEqual((stage1.calls, stage2.calls), (1, 1))

    def test_every_decision_state_escalates_at_the_same_threshold(self):
        for pmax in (0.25, 0.45, 0.5, 0.75, 0.8999):
            with self.subTest(pmax=pmax):
                stage1 = _Stage("stage1", answer=_answer("opt0", pmax))
                stage2 = _Stage("stage2", answer=_answer("opt1"))
                self.assertEqual(_pair(stage1, stage2)
                                 .choice(STATE, INSTRUCTION, CRITERIA).choice, "opt1")
                self.assertEqual(stage2.calls, 1)


class HardFallbackTest(unittest.TestCase):
    """Rules 3 and 4: a stage1 that raises is a fallback, not a failure."""

    def test_stage1_error_escalates_and_stage2_answers(self):
        second = _answer("opt1", 0.70)
        stage1 = _Stage("stage1", error=_Boom("d1 down"))
        stage2 = _Stage("stage2", answer=second)
        pair = _pair(stage1, stage2)
        got = pair.choice(STATE, INSTRUCTION, CRITERIA)
        self.assertIs(got, second)
        self.assertEqual(stage2.calls, 1)
        self.assertEqual(pair.last_stage, STAGE2)
        self.assertEqual(pair.last_reason, STAGE1_ERROR)
        self.assertTrue(pair.last_escalated)
        self.assertEqual(pair.last_stage1_conf, 0.0, "a stage that raised measured nothing")

    def test_stage1_error_is_the_default_policy(self):
        stage1 = _Stage("stage1", error=_Boom("d1 down"))
        stage2 = _Stage("stage2", answer=_answer("opt1"))
        pair = PairBackend([stage1, stage2], cfg={"tau": TAU})  # escalate_on_error unset
        self.assertTrue(pair.escalate_on_error)
        self.assertEqual(pair.choice(STATE, INSTRUCTION, CRITERIA).choice, "opt1")

    def test_stage1_error_without_the_fallback_propagates(self):
        stage1 = _Stage("stage1", error=_Boom("d1 down"))
        stage2 = _Stage("stage2", answer=_answer("opt1"))
        pair = _pair(stage1, stage2, escalate_on_error=False)
        with self.assertRaises(_Boom):
            pair.choice(STATE, INSTRUCTION, CRITERIA)
        self.assertEqual(stage2.calls, 0, "the fallback is off, so stage2 is not a fallback")
        self.assertEqual(pair.last_reason, STAGE1_ERROR)
        self.assertEqual(pair.last_stage, "", "nothing answered")
        self.assertFalse(pair.last_escalated)

    def test_a_stage1_with_no_usable_answer_counts_as_an_error(self):
        for answer in (None, ChoiceAnswer("", {}, 0.0), ChoiceAnswer("opt0", {}, 0.9)):
            with self.subTest(answer=answer):
                stage1 = _Stage("stage1", answer=answer)
                stage2 = _Stage("stage2", answer=_answer("opt1"))
                pair = _pair(stage1, stage2)
                self.assertEqual(pair.choice(STATE, INSTRUCTION, CRITERIA).choice, "opt1")
                self.assertEqual(pair.last_reason, STAGE1_ERROR)

    def test_both_stages_failing_raises(self):
        stage1 = _Stage("stage1", error=_Boom("d1 down"))
        stage2 = _Stage("stage2", error=_Boom("hosted down"))
        pair = _pair(stage1, stage2)
        with self.assertRaises(RuntimeError) as ctx:
            pair.choice(STATE, INSTRUCTION, CRITERIA)
        self.assertIn("no stage produced a choice", str(ctx.exception))
        self.assertEqual((pair.last_stage, pair.last_reason), ("", STAGE2_ERROR))

    def test_the_failure_names_what_a_fresh_install_must_set(self):
        """The composed stage detail used to reach log.debug and nothing else.

        A machine with no llama.cpp server and no hosted key is the state RELEASE.md
        §8 says a fresh install is in, so the one message it gets has to name the
        keys rather than stopping at "no stage produced a choice"
        (docs/audit-2026-11-03.md row 4).
        """
        stage1 = _Stage("d1_systemone", error=_Boom("d1 down"))
        stage2 = _Stage("openrouter", error=_Boom("hosted down"))
        with self.assertRaises(RuntimeError) as ctx:
            _pair(stage1, stage2).choice(STATE, INSTRUCTION, CRITERIA)
        for fragment in ("decide.d1_systemone.base_url", "decide.openrouter.key_env",
                         "TI_OPENROUTER_API_KEY", "decide.backends_in_use.choice",
                         "heuristic_micro", "stage1: d1 down", "stage2: hosted down"):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, str(ctx.exception))

    def test_an_undescribed_stage_gets_no_invented_prerequisite(self):
        """A stage a pair was configured with by hand is left to its own error."""
        stage1 = _Stage("some_future_backend", error=_Boom("down"))
        stage2 = _Stage("another_backend", error=_Boom("down too"))
        with self.assertRaises(RuntimeError) as ctx:
            _pair(stage1, stage2).choice(STATE, INSTRUCTION, CRITERIA)
        message = str(ctx.exception)
        self.assertNotIn("llama.cpp", message)
        self.assertNotIn("TI_OPENROUTER_API_KEY", message)
        self.assertIn("stage1: down", message)


class DegradedStage2Test(unittest.TestCase):
    """Rule 5: a failed escalation must not turn a usable answer into an error."""

    def test_stage2_error_returns_stage1s_answer(self):
        first = _answer("opt0", 0.45, conf=0.45)
        stage1 = _Stage("stage1", answer=first)
        stage2 = _Stage("stage2", error=_Boom("hosted down"))
        pair = _pair(stage1, stage2)
        got = pair.choice(STATE, INSTRUCTION, CRITERIA)
        self.assertIs(got, first, "stage1's answer, as-is: its own confidence is the signal")
        self.assertEqual(got.confidence, 0.45, "the confidence is not rewritten upward")
        self.assertEqual(pair.last_stage, STAGE1)
        self.assertEqual(pair.last_reason, STAGE2_ERROR)
        self.assertTrue(pair.last_escalated)
        self.assertEqual(pair.last_stage1_conf, 0.45)

    def test_stage2_with_no_usable_answer_returns_stage1s_answer(self):
        # openrouter degrades an unusable reply to ChoiceAnswer("", {}, 0.0); an empty
        # decision must not replace one stage1 had.
        first = _answer("opt0", 0.45)
        stage1 = _Stage("stage1", answer=first)
        stage2 = _Stage("stage2", answer=ChoiceAnswer("", {}, 0.0))
        pair = _pair(stage1, stage2)
        self.assertIs(pair.choice(STATE, INSTRUCTION, CRITERIA), first)
        self.assertEqual(pair.last_reason, STAGE2_ERROR)

    def test_the_degraded_answer_still_reaches_the_slot_gate(self):
        from ti.core.gates import GateBlocked, Gate, check_pre

        class _Artifact:
            conf = 0.45

        class _Store:
            def try_get(self, key):
                return _Artifact() if key == "picked_file" else None

        stage1 = _Stage("stage1", answer=_answer("opt0", 0.45))
        stage2 = _Stage("stage2", error=_Boom("hosted down"))
        answer = _pair(stage1, stage2).choice(STATE, INSTRUCTION, CRITERIA)
        gate = Gate(pre=[{"artifact": "picked_file", "kind": "conf_gte", "min": 0.7}])
        with self.assertRaises(GateBlocked):
            check_pre(gate, "decide_pick", _Store())
        self.assertLess(answer.confidence, 0.7, "the degraded answer is visibly below the gate")


class ObservabilityTest(unittest.TestCase):
    """The four fields, one row per path — a measurement workstream reads them."""

    def _record(self, stage1_answer, stage2_answer=None, stage2_error=None, **cfg):
        stage1 = _Stage("stage1", answer=stage1_answer,
                        error=stage1_answer if isinstance(stage1_answer, Exception) else None)
        stage2 = _Stage("stage2", answer=stage2_answer, error=stage2_error)
        pair = _pair(stage1, stage2, **cfg)
        try:
            pair.choice(STATE, INSTRUCTION, CRITERIA)
        except Exception:
            pass
        return (pair.last_stage, pair.last_escalated, pair.last_reason, pair.last_stage1_conf)

    def test_every_path(self):
        cases = {
            "confident": (_answer("opt0", 0.95), _answer("opt1"),
                          (STAGE1, False, CONFIDENT, 0.95)),
            "low_confidence": (_answer("opt0", 0.40), _answer("opt1"),
                               (STAGE2, True, LOW_CONFIDENCE, 0.40)),
            "stage1_error": (_Boom("d1 down"), _answer("opt1"),
                             (STAGE2, True, STAGE1_ERROR, 0.0)),
            "stage1_error_no_fallback": (_Boom("d1 down"), _answer("opt1"),
                                         ("", False, STAGE1_ERROR, 0.0)),
            "stage2_error": (_answer("opt0", 0.40), None,
                             (STAGE1, True, STAGE2_ERROR, 0.40)),
        }
        for name, (stage1, stage2, expected) in cases.items():
            with self.subTest(path=name):
                cfg = {"escalate_on_error": False} if name == "stage1_error_no_fallback" else {}
                got = self._record(stage1, stage2,
                                   stage2_error=_Boom("hosted down") if stage2 is None else None,
                                   **cfg)
                self.assertEqual(got, expected)

    def test_a_later_call_does_not_inherit_the_previous_record(self):
        stage1 = _Stage("stage1", answer=_answer("opt0", 0.40))
        stage2 = _Stage("stage2", error=_Boom("hosted down"))
        pair = _pair(stage1, stage2)
        pair.choice(STATE, INSTRUCTION, CRITERIA)
        self.assertEqual(pair.last_reason, STAGE2_ERROR)
        stage2.error = None
        stage2.answer = _answer("opt1", 0.99)
        pair.choice(STATE, INSTRUCTION, CRITERIA)
        self.assertEqual((pair.last_stage, pair.last_reason), (STAGE2, LOW_CONFIDENCE))

    def test_a_confident_second_call_clears_the_escalation_flag(self):
        stage1 = _Stage("stage1", answer=_answer("opt0", 0.40))
        stage2 = _Stage("stage2", answer=_answer("opt1"))
        pair = _pair(stage1, stage2)
        pair.choice(STATE, INSTRUCTION, CRITERIA)
        self.assertTrue(pair.last_escalated)
        stage1.answer = _answer("opt0", 0.99)
        pair.choice(STATE, INSTRUCTION, CRITERIA)
        self.assertEqual((pair.last_stage, pair.last_escalated, pair.last_reason),
                         (STAGE1, False, CONFIDENT))

    def test_a_debug_line_carries_stage_reason_and_confidence(self):
        stage1 = _Stage("stage1", answer=_answer("opt0", 0.40))
        stage2 = _Stage("stage2", answer=_answer("opt1"))
        pair = _pair(stage1, stage2)
        with self.assertLogs("ti.decide.pair", level="DEBUG") as logs:
            pair.choice(STATE, INSTRUCTION, CRITERIA)
        line = "\n".join(logs.output)
        self.assertIn("stage=stage2", line)
        self.assertIn(f"reason={LOW_CONFIDENCE}", line)
        self.assertIn("stage1_p_max=0.400", line)


class NoulTest(unittest.TestCase):
    """noul has no p_max, so there is no threshold: escalate on error only."""

    def test_a_low_noul_does_not_escalate(self):
        stage1 = _Stage("stage1", noul_value=0.0)
        stage2 = _Stage("stage2", noul_value=1.0)
        pair = _pair(stage1, stage2)
        self.assertEqual(pair.noul(STATE, "is it relevant?"), 0.0)
        self.assertEqual(stage2.noul_calls, 0, "no tau on noul: the value is the answer")
        self.assertEqual(pair.last_stage, STAGE1)

    def test_stage1_error_escalates(self):
        stage1 = _Stage("stage1", noul_error=_Boom("d1 down"))
        stage2 = _Stage("stage2", noul_value=0.83)
        pair = _pair(stage1, stage2)
        self.assertEqual(pair.noul(STATE, "is it relevant?"), 0.83)
        self.assertEqual((pair.last_stage, pair.last_escalated, pair.last_reason),
                         (STAGE2, True, STAGE1_ERROR))

    def test_both_failing_raises_the_second_error(self):
        stage1 = _Stage("stage1", noul_error=_Boom("d1 down"))
        stage2 = _Stage("stage2", noul_error=ValueError("hosted down"))
        pair = _pair(stage1, stage2)
        with self.assertRaises(ValueError):
            pair.noul(STATE, "is it relevant?")
        self.assertEqual((pair.last_stage, pair.last_reason), ("", STAGE2_ERROR))

    def test_a_stage1_error_is_the_only_escalation_noul_has(self):
        # A zero is data here, not a low-confidence verdict — the distinction the
        # threshold would have erased.
        stage1 = _Stage("stage1", noul_value=0.0)
        stage2 = _Stage("stage2", noul_value=0.99)
        self.assertEqual(_pair(stage1, stage2).noul(STATE, "is it relevant?"), 0.0)
        self.assertEqual(stage2.noul_calls, 0)


class RegistrationTest(unittest.TestCase):
    """The pair is selectable by name, and the shipped config selects it."""

    def test_the_shipped_config_selects_the_pair(self):
        self.assertEqual(load()["decide"]["backends_in_use"]["choice"], "pair")

    def test_the_shipped_pair_block(self):
        cfg = load()["decide"]["pair"]
        self.assertEqual(cfg, {"stage1": "d1_systemone", "stage2": "openrouter",
                               "tau": 0.90, "escalate_on_error": True})

    def test_construction_reads_the_block(self):
        pair = make_backend("pair")  # builds both stages; no call, so no network
        self.assertIsInstance(pair, PairBackend)
        self.assertEqual([stage.name for stage in pair._stages],
                         ["d1_systemone", "openrouter"])
        self.assertEqual(pair.tau, 0.90)
        self.assertTrue(pair.escalate_on_error)

    def test_a_pair_needs_two_stages(self):
        with self.assertRaises(ValueError):
            PairBackend([_Stage("stage1", answer=_answer())], cfg={"tau": TAU})

    def test_the_stage_names_are_required(self):
        with self.assertRaises(ValueError):
            PairBackend(None, cfg={"tau": TAU})  # neither stage named
        with self.assertRaises(ValueError):
            PairBackend(None, cfg={"stage1": "d1_systemone", "tau": TAU})  # no stage2

    def test_resolve_hands_back_the_pair_untouched(self):
        # resolve.py needs no change for a backend that wraps two others: it is
        # looked up by name like any other. Only the fast-path facade wraps it.
        decide = load()["decide"]
        before = (decide["cascade"]["enabled"], decide["backends_in_use"]["choice"])
        decide["cascade"]["enabled"] = False
        self.addCleanup(lambda: self._restore(before))
        backend = resolve("choice")
        self.assertIsInstance(backend, FastPathBackend)
        self.assertIsInstance(backend.primary, PairBackend)
        self.assertEqual(backend.name, "pair")

    def _restore(self, before):
        decide = load()["decide"]
        decide["cascade"]["enabled"], decide["backends_in_use"]["choice"] = before
        cache_mod.clear()


COLD = """
import sys
sys.path.insert(0, {service!r})
from ti.decide.primitives import make_backend
pair = make_backend("pair")
print(pair.name, [stage.name for stage in pair._stages], pair.tau, pair.escalate_on_error)
"""


class ColdResolveTest(unittest.TestCase):
    """make_backend("pair") resolves in a fresh interpreter, with no prior import."""

    def test_cold_resolve(self):
        proc = subprocess.run([sys.executable, "-c", COLD.format(service=str(SERVICE))],
                              capture_output=True, text=True, timeout=120, cwd=str(ROOT))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(),
                         "pair ['d1_systemone', 'openrouter'] 0.9 True")


if __name__ == "__main__":
    unittest.main()
