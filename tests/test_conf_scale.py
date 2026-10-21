"""conf_scale + d1_systemone.noul regression: formula fidelity, no network."""
import json
import math
import pathlib
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))

from ti.decide import conf_scale  # noqa: E402


def setUpModule():
    # load every backend: registration happens on import
    from ti.decide.primitives import _ensure_backends_loaded
    _ensure_backends_loaded()

PROBS = [
    [0.5, 0.5],
    [0.7, 0.2, 0.1],
    [0.25, 0.25, 0.25, 0.25],
    [0.9, 0.05, 0.05],
    [0.87, 0.13 / 7, 0.13 / 7, 0.13 / 7, 0.13 / 7, 0.13 / 7, 0.13 / 7, 0.13 / 7],
]


def entropy_conf_reference(probs):
    """Independent transcription of logit_read's pre-refactor _entropy_conf."""
    h = -sum(p * math.log(p) for p in probs if p > 0)
    return float(1.0 - h / math.log(len(probs)))


class ConfScaleTest(unittest.TestCase):
    def test_pmax_from_probs(self):
        self.assertEqual(conf_scale.pmax_from_probs([0.1, 0.7, 0.2]), 0.7)
        self.assertEqual(conf_scale.pmax_from_probs([0.5, 0.5]), 0.5)

    def test_d1_conf_matches_server_formula(self):
        # server-decision.cpp:1121-1128: max(0, (p_max - 1/K) / (1 - 1/K))
        self.assertAlmostEqual(conf_scale.d1_conf(0.74, 8), (0.74 - 0.125) / 0.875)
        self.assertAlmostEqual(conf_scale.d1_conf(0.87, 8), (0.87 - 0.125) / 0.875)
        self.assertAlmostEqual(conf_scale.d1_conf(1.0, 8), 1.0)
        self.assertEqual(conf_scale.d1_conf(0.05, 8), 0.0)       # floored at 0
        self.assertEqual(conf_scale.d1_conf(0.9, 1), 1.0)        # K<2 -> 1.0

    def test_entropy_conf_matches_refactor_reference_exactly(self):
        for probs in PROBS:
            self.assertEqual(conf_scale.entropy_conf(probs),
                             entropy_conf_reference(probs))

    def test_logit_read_entropy_conf_is_behaviour_identical(self):
        from ti.decide.backends.logit_read import LogitReadBackend
        for probs in PROBS:
            self.assertEqual(LogitReadBackend._entropy_conf(probs),
                             entropy_conf_reference(probs))

    def test_to_common_scale_d1_is_exact_pmax(self):
        for probs in PROBS:
            self.assertAlmostEqual(conf_scale.to_common_scale("d1_systemone", probs),
                                   conf_scale.pmax_from_probs(probs))

    def test_to_common_scale_logit_read_round_trips_entropy(self):
        for probs in PROBS:
            p = conf_scale.to_common_scale("logit_read", probs)
            k = len(probs)
            rebuilt = [p] + [(1.0 - p) / (k - 1)] * (k - 1)
            self.assertAlmostEqual(conf_scale.entropy_conf(rebuilt),
                                   conf_scale.entropy_conf(probs), places=6)

    def test_scales_agree_once_normalized_at_the_0_7_gate(self):
        # finding 7: d1 conf 0.70 ~ p_max 0.74, logit_read conf 0.70 ~ p_max 0.87
        d1_vec = [0.74] + [(1.0 - 0.74) / 7] * 7
        logit_vec = [0.87] + [(1.0 - 0.87) / 7] * 7
        d1_pmax = conf_scale.to_common_scale("d1_systemone", d1_vec)
        logit_pmax = conf_scale.to_common_scale("logit_read", logit_vec)
        self.assertAlmostEqual(d1_pmax, 0.74, places=2)
        self.assertAlmostEqual(logit_pmax, 0.87, places=2)
        self.assertAlmostEqual(conf_scale.d1_conf(d1_pmax, 8), 0.70, places=2)
        self.assertAlmostEqual(conf_scale.entropy_conf(logit_vec), 0.70, places=1)
        self.assertNotAlmostEqual(d1_pmax, logit_pmax, places=2)

    def test_to_common_scale_rejects_unknown_backend(self):
        with self.assertRaises(ValueError):
            conf_scale.to_common_scale("nope", [0.5, 0.5])


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class D1NoulRequestTest(unittest.TestCase):
    """d1_systemone.noul must build a schema-valid /v1/systemone body.

    server-decision.cpp:201-204 requires a non-null ``instructions`` on every
    question; ``question`` is not an accepted alias.
    """

    def _capture(self):
        from ti.decide.backends import d1_systemone
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["body"] = json.loads(req.data.decode())
            return _FakeResponse({"answers": {"check": {"type": "noul", "noul": 0.42}}})

        backend = d1_systemone.D1SystemoneBackend()
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            value = backend.noul("state text", "is this relevant?")
        return captured, value

    def test_noul_posts_instructions_field(self):
        captured, value = self._capture()
        self.assertEqual(captured["url"], "http://127.0.0.1:8080/v1/systemone")
        question = captured["body"]["questions"]["check"]
        self.assertEqual(question["type"], "noul")
        self.assertEqual(question["instructions"], "is this relevant?")
        self.assertNotIn("question", question)

    def test_noul_body_satisfies_parse_questions(self):
        captured, _ = self._capture()
        body = captured["body"]
        self.assertIn("state", body)
        self.assertIsInstance(body["questions"], dict)
        self.assertTrue(body["questions"])
        for q in body["questions"].values():
            self.assertIsNotNone(q.get("instructions"))
            self.assertIn(q["type"], {"choice", "score", "noul"})

    def test_noul_returns_server_value(self):
        _, value = self._capture()
        self.assertEqual(value, 0.42)


if __name__ == "__main__":
    unittest.main()
