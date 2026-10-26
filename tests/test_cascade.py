"""Cascade gate tests — banded policy against the stored per-row lane data.

The offline validation (toolInterfacerExtras/research/round2/A2-frontier-corrected.md
§4.2/§5.2) fit the banded sys→lr policy on per-row probabilities recorded from the
dell re-run. These tests rebuild that loop out of injectable stage backends fed
from the same stored rows, so the shipped implementation is checked against the
numbers the design was accepted on — not against a re-derivation of them.

Reference values are quoted from A2-frontier-corrected.json (`loops.*`,
`oof_loops.banded_sys_first`) next to each assertion.

The lane files live outside the repo; the reproduction classes skip when they are
absent (set TI_LANE_DIR to point somewhere else). Everything else is synthetic and
always runs. No network: the stage backends are fakes.
"""
import json
import os
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))

from ti.decide import conf_scale  # noqa: E402
from ti.decide import cascade as cascade_mod  # noqa: E402
from ti.decide.cache import clear as cache_clear  # noqa: E402
from ti.decide.cascade import CascadeBackend, budget_warning  # noqa: E402
from ti.decide.primitives import ChoiceAnswer, DecideBackend, make_backend  # noqa: E402

LANE_DIR = pathlib.Path(os.environ.get("TI_LANE_DIR", str(ROOT / "tests" / "data" / "lanes")))
LANE_FILES = {"d1_systemone": "v2-d1-3B-Q4-systemone.json",
              "logit_read": "v2-d1-3B-Q4-logit_read.json"}
LANES_PRESENT = all((LANE_DIR / name).is_file() for name in LANE_FILES.values())

CRITERIA = {"opt0": "", "opt1": ""}
N_ROWS = 150

# A2-frontier-corrected.json -> loops.c_banded_agreement_sys_then_lr.op_points
# ["thi0.7375_tlo0.4_t20.5"]: the stored op point this policy has to reproduce.
BANDED_OP_POINT = {
    "tau_hi": 0.7375, "tau_lo": 0.40, "tau2": 0.50,
    "committed": 123, "correct": 104, "coverage": 0.82, "precision": 0.8455,
    "ask_rate": 0.18, "second_call_rate": 0.3267, "band_skip_rate": 0.1133,
    "paths": {"stage1": (84, 0.8690), "agreement": (34, 0.7941), "stage2": (5, 0.80)},
}
# Same artifact, frontier entry (tau_hi 0.70, tau_lo 0.40, tau2 0.50) — the
# thresholds shipped in conf/defaults.json: 125 committed at precision 0.848.
# oof_loops.banded_sys_first["0.85"] (params refit per fold): 125/150 @ 0.832.
CONFIGURED_LOOP = {"committed": 125, "correct": 106, "coverage": 0.8333,
                   "precision": 0.848, "oof_coverage": 0.8333, "oof_precision": 0.832,
                   "band_skip_rate": 0.1133, "second_call_rate": 0.30,
                   "paths": {"stage1": (88, 0.8750), "agreement": (32, 0.78125),
                             "stage2": (5, 0.80)}}
# loops.b_logit_read_only.op_points.fit_0.85 — the single-backend gate.
SINGLE_LANE_GATE = {"tau": 0.444083, "committed": 114, "correct": 97,
                    "coverage": 0.76, "precision": 0.8509}


def _flat(pmax, k=8):
    """K-option vector whose peak is `pmax` and whose remainder is uniform."""
    return {f"opt{i}": (pmax if i == 0 else (1.0 - pmax) / (k - 1)) for i in range(k)}


class _FixedBackend(DecideBackend):
    """Answers one fixed ChoiceAnswer; counts calls; native conf is a decoy."""

    def __init__(self, name, choice, probs, native_conf=0.99):
        self.name = name
        self.pick = choice
        self.probs = probs
        self.native_conf = native_conf
        self.calls = 0

    def choice(self, state, instructions, criteria):
        self.calls += 1
        return ChoiceAnswer(self.pick, dict(self.probs), self.native_conf)

    def noul(self, state, question):
        return 0.25


class _SilentBackend(DecideBackend):
    """No opinion (the shape heuristic_micro returns when nothing matches)."""

    name = "silent"

    def choice(self, state, instructions, criteria):
        return None

    def noul(self, state, question):
        return None


class _LaneBackend(DecideBackend):
    """Replays one lane's stored per-row answer, keyed by state["idx"].

    `confidence` is reported as a decoy constant: the cascade scores stages on
    p_max from the probability vector, so a backend reporting its own conf
    formula (or any other constant) must not move a single decision.
    """

    def __init__(self, name, rows, native_conf=0.99):
        self.name = name
        self.rows = rows
        self.native_conf = native_conf
        self.calls = 0

    def choice(self, state, instructions, criteria):
        self.calls += 1
        row = self.rows[int(state["idx"])]
        return ChoiceAnswer(row["choice"], dict(row["probabilities"]), self.native_conf)

    def noul(self, state, question):
        return 0.5


def _lane_rows(name):
    data = json.loads((LANE_DIR / LANE_FILES[name]).read_text())
    return {int(r["idx"]): {"choice": r.get("choice"),
                            "correct": bool(r.get("correct")),
                            "probabilities": {k: float(v)
                                              for k, v in (r.get("probabilities") or {}).items()}}
            for r in data["per_row"]}


def _evaluate(band, first_rows, second_rows=None):
    """Run the cascade over every stored row, tallying the analysis' counters."""
    committed = correct = asked = skipped = 0
    paths = {"stage1": [0, 0], "agreement": [0, 0], "stage2": [0, 0]}
    for i in sorted(first_rows):
        band.choice({"idx": i}, "route the hop", CRITERIA)
        stage = band.last_stage
        if stage in ("stage1", "agreement"):
            rows = first_rows
        elif stage == "stage2":
            rows = second_rows
        else:
            asked += 1
            skipped += int(stage == "band_skip")
            continue
        committed += 1
        correct += int(rows[i]["correct"])
        paths[stage][0] += 1
        paths[stage][1] += int(rows[i]["correct"])
    n = len(first_rows)
    return {"committed": committed, "correct": correct, "asked": asked,
            "coverage": committed / n,
            "precision": (correct / committed) if committed else None,
            "ask_rate": asked / n, "band_skip_rate": skipped / n,
            "paths": {k: (v[0], (v[1] / v[0]) if v[0] else None) for k, v in paths.items()}}


@unittest.skipUnless(LANES_PRESENT, f"stored lane data not found under {LANE_DIR}")
class BandedReproductionTest(unittest.TestCase):
    """The implemented policy must reproduce the validated loop on stored data."""

    def setUp(self):
        self.sys_rows = _lane_rows("d1_systemone")
        self.lr_rows = _lane_rows("logit_read")
        self.assertEqual(len(self.sys_rows), N_ROWS)
        self.assertEqual(set(self.sys_rows), set(self.lr_rows))

    def _band(self, cfg, stages=None):
        self.stage1 = _LaneBackend("d1_systemone", self.sys_rows)
        self.stage2 = _LaneBackend("logit_read", self.lr_rows)
        cfg = {"enabled": True, **cfg}
        return CascadeBackend(stages or [self.stage1, self.stage2], cfg=cfg)

    def test_reproduces_the_stored_banded_op_point(self):
        band = self._band({"tau_hi": BANDED_OP_POINT["tau_hi"],
                           "tau_lo": BANDED_OP_POINT["tau_lo"],
                           "tau2": BANDED_OP_POINT["tau2"]})
        got = _evaluate(band, self.sys_rows, self.lr_rows)
        self.assertEqual(got["committed"], BANDED_OP_POINT["committed"])
        self.assertEqual(got["correct"], BANDED_OP_POINT["correct"])
        self.assertAlmostEqual(got["coverage"], BANDED_OP_POINT["coverage"], places=4)
        self.assertAlmostEqual(got["precision"], BANDED_OP_POINT["precision"], places=4)
        self.assertAlmostEqual(got["ask_rate"], BANDED_OP_POINT["ask_rate"], places=4)
        self.assertAlmostEqual(self.stage2.calls / N_ROWS,
                               BANDED_OP_POINT["second_call_rate"], places=4)
        self.assertAlmostEqual(got["band_skip_rate"], BANDED_OP_POINT["band_skip_rate"],
                               places=4)
        for path, (n, precision) in BANDED_OP_POINT["paths"].items():
            self.assertEqual(got["paths"][path][0], n, f"{path} path count")
            self.assertAlmostEqual(got["paths"][path][1], precision, places=4,
                                   msg=f"{path} path precision")

    def test_configured_thresholds_hold_the_precision_budget(self):
        cfg = cascade_mod.cascade_cfg()
        band = self._band({"tau_hi": cfg["tau_hi"], "tau_lo": cfg["tau_lo"],
                           "tau2": cfg["tau2"], "precision_target": cfg["precision_target"]})
        got = _evaluate(band, self.sys_rows, self.lr_rows)
        self.assertEqual(got["committed"], CONFIGURED_LOOP["committed"])
        self.assertEqual(got["correct"], CONFIGURED_LOOP["correct"])
        self.assertAlmostEqual(got["coverage"], CONFIGURED_LOOP["coverage"], places=4)
        self.assertAlmostEqual(got["precision"], CONFIGURED_LOOP["precision"], places=4)
        self.assertGreaterEqual(got["precision"], cfg["precision_target"] - 0.01,
                                "in-sample precision should hold the declared budget")
        self.assertAlmostEqual(got["coverage"], CONFIGURED_LOOP["oof_coverage"], places=2,
                               msg="in-sample coverage should track the out-of-fold 0.833")
        self.assertAlmostEqual(self.stage2.calls / N_ROWS,
                               CONFIGURED_LOOP["second_call_rate"], places=4)
        for path, (n, precision) in CONFIGURED_LOOP["paths"].items():
            self.assertEqual(got["paths"][path][0], n, f"{path} path count")
            self.assertAlmostEqual(got["paths"][path][1], precision, places=4)

    def test_single_stage_mode_reproduces_the_stored_single_lane_gate(self):
        band = self._band({"tau_hi": SINGLE_LANE_GATE["tau"], "stage2": None},
                          stages=[_LaneBackend("logit_read", self.lr_rows)])
        got = _evaluate(band, self.lr_rows)
        self.assertEqual(got["committed"], SINGLE_LANE_GATE["committed"])
        self.assertEqual(got["correct"], SINGLE_LANE_GATE["correct"])
        self.assertAlmostEqual(got["coverage"], SINGLE_LANE_GATE["coverage"], places=4)
        self.assertAlmostEqual(got["precision"], SINGLE_LANE_GATE["precision"], places=4)

    def test_banding_buys_coverage_at_the_single_lane_precision(self):
        band = self._band({"tau_hi": cascade_mod.cascade_cfg()["tau_hi"]})
        banded = _evaluate(band, self.sys_rows, self.lr_rows)
        single = _evaluate(self._band({"tau_hi": SINGLE_LANE_GATE["tau"], "stage2": None},
                                      stages=[_LaneBackend("logit_read", self.lr_rows)]),
                           self.lr_rows)
        self.assertAlmostEqual(banded["precision"], single["precision"], delta=0.01)
        self.assertGreater(banded["coverage"], single["coverage"] + 0.05,
                           "A2 §5.1: banding commits ~+7pp more rows at equal precision")

    def test_native_confidence_never_moves_a_decision(self):
        loud = _evaluate(self._band({}), self.sys_rows, self.lr_rows)  # decoy conf 0.99
        quiet = _evaluate(self._band({}, stages=[_LaneBackend("d1_systemone", self.sys_rows,
                                                              native_conf=0.01),
                                                 _LaneBackend("logit_read", self.lr_rows,
                                                              native_conf=0.01)]),
                          self.sys_rows, self.lr_rows)
        self.assertEqual(loud, quiet)

    def test_every_returned_confidence_sits_on_the_p_max_axis(self):
        cfg = cascade_mod.cascade_cfg()
        band = self._band({"tau_hi": cfg["tau_hi"]})
        ceiling = max(cfg["path_conf"].values())
        for i in sorted(self.sys_rows):
            ans = band.choice({"idx": i}, "route the hop", CRITERIA)
            self.assertGreaterEqual(ans.confidence, 1.0 / 8, f"row {i}")
            self.assertLessEqual(ans.confidence, 1.0, f"row {i}")
            if band.last_stage == "defer":
                self.assertAlmostEqual(ans.confidence,
                                       conf_scale.pmax_from_probs(list(ans.probabilities.values())),
                                       places=9, msg=f"row {i}: defer must carry its own p_max")
            else:
                self.assertLessEqual(ans.confidence, ceiling, f"row {i}")


class DisabledModeTest(unittest.TestCase):
    """enabled=false must be a byte-identical single-backend path."""

    def test_disabled_returns_the_stage1_answer_verbatim(self):
        stage1 = _FixedBackend("stage1", "opt0", _flat(0.95), native_conf=0.13)
        stage2 = _FixedBackend("stage2", "opt1", _flat(0.99))
        band = CascadeBackend([stage1, stage2], cfg={"enabled": False, "tau_hi": 0.70})
        direct = stage1.choice({"state": 1}, "route", CRITERIA)
        got = band.choice({"state": 1}, "route", CRITERIA)
        self.assertEqual(got.choice, direct.choice)
        self.assertEqual(got.probabilities, direct.probabilities)
        self.assertEqual(got.confidence, direct.confidence)
        self.assertEqual(got.confidence, 0.13, "no path-conf substitution when disabled")
        self.assertEqual(stage2.calls, 0)
        self.assertEqual(band.last_stage, "pass_through")

    def test_disabled_delegates_noul_and_health(self):
        stage1 = _FixedBackend("stage1", "opt0", _flat(0.95))
        stage2 = _FixedBackend("stage2", "opt1", _flat(0.99))
        band = CascadeBackend([stage1, stage2], cfg={"enabled": False})
        self.assertEqual(band.noul({"state": 1}, "relevant?"), 0.25)
        self.assertTrue(band.health())

    def test_default_config_keeps_the_single_backend_behaviour(self):
        from ti.conf import load
        cfg = cascade_mod.cascade_cfg()
        self.assertFalse(cfg["enabled"], "shipped config must not turn the cascade on")
        chosen = load()["decide"]["backends_in_use"]["choice"]
        self.assertNotEqual(chosen, "cascade", "single-backend default stays selected")


class BandSelectionTest(unittest.TestCase):
    """One case per branch of the banded policy, synthetic vectors."""

    def _band(self, p1, p2=None, choice1="opt0", choice2="opt0", cfg=None):
        self.stage1 = _FixedBackend("stage1", choice1, p1)
        self.stage2 = (_FixedBackend("stage2", choice2, p2) if p2 is not None else None)
        stages = [self.stage1] + ([self.stage2] if self.stage2 else [])
        return CascadeBackend(stages, cfg={"enabled": True, **(cfg or {})})

    def test_confident_stage1_commits_without_stage2(self):
        band = self._band(_flat(0.90))
        ans = band.choice({"state": 1}, "route", CRITERIA)
        self.assertEqual(ans.choice, "opt0")
        self.assertEqual(band.last_stage, "stage1")
        self.assertEqual(ans.confidence, cascade_mod.DEFAULT_PATH_CONF["stage1_confident"])

    def test_confident_stage1_does_not_call_stage2(self):
        band = self._band(_flat(0.90), _flat(0.99))
        band.choice({"state": 1}, "route", CRITERIA)
        self.assertEqual(self.stage2.calls, 0)

    def test_below_the_band_defers_without_paying_for_stage2(self):
        band = self._band(_flat(0.20), _flat(0.99), choice2="opt1")
        ans = band.choice({"state": 1}, "route", CRITERIA)
        self.assertEqual(band.last_stage, "band_skip")
        self.assertEqual(self.stage2.calls, 0)
        self.assertEqual(ans.choice, "opt0")
        self.assertAlmostEqual(ans.confidence, 0.20, places=9)

    def test_middle_band_agreement_commits(self):
        band = self._band(_flat(0.50), _flat(0.30), choice1="opt0", choice2="opt0")
        ans = band.choice({"state": 1}, "route", CRITERIA)
        self.assertEqual(band.last_stage, "agreement")
        self.assertEqual(self.stage2.calls, 1)
        self.assertEqual(ans.choice, "opt0")
        self.assertEqual(ans.confidence, cascade_mod.DEFAULT_PATH_CONF["agreement"])

    def test_middle_band_disagreement_with_confident_stage2_commits_stage2(self):
        band = self._band(_flat(0.50), _flat(0.60), choice1="opt0", choice2="opt1")
        ans = band.choice({"state": 1}, "route", CRITERIA)
        self.assertEqual(band.last_stage, "stage2")
        self.assertEqual(ans.choice, "opt1")
        self.assertEqual(ans.confidence, cascade_mod.DEFAULT_PATH_CONF["stage2_alone"])

    def test_middle_band_disagreement_and_unsure_stage2_defers(self):
        band = self._band(_flat(0.50), _flat(0.30), choice1="opt0", choice2="opt1")
        ans = band.choice({"state": 1}, "route", CRITERIA)
        self.assertEqual(band.last_stage, "defer")
        self.assertEqual(ans.choice, "opt0", "the better readout is kept")
        self.assertAlmostEqual(ans.confidence, 0.50, places=9)

    def test_middle_band_defers_on_the_better_readout(self):
        band = self._band(_flat(0.50), _flat(0.55), choice1="opt0", choice2="opt1",
                          cfg={"tau2": 0.60})
        ans = band.choice({"state": 1}, "route", CRITERIA)
        self.assertEqual(band.last_stage, "defer")
        self.assertEqual(ans.choice, "opt1")
        self.assertAlmostEqual(ans.confidence, 0.55, places=9)

    def test_single_stage_mode_commits_above_tau_hi(self):
        band = self._band(_flat(0.50), cfg={"tau_hi": SINGLE_LANE_GATE["tau"]})
        ans = band.choice({"state": 1}, "route", CRITERIA)
        self.assertEqual(band.last_stage, "stage1")
        self.assertEqual(ans.confidence, cascade_mod.DEFAULT_PATH_CONF["stage1_confident"])

    def test_single_stage_mode_defers_below_tau_hi(self):
        band = self._band(_flat(0.40), cfg={"tau_hi": SINGLE_LANE_GATE["tau"]})
        ans = band.choice({"state": 1}, "route", CRITERIA)
        self.assertEqual(band.last_stage, "defer")
        self.assertAlmostEqual(ans.confidence, 0.40, places=9)

    def test_silent_stage1_falls_through_to_stage2(self):
        stage2 = _FixedBackend("stage2", "opt1", _flat(0.70))
        band = CascadeBackend([_SilentBackend(), stage2], cfg={"enabled": True})
        ans = band.choice({"state": 1}, "route", CRITERIA)
        self.assertEqual(band.last_stage, "stage2")
        self.assertEqual(ans.choice, "opt1")
        self.assertAlmostEqual(ans.confidence, 0.70, places=9)

    def test_no_stage_opinion_raises_rather_than_leaking_none(self):
        band = CascadeBackend([_SilentBackend()], cfg={"enabled": True})
        with self.assertRaises(RuntimeError):
            band.choice({"state": 1}, "route", CRITERIA)

    def test_stage_count_is_validated(self):
        stage = _FixedBackend("stage1", "opt0", _flat(0.9))
        with self.assertRaises(ValueError):
            CascadeBackend([stage, stage, stage], cfg={"enabled": True})


class CommonScaleTest(unittest.TestCase):
    """Thresholds are p_max, never a backend's own confidence formula."""

    # logit_read row 115 of the stored lane: p_max 0.4042 but a 1-H/lnK entropy
    # inversion of 0.6702 — the two scales straddle tau_hi 0.55 below.
    SKEWED = {"filesys_read": 0.0566, "filesys_glob": 0.3951, "filesys_grep": 0.4042,
              "decide_pick": 0.1170, "decide_extract": 0.0144, "web_search": 0.0066,
              "web_fetch": 0.0031, "reddit_digest": 0.0029}

    def test_gate_uses_p_max_not_the_entropy_inversion(self):
        pmax = conf_scale.pmax_from_probs(list(self.SKEWED.values()))
        inverted = conf_scale.to_common_scale("logit_read", list(self.SKEWED.values()))
        self.assertLess(pmax, 0.55)
        self.assertGreater(inverted, 0.55, "the entropy inversion would have committed here")
        stage1 = _FixedBackend("logit_read", "filesys_grep", self.SKEWED)
        stage2 = _FixedBackend("logit_read", "filesys_glob", self.SKEWED)
        band = CascadeBackend([stage1, stage2],
                             cfg={"enabled": True, "tau_hi": 0.55, "tau_lo": 0.10})
        band.choice({"state": 1}, "route", CRITERIA)
        self.assertNotEqual(band.last_stage, "stage1",
                            "stage1 must be scored on p_max, not on an inverted entropy conf")
        self.assertEqual(stage2.calls, 1)

    def test_commit_confidence_is_the_configured_path_precision(self):
        cfg = {"enabled": True, "path_conf": {"stage1_confident": 0.5, "agreement": 0.25}}
        band = CascadeBackend([_FixedBackend("s1", "opt0", _flat(0.90))], cfg=cfg)
        self.assertAlmostEqual(band.choice({"state": 1}, "route", CRITERIA).confidence, 0.5)

    def test_path_conf_is_overridable_partially(self):
        band = CascadeBackend([_FixedBackend("s1", "opt0", _flat(0.90))],
                             cfg={"enabled": True, "path_conf": {"agreement": 0.6}})
        self.assertEqual(band.path_conf["agreement"], 0.6)
        self.assertEqual(band.path_conf["stage2_alone"],
                         cascade_mod.DEFAULT_PATH_CONF["stage2_alone"])


class LoopIntegrationTest(unittest.TestCase):
    """A cascade answer has to survive the loop's conf_gte pre-gate.

    The slot gate (policy.gate_slot 0.70) compares against the artifact conf, so
    the p_max-axis confidence the cascade returns is what decides whether a
    commit actually runs or turns back into ask-back.

    The decision cache is process-global, so every case starts from an empty
    table: an answer another case stored for this state must not stand in for
    the injected fake.
    """

    def setUp(self):
        cache_clear()

    def _run_with(self, factory):
        from ti.conf import load
        from ti.service_entry import use_tool
        prim = sys.modules["ti.decide.primitives"]
        prim._REGISTRY["cascade_live"] = factory
        backends = load()["decide"]["backends_in_use"]
        previous = backends["choice"]
        backends["choice"] = "cascade_live"
        try:
            return use_tool({"intent": "read the file we discussed",
                             "digest": {"cwd": str(ROOT)}})
        finally:
            backends["choice"] = previous
            prim._REGISTRY.pop("cascade_live", None)

    def test_agreement_commit_runs_the_next_step(self):
        stage1 = _FixedBackend("s1", "README.md", _flat(0.50))
        stage2 = _FixedBackend("s2", "README.md", _flat(0.30))
        out = self._run_with(lambda: CascadeBackend([stage1, stage2], cfg={"enabled": True}))
        self.assertEqual(out["status"], "done", out)
        self.assertFalse(out["degraded"], out)
        self.assertEqual([h["step"] for h in out["hops"]],
                         ["candidates", "picked_file", "content"])
        self.assertAlmostEqual(out["confidence"],
                               cascade_mod.DEFAULT_PATH_CONF["agreement"])

    def test_deferral_asks_back_then_falls_back_to_a_best_guess(self):
        stage1 = _FixedBackend("s1", "README.md", _flat(0.30))
        out = self._run_with(lambda: CascadeBackend([stage1],
                                                    cfg={"enabled": True, "tau_hi": 0.46}))
        self.assertTrue(out["degraded"], out)
        self.assertIn("gate unresolved", " ".join(out["assumptions"]))
        # The deferral still raises the question ...
        self.assertIn("askback", [h["op"] for h in out["hops"]])
        # ... but with no answer available the step runs on the best guess instead
        # of returning nothing (spec §4: a degraded result has value, an abort does not).
        self.assertIn("content", [h["step"] for h in out["hops"]])
        self.assertIsNotNone(out["result"])


class ResolvePrecedenceTest(unittest.TestCase):
    """decide.cascade.enabled is authoritative for the "choice" kind.

    Before this rule the flag alone did nothing: resolve("choice") read only
    decide.backends_in_use, so turning the cascade on also meant hand-editing the
    map entry to "cascade" — two keys that had to agree, with the flag silently
    losing when they did not.

    resolve() wraps the choice backend in the fast-path facade, so a case that is
    about which backend was selected unwraps it through .primary.
    """

    def setUp(self):
        from ti.conf import load
        from ti.decide.resolve import resolve
        self._conf = load()["decide"]
        self._resolve = resolve
        self._before = (self._conf["cascade"]["enabled"],
                        self._conf["backends_in_use"]["choice"])
        self.addCleanup(self._restore)

    def _restore(self):
        self._conf["cascade"]["enabled"] = self._before[0]
        self._conf["backends_in_use"]["choice"] = self._before[1]

    def _set(self, enabled, choice):
        self._conf["cascade"]["enabled"] = enabled
        self._conf["backends_in_use"]["choice"] = choice

    def test_enabled_flag_beats_a_fixture_map_entry(self):
        self._set(True, "fixture")
        backend = self._resolve("choice")
        self.assertIsInstance(backend.primary, CascadeBackend)
        self.assertEqual(backend.name, "cascade")

    def test_enabled_flag_beats_any_named_choice_backend(self):
        for name in ("fixture", "heuristic_micro", "d1_systemone", "logit_read"):
            with self.subTest(backends_in_use_choice=name):
                self._set(True, name)
                self.assertIsInstance(self._resolve("choice").primary, CascadeBackend)

    def test_disabled_flag_leaves_the_map_entry_in_charge(self):
        self._set(False, "fixture")
        self.assertEqual(self._resolve("choice").name, "fixture")

    def test_map_entry_alone_does_not_turn_the_band_on(self):
        self._set(False, "cascade")
        backend = self._resolve("choice").primary
        self.assertIsInstance(backend, CascadeBackend)
        self.assertFalse(backend._enabled, "resolve() must not enable the band for you")

    def test_enabled_flag_leaves_the_other_kinds_on_the_map(self):
        self._set(True, "fixture")
        self.assertEqual(self._resolve("noul").name, "fixture")

    def test_stages_come_from_the_cascade_config_not_from_resolve(self):
        self._set(True, "fixture")
        backend = self._resolve("choice").primary
        cfg = cascade_mod.cascade_cfg()
        self.assertEqual([stage.name for stage in backend._stages],
                         [cfg["stage1"], cfg["stage2"]])
        for stage in backend._stages:
            self.assertNotIsInstance(stage, CascadeBackend,
                                     "a stage must never be another cascade")
            self.assertIsNot(stage, backend, "stages are built, not resolved")


class ConfigTest(unittest.TestCase):
    def test_cascade_block_matches_the_validated_bands(self):
        cfg = cascade_mod.cascade_cfg()
        for key in ("enabled", "stage1", "stage2", "tau_hi", "tau_lo", "tau2",
                    "precision_target"):
            self.assertIn(key, cfg)
        self.assertGreaterEqual(cfg["tau_hi"], 0.65)
        self.assertLessEqual(cfg["tau_hi"], 0.70, "tau_hi outside the validated 0.65-0.70 band")
        self.assertAlmostEqual(cfg["tau_lo"], 0.40)
        self.assertGreaterEqual(cfg["tau2"], 0.50)
        self.assertLessEqual(cfg["tau2"], 0.60, "tau2 outside the validated 0.50-0.60 band")
        self.assertLessEqual(cfg["precision_target"], cascade_mod.RECOMMENDED_MAX_PRECISION)
        for path, value in cfg["path_conf"].items():
            self.assertTrue(0.0 < value <= 1.0, path)

    def test_budget_ceiling_is_documented_not_enforced(self):
        self.assertIsNone(budget_warning(0.85))
        note = budget_warning(0.90)
        self.assertIsInstance(note, str)
        self.assertIn("0.90", note)
        band = CascadeBackend([_FixedBackend("s1", "opt0", _flat(0.90))],
                             cfg={"enabled": True, "precision_target": 0.92})
        self.assertEqual(band.precision_target, 0.92)
        self.assertEqual(band.choice({"state": 1}, "route", CRITERIA).choice, "opt0")

    def test_cascade_is_reachable_through_the_primitive_wall(self):
        backend = make_backend("cascade")
        self.assertIsInstance(backend, CascadeBackend)
        self.assertEqual([stage.name for stage in backend._stages],
                         [cascade_mod.cascade_cfg()["stage1"],
                          cascade_mod.cascade_cfg()["stage2"]])

    def test_missing_stage1_config_is_an_error(self):
        with self.assertRaises(ValueError):
            CascadeBackend(cfg={"enabled": True})


if __name__ == "__main__":
    unittest.main()
