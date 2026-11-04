"""router_policy: the router must not be able to see the answer.

The defect this guards against is one line — `choice_map[labels[i]]` instead of
`choice_map[predicted_class[i]]` — and it is invisible in an accuracy column: the
oracle scores 84.5% and reads as a router. So the first two cases perturb the
*test-fold labels* and assert the polarity directly: every deployable policy's
decisions must be byte-identical, and the oracle's must not be. A test that only
checked the deployable side would pass just as well if the perturbation was
silently a no-op, which is why the oracle's movement is also asserted.

The rest pins the arithmetic a published number depends on: the recording is
aligned to the dataset it is scored on, the per-class map is a function of
training rows only, and the escalation rate a cascade row reports is the rate its
own predictions escalate at.
"""
import copy
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))
sys.path.insert(0, str(ROOT / "bench"))

import eval_harness as H  # noqa: E402
import router_policy as R  # noqa: E402

ARTIFACT = ROOT / "bench/results/router-policy-2026-11-04.json"
DEPLOYABLE = ("lr", "router[static]", "router[per-class, A]", "router[per-class, B]",
              "router[A, 1nn dispatch]", "cascade[nested tau]")


def _perturbed(ds: H.Dataset, rows: list[int]) -> H.Dataset:
    """The same dataset with the labels of `rows` moved to a different class."""
    out = copy.deepcopy(ds)
    labels = list(ds.labels)
    for i in rows:
        current = labels.index(out.examples[i].label)
        out.examples[i].label = labels[(current + 1) % len(labels)]
    return out


class TestNoGroundTruthAtInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = H.load_v2(grouping="boundary")
        cls.three_b = R.ThreeB().align(cls.ds)
        cls.fold_of = H.folds_for(cls.ds, "group", H.DEFAULT_K, 0)
        cls.test_idx = [i for i, f in enumerate(cls.fold_of) if f == 0]
        cls.base = R._fold_pass(cls.ds, cls.fold_of, 0, 0, cls.three_b, 3, 3)
        cls.pert = R._fold_pass(_perturbed(cls.ds, cls.test_idx), cls.fold_of, 0, 0,
                                cls.three_b, 3, 3)

    def test_deployable_decisions_ignore_the_test_labels(self):
        for name in DEPLOYABLE:
            with self.subTest(policy=name):
                self.assertEqual(self.base["decisions"][name],
                                 self.pert["decisions"][name],
                                 f"{name} changed when the scored rows' labels "
                                 f"changed — it is reading the answer")

    def test_the_perturbation_is_visible_to_an_oracle(self):
        """The control: if this passes trivially the test above proves nothing."""
        self.assertNotEqual(self.base["decisions"]["oracle[per-class]"],
                            self.pert["decisions"]["oracle[per-class]"],
                            "perturbing the scored labels moved nothing, not even "
                            "the oracle — the perturbation is a no-op")

    def test_oracle_and_router_are_different_policies(self):
        self.assertNotEqual(self.base["decisions"]["oracle[per-class]"],
                            self.base["decisions"]["router[per-class, A]"])

    def test_the_map_is_a_function_of_training_rows_only(self):
        self.assertEqual(self.base["map_true"], self.pert["map_true"])
        self.assertEqual(self.base["map_pred"], self.pert["map_pred"])
        self.assertEqual(self.base["tau"], self.pert["tau"])


class TestRecordingAndGates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = H.load_v2(grouping="boundary")
        cls.three_b = R.ThreeB().align(cls.ds)
        cls.fold_of = H.folds_for(cls.ds, "group", H.DEFAULT_K, 0)
        cls.out = R._fold_pass(cls.ds, cls.fold_of, 0, 0, cls.three_b, 3, 3)

    def test_recording_must_match_the_scored_dataset(self):
        other = H.load_382(grouping="row")
        with self.assertRaises(SystemExit):
            R.ThreeB().align(other)

    def test_cascade_prediction_follows_its_gate(self):
        """Every cascade row must be the cheap answer above τ and the 3B below it."""
        cheap = self.out["cheap_pred"]
        b3 = self.out["b3"]
        for tau in R.TAUS:
            preds = self.out["decisions"][f"cascade[maxp>={tau}]"]
            gate = self.out["decisions"][f"escalate[maxp>={tau}]"]
            for r in range(len(preds)):
                expected = b3[r] if gate[r] == 1.0 else cheap[r]
                self.assertEqual(preds[r], expected)
                self.assertEqual(gate[r], 0.0 if self.out["cheap_maxp"][r] >= tau
                                 else 1.0)

    def test_every_policy_answers_every_row(self):
        """`rules` abstains by design (`rules_predict` may return None); every
        other policy here is a decision and must always produce a label."""
        for name, preds in self.out["decisions"].items():
            if name == "rules":
                continue
            with self.subTest(policy=name):
                self.assertEqual(len(preds), len(self.out["test_idx"]))
                self.assertTrue(all(p is not None for p in preds))


class TestPublishedArtifact(unittest.TestCase):
    """The committed JSON must carry split, n and an interval on every number."""

    @classmethod
    def setUpClass(cls):
        if not ARTIFACT.exists():
            raise unittest.SkipTest("artifact not generated in this tree")
        cls.data = json.loads(ARTIFACT.read_text())

    def test_every_accuracy_carries_n_split_and_ci(self):
        protocol = self.data["protocol"]
        for name, row in self.data["results"].items():
            if "acc" not in row:
                continue
            with self.subTest(policy=name):
                self.assertEqual(row["n"], self.data["dataset"]["n"])
                self.assertEqual(row["split"], protocol["split"])
                self.assertEqual(len(row["ci95"]), 2)
                self.assertLessEqual(row["ci95"][0], row["acc"])
                self.assertGreaterEqual(row["ci95"][1], row["acc"])

    def test_the_oracle_is_labelled_and_below_the_union_ceiling(self):
        r = self.data["results"]
        union = self.data["unions"]["union(lr|3b)"]["acc"]
        self.assertGreater(r["oracle[per-class]"]["acc"], r["router[per-class, A]"]["acc"])
        self.assertGreater(union, r["oracle[per-class]"]["acc"])

    def test_the_caveat_is_in_the_file(self):
        caveat = self.data["dataset"]["caveat"].lower()
        self.assertIn("generalisation", caveat)
        self.assertGreater(self.data["dataset"]["group_report"]["singleton_row_fraction"],
                           0.5)


if __name__ == "__main__":
    unittest.main()
