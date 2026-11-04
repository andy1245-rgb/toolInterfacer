"""eval_harness: the split guarantees, the family reconstruction, the interval.

These cases are the contract the rest of the harness rests on. If the group
split ever lets a family straddle two folds, every number measured through this
module silently becomes a row-wise number again — so that is asserted directly
rather than trusted.

Two of the cases pin the *reconstruction* to the implementations it mirrors
(`bench/classical_baseline.py::template_key` and `bench/local_clf.py::build_groups`),
because a harness that quietly grew a third family definition would make its
numbers incomparable with the published ones.

No model is fitted here: `local_clf.Featurizer` is imported only so the grouping
function can be compared, and the accuracy cases use synthetic predictors.
"""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))
sys.path.insert(0, str(ROOT / "bench"))

import eval_harness as H  # noqa: E402


class _Constant(H.Predictor):
    """Always answers one label. A stand-in for a signal with no skill."""

    def __init__(self, label, name="constant"):
        self.label = label
        self.name = name

    def fit(self, train):
        pass

    def predict(self, text):
        return self.label


class _Prefix(H.Predictor):
    """Answers from the label most often seen with the same opening tokens.

    This is the cleanest probe of a leaking split: it has no capacity for
    anything except the phrasing skeleton, so under a row split its siblings are
    still in the training fold and it scores, while under a template-held-out
    split the whole skeleton is unseen and it abstains.
    """

    name = "prefix"

    def __init__(self, n: int = 3):
        self.n = n
        self.memo = {}

    def _key(self, text):
        return " ".join(text.lower().split()[: self.n])

    def fit(self, train):
        tally = {}
        for e in train:
            tally.setdefault(self._key(e.text), {}).setdefault(e.label, 0)
            tally[self._key(e.text)][e.label] += 1
        self.memo = {k: max(v.items(), key=lambda kv: kv[1])[0] for k, v in tally.items()}

    def predict(self, text):
        return self.memo.get(self._key(text))


class GroupSplitTest(unittest.TestCase):
    def test_no_group_straddles_two_folds(self):
        for ds in (H.load_v2(), H.load_v2(grouping="surface"), H.load_382(),
                   H.load_382(grouping="topic")):
            for seed in range(5):
                fold_of = H.folds_for(ds, "group", 5, seed)
                seen = {}
                for i, g in enumerate(ds.groups):
                    self.assertEqual(
                        seen.setdefault(g, fold_of[i]), fold_of[i],
                        f"{ds.name} seed={seed}: group {g!r} straddles folds")

    def test_every_fold_is_used_and_nonempty(self):
        ds = H.load_382()
        for seed in range(3):
            fold_of = H.folds_for(ds, "group", 5, seed)
            self.assertEqual(sorted(set(fold_of)), [0, 1, 2, 3, 4])

    def test_row_split_ignores_groups(self):
        ds = H.load_v2()
        fold_of = H.folds_for(ds, "row", 5, 0)
        straddling = 0
        seen = {}
        for i, g in enumerate(ds.groups):
            if seen.setdefault(g, fold_of[i]) != fold_of[i]:
                straddling += 1
        self.assertGreater(straddling, 0, "a row split should straddle groups")

    def test_seeds_produce_different_splits(self):
        ds = H.load_v2()
        a = H.folds_for(ds, "group", 5, 0)
        b = H.folds_for(ds, "group", 5, 1)
        self.assertNotEqual(a, b, "repeated splits are only repeated if seeds differ")

    def test_unknown_split_is_rejected(self):
        ds = H.load_v2()
        with self.assertRaises(ValueError):
            H.folds_for(ds, "family", 5, 0)
        with self.assertRaises(ValueError):
            H.measure(ds, [_Constant("filesys_read")], split="leaky")


class FamilyReconstructionTest(unittest.TestCase):
    def test_v2_has_no_family_field_and_yields_the_documented_groups(self):
        # the headline: 150 rows carry utterance/label/band only, so the declared
        # families are the t3 boundary pairs and nothing else
        import json
        rows = json.loads((ROOT / "tests/data/utterances_v2.json").read_text())
        self.assertEqual(sorted(rows[0]), ["band", "label", "utterance"])

        boundary = H.load_v2(grouping="boundary").group_report()
        self.assertEqual(boundary["groups"], 106)
        self.assertEqual(boundary["singleton_groups"], 95)
        surface = H.load_v2(grouping="surface").group_report()
        self.assertEqual(surface["groups"], 102)
        self.assertEqual(surface["singleton_groups"], 87)

    def test_v2_surface_matches_local_clf_build_groups(self):
        import json
        import local_clf
        rows = json.loads((ROOT / "tests/data/utterances_v2.json").read_text())
        theirs = local_clf.build_groups(rows)
        # the ids differ by construction; the *partition* must be identical
        mine = H.v2_surface_groups(rows)
        self.assertEqual(_partition(theirs), _partition(mine))

    def test_r382_template_groups_match_classical_baseline(self):
        import json
        import classical_baseline
        rows = json.loads((ROOT / "tests/data/routing_382.json").read_text())
        registry = json.loads((ROOT / "bench/registry_382.json").read_text())
        templates = {t["id"]: t["templates"] for t in registry["tools"]}
        theirs = [classical_baseline.template_key(r, templates) for r in rows]
        mine = H.r382_template_groups(rows, registry)
        self.assertEqual(theirs, mine)
        self.assertEqual(len(set(mine)), 110)

    def test_r382_topic_families_are_class_disjoint(self):
        """The shipped `family` column groups whole label classes, so holding one
        out is leave-classes-out, not phrasing-held-out. Pinned here because a
        reader would otherwise take 'family-held-out 0%' as a model result."""
        ds = H.load_382(grouping="topic")
        topic_of_label = {}
        for e in ds.examples:
            topic_of_label.setdefault(e.label, set()).add(e.meta["family"])
        self.assertTrue(all(len(v) == 1 for v in topic_of_label.values()))
        # and no label spans two topics
        classes_per_topic = {}
        for e in ds.examples:
            classes_per_topic.setdefault(e.meta["family"], set()).add(e.label)
        overlap = sum(len(v) for v in classes_per_topic.values()) - \
            len({l for v in classes_per_topic.values() for l in v})
        self.assertEqual(overlap, 0)


class IntervalTest(unittest.TestCase):
    def test_cluster_bootstrap_brackets_the_point_estimate(self):
        ds = H.load_v2()
        outcomes = [1.0 if i % 3 else 0.0 for i in range(ds.n)]
        lo, hi = H.cluster_bootstrap_ci(outcomes, ds.groups, n_boot=400)
        point = sum(outcomes) / len(outcomes)
        self.assertLessEqual(lo, point)
        self.assertGreaterEqual(hi, point)

    def test_clustered_data_gets_a_wider_interval_than_independence_assumes(self):
        """When whole groups are all-right or all-wrong, rows are not independent
        draws and an interval built for independent rows is too narrow."""
        groups = [f"g{i // 4}" for i in range(120)]
        outcomes = [1.0 if int(g[1:]) % 2 else 0.0 for g in groups]
        lo, hi = H.cluster_bootstrap_ci(outcomes, groups, n_boot=800, seed=1)
        width_cluster = hi - lo

        n = len(outcomes)
        p = sum(outcomes) / n
        z = 1.96
        denom = 1 + z * z / n
        half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
        width_wilson = 2 * half

        self.assertGreater(width_cluster, width_wilson)

    def test_singleton_groups_reproduce_the_independent_interval(self):
        ds = H.load_v2(grouping="row")
        outcomes = [1.0 if i % 3 else 0.0 for i in range(ds.n)]
        lo, hi = H.cluster_bootstrap_ci(outcomes, ds.groups, n_boot=400)
        point = sum(outcomes) / len(outcomes)
        # with every row its own group there is nothing to cluster, so the
        # interval must look like a row-level one
        self.assertLess(hi - lo, 0.25)
        self.assertLessEqual(lo, point)
        self.assertGreaterEqual(hi, point)


class MeasureTest(unittest.TestCase):
    def test_result_carries_split_n_and_groups(self):
        ds = H.load_v2()
        label = ds.examples[0].label
        (res,) = H.measure(ds, [_Constant(label)], split="group", k=5, seeds=(0,),
                           n_boot=200)
        self.assertEqual(res.split, "group")
        self.assertEqual(res.n, 150)
        self.assertEqual(res.n_groups, 106)
        self.assertEqual(res.dataset, ds.name)
        self.assertGreater(res.atomic_row_fraction, 0.5)

    def test_frozen_predictor_is_split_invariant(self):
        ds = H.load_382()
        p = _Constant("timer_create", name="frozen_const")
        p.frozen = True
        a = H.measure(ds, [p], split="group", n_boot=200)[0]
        b = H.measure(ds, [p], split="row", n_boot=200)[0]
        self.assertEqual(a.acc_mean, b.acc_mean)
        self.assertEqual(a.seeds, 1)

    def test_phrasing_skeleton_scores_only_when_families_stay_in_train(self):
        """The instrument's sanity check. A model with no capacity beyond the
        opening tokens must score under a row split — where siblings of one
        template are still in the training fold — and must collapse when whole
        templates are held out. If those two numbers came out equal, the group
        split would not be grouping."""
        ds = H.load_382()
        grouped = H.measure(ds, [_Prefix()], split="group", k=5, seeds=(0, 1),
                            n_boot=200)[0]
        row = H.measure(ds, [_Prefix()], split="row", k=5, seeds=(0, 1),
                        n_boot=200)[0]
        self.assertGreater(row.acc_mean, grouped.acc_mean)
        self.assertLess(grouped.acc_mean, 0.30)


def _partition(ids):
    """Group ids -> the set of index sets, so two different labellings of the
    same partition compare equal."""
    bucket = {}
    for i, g in enumerate(ids):
        bucket.setdefault(g, set()).add(i)
    return sorted((frozenset(v) for v in bucket.values()), key=sorted)


if __name__ == "__main__":
    unittest.main()
