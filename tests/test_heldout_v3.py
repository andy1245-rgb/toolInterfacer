"""heldout_v3 — the one set whose split is declared before its text exists.

`tests/data/utterances_v2.json` cannot hold out a phrasing: 95 of its 150 rows are
atomic, so a group split there resolves to a row split and every number it produces
carries an optimism of unknown size. `bench/data/heldout_v3.json` is generated from the
authored frames in `tools/distill/frames.py` precisely so that it can.

These cases pin the properties that make it usable, and they are the ones whose silent
loss would be invisible in a published figure:

* **no family straddles the split** — the guard that caught the distill corpus, applied
  to the set that ships the split;
* **the families are real** — if a regeneration lowered the variant cap, every family
  would become a singleton and the split would quietly decay into a row split while
  still *claiming* to be a family split. The multi-row row count is therefore pinned;
* **the artefact is reproducible** — the committed rows must be what the generator
  produces from the committed inventory, so a hand-edited row or a silent `frames.py`
  change fails here instead of shipping;
* **the split is not a repartition of the same rows** — `split="row"` must cross
  families, or the mirage comparison the harness prints would be comparing a set with
  itself.

Nothing here fits a model or reads a recording: the two committed JSON files and the
generator's own output are the whole subject.
"""
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (str(ROOT / "bench"), str(ROOT / "tools"), str(ROOT / "service")):
    if p not in sys.path:
        sys.path.insert(0, p)

import eval_harness as H  # noqa: E402

DATA = ROOT / "bench" / "data"
ROWS_PATH = DATA / "heldout_v3.json"
MANIFEST_PATH = DATA / "heldout_v3_families.json"
OPS = {"decide_extract", "decide_pick", "filesys_glob", "filesys_grep",
       "filesys_read", "reddit_digest", "web_fetch", "web_search"}
BANDS = {"t1", "t2", "t3"}


def rows():
    return json.loads(ROWS_PATH.read_text(encoding="utf-8"))


def manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


class DatasetShapeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = rows()

    def test_size_supports_a_per_class_interval(self):
        self.assertGreaterEqual(len(self.rows), 350)
        per_class = {}
        for r in self.rows:
            per_class[r["label"]] = per_class.get(r["label"], 0) + 1
        self.assertEqual(set(per_class), OPS)
        thin = {op: n for op, n in per_class.items() if n < 30}
        self.assertEqual(thin, {}, f"classes below the 30-row floor: {thin}")

    def test_every_row_carries_the_fields_a_reader_needs(self):
        for r in self.rows:
            self.assertEqual(set(r), {"id", "utterance", "label", "band", "tier_target",
                                      "family", "split"})
            self.assertIn(r["label"], OPS)
            self.assertIn(r["band"], BANDS)
            self.assertIn(r["split"], ("train", "test"))
            self.assertTrue(r["family"].strip())
            self.assertGreaterEqual(len(r["utterance"]), 8)
            self.assertLessEqual(len(r["utterance"]), 120)

    def test_utterances_are_unique(self):
        seen = {}
        for r in self.rows:
            norm = " ".join(r["utterance"].lower().split())
            self.assertNotIn(norm, seen, f"{r['id']} restates {seen.get(norm)}")
            seen[norm] = r["id"]

    def test_t3_rows_name_a_rival_and_others_do_not(self):
        for r in self.rows:
            if r["band"] == "t3":
                self.assertIn(r["tier_target"], OPS, f"{r['id']}: no rival named")
                self.assertNotEqual(r["label"], r["tier_target"], f"{r['id']}: label == rival")
            else:
                self.assertIsNone(r["tier_target"], f"{r['id']}: rival on a {r['band']} row")

    def test_both_sides_carry_every_class_and_band(self):
        for side in ("train", "test"):
            labels = {r["label"] for r in self.rows if r["split"] == side}
            bands = {r["band"] for r in self.rows if r["split"] == side}
            self.assertEqual(labels, OPS, f"{side} is missing {OPS - labels}")
            self.assertEqual(bands, BANDS, f"{side} is missing {BANDS - bands}")


class FamilySplitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = rows()
        cls.manifest = manifest()

    def test_no_family_crosses_the_split(self):
        side = {}
        for r in self.rows:
            side.setdefault(r["family"], set()).add(r["split"])
        crossing = sorted(f for f, s in side.items() if len(s) > 1)
        self.assertEqual(crossing, [], f"families on both sides of the split: {crossing}")

    def test_a_row_agrees_with_the_assignment_record(self):
        assignment = self.manifest["assignment"]
        for r in self.rows:
            self.assertIn(r["family"], assignment, f"{r['id']}: family not in the record")
            self.assertEqual(r["split"], assignment[r["family"]]["split"],
                             f"{r['id']}: row side disagrees with the family record")
            self.assertEqual(r["label"], assignment[r["family"]]["label"])

    def test_the_record_covers_every_family_and_matches_the_counts(self):
        assignment = self.manifest["assignment"]
        sizes = {}
        for r in self.rows:
            sizes[r["family"]] = sizes.get(r["family"], 0) + 1
        self.assertEqual(set(assignment), set(sizes))
        for fam, rec in assignment.items():
            self.assertEqual(rec["rows"], sizes[fam], f"{fam}: recorded rows drift")
        per_side = {"train": 0, "test": 0}
        for rec in assignment.values():
            per_side[rec["split"]] += 1
        self.assertEqual(per_side, self.manifest["families"]["families_per_side"])

    def test_the_seed_is_recorded(self):
        self.assertIsInstance(self.manifest["seed"], int)
        self.assertEqual(self.manifest["generator"], "bench/gen_heldout_v3.py")
        self.assertTrue(self.manifest["rule"])

    def test_the_families_are_larger_than_singletons(self):
        """The property that distinguishes this set from v2: most rows are paraphrases
        of a skeleton held out whole, not atomic items."""
        sizes = {}
        for r in self.rows:
            sizes[r["family"]] = sizes.get(r["family"], 0) + 1
        multi = sum(n for n in sizes.values() if n > 1)
        self.assertGreaterEqual(multi / len(self.rows), 0.75,
                                f"only {multi}/{len(self.rows)} rows sit in multi-row families")
        self.assertGreaterEqual(self.manifest["families"]["largest_family"], 4)



class GeneratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "bench"))
        import gen_heldout_v3 as G
        cls.G = G
        cls.rows = rows()
        cls.manifest = manifest()

    def test_the_committed_set_is_what_the_generator_produces(self):
        built, _manifest, _refused = self.G.build()
        self.assertEqual(built, self.rows,
                         "the committed set drifted from the frame inventory — "
                         "regenerate with `python3 bench/gen_heldout_v3.py`")

    def test_generation_is_deterministic(self):
        first, _m1, _r1 = self.G.build()
        second, _m2, _r2 = self.G.build()
        self.assertEqual(first, second)

    def test_the_split_is_assigned_before_any_utterance_exists(self):
        """`plan()` returns families with sides and no text; `render()` cannot move one."""
        families, _budgets, _dupes = self.G.plan()
        self.assertTrue(all(f.split in ("train", "test") for f in families))
        self.assertTrue(all(f.size >= 1 for f in families))
        dropped = {d["family"] for d in self.manifest["dropped_families"]}
        before = {f.id: f.split for f in families}
        after = {}
        for r in self.rows:
            after.setdefault(r["family"], set()).add(r["split"])
        missing = set(before) - set(after)
        self.assertEqual(missing, dropped,
                         "a family vanished without being recorded as dropped")
        for fam, side in before.items():
            if fam in dropped:
                continue
            self.assertEqual(after.get(fam), {side}, f"{fam}: side moved during render")

    def test_duplicate_skeletons_were_dropped_from_the_inventory(self):
        """Two frames that render the same sentence must not become two families."""
        from distill import similarity

        inventory, dropped = self.G.inventory()
        norms = [similarity.normalize(f.text) for f in inventory]
        self.assertEqual(len(norms), len(set(norms)), "a duplicate skeleton survived")
        self.assertGreaterEqual(len(dropped), 3)
        self.assertTrue(all(d["why"] for d in dropped))
        self.assertEqual(
            len(self.manifest["frames"]["duplicates"]), len(dropped),
            "the record must carry every inventory exclusion")

    def test_no_family_is_split_across_the_plan(self):
        families, _budgets, _dupes = self.G.plan()
        sides = {}
        for f in families:
            sides.setdefault(f.id, set()).add(f.split)
        self.assertEqual([f for f, s in sides.items() if len(s) > 1], [])


class HarnessWiringTest(unittest.TestCase):
    def test_group_split_returns_two_sides_of_declared_families(self):
        train, test = H.load_v3(split="group")
        self.assertGreater(train.n, 0)
        self.assertGreater(test.n, 0)
        self.assertEqual(set(train.groups) & set(test.groups), set())
        for ds in (train, test):
            self.assertEqual(set(ds.labels), OPS)
            self.assertEqual({e.label for e in ds.examples}, OPS)

    def test_row_split_keeps_the_sizes_and_crosses_families(self):
        train_g, test_g = H.load_v3(split="group")
        train_r, test_r = H.load_v3(split="row")
        self.assertEqual((train_r.n, test_r.n), (train_g.n, test_g.n))
        self.assertNotEqual(set(train_r.groups) & set(test_r.groups), set(),
                            "the row split must cross families, or it is the same split")

    def test_unsplit_returns_the_whole_set(self):
        ds = H.load_v3(split=None)
        self.assertEqual(ds.n, len(rows()))
        self.assertIn("test", {e.meta["split"] for e in ds.examples})

    def test_unknown_split_is_rejected(self):
        with self.assertRaises(ValueError):
            H.load_v3(split="stratified")

    def test_a_crossing_family_is_refused_by_the_guard(self):
        train, test = H.load_v3(split="group")
        stolen = train.examples.pop()
        test.examples.append(stolen)
        with self.assertRaises(ValueError):
            H.assert_no_family_crossing(train, test)

    def test_score_split_fits_on_train_and_scores_the_test_families(self):
        train, test = H.load_v3(split="group")

        class FirstSeen(H.Predictor):
            name = "first-seen"
            frozen = True

            def __init__(self):
                self.label = None

            def fit(self, tr):
                self.label = sorted({e.label for e in tr})[0]

            def predict(self, text):
                return self.label

        res = H.score_split(train, test, [FirstSeen()], n_boot=200)[0]
        self.assertEqual(res.n, test.n)
        self.assertEqual(res.n_groups, len(set(test.groups)))
        self.assertGreaterEqual(res.acc_mean, 0.0)
        self.assertLessEqual(res.acc_mean, 1.0)
        self.assertLessEqual(res.ci_lo, res.acc_mean)
        self.assertGreaterEqual(res.ci_hi, res.acc_mean)


if __name__ == "__main__":
    unittest.main()
