"""The distill-overlap guard — a documented caveat, made an enforced one.

`bench/data/heldout_v3.json` is generated from the same authored frame inventory as
`data/distill/`, which is correct: they are the same 8-op question in two registers.
The consequence is not: a model fitted on `data/distill/` has already seen the
skeletons behind **160 of v3's 207 test-side rows**, so scoring it on the whole test
side measures recall of phrasings it was trained on.

That fact lived in a `note` field in `heldout_v3_families.json`, which is cheaper to
ignore than to obey. These cases pin three things: the join itself (recomputed here
from the raw corpus files, so the manifest is not the only witness to its own
number), the exclusion count every measurement of a distill-trained predictor now
carries, and the refusal of any mode that would include a contaminated row.
"""
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))
sys.path.insert(0, str(ROOT / "bench"))

import eval_harness as H  # noqa: E402

V3 = ROOT / "bench/data/heldout_v3.json"
MANIFEST = ROOT / "bench/data/heldout_v3_families.json"
DISTILL = ROOT / "data/distill"


def corpus_family_sets() -> dict[str, set[str]]:
    """The corpus side of the join, read here — no harness, no manifest."""
    out = {}
    for name in ("train.jsonl", "val.jsonl"):
        lines = (DISTILL / name).read_text(encoding="utf-8").splitlines()
        out[name] = {json.loads(line)["family"] for line in lines if line.strip()}
    return out


class _Always(H.Predictor):
    """A frozen signal that answers one label. Nothing is fitted, so the only thing
    that can move its row count is the overlap guard."""

    frozen = True

    def __init__(self, name="const", trained_on=(), label="filesys_read"):
        self.name = name
        self.trained_on = trained_on
        self.label = label

    def fit(self, train):
        pass

    def predict(self, text):
        return self.label


class JoinTest(unittest.TestCase):
    """Which count is the join, and why the older pair was not one."""

    @classmethod
    def setUpClass(cls):
        cls.rows = json.loads(V3.read_text())
        cls.manifest = json.loads(MANIFEST.read_text())["distill_overlap"]
        cls.families = corpus_family_sets()
        cls.v3 = {r["family"] for r in cls.rows}
        cls.train = cls.families["train.jsonl"]
        cls.whole = cls.train | cls.families["val.jsonl"]
        cls.test_rows = [r for r in cls.rows if r["split"] == "test"]

    def test_the_corpus_side_counts_cannot_be_intersections(self):
        # 239 families in train.jsonl, 293 in train+val. The set under test has 133
        # families in total, so neither figure can be "of this set's families, N are
        # in the corpus" — the intersection is 89 and 106 respectively.
        self.assertEqual(len(self.train), 239)
        self.assertEqual(len(self.whole), 293)
        self.assertEqual(len(self.v3), 133)
        self.assertEqual(len(self.v3 & self.train), 89)
        self.assertEqual(len(self.v3 & self.whole), 106)
        self.assertGreater(len(self.whole), len(self.v3),
                           "293 > 133: it counts the corpus, not the join")

    def test_the_row_counts_are_160_and_145_on_the_test_side(self):
        on_whole = [r for r in self.test_rows if r["family"] in self.whole]
        on_train = [r for r in self.test_rows if r["family"] in self.train]
        self.assertEqual(len(on_whole), 160)
        self.assertEqual(len(on_train), 145)
        # the two row counts are right for their own declaration; they differ only
        # by whether val.jsonl is part of the corpus a model was fitted on
        self.assertEqual(len(on_whole) - len(on_train),
                         len([r for r in self.test_rows
                              if r["family"] in self.families["val.jsonl"]
                              and r["family"] not in self.train]))

    def test_the_whole_set_join_is_295_of_415_in_106_families(self):
        hit = [r for r in self.rows if r["family"] in self.whole]
        self.assertEqual(len(hit), 295)
        self.assertEqual(len({r["family"] for r in hit}), 106)
        self.assertEqual((sum(1 for r in hit if r["split"] == "train"),
                          sum(1 for r in hit if r["split"] == "test")), (135, 160))

    def test_the_manifest_records_the_join_not_only_the_corpus_count(self):
        join = self.manifest["join"]
        self.assertEqual(join["files"], ["train.jsonl", "val.jsonl"])
        self.assertEqual(join["corpus_families"], 293)
        self.assertEqual(join["overlapping_families"], 106)
        self.assertEqual(join["set_families"], 133)
        self.assertEqual((join["rows"], join["test_rows"]), (295, 160))
        train_only = self.manifest["train_only_join"]
        self.assertEqual(train_only["files"], ["train.jsonl"])
        self.assertEqual(train_only["corpus_families"], 239)
        self.assertEqual(train_only["overlapping_families"], 89)
        self.assertEqual(train_only["test_rows"], 145)
        self.assertEqual(self.manifest["kept_after_join"], {"rows": 120, "test_rows": 47})

    def test_the_manifest_join_agrees_with_a_raw_recomputation(self):
        for block, families in ((self.manifest["join"], self.whole),
                                (self.manifest["train_only_join"], self.train)):
            hit = [r for r in self.rows if r["family"] in families]
            self.assertEqual(block["corpus_families"], len(families))
            self.assertEqual(block["overlapping_families"], len(self.v3 & families))
            self.assertEqual(block["rows"], len(hit))
            self.assertEqual(block["test_rows"],
                             sum(1 for r in hit if r["split"] == "test"))

    def test_the_superseded_pair_is_kept_with_its_scale_error_written_down(self):
        old = self.manifest["superseded"]
        self.assertEqual(old["families_in_distill"], 293)
        self.assertEqual((old["resolved_rows"], old["resolved_test_rows"]), (295, 160))
        self.assertEqual(old["train_only_pair"],
                         {"families_in_distill": 239, "resolved_test_rows": 145})
        self.assertIn("corpus's own families", old["why"])
        self.assertIn("295", old["why"])


class GuardTest(unittest.TestCase):
    """A distill-trained predictor meets a v3 row: what the harness does."""

    @classmethod
    def setUpClass(cls):
        cls.train, cls.test = H.load_v3(split="group")

    def test_a_distill_trained_predictor_is_scored_on_47_of_the_207_test_rows(self):
        res = H.score_split(self.train, self.test,
                            [_Always("distill-lr", ("distill",))], n_boot=200)[0]
        self.assertEqual(res.n, 47)
        self.assertEqual(res.n_groups, 10)      # 68 test families - 58 overlapping
        self.assertIsNotNone(res.overlap)
        self.assertEqual(res.overlap.excluded_rows, 160)
        self.assertEqual(res.overlap.excluded_by_split, {"test": 160})
        self.assertEqual(len(res.overlap.overlapping_families), 58)
        self.assertEqual(res.overlap.kept_rows, 47)
        self.assertEqual(res.overlap.declared, ("distill",))
        self.assertEqual(res.overlap.files, ("train.jsonl", "val.jsonl"))

    def test_the_dropped_rows_are_exactly_the_families_the_corpus_holds(self):
        scored, overlap = H.scoreable(self.test, _Always("d", ("distill",)))
        corpus = corpus_family_sets()["train.jsonl"] | corpus_family_sets()["val.jsonl"]
        expected = {e.meta["id"] for e in self.test.examples if e.group in corpus}
        got = {e.meta["id"] for e in self.test.examples} - {e.meta["id"]
                                                            for e in scored.examples}
        self.assertEqual(got, expected)
        self.assertEqual(len(got), 160)
        self.assertEqual(len(scored.examples), 47)
        self.assertEqual(overlap.kept_rows, 47)

    def test_the_exclusion_is_in_the_reported_output_not_only_in_the_object(self):
        res = H.score_split(self.train, self.test,
                            [_Always("distill-lr", ("distill",))], n_boot=200)
        table = H.render_table(res, "v3 test side")
        self.assertIn("**160**", table)
        self.assertIn("160 of 207 rows", table)
        self.assertIn("train.jsonl+val.jsonl", table)
        row = res[0].row()
        self.assertEqual(row["overlap"]["excluded_rows"], 160)
        self.assertEqual(row["overlap"]["kept_rows"], 47)
        self.assertIn("excluded 160", row["note"])

    def test_a_train_only_declaration_drops_the_train_join_and_no_more(self):
        scored, overlap = H.scoreable(self.test, _Always("d", ("distill:train",)))
        self.assertEqual((overlap.excluded_rows, scored.n), (145, 62))
        self.assertEqual(len(overlap.overlapping_families), 53)
        self.assertEqual(overlap.files, ("train.jsonl",))

    def test_a_val_only_declaration_drops_the_val_join(self):
        scored, overlap = H.scoreable(self.test, _Always("d", ("distill:val",)))
        self.assertEqual((overlap.excluded_rows, scored.n), (15, 192))

    def test_a_predictor_that_declares_no_corpus_loses_no_rows(self):
        res = H.score_split(self.train, self.test, [_Always("plain")], n_boot=200)[0]
        self.assertEqual(res.n, self.test.n)
        self.assertEqual(self.test.n, 207)
        self.assertIsNone(res.overlap)

    def test_the_whole_set_reports_both_sides(self):
        ds = H.load_v3(split=None)
        res = H.measure(ds, [_Always("d", ("distill",))], split="group",
                        seeds=(0,), n_boot=200)[0]
        self.assertEqual(res.n, 120)
        self.assertEqual(res.overlap.excluded_by_split, {"train": 135, "test": 160})
        self.assertEqual(res.overlap.excluded_rows, 295)
        self.assertEqual(len(res.overlap.overlapping_families), 106)

    def test_a_declared_but_clean_dataset_reports_zero_rather_than_nothing(self):
        ex = [H.Example(text="how do I boil an egg", label="decide_pick",
                        group="t1:not_a_frame:99")]
        ds = H.Dataset(name="synthetic", examples=ex, labels=("decide_pick",),
                       family_kind="frame")
        scored, overlap = H.scoreable(ds, _Always("d", ("distill",)))
        self.assertEqual(scored.n, 1)
        self.assertIsNotNone(overlap, "checked and clean must differ from not checked")
        self.assertEqual(overlap.excluded_rows, 0)


class RefusalTest(unittest.TestCase):
    """Every route that would include a contaminated row is closed."""

    @classmethod
    def setUpClass(cls):
        cls.train, cls.test = H.load_v3(split="group")

    def test_raise_mode_refuses_the_measurement(self):
        with self.assertRaises(H.OverlapError) as ctx:
            H.score_split(self.train, self.test,
                          [_Always("d", ("distill",))], on_contamination="raise")
        self.assertIn("160", str(ctx.exception))
        with self.assertRaises(H.OverlapError):
            H.measure(self.test, [_Always("d", ("distill",))], split="group",
                      on_contamination="raise")

    def test_there_is_no_mode_that_scores_the_contaminated_rows(self):
        for mode in ("include", "keep", "ignore", "warn", ""):
            with self.assertRaises(ValueError):
                H.score_split(self.train, self.test, [_Always("d", ("distill",))],
                              on_contamination=mode)
            with self.assertRaises(ValueError):
                H.measure(self.test, [_Always("d", ("distill",))], split="group",
                          on_contamination=mode)

    def test_the_private_engines_refuse_an_unfiltered_dataset(self):
        # reaching past measure/score_split is the bypass a future agent would take
        with self.assertRaises(H.OverlapError):
            H._measure_frozen(self.test, _Always("d", ("distill",)))
        with self.assertRaises(H.OverlapError):
            H._measure_cv(self.test, _Always("d", ("distill",)), "group", 3, (0,))

    def test_an_unknown_corpus_key_is_rejected(self):
        with self.assertRaises(ValueError):
            H.score_split(self.train, self.test, [_Always("d", ("distill:test",))])
        with self.assertRaises(ValueError):
            H.corpus_files("imagenet")

    def test_a_dataset_without_a_join_key_is_refused_rather_than_assumed_clean(self):
        ds = H.load_v2()
        with self.assertRaises(ValueError) as ctx:
            H.measure(ds, [_Always("d", ("distill",))], split="group")
        self.assertIn("join key", str(ctx.exception))

    def test_more_than_one_declaration_is_the_union(self):
        both = H.corpus_files("distill", "distill:train")
        self.assertEqual(both, ("train.jsonl", "val.jsonl"))
        self.assertEqual(set(H.corpus_files("distill:train", "distill:val")),
                         {"train.jsonl", "val.jsonl"})
        scored, overlap = H.scoreable(self.test,
                                      _Always("d", ("distill:train", "distill:val")))
        self.assertEqual(overlap.excluded_rows, 160)
        self.assertEqual(scored.n, 47)


if __name__ == "__main__":
    unittest.main()
