"""The corpus-trained classifier: what it is allowed to be, and that it stays it.

Four properties, none of which the numbers alone would catch:

* **deterministic** — two fits of the same configuration produce the same
  artefact sha, so `data/distill/clf_recipe.json` is a reproducible claim rather
  than a note about one run;
* **trained on the corpus and nothing else** — the fit reads
  `data/distill/train.jsonl`, and every one of its training utterances is
  disjoint from all 150 rows of `tests/data/utterances_v2.json`. The corpus's own
  certification establishes the disjointness; this asserts it at the point where
  a leaked row would actually inflate a number;
* **labelled by `intended`, not by the teacher** — the shipped rows carry no
  `target` or `teacher` field at all, so the teacher's answer is structurally
  unavailable to the trainer rather than merely discouraged. 36 of 412 rows have
  the teacher disagreeing with the authored label;
* **reported per band** — `train.score` returns a `bands` block, because a pooled
  number on this corpus is 33% t1 rows whose utterance names the operation.
"""
from __future__ import annotations

import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from distill import clf, train  # noqa: E402

RECIPE = ROOT / "data" / "distill" / "clf_recipe.json"
SHIPPED = clf.DEFAULT


class DeterminismTest(unittest.TestCase):
    def test_two_fits_agree_byte_for_byte(self):
        a = clf.DistillClassifier.from_config(SHIPPED).fit(clf.load_split("train"))
        b = clf.DistillClassifier.from_config(SHIPPED).fit(clf.load_split("train"))
        self.assertEqual(a.artefact_sha(), b.artefact_sha())

    def test_recipe_reproduces(self):
        self.assertTrue(RECIPE.exists(), "data/distill/clf_recipe.json is the recipe")
        committed = json.loads(RECIPE.read_text())["artefact_sha256"]
        model = clf.DistillClassifier.from_config(SHIPPED).fit(clf.load_split("train"))
        self.assertEqual(model.artefact_sha(), committed,
                         "the committed recipe no longer reproduces — either the "
                         "config moved or the fit stopped being deterministic")

    def test_recipe_records_the_corpus_it_was_fitted_to(self):
        recipe = json.loads(RECIPE.read_text())
        self.assertEqual(recipe["corpus"]["train"]["sha"],
                         train.sha256_file(clf.CORPUS_DIR / "train.jsonl"))
        self.assertEqual(recipe["corpus"]["val"]["sha"],
                         train.sha256_file(clf.CORPUS_DIR / "val.jsonl"))

    def test_abstraction_is_a_function_of_the_utterance(self):
        # Object typing has to work on nouns the corpus never saw, or the whole
        # point of it — transferring to a different authoring pass — is lost.
        self.assertIn("<url>", clf.abstract("fetch https://never.seen.example/x"))
        self.assertIn("<path>", clf.abstract("open tests/data/never_seen.json"))
        self.assertIn("<ident>", clf.abstract("where is never_seen_symbol defined"))
        self.assertIn("<sub>", clf.abstract("what does r/NeverSeen say"))


class TrainingDataTest(unittest.TestCase):
    def test_training_rows_are_disjoint_from_the_eval_set(self):
        train_texts = {r["utterance"] for r in clf.load_split("train")}
        val_texts = {r["utterance"] for r in clf.load_split("val")}
        v2 = json.loads((ROOT / "tests/data/utterances_v2.json").read_text())
        v2_texts = {r["utterance"] for r in v2}
        self.assertEqual(len(v2_texts), 150)
        self.assertEqual(train_texts & v2_texts, set())
        self.assertEqual(val_texts & v2_texts, set())

    def test_label_is_the_authored_intended_op(self):
        for split in ("train", "val"):
            for row in clf.load_split(split):
                self.assertNotIn("target", row, "the teacher's answer must not ship")
                self.assertNotIn("teacher", row)
                self.assertIn(row["label"], clf.LABELS)
                self.assertIn(row["label"], row["candidates"],
                              "a label off its own ballot is a labelling impossibility")

    def test_default_fit_reads_one_split(self):
        # `train_default` fits on train.jsonl. If someone points it at val, the
        # corpus's own holdout disappears without any number changing visibly.
        model = clf.train_default()
        self.assertEqual(model.n_train, len(clf.load_split("train")))


class ReportedAccuracyTest(unittest.TestCase):
    """Regression floors, not claims. Each is ~4 points under what the shipped
    model scores, so a real breakage fails and a re-seed does not."""

    @classmethod
    def setUpClass(cls):
        cls.model = clf.train_default()
        cls.result = train.score(cls.model, clf.load_split("val"))

    def test_bands_are_reported_separately(self):
        self.assertEqual(set(self.result["bands"]), {"t1", "t2", "t3", "all"})
        for band in ("t1", "t2", "t3"):
            self.assertGreater(self.result["bands"][band]["n"], 0)

    def test_val_floors(self):
        bands = self.result["bands"]
        self.assertGreaterEqual(bands["t1"]["acc"], 0.80)
        self.assertGreaterEqual(bands["t2"]["acc"], 0.45)
        self.assertGreaterEqual(bands["t3"]["acc"], 0.70)
        self.assertGreaterEqual(bands["all"]["acc"], 0.68)

    def test_per_class_covers_the_eight_ops(self):
        self.assertEqual(set(self.result["per_class"]), set(clf.LABELS))
