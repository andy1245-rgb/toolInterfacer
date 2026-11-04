"""the frozen training set — pinned so a regeneration cannot quietly reintroduce eval content.

`data/distill/` is the committed artefact a trainer is expected to fit on. Its two
claims are the ones that, if silently violated, produce a good-looking number that
means nothing:

* **the rows do not restate the frozen 150.** The label store they came from is
  git-ignored, so nothing about it is reviewable and the next run could change it.
  This test re-runs the leakage guard over the committed rows themselves;
* **the split does not leak families.** A frame on both sides of train/val makes
  the corpus's own validation number a memorisation score.

It also pins the label contract — `label` present, on the row's own ballot, and
distinct from `tier_target` — because a row whose label is not one of its candidates
is not a hard row, it is an unanswerable one.

Nothing here reads `var/distill/`: the committed artefact is the subject, so the
test passes on a fresh clone.
"""
import hashlib
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from distill import corpus, similarity  # noqa: E402

DATA = ROOT / "data" / "distill"
MANIFEST = DATA / "manifest.json"


def rows(name: str) -> list[dict]:
    return [json.loads(line) for line in
            (DATA / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def frozen_rows() -> list[dict]:
    return json.loads(corpus.FROZEN_PATH.read_text(encoding="utf-8"))


class FrozenCorpusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.train = rows("train.jsonl")
        cls.val = rows("val.jsonl")
        cls.excluded = rows("excluded.jsonl")
        cls.ballot = json.loads((DATA / "ballot.json").read_text(encoding="utf-8"))

    def test_counts_match_the_manifest(self):
        counts = self.manifest["counts"]
        self.assertEqual(len(self.train), counts["train"])
        self.assertEqual(len(self.val), counts["val"])
        self.assertEqual(len(self.excluded), counts["excluded"])
        self.assertEqual(len(self.train) + len(self.val), counts["retained"])
        self.assertEqual(len(self.train) + len(self.val) + len(self.excluded),
                         counts["labels"])

    def test_file_hashes_match_the_manifest(self):
        """The manifest's digests are the record; a hand-edit after freezing breaks them."""
        for name, record in self.manifest.get("files", {}).items():
            path = DATA / name
            self.assertTrue(path.exists(), f"{name} named in the manifest is missing")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
            self.assertEqual(digest, record["sha"], f"{name} changed since it was frozen")

    def test_contamination_digests_match_the_eval_sets_they_name(self):
        """`contamination.*.sha` records an eval set the repo does not track.

        The label store those sets live in is git-ignored, so this field was kept
        by hand and nothing verified it: the leakage result is a measurement
        *against a specific file*, and a digest that has drifted from that file
        means the certification is describing an eval set nobody can identify.
        `files` is covered above; this is the other half of the manifest's
        digests, and it was only being caught by a reader noticing.
        """
        for name, record in self.manifest["contamination"]["eval_sets"].items():
            path = ROOT / record["path"]
            self.assertTrue(path.exists(), f"{name}: {record['path']} is missing")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
            self.assertEqual(digest, record["sha"],
                             f"{name}: {record['path']} is not the file the "
                             f"contamination sweep measured")

    def test_no_row_restates_the_frozen_eval_set(self):
        frozen = corpus.frozen()
        for row in self.train + self.val:
            comparison, hit = frozen.match(row["utterance"])
            self.assertFalse(
                comparison.leaks,
                f"{row['id']} collides with the frozen 150: {row['utterance']!r} ~ {hit!r} "
                f"(jaccard={comparison.jaccard:.2f} ratio={comparison.ratio:.2f})")

    def test_no_row_subsumes_a_frozen_row(self):
        """The deleted-clause case: a strict token-subset the ratio bar is blind to."""
        refs = [similarity.normalize(r["utterance"]) for r in frozen_rows()]
        for row in self.train + self.val:
            mine = set(similarity.normalize(row["utterance"]).split())
            for reference in refs:
                theirs = set(reference.split())
                if not mine or not theirs or not (mine < theirs or theirs < mine):
                    continue
                ratio = similarity.ratio_norm(similarity.normalize(row["utterance"]), reference)
                self.assertLess(ratio, 0.70,
                                f"{row['id']} is a subset of a frozen row at ratio {ratio:.2f}: "
                                f"{row['utterance']!r} ~ {reference!r}")

    def test_no_family_crosses_the_split(self):
        side: dict[str, set[str]] = {}
        for row in self.train:
            side.setdefault(row["family"], set()).add("train")
        for row in self.val:
            side.setdefault(row["family"], set()).add("val")
        crossing = sorted(f for f, s in side.items() if len(s) > 1)
        self.assertEqual(crossing, [], f"families on both sides of the split: {crossing}")

    def test_label_is_on_the_row_ballot(self):
        for row in self.train + self.val:
            self.assertIn(row["label"], row["candidates"],
                          f"{row['id']}: label {row['label']!r} not among its own candidates")
            for key in row["candidates"]:
                self.assertIn(key, self.ballot, f"{row['id']}: unknown op {key!r}")

    def test_label_is_never_the_decoy(self):
        for row in self.train + self.val:
            if row["band"] == "t3":
                self.assertIsNotNone(row["tier_target"], f"{row['id']}: t3 without a decoy")
                self.assertNotEqual(row["label"], row["tier_target"],
                                    f"{row['id']}: label equals the decoy")
            else:
                self.assertIsNone(row["tier_target"],
                                  f"{row['id']}: decoy on a non-t3 row")

    def test_bands_and_labels_are_in_range(self):
        for row in self.train + self.val:
            self.assertIn(row["band"], ("t1", "t2", "t3"))
            self.assertIn(row["split"], ("train", "val"))
            self.assertTrue(row["utterance"].strip())

    def test_every_exclusion_carries_a_reason(self):
        rules = self.manifest["exclusions"]["rules"]
        for record in self.excluded:
            self.assertIn(record["rule"], rules, f"{record['id']}: unknown exclusion rule")
            self.assertTrue(record.get("why"), f"{record['id']}: excluded with no reason")
            self.assertTrue(record.get("detail"), f"{record['id']}: excluded with no numbers")
        excluded_ids = {r["id"] for r in self.excluded}
        shipped = {r["id"] for r in self.train + self.val}
        self.assertEqual(shipped & excluded_ids, set(), "an excluded row is in the frozen set")

    def test_teacher_choice_is_not_shipped_as_a_label(self):
        """The teacher contradicts the authored label on 36 rows; it must not leak in."""
        for row in self.train + self.val:
            self.assertNotIn("teacher_choice", row)
            self.assertNotIn("target", row)


if __name__ == "__main__":
    unittest.main()
