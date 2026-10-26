"""shape.classify scored on a held-out set it was not tuned on.

The rows were written before the inventory-frame guard was implemented, from the
same families the review's set covered: 29 listing phrasings and 41 hard negatives
— the semantic `which <file set>` questions the loose rule mis-routed, plus
content-search, extract, pick, read, web and reddit rows that name a file set on
the way past it.

The number this set reports is the guard's own, and the misses are pinned rather
than tuned away: they are vocabulary gaps in the file-set noun list ("templates",
"notebooks", "helpers") and in the object gate (a bare file name such as
"Procfile" names no class), not failures of the inventory frame. Widening the
vocabulary to close them is a separate change, and making it against this set
would spend the measurement it exists to provide.
"""
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))

from ti.planner import shape  # noqa: E402

ROWS = json.loads((ROOT / "tests" / "data" / "heldout_phase4.json").read_text())

# The five listings the classifier still misses, measured at 29/29 precision and
# 24/29 recall. Kept as an exact list so that a new miss is a failure and a
# closed gap is a deliberate edit.
KNOWN_MISSES = [
    "give me the names of the shell helpers in bin/",
    "is there a Procfile anywhere",
    "list everything in the vendor directory",
    "list of every notebook in analysis",
    "what html templates are in the theme folder",
]


class HeldOutSetTest(unittest.TestCase):
    def test_the_set_is_balanced_and_large_enough(self):
        listings = [r for r in ROWS if r["expect"] == shape.ENUMERATIVE]
        other = [r for r in ROWS if r["expect"] != shape.ENUMERATIVE]
        self.assertGreaterEqual(len(ROWS), 60)
        self.assertGreaterEqual(len(listings), 25)
        self.assertGreaterEqual(len(other), 35)
        # every hard-negative family the review measured is represented
        families = {r["family"] for r in other}
        for family in ("semantic-which", "content-search", "extract", "pick",
                       "read", "web", "reddit"):
            with self.subTest(family=family):
                self.assertIn(family, families)

    def test_no_singular_intent_is_classified_as_a_listing(self):
        wrong = [(r["family"], r["utterance"]) for r in ROWS
                 if r["expect"] != shape.ENUMERATIVE
                 and shape.classify(r["utterance"]) == shape.ENUMERATIVE]
        self.assertEqual(wrong, [])

    def test_the_listing_misses_are_exactly_the_pinned_ones(self):
        missed = sorted(r["utterance"] for r in ROWS
                        if r["expect"] == shape.ENUMERATIVE
                        and shape.classify(r["utterance"]) != shape.ENUMERATIVE)
        self.assertEqual(missed, sorted(KNOWN_MISSES))


if __name__ == "__main__":
    unittest.main()
