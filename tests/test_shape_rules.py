"""shape.classify pins its ordered rule list against measured misfires.

Every case here is a wording that the previous rule set read the wrong way, or a
wording it read correctly and must keep reading correctly:

  * `\\blist\\b` fired on the noun in "candidate list" and "list the exact
    identifiers", turning pick and extract rows into listings.
  * `which files` fired on content questions, turning grep rows (and one read)
    into listings.
  * "find every .sh", "find the README" and "which config files live in the
    conf/ directory?" all fell through to SINGULAR.

The ordering is what fixes them, so the rule that fired is asserted alongside the
shape: a content question must be caught by R2 before R4/R6 can claim it, an
extract row by R1 before R5/R6 see its disk-entry nouns, and a prior set by R3
before R6 reads it as an inventory.
"""
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))

from ti.planner import shape  # noqa: E402

DATA = ROOT / "tests" / "data"

# The six misfires that were measured: listing vocabulary in an intent whose
# answer is a single string or a single file.
MISFIRES = (
    "from the candidate list, choose one to fetch first",
    "list the exact identifiers the user mentioned",
    "which files mention the scheduler hooks?",
    "which files talk about the eviction policy?",
    "which files under src discuss the shard map?",
    "what files make use of the old env var?",
)

# The positives the previous rule set could not see.
MISSED_POSITIVES = (
    "find every .sh under scripts/",
    "find the README",
    "which config files live in the conf/ directory?",
)

# A semantic question that names a file set: every row asks about the contents of
# the set, not its extent, and every row was routed to the listing shape by the
# loose `which <file set>` rule, which terminated at the candidate list with no
# gate to stop it. The predicate is what tells them apart, and content predicates
# cannot be enumerated, so the rule is gated on the inventory vocabulary instead.
SEMANTIC_WH = (
    "which files do I need to change",
    "which docs should I read first",
    "which scripts run at boot",
    "which paths does the loader walk",
    "show me the files the deploy step writes",
    "which files are involved",
    "which files are read at startup",
    "which files implement the rate limiter",
    "which files handle the websocket upgrade",
    "which scripts does the cron job invoke",
    "which docs explain the shard layout",
    "which paths hit the legacy conversion",
    "which files get rewritten by the formatter",
    "which files should be deleted",
    "which files use the deprecated helper",
    "which files hold the migration order",
    "which docs mention the capacity plan",
    "which scripts back up the database",
    "which paths need the new permission",
    "which files cache the parsed config",
    "which docs are stale",
    "which scripts validate the schema",
)

# The same wh-frame asking what exists: stative predicate, no clause after it.
INVENTORY_WH = (
    "which config files live in the conf/ directory?",
    "which directories live under service/",
    "which files are in the bench directory",
    "what files sit in conf/",
    "what json files exist in the config dir",
    "which fixtures are under tests/data",
)

# Phrasings the listing shape shipped with: they must keep reaching it.
LISTING_PHRASES = (
    "ls",
    "list the files",
    "show me the files",
    "what files are in the bench directory",
    "what's in this directory",
)

READ_PHRASES = (
    "read README.md",
    "show me README.md",
    "what's in README.md",
    "",
)


class MisfireTest(unittest.TestCase):
    def test_listing_vocabulary_in_a_singular_intent(self):
        for intent in MISFIRES:
            with self.subTest(intent=intent):
                self.assertEqual(shape.classify(intent), shape.SINGULAR)

    def test_the_rule_that_caught_each_misfire_is_the_guard(self):
        # R2 reads a file as a container of text; R1 as a string artefact; R3 as
        # a choice over a set already in hand.
        expected = {
            "from the candidate list, choose one to fetch first":
                "R3_PRIOR_SET_REFERENCE",
            "list the exact identifiers the user mentioned":
                "R1_EXTRACT_PRODUCTION",
            "which files mention the scheduler hooks?":
                "R2_CONTENT_PREDICATE",
            "which files talk about the eviction policy?":
                "R2_CONTENT_PREDICATE",
            "which files under src discuss the shard map?":
                "R2_CONTENT_PREDICATE",
            "what files make use of the old env var?":
                "R2_CONTENT_PREDICATE",
        }
        for intent, rule in expected.items():
            with self.subTest(intent=intent):
                self.assertEqual(shape.rule_for(intent), rule)


class PositiveTest(unittest.TestCase):
    def test_previously_missed_positives_are_listings(self):
        for intent in MISSED_POSITIVES:
            with self.subTest(intent=intent):
                self.assertEqual(shape.classify(intent), shape.ENUMERATIVE)

    def test_a_name_criterion_is_what_admits_them(self):
        self.assertEqual(shape.rule_for("find every .sh under scripts/"),
                         "R4_NAME_CRITERION")
        self.assertEqual(shape.rule_for("find the README"), "R4_NAME_CRITERION")
        # no name criterion here, so the inventory frame has to carry it
        self.assertEqual(shape.rule_for("which config files live in the conf/ directory?"),
                         "R6_EXISTENTIAL_INVENTORY")

    def test_a_file_class_without_an_existence_frame_is_a_read(self):
        for intent in ("what does the makefile actually do?",
                       "check the top of CHANGELOG.md",
                       "give me the body of the root makefile"):
            with self.subTest(intent=intent):
                self.assertEqual(shape.classify(intent), shape.SINGULAR)


class PreservedContractTest(unittest.TestCase):
    def test_listing_phrasings_keep_reaching_the_listing_shape(self):
        for intent in LISTING_PHRASES:
            with self.subTest(intent=intent):
                self.assertEqual(shape.classify(intent), shape.ENUMERATIVE)

    def test_read_phrasings_keep_the_read_shape(self):
        for intent in READ_PHRASES:
            with self.subTest(intent=intent):
                self.assertEqual(shape.classify(intent), shape.SINGULAR)

    def test_a_container_noun_is_what_makes_what_is_in_a_listing(self):
        self.assertEqual(shape.classify("what's in this directory"), shape.ENUMERATIVE)
        self.assertEqual(shape.classify("what's in README.md"), shape.SINGULAR)
        self.assertEqual(shape.classify("what's the retry flow inside bench/replay.py?"),
                         shape.SINGULAR)


class InventoryGateTest(unittest.TestCase):
    def test_an_existence_question_off_disk_is_not_a_listing(self):
        # "is there" alone is an inventory frame; without a file-set object the
        # gate refuses it, and the question is not about the disk at all.
        for intent in ("is there anything published about the tokenizer patch?",
                       "any recent coverage of the torch.compile release?",
                       "any chatter elsewhere about the sigill crash?"):
            with self.subTest(intent=intent):
                self.assertEqual(shape.classify(intent), shape.SINGULAR)

    def test_how_many_counts_the_file_set_itself(self):
        self.assertEqual(shape.classify("how many .md docs are there in docs/guide?"),
                         shape.ENUMERATIVE)
        self.assertEqual(shape.classify("how many lines does the main file have"),
                         shape.SINGULAR)


class InventoryFrameTest(unittest.TestCase):
    """A wh-word plus a file-set noun is not yet an inventory question.

    The shape decides where the plan terminates, and a listing has no gate after
    it: it returns confidence 1.0 whatever the set is asked to mean. So the
    ambiguous reading has to be SINGULAR by default and ENUMERATIVE only when the
    wording is asking what exists.
    """

    def test_a_semantic_which_question_is_not_an_inventory(self):
        for intent in SEMANTIC_WH:
            with self.subTest(intent=intent):
                self.assertEqual(shape.classify(intent), shape.SINGULAR)

    def test_a_stative_predicate_is_what_admits_a_which_question(self):
        for intent in INVENTORY_WH:
            with self.subTest(intent=intent):
                self.assertEqual(shape.classify(intent), shape.ENUMERATIVE)
                self.assertEqual(shape.rule_for(intent), "R6_EXISTENTIAL_INVENTORY")

    def test_a_clause_between_wh_word_and_noun_is_not_an_object(self):
        # the file set is the object of the wh-word or it is the subject of a
        # clause; only the first is an inventory
        self.assertEqual(shape.rule_for("which config files live in conf/"),
                         "R6_EXISTENTIAL_INVENTORY")
        self.assertIsNone(shape.rule_for("which files do we keep in conf/"))

    def test_a_relative_clause_after_the_object_is_not_a_bare_command(self):
        # "show me the files" is the request; "show me the files the deploy step
        # writes" is a question about three files' worth of reading
        self.assertEqual(shape.rule_for("show me the files"),
                         "R5_IMPLIED_OBJECT_COMMAND")
        self.assertEqual(shape.classify("show me the files the deploy step writes"),
                         shape.SINGULAR)
        self.assertEqual(shape.classify("list the files the build touches"),
                         shape.SINGULAR)
        for intent in ("show me the files in the bench dir",
                       "show all files prefixed with sync_",
                       "show the contents of the templates dir"):
            with self.subTest(intent=intent):
                self.assertEqual(shape.classify(intent), shape.ENUMERATIVE)

    def test_a_prefix_frame_needs_the_prefix(self):
        self.assertEqual(shape.classify("find files that begin with test_"),
                         shape.ENUMERATIVE)
        self.assertEqual(shape.classify("which documentation should I start with"),
                         shape.SINGULAR)


class DefaultTest(unittest.TestCase):
    def test_unrecognised_intents_stay_singular(self):
        for intent in ("the thing we discussed", "handle that", "open it",
                       "show me that file", "the file we discussed", ""):
            with self.subTest(intent=intent):
                self.assertEqual(shape.classify(intent), shape.SINGULAR)

    def test_no_rule_fires_on_a_bare_read(self):
        self.assertIsNone(shape.rule_for("read README.md"))
        self.assertIsNone(shape.rule_for(""))


class V2TableTest(unittest.TestCase):
    """classify over all 150 v2 utterances against the pinned expectation table.

    The table is data, in tests/data/shape_expectations.json: utterance ->
    expected shape. Exactly the 21 filesys_glob rows are listings; every other
    row — read, grep, pick, extract, and the three web ops — is singular, because
    a listing shape would make those questions unanswerable.
    """

    def setUp(self):
        self.rows = json.loads((DATA / "utterances_v2.json").read_text())
        self.expect = json.loads((DATA / "shape_expectations.json").read_text())

    def test_the_table_covers_every_row_once(self):
        self.assertEqual(len(self.rows), 150)
        self.assertEqual(len(self.expect), 150)
        self.assertEqual([r["utterance"] for r in self.rows],
                         [e["utterance"] for e in self.expect])

    def test_the_listing_rows_in_the_table_are_the_glob_rows(self):
        listings = [e["utterance"] for e in self.expect if e["shape"] == shape.ENUMERATIVE]
        globs = [r["utterance"] for r in self.rows if r["label"] == "filesys_glob"]
        self.assertEqual(len(listings), 21)
        self.assertEqual(listings, globs)

    def test_classify_matches_the_table_on_every_row(self):
        wrong = [(idx, e["utterance"], e["shape"], shape.classify(e["utterance"]))
                 for idx, e in enumerate(self.expect)
                 if shape.classify(e["utterance"]) != e["shape"]]
        self.assertEqual(wrong, [])

    def test_exactly_the_glob_rows_are_enumerative(self):
        fired = [idx for idx, e in enumerate(self.expect)
                 if shape.classify(e["utterance"]) == shape.ENUMERATIVE]
        self.assertEqual(fired, [1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 96,
                                 106, 116, 126, 129, 135, 141, 143, 146])


if __name__ == "__main__":
    unittest.main()
