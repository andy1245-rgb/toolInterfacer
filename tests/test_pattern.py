"""pattern.build pins which names a listing asks for.

scope.py answers which directory; this answers which entries inside it. The table
below is the 21 filesys_glob rows of tests/data/utterances_v2.json with the glob
each one should produce: eight of them state no name criterion at all and must
come back as `**/*` with an assumption that says so, rather than a pattern that
looks specific and is not.

Dotfiles get their own case: the honest pattern is `**/.*`, and the filesys walk
skips every entry beginning with "." before it can match, so the assumption has to
record that the dotfiles were not enumerated.
"""
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))

from ti.planner import pattern  # noqa: E402

DATA = ROOT / "tests" / "data"

# idx of the row in tests/data/utterances_v2.json -> the glob it asks for.
TABLE = {
    1: "**/*.toml",        # list every file ending in .toml anywhere in the repo
    9: "**/*",             # which config files live in the conf/ directory?
    17: "**/*",            # find all test fixtures under tests/fixtures
    25: "**/*.proto",      # names of all .proto files under api/
    33: "**/*.sh",         # find every .sh under scripts/
    41: "**/*dockerfile*",  # do we have any dockerfile at the top level?
    49: "**/*.md",         # how many .md docs are there in docs/guide?
    57: "**/*bench_*",     # which files here start with bench_?
    65: "**/*",            # where do we keep the probe fixtures?
    73: "**/*",            # is there a migration script for the settings table?
    81: "**/*README*",     # find the README
    89: "**/.*",           # which dotfiles did the setup leave behind?
    96: "**/*replay*",     # which files have 'replay' in their name?
    106: "**/*.har",       # are there any .har files in the captures dir?
    116: "**/*.bak",       # files ending in .bak anywhere in the project?
    126: "**/*",           # which files match the pattern used for backups?
    129: "**/*",           # list the matches for the pattern in the user's note
    135: "**/*.patch",     # which ones in here end in .patch?
    141: "**/*",           # any config files lying around that look relevant?
    143: "**/*manifest*",  # work out the paths for every manifest under deploy/
    146: "**/*",           # give me the collection of test data files we have
}

KIND = {
    1: pattern.EXTENSION, 9: pattern.NONE, 17: pattern.NONE, 25: pattern.EXTENSION,
    33: pattern.EXTENSION, 41: pattern.FILE_CLASS, 49: pattern.EXTENSION,
    57: pattern.PREFIX, 65: pattern.NONE, 73: pattern.NONE, 81: pattern.FILE_CLASS,
    89: pattern.DOTFILE, 96: pattern.SUBSTRING, 106: pattern.EXTENSION,
    116: pattern.EXTENSION, 126: pattern.NONE, 129: pattern.NONE,
    135: pattern.EXTENSION, 141: pattern.NONE, 143: pattern.FILE_CLASS,
    146: pattern.NONE,
}


class V2PatternTableTest(unittest.TestCase):
    def setUp(self):
        self.rows = json.loads((DATA / "utterances_v2.json").read_text())

    def test_every_glob_row_is_in_the_table(self):
        globs = [i for i, r in enumerate(self.rows) if r["label"] == "filesys_glob"]
        self.assertEqual(sorted(TABLE), globs)
        self.assertEqual(len(TABLE), 21)

    def test_the_table_pins_the_pattern_and_the_kind(self):
        for idx, want in sorted(TABLE.items()):
            utterance = self.rows[idx]["utterance"]
            with self.subTest(idx=idx, utterance=utterance):
                got, kind, _assumption = pattern.build(utterance)
                self.assertEqual(got, want)
                self.assertEqual(kind, KIND[idx])


class BuildTest(unittest.TestCase):
    def test_an_extension_is_lowercased_as_typed(self):
        self.assertEqual(pattern.build("find every .SH here")[0], "**/*.sh")
        self.assertEqual(pattern.build("list the .TOML files")[0], "**/*.toml")

    def test_the_extension_frame_wins_over_a_stray_dot(self):
        got, kind, _ = pattern.build("list every file ending in .proto under api.v2/")
        self.assertEqual((got, kind), ("**/*.proto", pattern.EXTENSION))

    def test_a_prefix_and_a_substring_both_widen_to_a_star(self):
        self.assertEqual(pattern.build("which files begin with tmp_?")[:2],
                         ("**/*tmp_*", pattern.PREFIX))
        self.assertEqual(pattern.build("files whose name has mid")[:2],
                         ("**/*mid*", pattern.SUBSTRING))

    def test_a_file_class_pattern_keeps_the_case_it_was_written_in(self):
        self.assertEqual(pattern.build("find the LICENSE")[:2],
                         ("**/*LICENSE*", pattern.FILE_CLASS))
        self.assertEqual(pattern.build("do we have a lockfile?")[:2],
                         ("**/*lockfile*", pattern.FILE_CLASS))

    def test_a_file_class_in_a_read_contributes_no_criterion(self):
        # shape.py keeps this singular; pattern.py must not invent a glob for it
        got, kind, assumption = pattern.build("what does the makefile actually do?")
        self.assertEqual((got, kind), ("**/*", pattern.NONE))
        self.assertIsNotNone(assumption)


class NoConstraintTest(unittest.TestCase):
    def test_no_name_constraint_still_lists_everything_and_says_so(self):
        got, kind, assumption = pattern.build("which config files live in the conf/ directory?")
        self.assertEqual((got, kind), ("**/*", pattern.NONE))
        self.assertIn("no name constraint", assumption)
        self.assertIn("every entry", assumption)

    def test_a_constraint_carries_no_assumption(self):
        for intent in ("find every .sh", "find the README", "which files start with bench_?"):
            with self.subTest(intent=intent):
                self.assertIsNone(pattern.build(intent)[2])


class DotfileLimitationTest(unittest.TestCase):
    def test_the_dotfile_case_names_the_walk_as_the_limitation(self):
        got, kind, assumption = pattern.build("which dotfiles did the setup leave behind?")
        self.assertEqual((got, kind), ("**/.*", pattern.DOTFILE))
        self.assertIn("dotfiles", assumption)
        self.assertIn("could not be enumerated", assumption)

    def test_a_named_dotfile_takes_the_same_branch(self):
        for intent in ("find the .env", "where is the .gitignore?"):
            with self.subTest(intent=intent):
                self.assertEqual(pattern.build(intent)[:2], ("**/.*", pattern.DOTFILE))


if __name__ == "__main__":
    unittest.main()
