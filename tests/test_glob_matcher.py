"""filesys_glob matches a glob, not a regex fragment.

The op used to do `re.match(pattern.replace("**", ".*"), rel)`, which is a regex
fragment wearing a glob's clothes. The measured consequences, each reproduced as a
case below over a fixture tree:

    **/bench_*      -> the literal "/" after ** was required, so a top-level
                       bench_* entry could never match
    **/*bench_*     -> degenerated to "contains bench", so bench/lane.py matched
    **/.*           -> `.*` matched everything, so the "dotfile" pattern returned
                       every non-dotfile in the tree
    **/*dockerfile* -> case-sensitive, so Dockerfile was missed
    **/*.sh         -> "." was an any-char, so session_push.ts matched

The singular read path walks with `**/*`, so that pattern's candidate set is
asserted unchanged against the fixture tree: it is the one pattern whose behaviour
everything downstream of decide_pick depends on.
"""
import os
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))

import ti.ops.filesys.impl  # noqa: E402,F401  (registers the ops)
from ti.ops.filesys import confine  # noqa: E402
from ti.ops.op_api import get  # noqa: E402

FILES = {
    # top level
    "README.md": "readme",
    "pyproject.toml": "toml",
    "bench_lane.py": "top-level bench_ entry",
    "Dockerfile": "docker",
    "notes.txt": "notes",
    # a nested tree
    "sub/README.txt": "readme in sub",
    "sub/lane.sh": "sh",
    "sub/run.sh": "sh",
    "sub/bench_train.py": "nested bench_ entry",
    "sub/one.patch": "patch",
    "sub/Dockerfile": "nested dockerfile",
    "sub/deep/inner.sh": "deep sh",
    "sub/deep/inner.toml": "deep toml",
    "sub/deep/bench_deep.py": "deep bench_ entry",
    # a directory whose name contains the criterion word but no entry that does
    "bench/lane.py": "not a bench_* name, only inside a 'bench' dir",
    "bench_x/lane.py": "not a bench_* name, only inside a 'bench_'-prefixed dir",
    "bench/serve_local.sh": "sh in the bench dir",
    # decoys: what the regex fragment matched and a glob must not
    "other/session_push.ts": "not a shell script",
    "other/machine.toml.bak": "bak",
}


def _build(root: pathlib.Path) -> None:
    for rel, body in FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    # skipped by the walk, exactly as before: vendored trees and dot entries
    for rel in ("build/ignored.sh", "node_modules/pkg/index.sh",
                ".hidden/inside.sh", ".secret"):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")


class GlobMatcherTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = pathlib.Path(self._tmp.name)
        _build(self.root)
        # Every path an op acts on resolves against the run root; use_tool() and
        # the HTTP layer set it per run, a direct op call sets it here.
        token = confine.set_root(str(self.root))
        self.addCleanup(confine.reset_root, token)

    def hits(self, pattern: str, **params) -> list[str]:
        _kind, data, _conf = get("filesys_glob").run(
            {"pattern": pattern, "cwd": str(self.root), **params}, store=None)
        return sorted(data)

    def assertExact(self, pattern: str, expected: set[str]) -> None:
        got = set(self.hits(pattern))
        self.assertEqual(got - expected, set(), f"{pattern}: violating hits")
        self.assertEqual(expected - got, set(), f"{pattern}: missed hits")

    def test_a_star_star_prefix_matches_a_top_level_entry(self):
        # `**/` also matches zero directories, so `**/bench_*` needs no directory
        # in front of the name: the literal-slash reading returned nothing at all
        self.assertExact("**/bench_*",
                         {"bench_lane.py", "sub/bench_train.py", "sub/deep/bench_deep.py"})

    def test_bench_widened_to_a_name_criterion_does_not_match_a_directory(self):
        # "contains bench" would have returned bench/lane.py, and fnmatch's `*`
        # crossing '/' returned bench_x/lane.py; a glob matches the entry name
        self.assertExact("**/*bench_*",
                         {"bench_lane.py", "sub/bench_train.py", "sub/deep/bench_deep.py"})
        self.assertNotIn("bench_x/lane.py", self.hits("**/*bench_*"))
        self.assertNotIn("bench_x/lane.py", self.hits("**/bench_*"))

    def test_the_dotfile_pattern_matches_no_visible_entry(self):
        # the walk skips dot entries before matching, so `**/.*` is empty by
        # construction — what it must never do is return every visible path
        self.assertExact("**/.*", set())

    def test_a_file_class_matches_case_insensitively(self):
        self.assertExact("**/*dockerfile*", {"Dockerfile", "sub/Dockerfile"})

    def test_a_dot_in_the_pattern_is_a_dot(self):
        self.assertExact("**/*.sh", {"sub/lane.sh", "sub/run.sh", "sub/deep/inner.sh",
                                      "bench/serve_local.sh"})

    def test_every_other_pinned_pattern(self):
        self.assertExact("**/*.toml", {"pyproject.toml", "sub/deep/inner.toml"})
        self.assertExact("**/*.patch", {"sub/one.patch"})
        self.assertExact("**/*README*", {"README.md", "sub/README.txt"})
        self.assertExact("**/*.bak", {"other/machine.toml.bak"})

    def test_star_star_star_still_returns_every_visible_path(self):
        visible = {rel for rel in FILES}
        self.assertExact("**/*", visible)

    def test_the_vendored_and_dot_skips_are_untouched(self):
        got = self.hits("**/*.sh")
        self.assertNotIn("build/ignored.sh", got)
        self.assertNotIn("node_modules/pkg/index.sh", got)
        self.assertNotIn(".hidden/inside.sh", got)
        self.assertNotIn(".secret", self.hits("**/*"))

    def test_base_keeps_the_reporting_frame(self):
        frame = self.root.parent
        got = set(self.hits("**/*.sh", base=str(frame)))
        shown = {str(pathlib.Path(self.root.name, rel)) for rel in
                 ("sub/lane.sh", "sub/run.sh", "sub/deep/inner.sh",
                  "bench/serve_local.sh")}
        self.assertEqual(got, shown)
        self.assertTrue(os.path.isabs(str(frame)))

    def test_the_walk_root_is_independent_of_the_base_frame(self):
        # the pattern still applies to paths relative to the walk root
        self.assertEqual(self.hits("**/*bench_*", base=str(self.root.parent)),
                         [str(pathlib.Path(self.root.name, rel)) for rel in
                          ("bench_lane.py", "sub/bench_train.py", "sub/deep/bench_deep.py")])


class SingularCandidateSetTest(unittest.TestCase):
    """The read path's candidate set is the `**/*` walk: it must not move."""

    def test_star_star_star_matches_every_walk_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _build(root)
            walked = []
            for dirpath, files in ti.ops.filesys.impl._walk(str(root)):
                for name in files:
                    if name.startswith("."):
                        continue
                    walked.append(os.path.relpath(os.path.join(dirpath, name), str(root)))
            self.assertEqual(sorted(walked), sorted(FILES))
            token = confine.set_root(str(root))
            try:
                _kind, data, _conf = get("filesys_glob").run(
                    {"pattern": "**/*", "cwd": tmp}, store=None)
            finally:
                confine.reset_root(token)
            self.assertEqual(sorted(data), sorted(walked))


if __name__ == "__main__":
    unittest.main()
