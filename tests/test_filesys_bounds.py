"""The filesys ops' bounds: confinement, the deny list, every cap, and the regex.

Each case is one line of the independent review of 2026-11-03 (B4) turned into an
assertion: a walk rooted wherever the caller said with no depth, hit or time cap;
`params.path` naming a file outside the run; a caller-supplied regex compiled and
run with no budget; and a `conf` of 1.0 no matter how little of the tree the
answer covered.

The cases drive the ops directly — the confinement and the caps live in the ops,
so that is where they are pinned. The HTTP layer's own cases (token, allowlist,
status codes) are in test_http_bounds.py.
"""
import copy
import json
import os
import pathlib
import shutil
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))

import ti.ops.filesys.impl  # noqa: E402,F401  (registers the ops)
from ti.conf import load  # noqa: E402
from ti.ops.filesys import confine, regexguard, walk as walk_mod  # noqa: E402
from ti.ops.op_api import get  # noqa: E402


def _write(path: pathlib.Path, body: str = "x") -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


class _Clock:
    """A monotonic clock that advances a fixed step per read: no test sleeps."""

    def __init__(self, step: float = 1.0) -> None:
        self.now = 0.0
        self.step = step

    def monotonic(self) -> float:
        self.now += self.step
        return self.now


class FilesysCase(unittest.TestCase):
    """A tmp tree as the run root, with `ops.filesys` restored afterwards."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = pathlib.Path(self._tmp.name)
        token = confine.set_root(str(self.root))
        self.addCleanup(confine.reset_root, token)
        self._filesys = copy.deepcopy(load()["ops"]["filesys"])
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        load()["ops"]["filesys"] = self._filesys

    def set_conf(self, **values) -> None:
        section = load()["ops"]["filesys"]
        for key, value in values.items():
            if isinstance(value, dict) and isinstance(section.get(key), dict):
                section[key] = {**section[key], **value}
            else:
                section[key] = value

    def glob(self, pattern: str, **params) -> tuple[list[str], float]:
        _kind, data, conf = get("filesys_glob").run(
            {"pattern": pattern, "cwd": str(self.root), **params}, store=None)
        return list(data), conf

    def read(self, path: str, **params) -> tuple[str, float]:
        _kind, data, conf = get("filesys_read").run({"path": path, **params}, store=None)
        return str(data), conf

    def grep(self, pattern: str, **params) -> tuple[list[dict], float]:
        _kind, data, conf = get("filesys_grep").run(
            {"pattern": pattern, "cwd": str(self.root), **params}, store=None)
        return list(data), conf


class ConfineTest(FilesysCase):
    """A path outside the run root is refused, however it was written."""

    def test_a_sibling_path_is_refused(self):
        _write(self.root / "inside.txt")
        with self.assertRaises(confine.OutsideRoot) as caught:
            self.read(str(self.root.parent / "outside.txt"))
        self.assertIn("outside the permitted root", str(caught.exception))

    def test_dotdot_traversal_is_refused(self):
        _write(self.root / "inside.txt")
        with self.assertRaises(confine.OutsideRoot) as caught:
            self.read("../escaped.txt")
        self.assertIn("escaped.txt", str(caught.exception))

    def test_dotdot_out_of_a_subdirectory_is_refused(self):
        _write(self.root / "sub" / "inside.txt")
        with self.assertRaises(confine.OutsideRoot):
            self.read("sub/../../escaped.txt")

    def test_a_symlink_out_of_the_root_is_refused(self):
        with tempfile.TemporaryDirectory() as other:
            _write(pathlib.Path(other) / "secret.txt", "secret")
            os.symlink(other, self.root / "link")
            with self.assertRaises(confine.OutsideRoot) as caught:
                self.read("link/secret.txt")
            self.assertIn("symlink", str(caught.exception))

    def test_a_relative_path_resolves_against_the_run_root(self):
        _write(self.root / "sub" / "inside.txt", "content")
        text, conf = self.read("sub/inside.txt")
        self.assertEqual(text, "content")
        self.assertEqual(conf, 1.0)

    def test_the_run_root_is_where_use_tool_says_it_is(self):
        from ti.service_entry import use_tool
        _write(self.root / "sub" / "target.txt", "resolved by the run root")
        out = use_tool({"intent": "x", "digest": {"cwd": str(self.root)},
                        "params": {"tool": "filesys_read", "args": {"path": "sub/target.txt"}}})
        self.assertEqual(out["status"], "done", out)
        self.assertIn("resolved by the run root", str(out["result"]))


class PolicyRootsTest(unittest.TestCase):
    """`ops.filesys.roots` is what a run root must live inside — and its default.

    The default is `["~", "."]`, and `.` is the load-bearing half: it is the
    directory the service was started in. Without it a clone that does not live
    under the home directory cannot run the README's first example at all
    (`docs/quickstart-stranger-2026-11-03.md` §1 measured exactly that: a clone
    in `/tmp` answered `403 scope_refused` for every documented call).

    Widening it does not widen what a request may name: the requested cwd still
    has to resolve *inside* one of the roots, so `/etc` is refused unless the
    service happens to have been started there.
    """

    def setUp(self) -> None:
        self._filesys = copy.deepcopy(load()["ops"]["filesys"])
        self.addCleanup(self._restore)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _restore(self) -> None:
        load()["ops"]["filesys"] = self._filesys

    def test_the_shipped_default_covers_home_and_the_start_directory(self):
        shipped = json.loads(
            (ROOT / "service" / "ti" / "conf" / "defaults.json").read_text()
        )["ops"]["filesys"]["roots"]
        self.assertEqual(shipped, ["~", "."])

    def test_the_start_directory_is_a_permitted_root(self):
        # A clone anywhere works, because the service's own cwd is a root.
        cwd = os.getcwd()
        load()["ops"]["filesys"]["roots"] = ["~", "."]
        self.assertIn(os.path.realpath(cwd), confine.policy_roots())
        self.assertEqual(confine.confined_root(self._tmp.name)
                         if self._tmp.name.startswith(os.path.realpath(cwd))
                         else confine.confined_root("."),
                         os.path.realpath("."))

    def test_a_cwd_outside_every_root_is_still_refused(self):
        load()["ops"]["filesys"]["roots"] = ["~", "."]
        outside = tempfile.mkdtemp(prefix="ti-outside-")
        self.addCleanup(shutil.rmtree, outside, True)
        if os.path.realpath(outside).startswith(os.path.realpath(".")):
            self.skipTest("the temp dir is inside the start directory; no escape to test")
        with self.assertRaises(confine.PathRefused):
            confine.confined_root(outside)


class DenyListTest(FilesysCase):
    """A configured deny root is neither descended into nor nameable."""

    def setUp(self) -> None:
        super().setUp()
        self.dead = self.root / "mnt"
        _write(self.dead / "secret.txt", "must not be listed")
        _write(self.root / "visible.txt")
        self.set_conf(deny_roots=[str(self.dead)])

    def test_the_walk_does_not_descend_into_a_denied_root(self):
        hits, conf = self.glob("**/*")
        self.assertIn("visible.txt", hits)
        self.assertNotIn("mnt/secret.txt", hits)
        # not a complete listing: it pruned a tree, and says so
        self.assertEqual(conf, 0.7)

    def test_the_walk_records_what_it_refused(self):
        state = walk_mod.WalkState()
        listed = [files for _dirpath, files, _depth
                  in walk_mod.walk(str(self.root), walk_mod.Limits.from_conf(), state)]
        self.assertNotIn("secret.txt", [name for files in listed for name in files])
        self.assertEqual(state.refused, [str(self.dead)])

    def test_naming_a_denied_root_is_refused_rather_than_emptied(self):
        with self.assertRaises(confine.DeniedPath) as caught:
            get("filesys_glob").run({"pattern": "**/*", "cwd": str(self.dead)}, store=None)
        self.assertIn("denied root", str(caught.exception))

    def test_a_file_inside_a_denied_root_is_refused(self):
        with self.assertRaises(confine.DeniedPath):
            self.read("mnt/secret.txt")


class WalkBoundTest(FilesysCase):
    """Depth, file count and wall clock each stop the walk, and conf admits it."""

    def test_the_depth_cap_stops_the_descent(self):
        _write(self.root / "top.txt")
        _write(self.root / "a" / "one.txt")
        _write(self.root / "a" / "b" / "two.txt")
        _write(self.root / "a" / "b" / "c" / "three.txt")
        self.set_conf(max_depth=1)
        hits, conf = self.glob("**/*")
        self.assertIn("top.txt", hits)
        self.assertIn("a/one.txt", hits)
        self.assertNotIn("a/b/two.txt", hits)
        self.assertEqual(conf, 0.7)

    def test_the_hit_cap_stops_the_listing(self):
        for index in range(10):
            _write(self.root / f"file{index}.txt")
        self.set_conf(glob={"max_hits": 3})
        hits, conf = self.glob("**/*")
        self.assertEqual(len(hits), 3)
        self.assertEqual(conf, 0.7)

    def test_the_file_cap_stops_the_walk(self):
        for index in range(10):
            _write(self.root / f"file{index}.txt")
        state = walk_mod.WalkState()
        limits = walk_mod.Limits(max_depth=8, max_files=4, deadline_s=60.0)
        listed = [name for _dirpath, files, _depth
                  in walk_mod.walk(str(self.root), limits, state) for name in files]
        self.assertEqual(state.stopped, walk_mod.MAX_FILES)
        self.assertLessEqual(len(listed), 10)

    def test_the_deadline_stops_the_walk(self):
        for index in range(20):
            _write(self.root / f"file{index}.txt")
        clock = _Clock(step=1.0)
        fake = types.SimpleNamespace(monotonic=clock.monotonic)
        with mock.patch("ti.ops.filesys.walk.time", fake):
            hits, conf = self.glob("**/*")
        self.assertLess(len(hits), 20)
        self.assertEqual(conf, 0.5)

    def test_a_complete_walk_reports_one(self):
        _write(self.root / "one.txt")
        _write(self.root / "sub" / "two.txt")
        hits, conf = self.glob("**/*")
        self.assertEqual(sorted(hits), ["one.txt", "sub/two.txt"])
        self.assertEqual(conf, 1.0)


class ReadBoundTest(FilesysCase):
    """A read is capped in bytes as well as lines, and says when it truncated."""

    def test_the_byte_cap_truncates(self):
        _write(self.root / "big.txt", "0123456789" * 10)
        self.set_conf(read={"max_lines": 100, "max_bytes": 25})
        text, conf = self.read("big.txt")
        self.assertEqual(len(text), 25)
        self.assertEqual(conf, 0.7)

    def test_the_line_cap_truncates(self):
        _write(self.root / "lines.txt", "".join(f"line{i}\n" for i in range(10)))
        text, conf = self.read("lines.txt", max_lines=3)
        self.assertEqual(len(text.splitlines()), 3)
        self.assertEqual(conf, 0.7)

    def test_a_caller_cannot_raise_the_line_cap(self):
        _write(self.root / "lines.txt", "".join(f"line{i}\n" for i in range(20)))
        self.set_conf(read={"max_lines": 4, "max_bytes": 10000})
        text, _conf = self.read("lines.txt", max_lines=500)
        self.assertEqual(len(text.splitlines()), 4)


class GrepBoundTest(FilesysCase):
    """A pattern is refused or bounded before `re` is asked to run it."""

    def test_an_over_long_pattern_is_refused(self):
        self.set_conf(grep={"pattern_max_chars": 20})
        _write(self.root / "a.txt", "a" * 41)
        with self.assertRaises(regexguard.UnsafePattern) as caught:
            self.grep("a" * 21)
        self.assertIn("the limit is 20", str(caught.exception))

    def test_the_redos_case_is_refused(self):
        # the review's measured pattern: still running at a 25s kill over 41 bytes
        _write(self.root / "a.txt", "a" * 41)
        with self.assertRaises(regexguard.UnsafePattern) as caught:
            self.grep("(a+)+$")
        self.assertIn("nested quantifier", str(caught.exception))

    def test_adjacent_overlapping_quantifiers_are_refused(self):
        _write(self.root / "a.txt", "a" * 41)
        with self.assertRaises(regexguard.UnsafePattern) as caught:
            self.grep("a*a*b")
        self.assertIn("adjacent overlapping quantifiers", str(caught.exception))

    def test_an_ambiguous_alternation_is_refused(self):
        _write(self.root / "a.txt", "a" * 41)
        with self.assertRaises(regexguard.UnsafePattern) as caught:
            self.grep("(a|aa)+b")
        self.assertIn("alternation", str(caught.exception))

    def test_a_match_far_into_a_long_line_is_still_found(self):
        # 200k characters on one line: `re` only ever sees a window of it, and the
        # window walk still reaches the match at the end
        _write(self.root / "wide.txt", "a" * 200000 + "needle\n")
        started = time.monotonic()
        hits, _conf = self.grep("needle")
        elapsed = time.monotonic() - started
        self.assertEqual(len(hits), 1)
        self.assertLess(elapsed, 2.0, "the windowed match did not stay bounded")
        self.assertLessEqual(len(hits[0]["excerpt"]), 200)

    def test_an_accepted_superlinear_pattern_stays_fast_on_a_long_line(self):
        # `\w+x` over 200k characters would be quadratic without a bound on the
        # subject; each window is at most line_max_chars wide
        _write(self.root / "wide.txt", "a" * 200000 + "\n")
        started = time.monotonic()
        hits, _conf = self.grep(r"\w+x")
        elapsed = time.monotonic() - started
        self.assertEqual(hits, [])
        self.assertLess(elapsed, 2.0)

    def test_the_hit_cap_stops_the_search(self):
        for index in range(6):
            _write(self.root / f"hit{index}.txt", "needle\n")
        self.set_conf(grep={"max_hits": 2})
        hits, conf = self.grep("needle")
        self.assertEqual(len(hits), 2)
        self.assertEqual(conf, 0.7)

    def test_the_name_filter_is_a_glob_not_a_regex(self):
        # it is translated by fnmatch, so its metacharacters are escaped and it
        # can never be the blowup the search pattern can: this matches nothing
        # rather than raising, and rather than running as a regex
        _write(self.root / "a.py", "needle\n")
        hits, _conf = self.grep("needle", glob="(a+)+")
        self.assertEqual(hits, [])

    def test_the_name_filter_is_length_capped(self):
        _write(self.root / "a.py", "needle\n")
        self.set_conf(grep={"pattern_max_chars": 10})
        with self.assertRaises(regexguard.UnsafePattern) as caught:
            self.grep("needle", glob="abcdefghijkl")
        self.assertIn("the limit is 10", str(caught.exception))

    def test_the_name_filter_still_filters(self):
        _write(self.root / "a.py", "needle\n")
        _write(self.root / "b.txt", "needle\n")
        hits, _conf = self.grep("needle", glob="*.py")
        self.assertEqual([hit["path"] for hit in hits], ["a.py"])


if __name__ == "__main__":
    unittest.main()
