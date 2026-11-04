"""The hop log: its default, what a record holds, and the mode it leaves behind.

Three claims, each measured here rather than argued in a docstring
(`docs/logging-2026-11-04.md`):

**Off unless asked.** `hops_log.enabled` ships false, so a fresh clone records nothing,
and a disabled log creates no file even when it is handed a record.

**A record is the execution shape plus the utterance.** The dataset pipeline reads `op`
and `soft.intent_context`; the record carries those, argument *names* instead of values,
and measurements. It carries no artifact text — `artifact_summary` was the first 200
characters of whatever a hop read, and nothing consumed it — and the utterance is written
whole, where 100 characters of it used to be.

**The file is the owner's.** 0600 in a 0700 directory, a looser existing file narrowed
rather than refused, and a directory that cannot be narrowed means no write at all plus a
`reason` instead of an exception on the tool path.
"""
import json
import os
import pathlib
import stat
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))
sys.path.insert(0, str(ROOT))

from ti.conf import load  # noqa: E402
from ti.decide import primitives as prim  # noqa: E402
from ti.decide.cache import clear as clear_decisions  # noqa: E402
from ti.decide.primitives import ChoiceAnswer, DecideBackend  # noqa: E402
from ti.hops import log as hop_log  # noqa: E402
from ti.hops.log import HopLog  # noqa: E402
from ti.ops.op_api import import_all  # noqa: E402
from ti.service_entry import use_tool  # noqa: E402
from tools.dataset.paths import logs_dir  # noqa: E402

DEFAULTS = ROOT / "service" / "ti" / "conf" / "defaults.json"


def mode_of(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def records(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class HopLogCase(unittest.TestCase):
    """Mutates the cached `hops_log` section in place, and puts it back."""

    def setUp(self):
        self.cfg = load()["hops_log"]
        self._saved = dict(self.cfg)
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="ti-hop-log-"))
        self.addCleanup(self._restore)

    def _restore(self):
        self.cfg.clear()
        self.cfg.update(self._saved)

    def enable(self, path: pathlib.Path | None = None) -> pathlib.Path:
        target = pathlib.Path(path or self.tmp / "logs" / "hops.jsonl")
        self.cfg.update({"enabled": True, "path": str(target)})
        return target


class ShippedDefaultTest(HopLogCase):
    """The default a stranger gets, read from the shipped file, not from a mutated cache."""

    def setUp(self):
        super().setUp()
        self.defaults = json.loads(DEFAULTS.read_text(encoding="utf-8"))["hops_log"]

    def test_the_log_is_off(self):
        self.assertIs(self.defaults["enabled"], False)

    def test_nothing_is_written_while_it_is_off(self):
        target = self.tmp / "off" / "hops.jsonl"
        self.cfg.update({"enabled": False, "path": str(target)})
        log = HopLog()
        self.assertFalse(log.enabled)
        log.append({"op": "filesys_glob"})
        self.assertFalse(target.exists())

    def test_the_path_is_the_users_state_directory_not_the_working_directory(self):
        shipped = os.path.expanduser(self.defaults["path"])
        self.assertFalse(os.path.isabs(self.defaults["path"]), "ship a portable path, not $HOME")
        self.assertTrue(os.path.isabs(shipped), shipped)
        self.assertEqual(pathlib.Path(shipped).parent, pathlib.Path.home() / ".local/state/ti/logs")

    def test_the_dataset_pipeline_reads_the_directory_the_writer_uses(self):
        shipped = pathlib.Path(os.path.expanduser(self.defaults["path"])).parent
        self.assertEqual(os.path.abspath(logs_dir()), os.path.abspath(shipped))


class RecordShapeTest(HopLogCase):
    """One real L1 run — glob, pick, read — and what it leaves in the log."""

    INTENT = "open the alpha text file please"
    MARKER = "SECRET-CONTENT-MARKER"

    def setUp(self):
        super().setUp()
        import_all()
        self.tree = self.tmp / "tree"
        self.tree.mkdir()
        (self.tree / "alpha.txt").write_text(self.MARKER + "\n", encoding="utf-8")
        self.path = self.enable()

    def _pin_pick(self):
        """Point the `choice` decision at a stub so the run needs no model."""

        class Counter(DecideBackend):
            name = "hop_log_counter"

            def choice(self, state, instructions, criteria):
                pick = "alpha.txt" if "alpha.txt" in criteria else next(iter(criteria))
                rest = max(len(criteria) - 1, 1)
                probs = {k: (0.95 if k == pick else 0.05 / rest) for k in criteria}
                return ChoiceAnswer(pick, probs, 0.95)

            def noul(self, state, question):
                return 0.9

        decide = load()["decide"]
        saved = decide["backends_in_use"]["choice"]
        prim._REGISTRY["hop_log_counter"] = Counter
        decide["backends_in_use"]["choice"] = "hop_log_counter"
        clear_decisions()
        self.addCleanup(self._unpin, decide, saved)

    def _unpin(self, decide, saved):
        decide["backends_in_use"]["choice"] = saved
        prim._REGISTRY.pop("hop_log_counter", None)
        clear_decisions()

    def run_intent(self, intent=None):
        return use_tool({"intent": intent or self.INTENT,
                         "digest": {"cwd": str(self.tree)}})

    def test_a_read_hop_logs_no_bytes_of_the_file_it_read(self):
        self._pin_pick()
        out = self.run_intent()
        self.assertEqual(out["status"], "done", out)
        self.assertIn("filesys_read", [h["op"] for h in out["hops"]])
        hops = records(self.path)
        self.assertIn("filesys_read", [r["op"] for r in hops])
        self.assertNotIn(self.MARKER, self.path.read_text(encoding="utf-8"))
        for rec in hops:
            self.assertNotIn("artifact_summary", rec)

    def test_a_record_is_the_trace_and_the_utterance_and_nothing_else(self):
        self._pin_pick()
        self.run_intent()
        written = records(self.path)
        for rec in written:
            self.assertLessEqual(set(rec), {"conf", "kind", "ms", "op", "params_keys", "soft", "ts"})
            # `soft` rides along only on the hops whose step saw the caller's frame, and
            # it holds the utterance alone — never the other soft hints (paths, answers).
            if "soft" in rec:
                self.assertEqual(sorted(rec["soft"]), ["intent_context"])
                self.assertEqual(rec["soft"]["intent_context"], self.INTENT)
        self.assertTrue(any("soft" in rec for rec in written), written)

    def test_argument_names_are_logged_and_argument_values_are_not(self):
        self._pin_pick()
        self.run_intent()
        glob = [r for r in records(self.path) if r["op"] == "filesys_glob"][0]
        self.assertIn("cwd", glob["params_keys"])
        self.assertNotIn(str(self.tree), self.path.read_text(encoding="utf-8"))

    def test_a_long_utterance_is_not_clipped(self):
        self._pin_pick()
        intent = ("open the alpha text file please, the one with the marker in it, "
                  "and just tell me what the first line says without pasting the rest")
        self.assertGreater(len(intent), 100)
        self.run_intent(intent)
        logged = [r["soft"]["intent_context"] for r in records(self.path) if "soft" in r]
        self.assertIn(intent, logged)

    def test_the_default_run_writes_no_log_at_all(self):
        self.cfg["enabled"] = False
        self._pin_pick()
        out = self.run_intent()
        self.assertEqual(out["status"], "done", out)
        self.assertFalse(self.path.exists())


class FileModeTest(HopLogCase):
    def test_a_new_file_is_0600_in_a_0700_directory(self):
        path = self.enable()
        HopLog().append({"op": "filesys_glob"})
        self.assertEqual(mode_of(path), 0o600)
        self.assertEqual(mode_of(path.parent), 0o700)

    def test_a_wider_existing_file_is_narrowed_and_appended_to(self):
        path = self.tmp / "loose" / "hops.jsonl"
        path.parent.mkdir(mode=0o755)
        os.chmod(path.parent, 0o755)
        path.write_text('{"op": "earlier"}\n', encoding="utf-8")
        os.chmod(path, 0o644)
        self.enable(path)
        log = HopLog()
        log.append({"op": "filesys_glob"})
        self.assertIsNone(log.reason)
        self.assertEqual(mode_of(path), 0o600)
        self.assertEqual(mode_of(path.parent), 0o700)
        self.assertEqual([r["op"] for r in records(path)], ["earlier", "filesys_glob"])

    def test_a_loose_directory_is_narrowed_before_the_first_write(self):
        directory = self.tmp / "wide"
        directory.mkdir(mode=0o777)
        os.chmod(directory, 0o777)
        path = self.enable(directory / "hops.jsonl")
        HopLog().append({"op": "filesys_glob"})
        self.assertEqual(mode_of(directory), 0o700)
        self.assertEqual(mode_of(path), 0o600)

    def test_a_directory_that_cannot_be_narrowed_gets_no_write_and_a_reason(self):
        directory = self.tmp / "theirs"
        directory.mkdir()
        os.chmod(directory, 0o777)
        path = self.enable(directory / "hops.jsonl")
        log = HopLog()
        denied = PermissionError(1, "Operation not permitted", str(directory))
        with mock.patch.object(hop_log.os, "chmod", side_effect=denied), redirect_stderr(StringIO()) as err:
            log.append({"op": "filesys_glob"})
        self.assertFalse(path.exists())
        self.assertIn("hop log off", str(log.reason))
        self.assertIn("hop log off", err.getvalue())   # said once, not silently

    def test_an_unwritable_place_never_raises_on_the_tool_path(self):
        directory = self.tmp / "read-only"
        directory.mkdir()
        os.chmod(directory, 0o500)
        self.addCleanup(os.chmod, directory, 0o700)
        self.enable(directory / "hops.jsonl")
        log = HopLog()
        with redirect_stderr(StringIO()):
            log.append({"op": "filesys_glob"})      # the call must not raise
        self.assertIsNotNone(log.reason)
        self.assertFalse((directory / "hops.jsonl").exists())

    def test_a_symlink_at_the_log_path_is_not_written_through(self):
        target = self.tmp / "elsewhere.txt"
        target.write_text("", encoding="utf-8")
        path = self.enable(self.tmp / "link.jsonl")
        path.symlink_to(target)
        log = HopLog()
        with redirect_stderr(StringIO()):
            log.append({"op": "filesys_glob"})
        self.assertEqual(target.read_text(encoding="utf-8"), "")
        self.assertIsNotNone(log.reason)


if __name__ == "__main__":
    unittest.main()
