"""The demo's rendering, held to the two things that actually broke while writing it.

`bench/demo.py` starts a real service, so the *scenes* are not a CI concern — but the
rendering is pure, and it had two defects worth a guard:

1. an artifact summary for a file read is the file's contents, newlines and all, so
   slicing it to 34 characters printed the rest of the file on its own line and broke
   the table;
2. `f"{colour(text):<16}"` pads *around* the ANSI escapes, so a coloured column counts
   the colour bytes as characters and drifts on exactly the terminal the demo is
   recorded on.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))


def _load_demo():
    spec = importlib.util.spec_from_file_location("ti_demo", ROOT / "bench" / "demo.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ti_demo"] = module
    spec.loader.exec_module(module)
    return module


demo = _load_demo()


class PadTest(unittest.TestCase):
    def test_padding_is_applied_to_the_plain_text_not_the_colour(self):
        # A coloured cell padded from outside counts the escape bytes as
        # characters and drifts by exactly that many columns; padded from inside
        # it does not. `_c` is a no-op when stdout is not a tty, so force it on.
        original = demo.USE_COLOUR
        demo.USE_COLOUR = True
        try:
            coloured = demo._c(demo._pad("1.00", 7, right=True), demo.GREEN)
        finally:
            demo.USE_COLOUR = original
        self.assertTrue(coloured.startswith(demo.GREEN))
        self.assertEqual(coloured[len(demo.GREEN):-len(demo.RESET)], "   1.00")
        self.assertEqual(len(demo._pad("1.00", 7, right=True)), 7)

    def test_right_and_left_padding(self):
        self.assertEqual(demo._pad("ms", 9, right=True), "       ms")
        self.assertEqual(demo._pad("op", 18), "op" + " " * 16)


class HopTableTest(unittest.TestCase):
    def _render(self, hops):
        import contextlib
        import io
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            demo._hop_table(hops)
        return buffer.getvalue().splitlines()

    def test_a_multi_line_artifact_summary_cannot_break_the_table(self):
        hops = [{"step": "content", "op": "filesys_read", "ms": 0.1, "conf": 1.0,
                 "artifact_summary": "file_content :: # Notes\n\nThe model runs.\n"}]
        lines = self._render(hops)
        # The invariant is "one row, no spill": the summary is clipped to its
        # column, and what matters is that its newlines never reach the terminal.
        self.assertEqual(len(lines), 2, lines)
        self.assertIn("file_content :: # Notes", lines[1])
        self.assertNotIn("\n", lines[1])

    def test_a_missing_confidence_renders_as_absent_not_zero(self):
        hops = [{"step": "queries", "op": "decide_extract", "ms": 0.2, "conf": None,
                 "artifact_summary": "value_list[3]"}]
        row = self._render(hops)[1]
        self.assertIn("—", row)
        self.assertNotIn("0.00", row)

    def test_no_hops_is_stated_rather_than_left_blank(self):
        self.assertEqual(len(self._render([])), 1)


class ResultPreviewTest(unittest.TestCase):
    def test_a_digest_is_summarised_not_dumped(self):
        out = {"result": {"sources": [{"url": "https://a.test"}, {"url": "https://b.test"}],
                          "queries": ["q1", "q2"], "skipped": []}}
        preview = demo._result_preview(out)
        self.assertIn("2 source(s)", preview)
        self.assertIn("queries: q1, q2", preview)
        self.assertIn("https://a.test", preview)
        self.assertNotIn("{", preview)

    def test_a_long_result_is_truncated_with_an_ellipsis(self):
        preview = demo._result_preview({"result": "x" * 500}, width=20)
        self.assertLessEqual(len(preview), 21)
        self.assertTrue(preview.endswith("…"))

    def test_a_path_list_shows_the_first_few(self):
        preview = demo._result_preview({"result": [f"f{i}.py" for i in range(20)]})
        self.assertIn("f0.py", preview)
        self.assertIn("…", preview)


if __name__ == "__main__":
    unittest.main()
