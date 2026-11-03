"""The cast→SVG renderer, held to the two defects that showed up in a real render.

Both were visible in the first picture this produced, which is the only reason they are
worth a test:

1. **Runs positioned by a guessed character width drift.** A `<text x=pad + col*w>` only
   aligns if `w` equals the font's real advance; it does not, and the confidence column
   landed ~200 px to the right of its header. Runs are `<tspan>`s inside one `<text>` per
   line now, so a monospace font aligns itself.
2. **The last screenful is not the picture.** The demo prints more lines than a terminal
   shows, so the "final frame" was the footer wedged mid-scene. Every line is rendered.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))


def _load():
    spec = importlib.util.spec_from_file_location("cast_to_svg", ROOT / "bench" / "cast_to_svg.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cast_to_svg"] = module
    spec.loader.exec_module(module)
    return module


svg = _load()


def _write_cast(lines: list[str]) -> pathlib.Path:
    """A minimal v2 cast: one output event per line, all at t=0."""
    handle = tempfile.NamedTemporaryFile("w", suffix=".cast", delete=False)
    handle.write(json.dumps({"version": 2, "width": 100, "height": 40}) + "\n")
    for line in lines:
        handle.write(json.dumps([0.0, "o", line]) + "\n")
    handle.close()
    return pathlib.Path(handle.name)


class StyledLinesTest(unittest.TestCase):
    def test_sgr_is_parsed_into_runs(self):
        cast = _write_cast(["\x1b[32mgreen\x1b[0m plain\n"])
        try:
            lines = svg.styled_lines(cast)
        finally:
            cast.unlink()
        runs = [r for r in lines[0] if r.text]
        self.assertEqual(runs[0].text, "green")
        self.assertEqual(runs[0].style.fg, svg.PALETTE[32])
        self.assertEqual(runs[1].text, " plain")
        self.assertEqual(runs[1].style.fg, svg.FOREGROUND)

    def test_bold_and_dim_survive_a_reset(self):
        cast = _write_cast(["\x1b[1;2mhot\x1b[22mcold\n"])
        try:
            lines = svg.styled_lines(cast)
        finally:
            cast.unlink()
        runs = [r for r in lines[0] if r.text]
        self.assertTrue(runs[0].style.bold and runs[0].style.dim)
        self.assertFalse(runs[1].style.bold or runs[1].style.dim,
                         "SGR 22 clears bold and dim together")


class WrapTest(unittest.TestCase):
    def test_a_long_run_is_broken_at_the_column_and_keeps_its_style(self):
        line = [svg.Run("x" * 25, svg.Style(svg.PALETTE[32], False, False))]
        wrapped = svg.wrap([line], 10)
        self.assertEqual(len(wrapped), 3)
        self.assertEqual([len(r.text) for r in wrapped[0]], [10])
        self.assertEqual(wrapped[2][0].style.fg, svg.PALETTE[32])

    def test_a_short_line_is_untouched(self):
        line = [svg.Run("hello", svg.Style())]
        self.assertEqual(len(svg.wrap([line], 80)), 1)


class RenderTest(unittest.TestCase):
    def test_runs_are_tspans_inside_one_text_element_per_line(self):
        lines = [[svg.Run("a", svg.Style(svg.PALETTE[31], True, False)),
                  svg.Run("b", svg.Style())],
                 [svg.Run("second", svg.Style())]]
        out = svg.render_svg(lines, 40)
        self.assertEqual(out.count("<text"), 2, "one <text> per line, not per run")
        self.assertNotIn("<text", out.split("<text")[2] if out.count("<text") > 1 else "")
        self.assertEqual(out.count("<tspan"), 3)
        self.assertIn('fill="' + svg.PALETTE[31] + '"', out)
        self.assertIn('font-weight="600"', out)

    def test_every_line_is_rendered_not_just_the_last_screenful(self):
        lines = [[svg.Run(f"line {i}", svg.Style())] for i in range(80)]
        out = svg.render_svg(lines, 40)
        self.assertIn("line 0", out)
        self.assertIn("line 79", out)

    def test_markup_characters_are_escaped(self):
        lines = [[svg.Run("a < b & c > d", svg.Style())]]
        out = svg.render_svg(lines, 40)
        self.assertIn("&lt;", out)
        self.assertIn("&amp;", out)
        self.assertIn("&gt;", out)

    def test_a_blank_line_emits_no_empty_text_element(self):
        lines = [[svg.Run("   ", svg.Style())], [svg.Run("real", svg.Style())]]
        out = svg.render_svg(lines, 40)
        self.assertEqual(out.count("<text"), 1)

    def test_the_document_is_standalone(self):
        out = svg.render_svg([[svg.Run("x", svg.Style())]], 20)
        self.assertTrue(out.startswith("<svg "))
        self.assertTrue(out.rstrip().endswith("</svg>"))
        # `xmlns="http://www.w3.org/2000/svg"` is a namespace name, not a fetch.
        body = out.replace('xmlns="http://www.w3.org/2000/svg"', "")
        for external in ("href", "@import", "url("):
            self.assertNotIn(external, body)


if __name__ == "__main__":
    unittest.main()
