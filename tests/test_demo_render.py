"""The demo's rendering, held to the things that actually broke while writing it.

`bench/demo.py` starts a real service, so the *scenes* are not a CI concern — but the
rendering is pure, and it had two defects worth a guard:

1. an artifact summary for a file read is the file's contents, newlines and all, so
   slicing it to 34 characters printed the rest of the file on its own line and broke
   the table;
2. `f"{colour(text):<16}"` pads *around* the ANSI escapes, so a coloured column counts
   the colour bytes as characters and drifts on exactly the terminal the demo is
   recorded on.

The rest of it guards the claim the demo exists to make. The header says which decision
backend answered and whether the CLIs were fixtures; that is only worth reading if it is
derived from the config the service actually boots, so `OfflineClaimTest` pushes the
demo's own overlay through `ti.conf` — the loader the service uses — instead of trusting
a header string to describe it. If the default mode ever quietly starts deciding with a
real model, that test fails.

`CommittedArtefactsTest` keeps the shipped picture honest: the SVG must be a render of
the cast in the repo, and the PNG a render of that SVG.
"""
from __future__ import annotations

import contextlib
import html
import importlib.util
import io
import json
import math
import os
import pathlib
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT / "service"))

CAST = ROOT / "demo" / "toolinterfacer.cast"
SVG = ROOT / "demo" / "toolinterfacer.svg"
PNG = ROOT / "demo" / "toolinterfacer.png"
RENDER_COLS = 112           # what demo/README.md records the SVG as rendered at
PNG_WIDTH = 1200


def _load_demo():
    spec = importlib.util.spec_from_file_location("ti_demo", ROOT / "bench" / "demo.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ti_demo"] = module
    spec.loader.exec_module(module)
    return module


def _load_cast():
    spec = importlib.util.spec_from_file_location("ti_cast_to_svg",
                                                  ROOT / "bench" / "cast_to_svg.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ti_cast_to_svg"] = module
    spec.loader.exec_module(module)
    return module


demo = _load_demo()
cast = _load_cast()


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


class DecideHeaderTest(unittest.TestCase):
    """The header's words per backend — the demo's only claim about what answered."""

    def test_a_fixture_is_always_labelled_a_recorded_answer(self):
        note = demo._backend_note("fixture")
        self.assertIn("recorded", note)
        self.assertNotIn("real", note)

    def test_keyword_rules_are_not_called_a_model(self):
        self.assertIn("no model", demo._backend_note("rule_extract"))

    def test_anything_else_is_named_the_configured_backend(self):
        for name in ("pair", "cascade", "openrouter"):
            self.assertIn("configured", demo._backend_note(name))

    def test_every_configured_kind_gets_a_row_with_choice_first(self):
        rows = demo._decide_rows({"extract": "rule_extract", "choice": "pair",
                                  "noul": "fixture", "other": "whatever"})
        self.assertEqual([r[0] for r in rows],
                         ["choice", "noul", "extract", "other"])
        self.assertEqual({r[0]: r[1] for r in rows}["noul"], "fixture")

    def test_the_fixture_kind_is_the_one_labelled_recorded(self):
        rows = demo._decide_rows({"choice": "fixture", "extract": "rule_extract"})
        self.assertEqual({kind: note for kind, _, note in rows}["choice"],
                         demo._backend_note("fixture"))

    def test_an_empty_map_produces_no_rows(self):
        self.assertEqual(demo._decide_rows({}), [])


class OfflineClaimTest(unittest.TestCase):
    """The default mode really is offline, and `--live` really is the shipped conf.

    The header's promise ("recorded answers — deterministic, offline") is a claim
    about the config the service boots with. Both cases below push the demo's
    overlay through `ti.conf`, so this cannot pass on a header that drifted away
    from the overlay: change the overlay to a real backend and it fails.
    """

    def setUp(self) -> None:
        from ti import conf
        self.conf = conf
        self._saved = os.environ.get("TI_CONF_OVERLAY")
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        if self._saved is None:
            os.environ.pop("TI_CONF_OVERLAY", None)
        else:
            os.environ["TI_CONF_OVERLAY"] = self._saved
        self.conf.reload()

    def _merged(self, live: bool) -> dict:
        service = demo.Service(live=live)
        self.addCleanup(shutil.rmtree, service.tmp, True)
        path = service.tmp / "overlay.json"
        path.write_text(json.dumps(service.overlay()), encoding="utf-8")
        os.environ["TI_CONF_OVERLAY"] = str(path)
        return self.conf.reload()

    def test_the_default_mode_cannot_reach_a_model(self):
        merged = self._merged(live=False)
        in_use = merged["decide"]["backends_in_use"]
        self.assertEqual(in_use["choice"], "fixture")
        for kind, name in in_use.items():
            self.assertIn(name, {"fixture", "rule_extract"},
                          f"the default mode must not decide {kind} with {name}")
        # A cache hit answers from an earlier run's decision, so it is off too.
        self.assertFalse(merged["decide"]["cache"]["enabled"])
        # …and the web ops, which are the other way a default run leaves the box.
        for op in ("web_search", "web_fetch"):
            self.assertEqual(merged["ops"][op]["script"], str(demo.FAKE_CLI), op)

    def test_live_puts_the_shipped_config_back(self):
        merged = self._merged(live=True)
        shipped = json.loads((ROOT / "service" / "ti" / "conf" / "defaults.json")
                             .read_text(encoding="utf-8"))
        self.assertEqual(merged["decide"]["backends_in_use"],
                         shipped["decide"]["backends_in_use"])
        self.assertNotEqual(merged["decide"]["backends_in_use"]["choice"], "fixture",
                            "--live must not keep the fixture backend")

    def test_live_leaves_the_configured_clis_alone(self):
        service = demo.Service(live=True)
        self.addCleanup(shutil.rmtree, service.tmp, True)
        self.assertNotIn("ops", service.overlay(),
                         "--live must not repoint the web ops at the fixtures")

    def test_the_overlay_only_ever_sets_the_port_and_the_hop_log(self):
        for live in (False, True):
            service = demo.Service(live=live)
            self.addCleanup(shutil.rmtree, service.tmp, True)
            self.assertTrue({"http", "hops_log"} <= set(service.overlay()))


class HonestReportTest(unittest.TestCase):
    """A scene that got no decision is described, not left to a red status line."""

    def _render(self, bodies):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            demo._honest_report(bodies)
        return buffer.getvalue()

    def test_a_clean_run_says_nothing(self):
        self.assertEqual(self._render([{"status": "done"}, {"status": "done"}]), "")

    def test_asking_back_is_not_reported_as_a_failure(self):
        self.assertEqual(self._render([{"status": "need_input"}]), "")

    def test_an_error_names_the_scene_the_reason_and_the_way_out(self):
        text = self._render([{"status": "done"},
                             {"status": "error",
                              "error": "pair: no stage produced a choice"}])
        self.assertIn("scene 2", text)
        self.assertIn("pair: no stage produced a choice", text)
        self.assertIn("no decision was made", text)
        self.assertIn("reachable", text)

    def test_a_repeated_reason_is_printed_once(self):
        text = self._render([{"status": "error", "error": "boom"},
                             {"status": "error", "error": "boom"}])
        self.assertEqual(text.count("the service said"), 1)
        self.assertIn("scenes 1, 2", text)


class CastRobustnessTest(unittest.TestCase):
    """`cast_to_svg` on the casts a recorder can actually leave behind."""

    def _cast(self, events, *, header=True) -> pathlib.Path:
        handle = tempfile.NamedTemporaryFile("w", suffix=".cast", delete=False,
                                             encoding="utf-8")
        if header:
            handle.write(json.dumps({"version": 2, "width": 100, "height": 40}) + "\n")
        for event in events:
            handle.write((event if isinstance(event, str) else json.dumps(event)) + "\n")
        handle.close()
        path = pathlib.Path(handle.name)
        self.addCleanup(path.unlink)
        return path

    def _svg(self, path: pathlib.Path, cols: int = RENDER_COLS) -> str:
        return cast.render_svg(cast.wrap(cast.styled_lines(path), cols), cols)

    def test_a_cast_with_no_output_events_still_renders(self):
        out = self._svg(self._cast([]))
        self.assertTrue(out.startswith("<svg "))
        self.assertTrue(out.rstrip().endswith("</svg>"))
        self.assertNotIn("<text", out)

    def test_a_cast_whose_only_event_is_not_output_still_renders(self):
        out = self._svg(self._cast([[0.1, "i", "echo hi\r\n"], [0.2, "m", "mark"]]))
        self.assertNotIn("<text", out)
        self.assertNotIn("echo hi", out)

    def test_an_empty_file_does_not_raise(self):
        self.assertTrue(self._svg(self._cast([], header=False)).startswith("<svg "))

    def test_unicode_box_drawing_survives_the_render(self):
        art = "┌───┐\n│ ✓ ─ · — ╬ │\n└───┘\n"
        out = self._svg(self._cast([[0.0, "o", art]]))
        for piece in ("┌───┐", "│ ✓ ─ · — ╬ │", "└───┘"):
            self.assertIn(piece, out)

    def test_a_runaway_cast_is_cut_at_the_guard(self):
        events = [[0.0, "o", f"line {i:03d}\n"] for i in range(500)]
        self.assertEqual(len(cast.styled_lines(self._cast(events))), cast.MAX_LINES)
        out = self._svg(self._cast(events))
        self.assertIn("line 000", out)
        self.assertIn(f"line {cast.MAX_LINES - 1:03d}", out)
        self.assertNotIn(f"line {cast.MAX_LINES:03d}", out)
        self.assertLess(len(out), 200_000)

    def test_a_truncated_final_event_does_not_lose_the_frames_before_it(self):
        path = self._cast([[0.0, "o", "kept\n"]])
        with path.open("a", encoding="utf-8") as handle:
            handle.write('[0.1, "o", "half a li')
        self.assertIn("kept", self._svg(path))

    def test_an_event_of_the_wrong_shape_is_skipped_not_fatal(self):
        self.assertIn("kept", self._svg(self._cast([[0.0, "o", "kept\n"], [0.1, "o"]])))

    def test_wrapping_at_zero_columns_terminates(self):
        line = [[cast.Run("x" * 12, cast.Style())]]
        self.assertEqual(len(cast.wrap(line, 0)), 12, "a zero width must not loop forever")

    def test_the_cli_refuses_a_zero_or_negative_width(self):
        path = self._cast([[0.0, "o", "hi\n"]])
        for cols in ("0", "-4"):
            done = subprocess.run([sys.executable, str(ROOT / "bench" / "cast_to_svg.py"),
                                   str(path), str(path.with_suffix(".svg")),
                                   "--cols", cols],
                                  capture_output=True, text=True, timeout=60)
            self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
            self.assertIn("--cols", done.stderr)

    def test_a_missing_cast_is_a_usage_error_not_a_traceback(self):
        done = subprocess.run([sys.executable, str(ROOT / "bench" / "cast_to_svg.py"),
                               str(ROOT / "demo" / "nope.cast")],
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(done.returncode, 2)
        self.assertNotIn("Traceback", done.stderr)


class CommittedArtefactsTest(unittest.TestCase):
    """What ships in demo/ has to be what this code renders from this cast."""

    def test_the_svg_is_a_render_of_the_committed_cast(self):
        fresh = cast.render_svg(cast.wrap(cast.styled_lines(CAST), RENDER_COLS),
                                RENDER_COLS)
        self.assertEqual(SVG.read_text(encoding="utf-8"), fresh,
                         "demo/toolinterfacer.svg is stale: re-render it from the cast")

    def test_the_svg_has_one_text_element_per_visible_line(self):
        wrapped = cast.wrap(cast.styled_lines(CAST), RENDER_COLS)
        visible = [line for line in wrapped if any(run.text.strip() for run in line)]
        self.assertGreater(len(visible), 20, "the recording looks empty")
        self.assertEqual(SVG.read_text(encoding="utf-8").count("<text"), len(visible))

    def test_all_three_scenes_are_in_the_committed_render(self):
        text = _svg_plain_text(SVG)
        for scene in ("scene 1", "scene 2", "scene 3"):
            self.assertIn(scene, text)
        self.assertIn("the caller made one call per scene", text)

    def test_no_rendered_line_breaks_a_url(self):
        lines = _svg_plain_text(SVG).split("\n")
        for marker in ("https://example.test/a#frag", "tests/data/fake_cli.py"):
            self.assertTrue(any(marker in line for line in lines),
                            f"{marker} is not on one line — check --cols")
        self.assertFalse([line for line in lines
                          if line.rstrip().endswith(("http", "https:", "https://"))],
                         "a rendered line ends mid-URL")

    def test_the_png_was_rendered_from_the_committed_svg(self):
        svg_width, svg_height = _svg_canvas(SVG)
        png_width, png_height = _png_size(PNG)
        expected = svg_height * PNG_WIDTH / svg_width
        self.assertEqual(png_width, PNG_WIDTH)
        self.assertLessEqual(abs(png_height - expected), 2,
                             f"demo/toolinterfacer.png is {png_width}x{png_height}; the "
                             f"committed SVG at {PNG_WIDTH}px is "
                             f"{PNG_WIDTH}x{math.ceil(expected)} — re-render it")


def _svg_plain_text(path: pathlib.Path) -> str:
    """The rendered picture as text, one entry per line, markup and entities removed."""
    out = []
    for element in re.findall(r"<text [^>]*>(.*?)</text>",
                              path.read_text(encoding="utf-8"), re.S):
        out.append(html.unescape(re.sub(r"<[^>]+>", "", element)))
    return "\n".join(out)


def _svg_canvas(path: pathlib.Path) -> tuple[int, int]:
    head = path.read_text(encoding="utf-8").split("\n", 1)[0]
    width = re.search(r'width="([0-9.]+)"', head)
    height = re.search(r'height="([0-9.]+)"', head)
    if not (width and height):
        raise AssertionError(f"no canvas in {path}: {head[:120]}")
    return round(float(width.group(1))), round(float(height.group(1)))


def _png_size(path: pathlib.Path) -> tuple[int, int]:
    head = path.read_bytes()[:33]
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError(f"{path} is not a PNG")
    return struct.unpack(">II", head[16:24])


if __name__ == "__main__":
    unittest.main()
