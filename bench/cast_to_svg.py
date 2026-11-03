#!/usr/bin/env python3
"""cast_to_svg — turn an asciinema v2 recording of a non-interactive command into an SVG.

    python3 bench/cast_to_svg.py demo/toolinterfacer.cast demo/toolinterfacer.svg

Why this exists: a `.cast` is the right artefact for a terminal and the wrong one for a
README, a slide or a post. `asciinema` 2.x cannot export an image and `agg` is a Rust
binary this box does not have — but `bench/demo.py` writes lines and never moves the
cursor, so the recording is a *transcript* and rendering it is a text layout problem.

Two decisions that the first version of this file got wrong, both visible in the render:

- **Runs are `<tspan>`s inside one `<text>` per line, not separately positioned
  `<text>` elements.** A `<text>` at `x = pad + column * char_width` only aligns if the
  guessed advance equals the font's real one, which it does not: the confidence column
  drifted 200 px right. Letting the text engine advance keeps a monospace font aligned
  by construction, whatever font the viewer actually has.
- **Every line is rendered, not the last screenful.** The demo prints ~55 lines and a
  34-row terminal scrolled the footer into the middle of scene 2, so the "final frame"
  was the wrong picture. The transcript is the picture.

The SVG is self-contained: no external CSS, no font files, monospace-family only.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from dataclasses import dataclass

PALETTE = {
    30: "#3d3d3d", 31: "#e06c75", 32: "#98c379", 33: "#e5c07b", 34: "#61afef",
    35: "#c678dd", 36: "#56b6c2", 37: "#d7dae0",
    90: "#5c6370", 91: "#e06c75", 92: "#98c379", 93: "#e5c07b", 94: "#61afef",
    95: "#c678dd", 96: "#56b6c2", 97: "#ffffff",
}
BACKGROUND = "#10131a"
FOREGROUND = "#d7dae0"
DIM_ALPHA = 0.55
FONT = ("ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, "
        "'DejaVu Sans Mono', monospace")
FONT_SIZE = 13.5
LINE_HEIGHT = 19.0
CHAR_WIDTH = 8.13          # advance of the size above in a monospace face
PAD = 26.0
MAX_LINES = 200            # a guard, not a policy: a runaway recording should not
                           # produce a megabyte of SVG

_CSI = re.compile(r"\x1b\[([0-9;]*)([A-Za-z])")


@dataclass(frozen=True)
class Style:
    fg: str = FOREGROUND
    bold: bool = False
    dim: bool = False


@dataclass
class Run:
    text: str
    style: Style


def _apply_sgr(style: Style, params: str) -> Style:
    fg, bold, dim = style.fg, style.bold, style.dim
    for code in ([int(p) for p in params.split(";") if p.isdigit()] or [0]):
        if code == 0:
            fg, bold, dim = FOREGROUND, False, False
        elif code == 1:
            bold = True
        elif code == 2:
            dim = True
        elif code == 22:
            bold = dim = False
        elif code in PALETTE:
            fg = PALETTE[code]
        elif code == 39:
            fg = FOREGROUND
    return Style(fg, bold, dim)


def styled_lines(cast: pathlib.Path) -> list[list[Run]]:
    """Every line the recording wrote, as styled runs."""
    raw = cast.read_text(encoding="utf-8").splitlines()
    if not raw:
        return []
    for line in raw[1:]:
        if line.strip():
            _t, kind, data = json.loads(line)
            if kind == "o":
                break
    chunks = [json.loads(line)[2] for line in raw[1:]
              if line.strip() and json.loads(line)[1] == "o"]
    text = "".join(chunks).replace("\r\n", "\n").replace("\r", "\n")

    lines: list[list[Run]] = [[]]
    style = Style()
    index = 0
    buffer: list[str] = []

    def flush(buffer: list[str], style: Style) -> None:
        if buffer:
            lines[-1].append(Run("".join(buffer), style))
            buffer.clear()

    while index < len(text):
        match = _CSI.match(text, index)
        if match:
            if match.group(2) == "m":
                flush(buffer, style)
                style = _apply_sgr(style, match.group(1))
            index = match.end()
            continue
        char = text[index]
        index += 1
        if char == "\n":
            flush(buffer, style)
            lines.append([])
        elif char == "\x1b":
            index += 1                      # skip an unsupported escape's final byte
        elif char >= " ":
            buffer.append(char)
    flush(buffer, style)
    return lines[:MAX_LINES]


def wrap(lines: list[list[Run]], width: int) -> list[list[Run]]:
    """Hard-wrap at `width` columns, carrying the style across the break."""
    out: list[list[Run]] = []
    for line in lines:
        column = 0
        current: list[Run] = []
        for run in line:
            remaining = run.text
            while remaining:
                room = width - column
                if room <= 0:
                    out.append(current)
                    current, column = [], 0
                    room = width
                head, remaining = remaining[:room], remaining[room:]
                current.append(Run(head, run.style))
                column += len(head)
        out.append(current)
    return out


def _escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def render_svg(lines: list[list[Run]], width_cols: int) -> str:
    width = width_cols * CHAR_WIDTH + PAD * 2
    height = max(len(lines), 1) * LINE_HEIGHT + PAD * 2
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
        f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'font-family="{FONT}" font-size="{FONT_SIZE}">',
        f'  <rect width="100%" height="100%" fill="{BACKGROUND}" rx="10"/>',
    ]
    for index, line in enumerate(lines):
        baseline = PAD + LINE_HEIGHT * (index + 0.76)
        if not any(run.text.strip() for run in line):
            continue
        spans = []
        for run in line:
            if not run.text:
                continue
            attrs = [f'fill="{run.style.fg}"']
            if run.style.bold:
                attrs.append('font-weight="600"')
            if run.style.dim:
                attrs.append(f'opacity="{DIM_ALPHA}"')
            spans.append(f'<tspan {" ".join(attrs)}>{_escape(run.text)}</tspan>')
        out.append(f'  <text x="{PAD:.0f}" y="{baseline:.1f}" '
                   f'xml:space="preserve">{"".join(spans)}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cast", help="an asciinema v2 .cast file")
    ap.add_argument("svg", nargs="?", help="output path (default: beside the cast)")
    ap.add_argument("--cols", type=int, default=100,
                    help="wrap width in columns (default %(default)s)")
    args = ap.parse_args()
    source = pathlib.Path(args.cast)
    target = pathlib.Path(args.svg) if args.svg else source.with_suffix(".svg")
    lines = wrap(styled_lines(source), args.cols)
    target.write_text(render_svg(lines, args.cols), encoding="utf-8")
    print(f"{source} -> {target} ({target.stat().st_size} bytes, {len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
