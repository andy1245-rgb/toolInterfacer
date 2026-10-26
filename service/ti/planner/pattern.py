"""pattern — which names a listing asks for, as a glob.

scope.py answers which directory a listing covers; this answers which entries
inside it. The two stay apart because the two answers come from different words
and are reported separately: listing every entry under the directory the caller
named is a good answer, and so is listing the right names in the directory the
caller meant.

Lexical only, like scope. A name stated as a criterion in the caller's own words
— an extension ("every .sh", "ending in .toml"), a prefix ("starts with bench_"),
a substring ("'replay' in their name"), a file class ("the README", "any
dockerfile") — becomes a glob over the listing root. A known file class only
counts when the intent asked for it to exist, so "what does the makefile actually
do?" contributes no pattern.

No name criterion is not a defect: a listing of every entry under the scope is a
legitimate answer, so build() returns `**/*` and says so in its assumption instead
of inventing a criterion to look specific.

Dotfiles are a known limitation. The filesys walk skips entries whose name begins
with ".", so `**/.*` cannot match anything through it today; the pattern is still
the honest statement of the request, and the assumption records that the walk
cannot honour it rather than letting an empty result read as "there are none".
"""
from __future__ import annotations

import re

from .shape import EXT, FILECLASS

# The kinds a pattern can be derived from. DOTFILE is a file-class case with its
# own emitted shape (`**/.*`), so it reports separately from FILE_CLASS.
DOTFILE = "dotfile"
EXTENSION = "extension"
PREFIX = "prefix"
SUBSTRING = "substring"
FILE_CLASS = "file_class"
NONE = "none"

_QUOTED = r"['\"`]([^'\"`]{1,40})['\"`]"

# A bare token is bounded the same way: an unbounded `[\w.-]+` followed by `\s+`
# backtracks over a long run of word characters, which made build() quadratic on a
# single 10 KB token (~0.6 s) and on unicode (~0.9 s).

# A dotfile is asked for by class ("the dotfiles") or by one of its usual names.
_DOTFILE = re.compile(r"\bdotfiles?\b|\.gitignore\b|\.env\b", re.IGNORECASE)

# Frames that put the extension in the sentence's own words, checked before the
# bare form so "ending in .toml" wins over any earlier dot in the sentence.
_EXT_FRAME = re.compile(r"\bend(?:s|ing)?\s+(?:in|with)\s+\.(" + EXT + r")\b")
_EXT_ANY = re.compile(r"\.(" + EXT + r")\b")

_PREFIX = re.compile(r"\b(?:start(?:s|ing)?|begin(?:s|ning)?|prefix(?:ed)?)\s+with\s+"
                     r"(?:" + _QUOTED + r"|([\w.-]{1,40}))", re.IGNORECASE)

_IN_NAME = re.compile(r"(?:" + _QUOTED + r"|([\w.-]{1,40}))\s+in\s+(?:their|its)\s+names?\b",
                      re.IGNORECASE)
_WHOSE_NAME = re.compile(r"\bwhose\s+names?\s+(?:has|have|contains?|includes?)\s+"
                         r"(?:" + _QUOTED + r"|([\w.-]{1,40}))", re.IGNORECASE)

# A file class on its own is a read ("what does the makefile actually do?"); only
# an existence frame makes it a name criterion. Mirrors the conjunction shape.py
# uses to admit the same class as a listing.
_FILE_CLASS = re.compile(FILECLASS, re.IGNORECASE)
_EXISTENCE_FRAME = re.compile(
    r"\bfind\b|\blocate\b|\b(?:is|are)\s+there\b|\bdo\s+we\s+have\b|\bhave\s+we\b"
    r"|\bwhich\b|\ball\b|\bevery\b|\bany\b|\bhow\s+many\b|\bleft\s+behind\b"
    r"|\blying\s+around\b", re.IGNORECASE)

_NO_CONSTRAINT = ("listing: no name constraint stated, so the listing covers "
                  "every entry under the listing root")
_DOTFILE_LIMIT = ("listing: dotfiles were asked for, but the filesys walk skips "
                  "entries beginning with '.', so the dotfiles could not be "
                  "enumerated")


def build(intent: str) -> tuple[str, str, str | None]:
    """The glob a listing intent asks for, the kind it came from, and any
    assumption the pattern itself establishes."""
    text = intent or ""
    lowered = text.lower()

    if _DOTFILE.search(text):
        return "**/.*", DOTFILE, _DOTFILE_LIMIT

    frame = _EXT_FRAME.search(lowered)
    if frame:
        return "**/*." + frame.group(1), EXTENSION, None
    any_ext = _EXT_ANY.search(lowered)
    if any_ext:
        return "**/*." + any_ext.group(1), EXTENSION, None

    prefix = _PREFIX.search(text)
    if prefix:
        return "**/*" + _token(prefix) + "*", PREFIX, None

    inside = _IN_NAME.search(text) or _WHOSE_NAME.search(text)
    if inside:
        return "**/*" + _token(inside) + "*", SUBSTRING, None

    klass = _FILE_CLASS.search(text)
    if klass and _EXISTENCE_FRAME.search(text):
        return "**/*" + klass.group(0) + "*", FILE_CLASS, None

    return "**/*", NONE, _NO_CONSTRAINT


def _token(match: re.Match) -> str:
    return match.group(1) or match.group(2)
