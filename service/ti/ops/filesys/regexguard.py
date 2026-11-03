"""regexguard — may a caller-supplied pattern be compiled at all?

A caller's pattern is a regular expression, and a regular expression can be a
denial of service. Measured on this tree: `(a+)+$` over a 41-byte file was still
running at a 25-second kill. Python's `re` has no match budget, and the service
runs it in a handler thread that nothing can interrupt, so the only place to
bound it is before the match: refuse the shapes whose cost explodes in the
subject length, and keep the subject small.

Refused, with the reason and the offset in the error:

  length       longer than `ops.filesys.grep.pattern_max_chars`
  nesting      a quantified group that itself contains a quantified atom —
               `(a+)+`, `(\\w*\\s*)+`: exponential in the subject length
  ambiguity    two adjacent *unbounded* quantified atoms whose character sets can
               match the same character — `a*a*`, `\\w+\\w+`, `.*.*`: quadratic
  alternation  an unbounded-quantified group whose branches can start with the
               same character — `(a|aa)+`: exponential without nesting

What survives is matched against one line clipped to
`ops.filesys.grep.line_max_chars`, with the visit's deadline checked between
lines and every hit counted against a cap. Character sets are compared coarsely —
the exact characters when an atom names a few, a category set otherwise — so the
four checks are conservative: they can refuse a pattern that would have been
fine, and the error says which shape refused it.
"""
from __future__ import annotations

import fnmatch
import re
import string
from dataclasses import dataclass, field

_CATS = ("lower", "upper", "digit", "underscore", "space", "punct", "other")
_WORD_CATS = frozenset({"lower", "upper", "digit", "underscore"})
_DIGITS = frozenset("0123456789")
_SPACE = frozenset(string.whitespace)
_MAX_EXACT = 256

_BRACE = re.compile(r"\{[0-9]*(?:,[0-9]*)?\}")


class UnsafePattern(ValueError):
    """A pattern refused before compilation. ValueError: it is a bad argument."""


@dataclass(frozen=True)
class _Char:
    """What one atom can match its first character with.

    `exact` lists characters when they can be enumerated cheaply; `cats` is the
    coarse category set, always. Two atoms overlap when their categories intersect
    *and* they are not both exact with disjoint characters — so `a*b*` passes and
    `a*a*` does not.
    """
    exact: frozenset[str] | None
    cats: frozenset[str]


_ANY = _Char(None, frozenset(_CATS))


def _cat(ch: str) -> str:
    if ch == "_":
        return "underscore"
    if ch.isdigit():
        return "digit"
    if ch.isspace():
        return "space"
    if ch.islower():
        return "lower"
    if ch.isupper():
        return "upper"
    if ch.isalnum():
        return "other"
    return "punct"


def _char(ch: str) -> _Char:
    return _Char(frozenset({ch}), frozenset({_cat(ch)}))


def _union(sets: list[_Char]) -> _Char:
    cats: set[str] = set()
    exact: set[str] = set()
    enumerable = bool(sets)
    for item in sets:
        cats |= item.cats
        if item.exact is None or len(exact | item.exact) > _MAX_EXACT:
            enumerable = False
        else:
            exact |= item.exact
    return _Char(frozenset(exact) if enumerable else None, frozenset(cats))


def _overlap(left: _Char, right: _Char) -> bool:
    if not (left.cats & right.cats):
        return False
    if left.exact is not None and right.exact is not None:
        return bool(left.exact & right.exact)
    return True


@dataclass
class _Atom:
    chars: _Char
    at: int
    unbounded: bool = False
    nested: bool = False
    branches: list[_Char] | None = None


@dataclass
class _Frame:
    opened_at: int
    lookaround: bool = False
    atoms: list[_Atom] = field(default_factory=list)
    branches: list[_Char] = field(default_factory=list)
    branch_start: int = 0
    has_unbounded: bool = False

    def add(self, atom: _Atom) -> _Atom:
        self.atoms.append(atom)
        return atom

    def close_branch(self) -> None:
        if self.branch_start < len(self.atoms):
            self.branches.append(self.atoms[self.branch_start].chars)
        else:
            self.branches.append(_ANY)      # an empty branch matches anything next
        self.branch_start = len(self.atoms)

    def chars(self) -> _Char:
        return _union(self.branches) if self.branches else _ANY


def check(pattern: str, *, max_chars: int) -> None:
    """Refuse `pattern` if its shape cannot be matched in bounded time."""
    if len(pattern) > max_chars:
        raise UnsafePattern(f"pattern is {len(pattern)} characters; the limit is "
                            f"{max_chars} — narrow the search with a longer literal")
    _scan(pattern)


def compile_guarded(pattern: str, *, max_chars: int) -> "re.Pattern[str]":
    check(pattern, max_chars=max_chars)
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise UnsafePattern(f"pattern is not a regular expression: {exc}") from exc


def compile_glob(expr: str, *, max_chars: int) -> "re.Pattern[str]":
    """A glob name filter as a compiled regex, guarded like any other pattern.

    Translated with fnmatch (real glob semantics: `*` stays inside one path
    component's worth of characters, `?` is one character) after collapsing runs
    of `*`: a caller writing `**/*.py` means one star, and left as written it
    translates to `.*.*`, which is the ambiguous-adjacency shape the guard
    refuses.
    """
    collapsed = re.sub(r"\*{2,}", "*", expr)
    return compile_guarded(fnmatch.translate(collapsed), max_chars=max_chars)


def _scan(pattern: str) -> None:
    stack: list[_Frame] = [_Frame(opened_at=0)]
    prev: _Atom | None = None
    index = 0
    length = len(pattern)
    while index < length:
        ch = pattern[index]
        if ch == "\\":
            chars, index = _read_escape(pattern, index)
            if chars is None:                 # a zero-width assertion
                prev = None
                continue
            prev = _emit(stack[-1], chars, index)
            _, index, prev = _take_quantifier(pattern, index, stack[-1], prev)
            continue
        if ch == "[":
            chars, index = _read_class(pattern, index)
            prev = _emit(stack[-1], chars, index)
            _, index, prev = _take_quantifier(pattern, index, stack[-1], prev)
            continue
        if ch == "(":
            kind, index = _read_group_prefix(pattern, index)
            if kind in ("flag", "comment"):
                prev = None
                continue
            if kind == "backref":             # (?P=name): matches arbitrary text
                prev = _emit(stack[-1], _ANY, index)
                _, index, prev = _take_quantifier(pattern, index, stack[-1], prev)
                continue
            stack.append(_Frame(opened_at=index, lookaround=(kind == "lookaround")))
            prev = None
            continue
        if ch == ")":
            frame = stack.pop() if len(stack) > 1 else None
            index += 1
            if frame is None or frame.lookaround:   # unbalanced, or zero-width
                prev = None
                continue
            frame.close_branch()
            atom = _Atom(frame.chars(), frame.opened_at,
                         nested=frame.has_unbounded, branches=list(frame.branches))
            prev = stack[-1].add(atom)
            _, index, prev = _take_quantifier(pattern, index, stack[-1], prev)
            continue
        if ch == "|":
            stack[-1].close_branch()
            index += 1
            prev = None
            continue
        if ch in "*+?{":
            found, nxt, atom = _take_quantifier(pattern, index, stack[-1], prev)
            if not found:                         # a `{` that is not a quantifier
                prev = _emit(stack[-1], _char(ch), index)
                index += 1
                continue
            index, prev = nxt, atom
            continue
        prev = _emit(stack[-1], _char(ch), index)
        index += 1


def _emit(frame: _Frame, chars: _Char, at: int) -> _Atom:
    return frame.add(_Atom(chars, at))


def _take_quantifier(pattern: str, index: int, frame: _Frame,
                     prev: _Atom | None) -> tuple[bool, int, _Atom | None]:
    """Consume the quantifier at `index` (if any) and apply it to the last atom.

    A quantifier with nothing to quantify — the `**` a glob caller writes, a
    stray `*` — is modelled as `.*`: it is what the same string means after the
    caller's own translation, and `re.compile` is still the authority on whether
    the pattern is a pattern at all.
    """
    unbounded, nxt = _read_quantifier(pattern, index)
    if unbounded is None:
        return False, index, prev
    atom = prev if prev is not None else _emit(frame, _ANY, index)
    if unbounded:
        _refuse_unsafe(frame, atom)
        atom.unbounded = True
        frame.has_unbounded = True
    return True, nxt, atom


def _refuse_unsafe(frame: _Frame, atom: _Atom) -> None:
    if atom.nested:
        raise UnsafePattern(
            f"pattern rejects a nested quantifier at offset {atom.at}: a quantified "
            f"group that contains another quantifier backtracks exponentially — the "
            f"shape that hangs the service")
    branches = atom.branches or []
    for left in range(len(branches)):
        for right in range(left + 1, len(branches)):
            if _overlap(branches[left], branches[right]):
                raise UnsafePattern(
                    f"pattern rejects an ambiguous alternation at offset {atom.at}: a "
                    f"quantified group whose branches can start with the same character "
                    f"can be matched exponentially many ways")
    previous = frame.atoms[-2] if len(frame.atoms) >= 2 else None
    if previous is not None and previous.unbounded and _overlap(previous.chars, atom.chars):
        raise UnsafePattern(
            f"pattern rejects adjacent overlapping quantifiers at offset {atom.at}: the "
            f"same characters can be divided between them in many ways, which makes the "
            f"match quadratic in the line length")


def _read_quantifier(pattern: str, index: int) -> tuple[bool | None, int]:
    """(unbounded, next index) for the quantifier at `index`; (None, index) if none."""
    if index >= len(pattern):
        return None, index
    ch = pattern[index]
    if ch in "*+":
        nxt = index + 1
        while nxt < len(pattern) and pattern[nxt] in "?+":   # lazy / possessive
            nxt += 1
        return True, nxt
    if ch == "?":
        return False, index + 1
    if ch == "{":
        match = _BRACE.match(pattern, index)
        if match is None:
            return None, index
        return match.group(0).endswith(",}"), match.end()
    return None, index


def _read_group_prefix(pattern: str, index: int) -> tuple[str, int]:
    """Classify the group starting at `(`; returns (kind, index after the prefix)."""
    if pattern.startswith("(?P<", index):
        close = pattern.find(">", index)
        return "group", (close + 1 if close != -1 else index + 3)
    if pattern.startswith("(?P=", index):
        close = pattern.find(")", index)
        return "backref", (close + 1 if close != -1 else len(pattern))
    if pattern.startswith("(?:", index):
        return "group", index + 3
    if pattern.startswith("(?=", index) or pattern.startswith("(?!", index):
        return "lookaround", index + 3
    if pattern.startswith("(?<=", index) or pattern.startswith("(?<!", index):
        return "lookaround", index + 4
    if pattern.startswith("(?#", index):
        close = pattern.find(")", index)
        return "comment", (close + 1 if close != -1 else len(pattern))
    if pattern.startswith("(?", index):
        close = pattern.find(")", index)
        colon = pattern.find(":", index)
        if close != -1 and (colon == -1 or close < colon):
            return "flag", close + 1
        return "group", (colon + 1 if colon != -1 else index + 2)
    return "group", index + 1


def _read_escape(pattern: str, index: int) -> tuple[_Char | None, int]:
    """Parse the escape at `pattern[index]`; None means a zero-width assertion."""
    nxt = index + 1
    if nxt >= len(pattern):
        return _char("\\"), nxt
    ch = pattern[nxt]
    if ch in "AbBZz":
        return None, nxt + 1
    if ch == "d":
        return _Char(_DIGITS, frozenset({"digit"})), nxt + 1
    if ch == "w":
        return _Char(None, _WORD_CATS), nxt + 1
    if ch == "s":
        return _Char(_SPACE, frozenset({"space"})), nxt + 1
    if ch in "DWS":
        return _ANY, nxt + 1
    if ch.isdigit():                       # backreference: matches arbitrary text
        end = nxt
        while end < len(pattern) and pattern[end].isdigit():
            end += 1
        return _ANY, end
    if ch == "x" and len(pattern) >= nxt + 3:
        try:
            return _char(chr(int(pattern[nxt + 1:nxt + 3], 16))), nxt + 3
        except ValueError:
            return _ANY, nxt + 1
    if ch in "uU" and len(pattern) >= nxt + 5:
        try:
            return _char(chr(int(pattern[nxt + 1:nxt + 5], 16))), nxt + 5
        except ValueError:
            return _ANY, nxt + 1
    simple = {"n": "\n", "t": "\t", "r": "\r", "f": "\f", "v": "\v", "a": "\a"}
    if ch in simple:
        return _char(simple[ch]), nxt + 1
    return _char(ch), nxt + 1


def _read_class(pattern: str, index: int) -> tuple[_Char, int]:
    """Parse the character class at `pattern[index] == '['`."""
    pos = index + 1
    negated = False
    if pos < len(pattern) and pattern[pos] == "^":
        negated = True
        pos += 1
    exact: set[str] = set()
    cats: set[str] = set()
    enumerable = True
    first = True
    while pos < len(pattern):
        ch = pattern[pos]
        if ch == "]" and not first:
            pos += 1
            break
        first = False
        if ch == "\\":
            atom, pos = _read_escape(pattern, pos)
            if atom is None or atom.exact is None:
                enumerable = False
                cats |= atom.cats if atom is not None else frozenset({"punct"})
            else:
                exact |= set(atom.exact)
                cats |= atom.cats
            continue
        if pos + 2 < len(pattern) and pattern[pos + 1] == "-" and pattern[pos + 2] != "]":
            low, high = pattern[pos], pattern[pos + 2]
            span = [chr(code) for code in range(ord(low), ord(high) + 1)] \
                if ord(low) <= ord(high) else [low, "-", high]
            if len(span) > _MAX_EXACT:
                enumerable = False
            else:
                exact |= set(span)
            cats |= {_cat(item) for item in span}
            pos += 3
            continue
        exact.add(ch)
        cats.add(_cat(ch))
        pos += 1
    if negated or not enumerable or not exact:
        return _ANY, pos
    return _Char(frozenset(exact), frozenset(cats)), pos
