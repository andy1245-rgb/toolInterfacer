"""shape — is this disk intent enumerative or singular?

A listing and a read are different plans, not different confidences: "what files
are in X" has no single file to pick, so no confidence can make the read shape
right. This module answers that shape question and nothing else.

Ordered rules, first match wins — two families, and the order between them is the
whole design:

    R1 EXTRACT_PRODUCTION      -> SINGULAR       (object is a string artefact)
    R2 CONTENT_PREDICATE       -> SINGULAR       (file as container of text)
    R3 PRIOR_SET_REFERENCE     -> SINGULAR       (set already materialised)
    R4 NAME_CRITERION          -> ENUMERATIVE
    R5 IMPLIED_OBJECT_COMMAND  -> ENUMERATIVE
    R6 EXISTENTIAL_INVENTORY   -> ENUMERATIVE    (gated on an explicit object)
    DEFAULT                    -> SINGULAR

R2 must precede R4/R5/R6 because a content question can carry a name criterion:
"which files mention the scheduler hooks" names a file class and asks about text,
and the question is about text. R1 must precede R5/R6 because extract rows are
full of disk-entry nouns: "name the dirs the request refers to" wants a set of
strings handed back, not a walk. R3 must precede R6 because pick vocabulary sits
inside the disk-entry noun class: "keep the best three of these files" is a
choice over a set already in hand, not an inventory of a directory. R4 before R6
because R6 needs an inventory frame and a bare name criterion carries none; R5
before R6 because R6 is gated on an explicit object and R5 supplies one
implicitly ("ls", "show me the files").

The R6 gate — a file-set noun or an extension token somewhere in the intent — is
load-bearing in both directions: without it six positive rows fall to SINGULAR,
and with a loose one an existence question about something off-disk ("is there
anything published about the tokenizer patch?") turns into a listing. Its "how
many" clause is narrowed so the counted noun is the file set itself, which keeps
"how many lines does the main file have" singular.

Rule-first and lexical only: no model call, no network, no added latency. Anything
unrecognised is SINGULAR, the shape shipped before the split — the classifier is
only allowed to move an intent off the read path when the wording says listing.
"""
from __future__ import annotations

import re
from typing import Callable

ENUMERATIVE = "enumerative"
SINGULAR = "singular"

# Shared token classes. Composed (not interpolated) because the cardinalities and
# repetition counts inside them are regex braces, not format fields.
FILESET = (r"(?:files?|file\s*names?|filenames?|entries?|paths?|dirs?|directories?"
           r"|folders?|scripts?|docs?|documents?|fixtures?|manifests?|configs?"
           r"|artifacts?|sources?|dotfiles?)")

FILESET_PL = (r"(?:files|file\s*names|filenames|entries|paths|dirs|directories"
              r"|folders|scripts|docs|documents|fixtures|manifests|configs"
              r"|artifacts|sources|dotfiles|contents)")

FILECLASS = (r"(?:readme|licen[cs]es?|makefiles?|dockerfiles?|dotfiles?|lockfiles?"
             r"|changelog|\.gitignore|\.env|manifest)")

SETNOUN = (r"(?:candidates?|matches?|hits?|results?|pages?|threads?|files?|paths?"
           r"|queries|wordings?|drafts?|entries|options?|items?|ones?)")

CARD = r"(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|dozen|\d+)"

EXT = r"[a-z][a-z0-9]{0,5}"


def _clauses(*patterns: str) -> Callable[[str], bool]:
    """A rule satisfied by any single one of its clauses."""
    compiled = tuple(re.compile(p) for p in patterns)
    return lambda text: any(c.search(text) for c in compiled)


# R1 — the object is a string artefact to hand back, whatever disk nouns it names.
_R1_EXTRACT_PRODUCTION = _clauses(
    r"\b(?:write|pull|print|spit|type|hand)\s+out\b",
    r"\bextract\b",
    r"\bcome\s+up\s+with\b",
    r"\bturn\b.{0,60}\binto\b",
    r"\bphras(?:e|es|ed|ing|ings)\b",
    r"\bwordings?\b",
    r"\bvariants?\b",
    r"\bkeywords?\b",
    r"\bsearch\s+(?:strings?|terms?)\b",
    r"\b(?:name|list|write\s+out)\s+the\s+(?:\w+\s+){0,3}"
    r"(?:identifiers|phrasings|keywords|terms|wordings|queries|names|titles"
    r"|dirs?|directories|paths?|file\s*names?|filenames?)\b",
    r"\bwhat\s+(?:literal\s+|exact\s+)?(?:paths?|dirs?|directories|identifiers"
    r"|names?|keywords?|terms?|words?|strings?)\b.{0,40}"
    r"\b(?:show\s+up|appears?|came?\s+up|mentioned|to\s+(?:print|use|try)"
    r"|should|could|would|fit|capture)\b",
    r"\bgive\s+me\b.{0,50}\b(?:phrasings?|keywords?|terms?|ways|angles|titles?"
    r"|wordings?|identifiers|variants?|batch|strings?)\b",
)

# R2 — the file is a container of text, so the answer is a hit inside it, not a
# list of names. Any of these makes the disk walk a means, not the deliverable.
_R2_CONTENT_PREDICATE = _clauses(
    r"\bmention(?:s|ed|ing)?\b",
    r"\btalk(?:s|ed|ing)?\s+about\b",
    r"\bdiscuss(?:es|ed|ing)?\b",
    r"\breferenc(?:e|es|ed|ing)\b",
    r"\brefer(?:s|red|ring)?\s+to\b",
    r"\bcontains?\b|\bcontaining\b",
    r"\bdefin(?:e|es|ed|ing)\b",
    r"\bcall\s*sites?\b|\bcall(?:s|ed|ing)?\b",
    r"\bimport(?:s|ed|ing)?\b",
    r"\buse\s+of\b|\buses?\b",
    r"\bpull(?:s|ed)?\s+in\b",
    r"\bwir(?:e|es|ed|ing)\s+up\b",
    r"\btouch(?:es|ed)?\b",
    r"\bcatch(?:es)?\b",
    r"\bgrep\b",
    r"\bsearch\b",
    r"\blook(?:ing)?\s+for\b",
    r"\blocat(?:e|es|ed)\b",
    r"\btrack(?:ing)?\s+down\b",
    r"\bfind\s+out\b",
    r"\bcodebase\b",
)

# R3 — a set is already on the table, so the question is a choice over it.
_R3_PRIOR_SET_REFERENCE = _clauses(
    r"\b(?:these|those)\b(?:\s+\w+){0,3}\s+" + SETNOUN + r"\b",
    r"\bwhich\s+of\b",
    r"\b(?:among|out\s+of)\b",
    r"\bof\s+the\s+(?:\w+\s+){0,3}" + SETNOUN + r"\b",
    r"\b" + CARD + r"\s+(?:\w+\s+){0,2}" + SETNOUN + r"\b",
    r"\bsort\b|\bnarrow\b|\brank\b|\bchoose\s+(?:one|the)\b|\bpick\s+the\b",
    r"\bkeep\s+the\s+(?:best|top|most|" + CARD + r")\b",
    r"\btell\s+me\s+which\b|\bpoint\s+me\s+at\b|\bwhich\s+one\b",
    r"\bwhich\s+(?:is|are)\s+the\s+(?:right|best|strongest)\b",
    r"\bonly\s+care\b|\bmatters?\b",
    r"\b(?:came|comes?)\s+(?:back|up)\b|\breturned\b|\bpulled\b|\bwe\s+just\b",
    r"\bpile\s+of\b|\bqueue\b|\bon\s+the\s+table\b|\bso\s+far\b|\bgot\s+a\b",
    r"\bcandidate\s+list\b|\bshortlist\b",
)

# R4 — the intent states a criterion on the names themselves.
_R4_SIMPLE = _clauses(
    r"\bend(?:s|ing)?\s+(?:in|with)\s+\." + EXT + r"\b",
    r"\b(?:all|every|each|any)\s+\." + EXT + r"\b",
    r"\." + EXT + r"\s+(?:files?|docs?|entries|scripts?|ones?)\b",
    r"\." + EXT + r"\s+(?:under|in|inside|anywhere|across|at)\b",
    r"\bstart(?:s|ing)?\s+with\b|\bbegin(?:s|ning)?\s+with\b|\bprefixed?\s+with\b",
    r"\bin\s+(?:their|its)\s+names?\b|\bwhose\s+names?\b",
    r"\bthe\s+pattern\b",
)

# A known file class only counts as a criterion when the intent also asks for the
# entries to exist or to be found. That conjunction is what keeps "what does the
# makefile actually do?" and "check the top of CHANGELOG.md" singular: a class
# name inside a read is a read.
_FILE_CLASS = re.compile(FILECLASS)
_EXISTENCE_FRAME = re.compile(
    r"\bfind\b|\blocate\b|\b(?:is|are)\s+there\b|\bdo\s+we\s+have\b|\bhave\s+we\b"
    r"|\bwhich\b|\ball\b|\bevery\b|\bany\b|\bhow\s+many\b|\bleft\s+behind\b"
    r"|\blying\s+around\b")


def _name_criterion(text: str) -> bool:
    return bool(_R4_SIMPLE(text)
                or (_FILE_CLASS.search(text) and _EXISTENCE_FRAME.search(text)))


# R5 — a command whose object is implied: the listing itself is the request.
_R5_SIMPLE = _clauses(
    r"^\s*ls\b",
    r"\benumerat\w*",
    r"\blist\s+(?:of\s+|all\s+|every\s+|each\s+|the\s+)?" + FILESET_PL + r"\b",
    r"\bshow\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?" + FILESET_PL + r"\b",
)

# "what's in this directory" — the container noun is what makes it a listing;
# "what's in README.md" has none and stays singular.
_WHAT_IS_IN = re.compile(r"\bwhat(?:'s|\s+is|\s+are)\b.{0,40}\b(?:in|inside|under|within)\b")
_CONTAINER = re.compile(r"\b(?:dir|dirs|directory|directories|folder|folders|tree|repo)\b")


def _implied_object_command(text: str) -> bool:
    return bool(_R5_SIMPLE(text)
                or (_WHAT_IS_IN.search(text) and _CONTAINER.search(text)))


# R6 — an inventory frame, admitted only when the intent names an object to
# inventory: a file-set noun, or an extension token anywhere.
_R6_SIMPLE = _clauses(
    r"\bhow\s+many\s+(?:\." + EXT + r"\s+)?(?:\w+\s+){0,1}" + FILESET + r"\b",
    r"\bnames?\s+of\b",
    r"\bcollection\s+of\b|\bbatch\s+of\b|\blist\s+of\b|\bcount\b",
    r"\b(?:is|are)\s+there\b",
    r"\bdo\s+we\s+have\b|\bhave\s+we\s+got\b",
    r"\bwhich\b(?:\s+\w+){0,2}\s+" + FILESET + r"\b",
    r"\bwhat\s+" + FILESET + r"\b",
    r"\b(?:all|every|each|any)\b(?:\s+\w+){0,3}\s+" + FILESET + r"\b",
    r"\bwhere\b.{0,30}\b(?:keep|kept|live|lives|store[ds]?|sit|sits)\b",
    r"\blying\s+around\b|\bleft\s+behind\b",
    r"\bthe\s+" + FILESET + r"\s+for\s+every\b",
    r"\bgive\s+me\s+the\s+(?:collection|set|list)\b",
)

_OBJECT_GATE = re.compile(r"\b" + FILESET + r"\b|\." + EXT + r"\b")


def _gated(inner: Callable[[str], bool]) -> Callable[[str], bool]:
    return lambda text: bool(_OBJECT_GATE.search(text)) and inner(text)


_RULES: tuple[tuple[str, str, Callable[[str], bool]], ...] = (
    ("R1_EXTRACT_PRODUCTION", SINGULAR, _R1_EXTRACT_PRODUCTION),
    ("R2_CONTENT_PREDICATE", SINGULAR, _R2_CONTENT_PREDICATE),
    ("R3_PRIOR_SET_REFERENCE", SINGULAR, _R3_PRIOR_SET_REFERENCE),
    ("R4_NAME_CRITERION", ENUMERATIVE, _name_criterion),
    ("R5_IMPLIED_OBJECT_COMMAND", ENUMERATIVE, _implied_object_command),
    ("R6_EXISTENTIAL_INVENTORY", ENUMERATIVE, _gated(_R6_SIMPLE)),
)


def _fire(text: str) -> tuple[str | None, str]:
    for name, kind, matches in _RULES:
        if matches(text):
            return name, kind
    return None, SINGULAR


def rule_for(intent: str) -> str | None:
    """The first rule that fires, or None when the intent falls through."""
    return _fire((intent or "").lower())[0]


def classify(intent: str) -> str:
    return _fire((intent or "").lower())[1]
