"""bands — the t1/t2/t3 definitions the frozen set uses, as checkable predicates.

The definitions are the frozen set's own, from
``tests/data/utterances_v2_NOTES.md``:

  t1  unambiguous      intent stated directly; may carry op-signature vocabulary
                       ("grep the tree", "search the web")
  t2  veiled           intent stated with no op-signature keyword at all
  t3  boundary pair    sits between two ops, the rival being a defensible reading

Those are statements about *vocabulary*, so they are decidable mechanically, and
this module decides them. Two consumers read the same vocabulary and that is the
point: the template generator asserts its own frames land in the band they claim
(a t2 frame that leaks "grep" is a bug, not a hard row), and the session miner
bands real operator phrasings with no author to ask.

The cue lists are per op and deliberately include the *false friends* the notes
flag — "find" alone is not a cue, "find every call site" is; "open" alone is not a
cue because "open the pending url" is web_fetch.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

# op -> phrases that, in a request, point at that op. A phrase is matched with
# flexible whitespace and word boundaries, case-insensitively.
CUES: dict[str, tuple[str, ...]] = {
    "filesys_grep": (
        "grep", "ripgrep", "which files ~ mention", "which files ~ reference",
        "which files ~ talk about", "which files ~ discuss", "which files ~ use",
        "call site", "call sites", "where ~ defined", "where ~ used",
        "definition of", "usages of", "used anywhere",
        "search the tree", "search the repo", "search the codebase",
        "references to", "mentions of", "mentioned", "every file that",
        "anything that imports", "does any file import", "find every call",
        "find any use", "any use of", "is ~ referenced",
        "mention", "mentions", "the files that reference", "file where",
        "files where", "across the tree", "in the source tree",
        "the source for", "in the codebase", "which local files",
        "the local files", "the code, not",
    ),
    "filesys_glob": (
        "glob", "list file", "list the files", "list every file", "list all files",
        "which files exist", "files ending in", "files named", "file names",
        "filenames", "matching files", "find the files", "find all files",
        "find files", "files ~ named", "directory listing", "what files are there",
        "what files are in", "any config files", "any files called",
        "names only", "the names only", "matching the name",
        "every file matching", "the paths matching", "files named like",
        "on disk called", "actual files matching", "files are present",
        "file matching",
    ),
    "filesys_read": (
        "read", "open the file", "cat ", "print", "show me", "display",
        "dump", "contents", "what's in", "what is in", "view", "head of",
        "tail of", "full listing", "show the", "pull up the file",
        "on screen", "written in", "only want that file", "and stop there",
        "without searching", "give me the text, not", "the text, not the values",
        "i only want that",
    ),
    "web_search": (
        "search the web", "search online", "search the internet", "look up",
        "look it up", "google", "search for", "find recent", "find out about",
        "latest", "news about", "what is the current", "who is", "results about",
        "search results", "look for information", "find sources", "the web",
        "results", "search on", "do the search", "run the search",
        "external writing", "the internet",
    ),
    "web_fetch": (
        "fetch", "url", "urls", "link", "links", "page at", "download",
        "scrape", "open that address", "blog post", "the site", "that site",
        "load the page", "web page", "the article", "that address",
        "the content at", "the text at", "pull down", "what's at",
        "what it says", "whole page",
    ),
    "reddit_digest": (
        "reddit", "r/", "subreddit", "the sub", "that sub", "thread",
        "forum", "community saying", "community think", "the boards",
        "upvotes", "commenters", "what people say on",
    ),
    "decide_pick": (
        "pick", "choose", "select", "rank", "which one", "which of",
        "best of", "top 3", "top three", "shortlist", "among these",
        "among the", "among those", "narrow it down", "which should we",
        "which is the right", "the right one", "priority", "act on",
        "use first", "the ones found", "should we act",
    ),
    "decide_extract": (
        "extract", "pull out", "list out", "keywords", "search terms",
        "queries", "variants", "turn this into", "turn it into", "paraphrase",
        "synonyms", "rewrite as", "break this down into", "what terms",
        "literal values", "spell out", "enumerate", "out of the values",
        "the extracted values", "into the values", "values out of",
        "literal paths", "as written", "unpicked", "without choosing",
    ),
}

BANDS = ("t1", "t2", "t3")

_BOUNDARY = r"[\s\-]+"
_GAP = r".{0,40}?"  # a cue written "a ~ b" allows intervening words


def _pattern(phrase: str) -> re.Pattern[str]:
    parts = [_BOUNDARY.join(re.escape(word) for word in chunk.split())
             for chunk in phrase.split("~")]
    return re.compile(r"\b" + _GAP.join(parts) + r"\b", re.IGNORECASE)


_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    op: [_pattern(p) for p in phrases] for op, phrases in CUES.items()
}


@dataclass
class Band:
    """One utterance's band, the rival it straddles, and the cue evidence."""

    band: str
    rival: str | None = None
    hits: dict[str, list[str]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {"band": self.band, "rival": self.rival, "hits": self.hits}


def cue_hits(text: str) -> dict[str, list[str]]:
    """op -> the cue phrases `text` literally contains (empty dict when none)."""
    out: dict[str, list[str]] = {}
    for op, patterns in _PATTERNS.items():
        found = [p.pattern.strip("\\b").replace(_BOUNDARY, " ").replace(_GAP, "~")
                 for p in patterns if p.search(text)]
        if found:
            out[op] = found
    return out


def classify(text: str) -> Band:
    """The band a bare utterance belongs to, from its cue vocabulary alone.

    No cue -> veiled (t2). Exactly one op cued -> unambiguous (t1). Two or more ->
    a boundary row (t3), and the rival is the runner-up op: the reading that is
    cued but not maximally, which is the one the row is meant to defend against.
    """
    hits = cue_hits(text)
    if not hits:
        return Band("t2", None, {})
    ranked = sorted(hits, key=lambda op: (-len(hits[op]), op))
    if len(ranked) == 1:
        return Band("t1", None, hits)
    return Band("t3", ranked[1], hits)


def cued_ops(text: str) -> list[str]:
    return sorted(cue_hits(text))


def leaked_cues(text: str) -> dict[str, list[str]]:
    """Cue evidence for a row that claims to be veiled; empty means the claim holds."""
    return cue_hits(text)


def vocabulary() -> dict[str, tuple[str, ...]]:
    """The signature vocabulary itself, for the manifest a generator emits."""
    return {op: tuple(phrases) for op, phrases in CUES.items()}


def ops() -> Iterable[str]:
    return CUES.keys()
