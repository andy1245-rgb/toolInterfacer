#!/usr/bin/env python3
"""Deterministic pre-filter in front of the hosted routing call.

The point of this module is to make the "how many intents can a rule absorb
before the model is asked?" question measurable instead of assumed. It is a
plain regex cascade over the utterance: no model, no network, no state.

Rules are grouped in three tiers by how much of the surface form has to line up:

  tier 1  an unambiguous token (a URL, an r/ path, an explicit "look up")
  tier 2  a shape word plus a verb ("list" + "files", "where is" + "defined")
  tier 3  a single keyword that a neighbouring intent also uses

`RULES` order is priority order: first match wins. Precision and coverage are
*not* asserted here — bench/cost_curve.py measures both per rule against a
recorded run and ranks rules by measured precision, which is what the absorption
curve is built from.
"""
from __future__ import annotations

import re

# (name, label, tier, pattern)
RULES: list[tuple[str, str, int, re.Pattern]] = [
    ("url", "web_fetch", 1, re.compile(r"https?://\S+", re.I)),
    ("reddit_path", "reddit_digest", 1, re.compile(r"\br/[A-Za-z0-9_]+", re.I)),
    ("lookup_verb", "web_search", 1,
     re.compile(r"\b(look up|search the web|google|find out what|who ships|"
                r"who else has|is there anything published|what's out there|"
                r"anything out there|any chatter|find chatter|show me what's out there)\b", re.I)),
    ("call_site", "filesys_grep", 1,
     re.compile(r"\b(where is|where are|where do we|call site|call sites|"
                r"references? to|import yaml|grep the tree|locate each)\b", re.I)),
    ("ext_ending", "filesys_glob", 1,
     re.compile(r"\b(ending in|ends? in|end in)\s+\.?[a-z]{1,5}\b", re.I)),
    ("body_of_file", "filesys_read", 1,
     re.compile(r"\b(print the whole|print the|what's inside|what's written in|"
                r"body of the|dump the .* from|show .* on screen|read through|"
                r"full text of|full listing of|walk me through)\b", re.I)),
    ("phrase_variants", "decide_extract", 1,
     re.compile(r"\b(phrasings?|wordings?|keywords?|search terms|search string|"
                r"candidate titles|alternate ways to ask|distinct ways to ask|"
                r"keyword variants|angles to try|literal paths)\b", re.I)),
    ("pick_best", "decide_pick", 1,
     re.compile(r"\b(most useful|strongest match|keep the best|narrow these|"
                r"point me at the right|which one should we|sort these|sort the|"
                r"which is the right one)\b", re.I)),
    ("listing_verb", "filesys_glob", 2,
     re.compile(r"\b(list|which|names of|how many|find all|find every|are there any|"
                r"do we have any|work out the paths|the collection of)\b", re.I)),
    ("path_container", "filesys_glob", 2,
     re.compile(r"\b(files?|docs?|scripts?|dotfiles|fixtures|matches|paths?)\b.*"
                r"\b(in|under|here|left|live|lying|matching|start)\b", re.I)),
    ("digest_cue", "reddit_digest", 2,
     re.compile(r"\b(digest|summarize .* (posts|thread)|top threads|top comments|"
                r"the gist|catch me up|what are people in|community saying|"
                r"any threads|did anyone respond|megathread)\b", re.I)),
    ("fetch_cue", "web_fetch", 2,
     re.compile(r"\b(grab the page|load the|open the (blog|linked|arxiv|page)|"
                r"get me the content|get me whatever lives|fetch the|"
                r"download the text|carry on reading|click through)\b", re.I)),
    ("web_vocab", "web_search", 2,
     re.compile(r"\b(any recent|latest|benchmarks comparing|coverage of|official docs|"
                r"perf numbers|alternate writeups|debate|regression|release notes)\b", re.I)),
    ("extract_cue", "decide_extract", 2,
     re.compile(r"\b(extract|turn .* into|pull the|pull out the|name the |"
                r"write out the|list the exact)\b", re.I)),
    ("read_cue", "filesys_read", 2,
     re.compile(r"\b(pyproject\.toml|makefile|\.md\b|\.txt\b|\.yaml\b|\.yml\b|\.py\b)", re.I)),
]


def classify(text: str) -> tuple[str | None, str | None]:
    """(label, rule name) for one utterance; (None, None) when no rule fires."""
    for name, label, _tier, pattern in RULES:
        if pattern.search(text or ""):
            return label, name
    return None, None


def rule_of(name: str) -> tuple[str, str, int]:
    """(label, name, tier) for a rule name."""
    for rule_name, label, tier, _pattern in RULES:
        if rule_name == name:
            return label, rule_name, tier
    raise KeyError(name)
