"""shape — is this disk intent enumerative or singular?

A listing and a read are different plans, not different confidences: "what files
are in X" has no single file to pick, so no confidence can make the read shape
right. This module answers that shape question and nothing else.

Rule-first and lexical only: no model call, no network, no added latency. Anything
unrecognised is SINGULAR, the shape shipped before the split — the classifier is
only allowed to move an intent off the read path when the wording says listing.
"""
from __future__ import annotations

import re

ENUMERATIVE = "enumerative"
SINGULAR = "singular"

# Listing verbs and listing-shaped noun phrases. Checked against the lowercased
# intent; each rule is a phrase a caller actually types, not a bag of keywords.
_ENUMERATIVE_RULES = (
    re.compile(r"\bls\b"),
    re.compile(r"\blist\b"),
    re.compile(r"\benumerat\w*"),
    re.compile(r"\b(?:what|which)\s+(?:files|entries)\b"),
    re.compile(r"\bfiles\s+(?:in|under|inside|within)\b"),
    re.compile(r"\b(?:show|give|display|print)\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?"
               r"(?:files|entries|listing|contents)\b"),
)

# "what's in this directory" — the container noun is what makes it a listing;
# "what's in README.md" has none and stays singular.
_WHAT_IS_IN = re.compile(r"\bwhat(?:'s| is| are|s)?\b.*\b(?:in|inside|under|within)\b")
_CONTAINER = re.compile(r"\b(?:dir|dirs|directory|directories|folder|folders|tree)\b")


def classify(intent: str) -> str:
    text = (intent or "").lower()
    if any(rule.search(text) for rule in _ENUMERATIVE_RULES):
        return ENUMERATIVE
    if _WHAT_IS_IN.search(text) and _CONTAINER.search(text):
        return ENUMERATIVE
    return SINGULAR
