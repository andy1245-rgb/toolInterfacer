"""similarity — normalised-text near-duplicate detection (dedup and leakage guard).

One definition, used for two jobs that must not disagree:

  * dedup  a generated utterance that restates another generated utterance is
           wasted volume, so it is dropped at generation time;
  * leak   an utterance that restates one of the frozen 150 eval rows is
           contamination, so it is refused at generation *and* re-refused at
           label time before any row is written.

Three measures, cheapest first: normalised equality, token-set Jaccard, and
character-level ``difflib.SequenceMatcher`` ratio. A pair is a duplicate if any
one of them trips. The method string travels into the generator's manifest so a
report can state what "deduplicated" meant instead of asserting it.

Why not embeddings: stdlib only, and this decision is lexical by construction —
the frozen rows are themselves short lexical probes, so cosine on a local model
would only re-derive the Jaccard signal at the cost of a dependency.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import pathlib
import re
import unicodedata
from dataclasses import dataclass

# A pair is a duplicate when ANY measure trips. Both are deliberately
# loose-jaccard / tight-ratio: short imperatives share most of their tokens, so
# Jaccard alone would merge "read the pyproject but not the setup.cfg" into
# "read the setup.cfg"; the character ratio separates them again.
JACCARD_NEAR = 0.75
RATIO_NEAR = 0.85
EXACT = 1.0

# The leakage guard is deliberately *stricter* than dedup, because the two decisions
# have different costs. Dropping a near-dup costs one row; letting one through costs
# the credibility of every number measured on the held-out set. So the guard fires on
# paraphrases of an eval item ("fetch json.html" ~ "load ssl.html" is the same frame
# with a different object, and the eval is not allowed to be that close).
LEAK_JACCARD = 0.70
LEAK_RATIO = 0.80

# The ratio sweep is bounded: token-set Jaccard over every reference is cheap (it is a
# set intersection on cached token sets), while SequenceMatcher is not, so only the
# PATIENCE best Jaccard candidates get an exact character ratio. A pair can only trip
# LEAK_RATIO with a Jaccard far below the top of that list if the two strings share
# almost no tokens yet almost the same characters, which for word-shaped sentences is
# the empty set of cases — and this bound is what keeps a 400-row generation sweep
# costing a second instead of a minute.
PATIENCE = 3

METHOD = (f"leak: normalised-exact OR token-jaccard>={LEAK_JACCARD} OR "
          f"SequenceMatcher.ratio>={LEAK_RATIO}; "
          f"dedup: exact OR jaccard>={JACCARD_NEAR} OR ratio>={RATIO_NEAR} "
          f"(ratio over the top-{PATIENCE} Jaccard candidates; normalise: NFKC, "
          f"casefold, non-alphanumerics to space, whitespace collapse)")

_PUNCT = re.compile(r"[^0-9a-z]+")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """The comparison form of an utterance: NFKC, casefolded, punctuation-free."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    return _WS.sub(" ", _PUNCT.sub(" ", folded)).strip()


def tokens(text: str) -> set[str]:
    return set(normalize(text).split())


def jaccard(a: str, b: str) -> float:
    return _jaccard_norm(normalize(a), normalize(b))


def _jaccard_norm(na: str, nb: str) -> float:
    ta, tb = set(na.split()), set(nb.split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def ratio(a: str, b: str) -> float:
    return ratio_norm(normalize(a), normalize(b))


@dataclass(frozen=True)
class Comparison:
    exact: float
    jaccard: float
    ratio: float

    @property
    def duplicate(self) -> bool:
        return (self.exact >= EXACT or self.jaccard >= JACCARD_NEAR
                or self.ratio >= RATIO_NEAR)

    @property
    def leaks(self) -> bool:
        """Whether this pair collides with the held-out set (stricter than dedup)."""
        return (self.exact >= EXACT or self.jaccard >= LEAK_JACCARD
                or self.ratio >= LEAK_RATIO)

    @property
    def rule(self) -> str:
        """Which measure tripped — so a report can say what removed a row."""
        if self.exact >= EXACT:
            return "exact"
        if self.jaccard >= JACCARD_NEAR:
            return "jaccard"
        if self.ratio >= RATIO_NEAR:
            return "ratio"
        return ""

    def as_dict(self) -> dict[str, object]:
        return {"exact": round(self.exact, 3), "jaccard": round(self.jaccard, 3),
                "ratio": round(self.ratio, 3), "duplicate": self.duplicate,
                "leaks": self.leaks, "rule": self.rule}


def compare(a: str, b: str) -> Comparison:
    return compare_norm(normalize(a), normalize(b))


def ratio_norm(na: str, nb: str) -> float:
    """Character-level similarity of two already-normalised strings."""
    if not na and not nb:
        return 1.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


def compare_norm(na: str, nb: str) -> Comparison:
    """`compare` over already-normalised text — the cheap path, no ratio sweep."""
    if na and na == nb:
        return Comparison(EXACT, 1.0, 1.0)
    return Comparison(0.0, _jaccard_norm(na, nb), ratio_norm(na, nb))


class Corpus:
    """A set of reference utterances, queryable for the closest match.

    Token sets and normalised strings are cached at insertion, so a sweep is a set
    intersection per reference and an exact character ratio only for the few closest.
    """

    def __init__(self, utterances: list[str] | None = None, label: str = "reference") -> None:
        self.label = label
        self.utterances: list[str] = []
        self._normalized: list[str] = []
        self._tokens: list[set[str]] = []
        for text in utterances or []:
            self.add(text)

    def add(self, utterance: str) -> None:
        norm = normalize(utterance)
        self.utterances.append(utterance)
        self._normalized.append(norm)
        self._tokens.append(set(norm.split()))

    def __len__(self) -> int:
        return len(self.utterances)

    @property
    def sha(self) -> str:
        blob = "\n".join(sorted(self._normalized))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    @classmethod
    def from_json(cls, path: pathlib.Path | str, label: str | None = None) -> "Corpus":
        """Read a rows file: a JSON list of dicts with an ``utterance`` key, or JSONL."""
        p = pathlib.Path(path)
        text = p.read_text(encoding="utf-8")
        rows: list[dict]
        if p.suffix == ".jsonl" or (text.lstrip()[:1] not in ("[", "{")):
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        elif text.lstrip().startswith("["):
            rows = json.loads(text)
        else:  # a single object: {id: {...}} or {"rows": [...]}
            obj = json.loads(text)
            rows = obj if isinstance(obj, list) else obj.get("rows", [])
        return cls([str(r["utterance"]) for r in rows], label or p.stem)

    def match(self, utterance: str) -> tuple[Comparison, str | None]:
        """The closest reference row and how close it is."""
        if not self.utterances:
            return Comparison(0.0, 0.0, 0.0), None
        norm = normalize(utterance)
        union = set(norm.split())
        scored: list[tuple[float, int]] = []
        for index, tokens in enumerate(self._tokens):
            if self._normalized[index] == norm:
                return Comparison(EXACT, 1.0, 1.0), self.utterances[index]
            shared = len(union & tokens)
            if not shared:
                continue
            scored.append((shared / len(union | tokens), index))
        if not scored:
            return Comparison(0.0, 0.0, 0.0), None
        scored.sort(reverse=True)
        best: tuple[Comparison, str] | None = None
        for jaccard_value, index in scored[:PATIENCE]:
            candidate = Comparison(0.0, jaccard_value,
                                   ratio_norm(norm, self._normalized[index]))
            if best is None or candidate.leaks or (
                    not best[0].leaks and candidate.ratio > best[0].ratio):
                best = (candidate, self.utterances[index])
            if candidate.leaks:
                break
        comparison, hit = best if best else (Comparison(0.0, 0.0, 0.0), None)
        return comparison, hit
