"""rule_extract — deterministic, model-free slot extraction (the `extract` kind).

`extract` is not a classification: the caller hands a value_list schema and
wants the literals themselves back. This backend answers by rule — regex plus
stopword ranking, no model call, no network, no sampling — so `decide_extract`
is replayable and instant, and the two real uses (research queries, the literal
paths/urls a request names) work before any extractor model exists.

STATE / SCHEMA CONTRACT
    state  = {"context": <caller ctx: dict | list | str>, "instructions": str}
    schema = {"type": "value_list", "count": N}
    return = {"values": [str, ...], "basis": <how they were produced>}

NO CONFIDENCE, AND WHY
A rule extractor has no distribution to report: it matched a pattern or it did
not. The rubric this module used to return (0.92 urls, 0.72 queries, 0.30 for a
literal ask that found nothing, ...) was a set of hand-picked constants dressed
as measurements, and `decide_extract`'s conf is gated downstream, so a caller
that thresholds 0.92 against the slot gate is thresholding a number nobody
measured. So the backend reports `basis` instead: the pattern class that produced
the values, which is a fact about what happened.

    url_literals          http(s) literals, in text order
    path_literals         path-like tokens, in text order
    query_phrases         ranked keyword phrases
    quoted_spans          generic pass: quoted spans (+ distinctiveness fill)
    distinctive_tokens    generic pass: salient tokens only
    none                  nothing extractable — an honest empty answer

A basis that names a different class than the ask named is the degraded case the
old 0.30 cap encoded: "extract the urls" answered by `distinctive_tokens` is a
guess, and now it says so instead of pricing itself at 0.3.

SEAM FOR A MODEL-BACKED EXTRACTOR
Any backend registered as a DecideBackend whose `extract` takes that same state
and schema and returns `{"values", ...}` is a drop-in replacement: name it in
conf.decide.backends_in_use["extract"] ("rule_extract" -> e.g. "d1_systemone")
and nothing above the primitive wall moves — ops/decisions.py `run_extract`
reads values, and reads `conf` only when the backend measured one. A model-backed
extractor that really does produce a distribution can return `conf`; one that
cannot should omit it rather than invent it, and the run then reports no
confidence for that step. Keep the rule backend registered as the
offline/fixture lane for bench and tests.

MODE SELECTION (instructions name the ask; context supplies the payload)
    url      the ask names urls/links     -> http(s) literals, in text order
    path     the ask names files/paths    -> path-like tokens, in text order
    query    the ask names queries/terms  -> ranked keyword phrases (count=N)
    generic  no signal in the ask: fall back to the payload's own structure —
             urls, else paths, else quoted spans + distinctive tokens

A literal ask outranks a verb in a tie, because naming a literal type is the
stronger signal ("list the urls to search" is a url ask, not a query ask). A
literal ask that finds nothing degrades to the generic pass rather than
returning an empty list silently; the basis says which class produced the
values, so the degradation is visible. A URL in the ask counts as a literal too
(unambiguous); a slash in the ask usually is prose, so paths are only mined from
the context.

QUERY GENERATION (count clamped to 3..6; the shipped conf.research.query_count
is [3, 6] for the same reason — one query is not a search). Salient terms are
dropped through a stopword list, ranked by frequency then first occurrence (both
stable, so identical input gives identical output), and then recombined into
keyword phrases: the leading trigram, its bigram and head term, ranked-term
pairs, and document-order windows. Every phrase is distinct by construction.
"""
from __future__ import annotations

import re
from typing import Any

from ..primitives import ChoiceAnswer, DecideBackend, register

_MAX_CHARS = 20_000
_MAX_DEPTH = 8

_URL_RE = re.compile(r"https?://[^\s<>\"'`]+", re.I)
_QUOTED_RE = re.compile(r"\"([^\"\n]{2,120})\"|'([^'\n]{2,120})'|`([^`\n]{2,120})`")
_TOKEN_RE = re.compile(r"[^\s,;\"'`()\[\]{}<>]+")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#._-]*")
_SEGMENT_RE = re.compile(r"[\w.@+-]+")

_URL_ASK = tuple(re.compile(p, re.I) for p in (
    r"\burls?\b", r"\blinks?\b", r"\bhyperlinks?\b", r"\bhttps?\b",
    r"\bhrefs?\b", r"\bweb[ -]?address(?:es)?\b",
))
_PATH_ASK = tuple(re.compile(p, re.I) for p in (
    r"\bpaths?\b", r"\bfiles?(?:names?)?\b", r"\bdirs?\b", r"\bdirector(?:y|ies)\b",
    r"\bfolders?\b", r"\bglobs?\b", r"\bfilepaths?\b",
))
_QUERY_ASK = tuple(re.compile(p, re.I) for p in (
    r"\bquer(?:y|ies)\b", r"\bsearch(?:es|ing)?\b", r"\bkeywords?\b", r"\bterms?\b",
    r"\bphras(?:e|es|ing|ings)\b", r"\bwordings?\b", r"\bwords?\b", r"\bvariants?\b",
    r"\bstrings?\b",
))

_EXTENSIONS = frozenset("""
py pyi md txt rst json jsonl yaml yml toml cfg ini conf sh bash zsh csv tsv
html htm css scss js jsx ts tsx mjs cjs go rs c h cc cpp cxx hpp java kt rb
php sql xml pdf log ipynb lock env service proto gradle
""".split())

_STOPWORDS = frozenset("""
a about above after again against all also am an and any anything are around as
at back be because been before being below between both but by can cannot could
did do does doing done down during each either else enough etc even ever every
few find first for from further get gets give given go going got had has have
having he her here hers herself him himself his how however i if in into is it
its itself just know let like made make makes many may me might more most much
must my myself need needs no nor not now of off on once one ones only or other
others our ours out over own per please put rather really same say says see seem
seems several she should show so some something still such take tell than that
the their theirs them themselves then there these they thing things this those
though through thus to too two under until up upon us use used uses using via
very vs want wants was way we well were what whatever when where whether which
while who whom whose why will with within without would yes yet you your yours
""".split())


@register("rule_extract")
class RuleExtractBackend(DecideBackend):
    """Extract-only backend: answers the `extract` primitive, refuses the rest."""

    name = "rule_extract"

    def choice(self, state: Any, instructions: str, criteria: dict[str, str]) -> ChoiceAnswer:
        raise NotImplementedError(
            "rule_extract: extract-only backend; it serves no choice decisions. "
            "Point conf.decide.backends_in_use.choice (or the cascade) at a "
            "choice backend.")

    def noul(self, state: Any, question: str) -> float:
        raise NotImplementedError(
            "rule_extract: extract-only backend; it serves no noul decisions. "
            "Point conf.decide.backends_in_use.noul at a noul backend.")

    def extract(self, state: Any, schema: dict[str, Any]) -> dict[str, Any]:
        if isinstance(state, dict):
            context, instructions = state.get("context"), str(state.get("instructions") or "")
        else:
            context, instructions = state, ""
        payload = _flatten(context)
        count = _query_count(schema)
        values, basis = _run(payload, instructions, count)
        return {"values": values, "basis": basis}


def _run(payload: str, instructions: str, count: int) -> tuple[list[str], str]:
    """Extract by mode. Every branch returns (values, basis); none invents a number."""
    mode = _ask_mode(instructions) or _infer_mode(payload)
    if mode == "url":
        urls = _urls(payload, instructions)
        if urls:
            return urls, "url_literals"
        return _generic(payload, count)
    if mode == "path":
        paths = _paths(payload)
        if paths:
            return paths, "path_literals"
        return _generic(payload, count)
    if mode == "query":
        phrases = _queries(payload, count)
        if phrases:
            return phrases, "query_phrases"
        return [], "none"
    return _generic(payload, count)


def _ask_mode(instructions: str) -> str | None:
    """Which literal the ask names, or None when it names none of them."""
    hits = {"url": _hits(_URL_ASK, instructions),
            "path": _hits(_PATH_ASK, instructions),
            "query": _hits(_QUERY_ASK, instructions)}
    if not any(hits.values()):
        return None
    order = ("url", "path", "query")
    return max(order, key=lambda kind: (hits[kind], -order.index(kind)))


def _hits(patterns: tuple[re.Pattern[str], ...], text: str) -> int:
    return sum(1 for p in patterns if p.search(text))


def _infer_mode(payload: str) -> str:
    """No signal in the ask: trust the payload's own structure, most specific first.

    Path inference is strict (a separator must be present): without an ask that
    names files, a bare ``llama.cpp`` is a project, not a path.
    """
    if _URL_RE.search(payload):
        return "url"
    if _paths(payload, strict=True):
        return "path"
    return "generic"


def _query_count(schema: dict[str, Any] | None) -> int:
    """schema["count"] clamped to 3..6 — the query band conf.research mirrors."""
    raw = (schema or {}).get("count", 3)
    try:
        n = int(raw)
    except (TypeError, ValueError):
        n = 3
    return max(3, min(6, n))


def _flatten(value: Any, depth: int = 0) -> str:
    """Every string inside a JSON-safe context, in document order, as one blob.

    The context arrives as whatever the previous step stored — a digest dict, a
    list of hits, a raw page string, or {} when the step has no source. Keys are
    dropped: they carry step vocabulary ("candidates", "cwd"), not content.
    """
    parts: list[str] = []

    def walk(node: Any, level: int) -> None:
        if level > _MAX_DEPTH:
            return
        if isinstance(node, str):
            parts.append(node)
        elif isinstance(node, dict):
            for item in node.values():
                walk(item, level + 1)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item, level + 1)
        elif node is not None:
            parts.append(str(node))

    walk(value, depth)
    return "\n".join(parts)[:_MAX_CHARS]


def _urls(*texts: str) -> list[str]:
    """http(s) literals in text order, deduped, trailing punctuation trimmed."""
    out: list[str] = []
    for text in texts:
        for raw in _URL_RE.findall(text):
            url = raw.rstrip(".,;:!?)]}'\"")
            if url and url not in out:
                out.append(url)
    return out


def _paths(text: str, strict: bool = False) -> list[str]:
    """Path-like tokens in text order: ./x, ../x, ~/x, /a/b, dir/file.ext, name.ext."""
    out: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        token = raw.rstrip(".,;:!?")
        if _is_path(token, strict) and token not in out:
            out.append(token)
    return out


def _is_path(token: str, strict: bool = False) -> bool:
    if "://" in token or len(token) < 3:
        return False
    if token.startswith(("./", "../", "~/")) or token.startswith("/"):
        return True
    if "/" in token:
        segments = [s for s in token.split("/") if s]
        if len(segments) >= 2 and all(_SEGMENT_RE.fullmatch(s) for s in segments):
            # "and/or" is prose, not a path: an all-stopword split is not a literal
            return not all(s.lower() in _STOPWORDS for s in segments)
        return False
    # a bare name.ext is a path only when the ask already said files (_infer_mode is strict)
    return (not strict) and "." in token and token.rsplit(".", 1)[-1].lower() in _EXTENSIONS


def _queries(payload: str, count: int) -> list[str]:
    """3..count distinct keyword phrases from the payload, deterministic order."""
    order, counts = _terms(payload)
    if not order:
        return []
    return _phrases(order, _ranked(order, counts), count)


def _terms(text: str) -> tuple[list[str], dict[str, int]]:
    """Salient terms in first-occurrence order, plus their frequencies.

    A term is a letter-initial word (so "llama.cpp", "c++", "gpt-4o" survive)
    that is not a stopword. Digits-only tokens and stopwords are dropped.
    """
    order: list[str] = []
    counts: dict[str, int] = {}
    for raw in _WORD_RE.findall(text):
        word = raw.strip("._-+").lower()
        if len(word) < 3 or word in _STOPWORDS:
            continue
        if word not in counts:
            order.append(word)
        counts[word] = counts.get(word, 0) + 1
    return order, counts


def _ranked(order: list[str], counts: dict[str, int]) -> list[str]:
    """Frequency first, then first occurrence, then alphabetical — total order."""
    first = {term: i for i, term in enumerate(order)}
    return sorted(order, key=lambda term: (-counts[term], first[term], term))


def _phrases(order: list[str], ranked: list[str], count: int) -> list[str]:
    """Recombine ranked terms into distinct keyword phrases, most informative first."""
    phrases: list[str] = []

    def add(phrase: str) -> None:
        if phrase and phrase not in phrases:
            phrases.append(phrase)

    add(" ".join(order[:3]))          # the leading trigram: already a query shape
    add(" ".join(order[:2]))          # its bigram
    add(order[0])                     # the head term alone
    top = ranked[:6]
    for i in range(len(top)):         # ranked pairs (covers non-adjacent terms)
        for j in range(i + 1, len(top)):
            add(f"{top[i]} {top[j]}")
    for start in range(1, min(len(order), 6)):  # document-order windows
        add(" ".join(order[start:start + 2]))
    add(" ".join(ranked[:3]))         # the ranked trigram
    return phrases[:count]


def _generic(payload: str, count: int) -> tuple[list[str], str]:
    """Quoted spans first, then distinctive tokens; nothing at all -> ([], 'none')."""
    values: list[str] = []
    for match in _QUOTED_RE.finditer(payload):
        span = next((g for g in match.groups() if g), "").strip()
        if span and span not in values:
            values.append(span)
    quoted = len(values)
    order, counts = _terms(payload)
    first = {term: i for i, term in enumerate(order)}
    distinctive = sorted((t for t in order if len(t) >= 5),
                         key=lambda t: (-counts[t], -len(t), first[t], t))
    for term in distinctive:
        if len(values) >= count:
            break
        if term not in values:
            values.append(term)

    if not values:
        return [], "none"
    return values, ("quoted_spans" if quoted else "distinctive_tokens")
