"""digest_assemble — the research pipeline's terminal step: pages in, digest out.

Spec §7.7 asks the last step for "per-source claim digest + link + confidence".
That is assembly, not decision: the pages reaching this op already cleared the
plan's per-page `noul:is_relevant` gate, so the remaining work is to attribute
each one, lift a citable gist out of it, and grade the result. No model call
happens here — a source's relevance was decided upstream, and re-deciding it here
would put a second, differently-tuned verdict in the same plan.

    {"intent": str,
     "queries": [str],                      # what was searched
     "sources": [{"url", "title", "claim", "chars", "method", "conf"}],
     "skipped": [{"url", "reason"}],        # a page that carried no text
     "threads": {…} | None,                 # the reddit cross-check, when it ran
     "coverage": {"sources": int, "target": int},
     "confidence": float}

CLAIM is rule-derived, not extracted: the page's leading text, whitespace-flat,
cut on a word boundary at `research.digest.claim_chars`. A model-generated claim
would need an `extract` call per page and a claim schema the primitive wall does
not have yet, so v1 ships the gist and says so.

SOURCE CONF is the page's evidence grade, on `web_fetch`'s own scale — 1.0 full
text, 0.5 a degraded extraction (a meta-description shell). Empty pages never
become sources, so a listed source is at least readable.

CONF is coverage, not decision confidence (the two are different questions and
must not be averaged): `min(1, sources / research.good_pages_target)`. A digest
built from two of the four pages the plan wanted is half the answer, and saying
so is the point of returning a number at all. It is 0.0 when nothing survived —
an empty digest that reports high confidence is worse than an error.
"""
from __future__ import annotations

from typing import Any

from ..op_api import op
from ..web.search import normalize_url

DEFAULT_CLAIM_CHARS = 360
DEFAULT_TARGET = 4

#: Thread fields a digest cites. The rest of a reddit post is payload the caller
#: did not ask for and would have to pay to read.
THREAD_FIELDS = ("id", "title", "subreddit", "author", "permalink", "score")
MAX_THREADS = 5


@op("digest_assemble", kind="digest", required=["intent"],
    optional=["pages", "hits", "threads", "queries"])
def run(params: dict[str, Any], store: Any) -> tuple[str, Any, float]:
    _ = store
    intent = str(params.get("intent") or "").strip()
    if not intent:
        raise ValueError("digest_assemble requires the intent the digest answers")
    claim_chars, target = _degrees()

    titles = _titles(params.get("hits"))
    sources: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for page in _items(params.get("pages")):
        if not isinstance(page, dict):
            continue
        url = str(page.get("url") or "")
        content = str(page.get("content") or "")
        if not content.strip():
            skipped.append({"url": url, "reason": str(page.get("error") or "no text")})
            continue
        degraded = bool(page.get("degraded"))
        sources.append({"url": url,
                        "title": titles.get(normalize_url(url), ""),
                        "claim": claim_of(content, claim_chars),
                        "chars": len(content),
                        "method": page.get("method"),
                        "conf": 0.5 if degraded else 1.0})

    threads = _threads(params.get("threads"))
    confidence = min(1.0, len(sources) / target) if sources else 0.0
    data = {"intent": intent,
            "queries": [str(q) for q in _items(params.get("queries"))],
            "sources": sources,
            "skipped": skipped,
            "threads": threads,
            "coverage": {"sources": len(sources), "target": target},
            "confidence": round(confidence, 3)}
    return "digest", data, round(confidence, 3)


def claim_of(text: str, chars: int) -> str:
    """The page's leading gist: flat whitespace, cut on a word boundary."""
    flat = " ".join(text.split())
    if len(flat) <= chars:
        return flat
    head = flat[:chars]
    space = head.rfind(" ")
    if space > chars // 2:
        head = head[:space]
    return head.rstrip(" ,;:-") + "…"


def _degrees() -> tuple[int, int]:
    """(claim_chars, good-page target) — the degrees, which live in conf."""
    from ...conf import load
    research = load().get("research") or {}
    digest = research.get("digest") or {}
    try:
        chars = max(1, int(digest.get("claim_chars", DEFAULT_CLAIM_CHARS)))
    except (TypeError, ValueError):
        chars = DEFAULT_CLAIM_CHARS
    try:
        target = max(1, int(research.get("good_pages_target", DEFAULT_TARGET)))
    except (TypeError, ValueError):
        target = DEFAULT_TARGET
    return chars, target


def _items(raw: Any) -> list[Any]:
    if raw is None or raw == "(missing)":
        return []
    return raw if isinstance(raw, list) else [raw]


def _titles(hits: Any) -> dict[str, str]:
    """URL → title, keyed the way `web_search` dedupes, so attribution is exact."""
    out: dict[str, str] = {}
    for hit in _items(hits):
        if isinstance(hit, dict) and hit.get("url"):
            out[normalize_url(str(hit["url"]))] = str(hit.get("title") or "")
    return out


def _threads(raw: Any) -> dict[str, Any] | None:
    """The cross-check, cited rather than copied: a few posts and where they came from."""
    if not isinstance(raw, dict):
        return None
    posts = [post for post in (raw.get("posts") or []) if isinstance(post, dict)]
    return {"query": raw.get("query"),
            "subreddit": raw.get("subreddit"),
            "fetched_at": raw.get("fetched_at"),
            "posts": [{k: post.get(k) for k in THREAD_FIELDS}
                      for post in posts[:MAX_THREADS]]}
