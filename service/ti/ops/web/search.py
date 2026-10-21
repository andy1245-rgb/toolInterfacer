"""web.search — search queries in, one deduped URL list out.

Backend: a configured search CLI (``ops.web_search.script`` by default — the
``free-search`` skill script). One invocation per query, because neither CLI has
a batch form; ``--no-preview`` keeps stdout down to the machine footer, and the
artifact named by ``RESULT_FILE`` is what gets parsed.

Two contracts are supported, selected by the ``backend`` param (or
``ops.web_search.backend``):

    free-search  {"results":  [{title, url, snippet, source_count, sources}]}
    web-search   {"manifests": [{provider, results: [{title, url, snippet}]}]}

Both normalize into one item shape:

    {"title": str, "url": str, "snippet": str,
     "sources": [str], "source_count": int, "queries": [str]}

Dedupe is by normalized URL (fragment dropped, scheme and host lowercased) with
sources and originating queries unioned — the point of merging queries is to
know a URL was hit twice, not to keep two copies of it.

conf is a coverage score, not a decision confidence: 1.0 when every query
produced at least one hit, 0.5 when only some did, 0.0 when none did. It is what
the ``gate_slot`` check ahead of ``web_fetch`` reads, so a half-answered search
must neither authorize a full fetch fan-out nor look like total failure.

A per-query CLI failure is data — that query counts as unanswered and the
coverage score drops. A missing or unreadable configured script is a config
error and raises. So does "every query failed": an empty ``url_list`` and
"nothing worked at all" must not look alike.
"""
from __future__ import annotations

from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

from .. import cli
from ..op_api import op

DEFAULT_BACKEND = "free-search"

ArgvBuilder = Callable[[str, str, dict[str, Any]], list[str]]
ItemParser = Callable[[dict[str, Any]], list[dict[str, Any]]]

_BACKENDS = {DEFAULT_BACKEND, "web-search", "provider"}


@op("web_search", kind="url_list", required=["queries"], optional=["backend"])
def run(params: dict[str, Any], store: Any) -> tuple[str, Any, float]:
    _ = store
    ops = cli.ops_conf()
    queries = as_queries(params.get("queries"))
    if not queries:
        return "url_list", [], 0.0
    backend = str(params.get("backend") or ops.get("web_search", {}).get("backend")
                  or DEFAULT_BACKEND)
    script, build, parse = _select_backend(backend, ops)
    cli.launcher(str(script))   # a missing script is a config error, not a query result
    timeout_s = cli.timeout_from_conf(ops)

    merged: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    answered = 0
    for query in queries:
        try:
            run_, doc = cli.run_result(build(script, query, ops), timeout_s, tolerate=(1,))
        except cli.CliError as exc:
            failures.append(f"{query!r}: {exc}")
            continue
        if run_.exit_code != 0:
            failures.append(f"{query!r}: exit {run_.exit_code}, artifact still read")
        items = parse(doc)
        if items:
            answered += 1
            for item in items:
                _merge(merged, item, query)

    results = [dict(item) for item in merged.values()]
    if not results:
        if failures:
            raise cli.CliFailed(
                f"web_search: no results and every query reported a problem:\n  "
                + "\n  ".join(failures))
        return "url_list", [], 0.0
    return "url_list", results, 1.0 if answered == len(queries) else 0.5


def as_queries(raw: Any) -> list[str]:
    """Coerce the param to a list of strings.

    ``queries`` is injected from a ``value_list`` decide step, but the runner
    unwraps a 1-element ``picked_items`` artifact to a scalar, so a single query
    arrives as a bare string.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    return [str(q).strip() for q in raw if str(q).strip()]


def _select_backend(backend: str,
                    ops: dict[str, Any]) -> tuple[str, ArgvBuilder, ItemParser]:
    if backend in ("web-search", "provider"):
        cfg = ops.get("web_search_provider") or {}
        script = cfg.get("script")
        if not script:
            raise cli.CliMissingBinary("ops.web_search_provider.script is not configured")
        return str(script), _argv_web_search, _items_from_web_search
    if backend == DEFAULT_BACKEND:
        script = (ops.get("web_search") or {}).get("script")
        if not script:
            raise cli.CliMissingBinary("ops.web_search.script is not configured")
        return str(script), _argv_free_search, _items_from_free_search
    raise ValueError(f"unknown web_search backend {backend!r}; "
                     f"known: {DEFAULT_BACKEND}, web-search")


def _argv_free_search(script: str, query: str, ops: dict[str, Any]) -> list[str]:
    args = ["--no-preview"]
    brave = str((ops.get("web_search") or {}).get("brave") or "auto").lower()
    if brave == "off":
        args.append("--no-brave")
    elif brave == "forced":
        args.append("--brave")
    return cli.cli_argv(script, args + ["--", query], ops)


def _argv_web_search(script: str, query: str, ops: dict[str, Any]) -> list[str]:
    provider = str((ops.get("web_search_provider") or {}).get("provider") or "duckduckgo")
    return cli.cli_argv(script, ["--provider", provider, "--no-preview", "--", query], ops)


def _items_from_free_search(doc: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for raw in doc.get("results") or []:
        if not isinstance(raw, dict):
            continue
        url = str(raw.get("url") or "").strip()
        if not url:
            continue
        sources = [str(s) for s in raw.get("sources") or []]
        items.append({"title": str(raw.get("title") or ""),
                      "url": url,
                      "snippet": str(raw.get("snippet") or ""),
                      "sources": sources,
                      "source_count": int(raw.get("source_count") or len(sources) or 1)})
    return items


def _items_from_web_search(doc: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for manifest in doc.get("manifests") or []:
        if not isinstance(manifest, dict):
            continue
        provider = str(manifest.get("provider") or "web-search")
        for raw in manifest.get("results") or []:
            if not isinstance(raw, dict):
                continue
            url = str(raw.get("url") or "").strip()
            if not url:
                continue
            items.append({"title": str(raw.get("title") or ""),
                          "url": url,
                          "snippet": str(raw.get("snippet") or ""),
                          "sources": [provider],
                          "source_count": 1})
    return items


def _merge(merged: dict[str, dict[str, Any]], item: dict[str, Any], query: str) -> None:
    key = normalize_url(item["url"])
    current = merged.get(key)
    if current is None:
        merged[key] = {**item, "queries": [query]}
        return
    for source in item["sources"]:
        if source not in current["sources"]:
            current["sources"].append(source)
    current["source_count"] = max(current["source_count"], item["source_count"],
                                  len(current["sources"]))
    if query not in current["queries"]:
        current["queries"].append(query)


def normalize_url(url: str) -> str:
    """Dedupe key: no fragment, lowercase scheme and host, query kept."""
    parts = urlsplit(url.strip())
    if not parts.scheme or not parts.netloc:
        return url.strip()
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path,
                       parts.query, ""))
