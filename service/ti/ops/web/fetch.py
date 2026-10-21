"""web.fetch — URLs in, page payloads out.

Each URL is its own CLI invocation (``ops.web_fetch.script`` — the ``web-fetch``
skill script) with ``--no-preview``, so the machine footer names the artifact;
``--max-chars`` comes from ``ops.web_fetch.max_chars`` (20000, the CLI's own
default).

One URL per process on purpose: the CLI's ``--batch`` mode prints no footer,
mints one run directory per URL and exits 0 even when every URL failed, so batch
mode would cost the artifact-to-URL correlation that is the whole point here.

Data is one entry per input URL, in input order:

    {"url": str, "content": str, "method": str | None, "error": str | None}

A failed page is skipped in the only sense a downstream consumer cares about —
``content`` is ``""`` — but it keeps its entry and its error, because "this page
failed" and "this URL was never fetched" must not look alike. The CLI signals a
total failure by *omitting* ``text``, not by setting ``error``; ``degraded`` (a
meta-description shell extracted instead of the page) is passed through as-is.

conf is the mean over URLs of 1.0 (content present and not degraded), 0.5
(degraded) and 0.0 (nothing usable). A page that failed is data the picker and
the gate can act on, so it does not raise; a missing or unreadable configured
script is a config error and does.
"""
from __future__ import annotations

from typing import Any

from .. import cli
from ..op_api import op

DEFAULT_MAX_CHARS = 20000


@op("web_fetch", kind="pages", required=["urls"], optional=["budget_notes"])
def run(params: dict[str, Any], store: Any) -> tuple[str, Any, float]:
    _ = store
    ops = cli.ops_conf()
    urls = as_urls(params.get("urls"))
    if not urls:
        return "pages", [], 0.0
    fetch_cfg = ops.get("web_fetch") or {}
    script = fetch_cfg.get("script")
    if not script:
        raise cli.CliMissingBinary("ops.web_fetch.script is not configured")
    cli.launcher(str(script))   # a missing script is a config error, not a page result
    max_chars = int(fetch_cfg.get("max_chars") or DEFAULT_MAX_CHARS)
    timeout_s = cli.timeout_from_conf(ops)

    pages: list[dict[str, Any]] = []
    scores: list[float] = []
    for url in urls:
        argv = cli.cli_argv(str(script), ["--max-chars", str(max_chars), "--", url], ops)
        try:
            _run, doc = cli.run_result(argv, timeout_s, tolerate=(1,))
        except cli.CliError as exc:
            pages.append({"url": url, "content": "", "method": None, "error": str(exc)})
            scores.append(0.0)
            continue
        page, score = _page(url, doc)
        pages.append(page)
        scores.append(score)
    return "pages", pages, sum(scores) / len(scores)


def as_urls(raw: Any) -> list[str]:
    """Coerce the param to a list of strings.

    ``urls`` is injected from the ``picked`` step; the runner unwraps a
    1-element ``picked_items`` artifact to a scalar, so a single pick arrives as
    a bare string.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    return [str(u).strip() for u in raw if str(u).strip()]


def _page(url: str, doc: dict[str, Any]) -> tuple[dict[str, Any], float]:
    content = str(doc.get("text") or "")
    error = doc.get("error") or (None if content else "no text extracted")
    page = {"url": str(doc.get("url") or url),
            "content": content,
            "method": doc.get("source"),
            "error": str(error) if error else None}
    if not content:
        return page, 0.0
    return page, 0.5 if doc.get("degraded") else 1.0
