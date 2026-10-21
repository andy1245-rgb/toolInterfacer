"""reddit.digest — reddit-research CLI search, projected into a thread digest.

**The CLI this op drives is currently unavailable on this box.** The skill's
project lives on an external mount that is not mounted here, and the path
``service/ti/conf/defaults.json`` ships
(``~/Documents/PersonalCodingProjects/Research/redditScrapingSearching``) does
not exist either. So the op probes ``ops.reddit.cwd`` and every path in
``ops.reddit.cwd_candidates`` and, when none is a directory, raises
``CliMissingBinary`` listing what it tried. That is deliberate: an unmounted
project and a query that legitimately found no posts must not look alike, and a
no-op must not be reported as a success.

The contract it is written against comes from the ``reddit-research`` SKILL.md,
not from a running binary:

    cd <project> && <ops.reddit.cli…> search "<query>" [--subreddit N]
                      [--limit N] --json

That CLI prints no machine footer — its machine-readable handle is a *field*
(``raw_json``) inside the ``--json`` summary dict, a third advert convention
alongside ``RESULT_FILE:`` and ``--print-path-only``. The op reads the summary
off stdout, follows ``raw_json``, and takes that document's ``posts[]``.

``posts[]`` keys are **inferred, not observed**: no recorded response exists to
check them against, so each post is projected through a known-key list and any
absent field stays ``None`` rather than becoming an invented value. The first
session with the mount available should record a real response and replace the
guess — see toolInterfacerExtras/devskills/reddit-research/fixtures/README.md.

Data: ``{"query", "subreddit", "fetched_at", "posts": [post…]}`` with posts
capped at ``thread_cap`` (``ops.reddit.thread_cap``, default 5). conf is 1.0
with at least one post, else 0.0.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .. import cli
from ..op_api import op

DEFAULT_CLI = (".venv/bin/python", "-m", "reddit_research.cli")
DEFAULT_THREAD_CAP = 5

#: Projection of a reddit post onto the fields a digest can use. Inferred from
#: the SKILL.md's ``posts[]`` shape, not from an observed payload; unknown keys
#: are dropped and missing ones become None.
POST_FIELDS = ("id", "title", "subreddit", "author", "permalink", "url",
               "score", "num_comments", "created_utc", "selftext")


@op("reddit_digest", kind="thread_digest", required=["query"],
    optional=["subreddit", "thread_cap"])
def run(params: dict[str, Any], store: Any) -> tuple[str, Any, float]:
    _ = store
    ops = cli.ops_conf()
    reddit_cfg = ops.get("reddit") or {}
    query = str(params.get("query") or "").strip()
    if not query:
        raise ValueError("reddit_digest requires a non-empty query")
    subreddit = str(params.get("subreddit") or "").strip() or None
    cap = int(params.get("thread_cap") or _conf_thread_cap() or DEFAULT_THREAD_CAP)

    project = resolve_project(reddit_cfg)
    argv = _argv(project, reddit_cfg, query, subreddit, cap, ops)
    timeout_s = cli.timeout_from_conf(ops)
    _run, summary = cli.run_result(argv, timeout_s, cwd=str(project))

    raw_path = summary.get("raw_json")
    if not raw_path:
        raise cli.CliUnparseable(
            "reddit-research --json summary has no raw_json handle; "
            f"got keys {sorted(summary)}", _run)
    posts_doc = _read_posts(Path(os.path.expanduser(str(raw_path))), _run)
    posts = [_project(p) for p in posts_doc][:cap]
    data = {"query": str(summary.get("query") or query),
            "subreddit": summary.get("subreddit") or subreddit,
            "fetched_at": summary.get("fetched_at"),
            "posts": posts}
    return "thread_digest", data, 1.0 if posts else 0.0


def resolve_project(reddit_cfg: dict[str, Any]) -> Path:
    """First configured reddit-research project directory that exists."""
    candidates: list[str] = []
    if reddit_cfg.get("cwd"):
        candidates.append(str(reddit_cfg["cwd"]))
    candidates.extend(str(c) for c in reddit_cfg.get("cwd_candidates") or [] if c)
    tried: list[str] = []
    for candidate in candidates:
        path = Path(os.path.expanduser(candidate))
        if str(path) in tried:
            continue
        tried.append(str(path))
        if path.is_dir():
            return path
    if not tried:
        raise cli.CliMissingBinary(
            "reddit-research is not configured: set ops.reddit.cwd to the project "
            "directory")
    raise cli.CliMissingBinary(
        "reddit-research CLI unavailable: no project directory found; tried:\n  "
        + "\n  ".join(tried)
        + "\nThe project lives on an external mount — mount it, or point "
          "ops.reddit.cwd at it, then record a real response for the fixture tree.")


def _conf_thread_cap() -> int | None:
    from ...conf import load
    value = (load().get("reddit") or {}).get("thread_cap")
    return int(value) if value else None


def _argv(project: Path, reddit_cfg: dict[str, Any], query: str,
          subreddit: str | None, cap: int, ops: dict[str, Any]) -> list[str]:
    cli_parts = list(reddit_cfg.get("cli") or DEFAULT_CLI)
    entry = project / cli_parts[0]
    if not entry.is_file():
        raise cli.CliMissingBinary(
            f"reddit-research entry point missing: {entry} (run `uv sync` in {project})")
    args = ["search", query, "--json", "--limit", str(cap)]
    if subreddit:
        args += ["--subreddit", subreddit]
    mode = ops.get("dev_mode")
    mode_args = ["--dev-mode", str(mode)] if mode else []
    return cli.launcher(str(entry)) + mode_args + cli_parts[1:] + args


def _read_posts(path: Path, run: cli.CliRun | None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise cli.CliMissingResult(f"reddit-research: raw_json missing: {path}", run)
    try:
        doc = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except ValueError as exc:
        raise cli.CliUnparseable(f"reddit-research: {path} is not JSON: {exc}", run) from exc
    if isinstance(doc, list):
        raw_posts = doc
    elif isinstance(doc, dict):
        raw_posts = doc.get("posts") or []
    else:
        raw_posts = []
    return [p for p in raw_posts if isinstance(p, dict)]


def _project(post: dict[str, Any]) -> dict[str, Any]:
    return {key: post.get(key) for key in POST_FIELDS}
