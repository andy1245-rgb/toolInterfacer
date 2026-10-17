"""reddit.digest — scrape matching threads/posts (reddit-research CLI), digest.

P4 wiring; replayable like web ops.
"""
from __future__ import annotations

from typing import Any

from ..op_api import op


@op("reddit_digest", kind="thread_digest", required=["query"], optional=["subreddit", "thread_cap"])
def run(params: dict[str, Any], store: Any) -> tuple[str, Any, float]:
    _ = params, store
    raise NotImplementedError("P4")
