"""reddit_digest — reddit-research scrape → thread digest (P4 wiring)."""
from __future__ import annotations

from ...core.runner import Step
from ...conf import load


def apply(intent: str, digest: dict) -> list[Step]:
    cfg = load()["reddit"]
    return [
        Step(id="threads", op="reddit_digest",
             params={"query": intent, "thread_cap": int(cfg["thread_cap"])}),
    ]
