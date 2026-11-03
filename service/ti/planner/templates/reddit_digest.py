"""reddit_digest — reddit-research scrape → thread digest (P4 wiring).

One step, because the op already owns the scrape: `reddit_digest` runs the
CLI, follows its `raw_json` handle, projects each post and caps the list at
`reddit.thread_cap`. The template's job is to publish that shape and let the op
carry the degree.

`docs/ops.md` §6: the CLI is not vendored, so an unconfigured box raises
`CliMissingBinary` naming every path it tried. That error is the honest answer to
"what does r/… say" and it stays an error — an empty digest would look like a
community with nothing to say.
"""
from __future__ import annotations

from typing import Any

from ...core.runner import Step
from ...conf import load


def build(intent: str, digest: dict, params: dict[str, Any] | None = None,
          ) -> tuple[list[Step], str, list[str]]:
    cfg = load()["reddit"]
    steps = [
        Step(id="threads", op="reddit_digest",
             params={"query": intent, "thread_cap": int(cfg["thread_cap"])}),
    ]
    return steps, "threads", []
