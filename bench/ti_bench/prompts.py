"""prompts — the frozen probe: one instruction and the 8 op criteria.

These two constants *are* the measurement instrument: every lane in
docs/model-sweep-consolidated.md was scored by asking exactly this question with
exactly this option set, so a number produced with a different wording is not
comparable with them. They therefore live here, once, and both
``bench/live_probe.py`` and ``bench/run.py`` import them.
"""
from __future__ import annotations

INSTRUCTIONS = "Which single operation should handle the user command?"

OP_DESCRIPTIONS = {
    "filesys_read": "Read one file's contents from disk.",
    "filesys_glob": "List files matching a name pattern (paths only, no content search).",
    "filesys_grep": "Search inside file contents for a regular expression.",
    "decide_pick": "Pick the best item(s) from a candidate list.",
    "decide_extract": "Extract query strings or parameter values from the user command.",
    "web_search": "Search the web by keywords.",
    "web_fetch": "Fetch one URL's full contents.",
    "reddit_digest": "Summarize a Reddit subreddit or thread.",
}
