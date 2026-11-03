"""The question sets a decision record answers, and the letter mapping used in SFT.

Two criteria sets exist in this repo, and a record names the one it answers in
``criteria_id`` so the two tasks never mix in one export:

``routing.ops_v1``
    state = a user utterance/intent, options = the 8 routable ops with the
    descriptions ``bench/live_probe.py`` sends to the model. This is the task the
    frozen utterance files label and the task the bench harness measures, so it is
    the default SFT target.

``shape.singular_vs_enumerative``
    state = a listing utterance, options = whether the answer is one thing or a
    set. The held-out phase 3/4 files and ``shape_expectations.json`` label this.

The op descriptions are duplicated here rather than imported from ``bench/``:
``live_probe`` runs ``sys.path`` surgery and imports the service at module scope,
so importing it from a library module would drag service state into ingest.
``tests/test_dataset_pipeline.py`` asserts this copy still equals the harness's.
"""
from __future__ import annotations

import json
from typing import Any

ROUTING_OPS = "routing.ops_v1"
SHAPE = "shape.singular_vs_enumerative"

INSTRUCTIONS = "Which single operation should handle the user command?"

OP_DESCRIPTIONS: dict[str, str] = {
    "filesys_read": "Read one file's contents from disk.",
    "filesys_glob": "List files matching a name pattern (paths only, no content search).",
    "filesys_grep": "Search inside file contents for a regular expression.",
    "decide_pick": "Pick the best item(s) from a candidate list.",
    "decide_extract": "Extract query strings or parameter values from the user command.",
    "web_search": "Search the web by keywords.",
    "web_fetch": "Fetch one URL's full contents.",
    "reddit_digest": "Summarize a Reddit subreddit or thread.",
}

SHAPE_DESCRIPTIONS: dict[str, str] = {
    "singular": "The answer is one file, one match, one page — a single item.",
    "enumerative": "The answer is a set of items — a listing.",
}

CRITERIA: dict[str, dict[str, str]] = {
    ROUTING_OPS: OP_DESCRIPTIONS,
    SHAPE: SHAPE_DESCRIPTIONS,
}


def options(criteria_id: str) -> dict[str, str]:
    return CRITERIA[criteria_id]


def letters(criteria_id: str) -> dict[str, str]:
    """Option id -> option letter. Sorted ids so the map is stable across runs."""
    return {opt: _letter(i) for i, opt in enumerate(sorted(CRITERIA[criteria_id]))}


def letter_for(criteria_id: str, label: str) -> str:
    return letters(criteria_id)[label]


def label_for(criteria_id: str, letter: str) -> str:
    reverse = {v: k for k, v in letters(criteria_id).items()}
    if letter not in reverse:
        raise KeyError(f"{letter!r} is not an option letter of {criteria_id}; "
                       f"valid: {sorted(reverse)}")
    return reverse[letter]


def _letter(index: int) -> str:
    if not 0 <= index < 26:
        raise ValueError(f"option index {index} has no A-Z letter")
    return chr(ord("A") + index)


def system_prompt(criteria_id: str) -> str:
    """The instruction block the fine-tuned model is trained against."""
    by_letter = sorted(letters(criteria_id).items(), key=lambda kv: kv[1])
    block = "\n".join(f"  {letter}. {opt}: {CRITERIA[criteria_id][opt]}"
                      for opt, letter in by_letter)
    return (
        "You are a tool-routing decision model. You are given one user command "
        "and a lettered list of options.\n"
        f"\n{INSTRUCTIONS}\n\n"
        f"Options:\n{block}\n\n"
        "Answer with exactly one option letter (A-Z) and nothing else."
    )


def state_text(state: Any) -> str:
    """Render a record's state as the user message body.

    A string state is the utterance itself. Anything else (the dict shape the
    packer produces) is serialised compactly and in a stable key order so the
    same state always produces byte-identical training text.
    """
    if isinstance(state, str):
        return state
    return json.dumps(state, sort_keys=True, ensure_ascii=False, separators=(", ", ": "))
