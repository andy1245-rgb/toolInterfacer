"""Session digest — consumes the pi extension's rolling transcript digest.

Shape the extension sends per useTool call (~spec §6):

  {turns: [{"role": "user"|"assistant", "summary": str, "mentions": [str]}],
   antecedents: [{"phrase": "the file we discussed", "resolved": str|None}],
   cwd?: str}

sanitize() is the ingress bound, and it is structural: every turn is projected
onto those three keys and every antecedent onto its two, so a key the shape does
not list is dropped rather than forwarded, and every string that survives is
clipped. The window is TURN_CAP x TURN_CHARS, hardcoded — the conf key
conf.context.digest_max_tokens exists but nothing reads it.

context.packer.trim is the backstop one layer out: when a state carrying the
digest still overruns conf.context.state_max_tokens, the digest is replaced by a
stub. So a caller cannot inflate the prompt either way — sanitize makes the
digest arrive thin, and a caller who slips past it loses the digest rather than
widening the prompt. That asymmetry is why the bound belongs here, where the
digest is still a digest.
"""
from __future__ import annotations

from typing import Any, Callable

EMPTY: dict[str, Any] = {"turns": [], "antecedents": []}

# Recent-weighted window + per-turn cap. The extension applies both at its end
# (session_push.ts: last 8 turns, 240 chars each); the service re-applies them
# because ingress cannot assume the caller is the extension.
TURN_CAP = 8
TURN_CHARS = 240
MENTION_CAP = 6      # session_push.ts slices each turn's mentions to 6
CWD_CHARS = 1024     # a path, not a paragraph: room for any real directory

# What a digest entry may carry, decided by reading the consumers:
#   role, summary, mentions  — the extension's DigestTurn (session_push.ts:8-11)
#                              and the fields spec §4 fills a slot from.
#   phrase                   — session_digest.mentioned() reads it (:119).
#   resolved                 — the antecedent's absolute path; mentioned()
#                              returns str(antecedent), so this is what makes a
#                              hit usable, and session_push.ts resolves it.
# Nothing downstream reads any other key: planner/scope.py and
# planner/templates/disk_lookup.py read digest["cwd"] alone, planner/inspector.py
# and context/packer.py treat the digest as opaque.
TURN_KEYS = ("role", "summary", "mentions")
ANTECEDENT_KEYS = ("phrase", "resolved")


def sanitize(digest: Any) -> dict[str, Any]:
    """Bound a caller-supplied digest to the shape the service budgets for.

    For an extension-shaped digest this is the identity: <= 8 turns x 240 chars,
    <= 8 antecedents, `cwd` carried through (the L1 disk-slot-fill bridge reads
    it, and every path comes out of it). Anything else is projected onto the
    keys the consumers read and clipped: a turn's unknown keys are dropped, not
    forwarded, and no string survives longer than its bound.
    """
    if not isinstance(digest, dict):
        return dict(EMPTY)
    out = {"turns": _window(digest.get("turns"), _clip_turn),
           "antecedents": _window(digest.get("antecedents"), _clip_antecedent)}
    cwd = _text(digest.get("cwd"), CWD_CHARS)
    if cwd is not None:
        out["cwd"] = cwd
    return out


def _window(items: Any, clip: Callable[[dict[str, Any]], dict[str, Any]]) -> list[dict[str, Any]]:
    """Last TURN_CAP entries, each projected onto its key set and clipped.

    A non-list gives an empty window and a non-dict entry is dropped rather than
    passed through: an entry `clip` cannot read is an entry no consumer can
    read either, and mentioned() would raise on it.
    """
    if not isinstance(items, list):
        return []
    return [clip(item) for item in items[-TURN_CAP:] if isinstance(item, dict)]


def _clip_turn(turn: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in TURN_KEYS:
        if key == "mentions":
            mentions = _mentions(turn.get(key))
            if mentions:
                out[key] = mentions
            continue
        value = _text(turn.get(key), TURN_CHARS)
        if value is not None:
            out[key] = value
    return out


def _clip_antecedent(ant: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ANTECEDENT_KEYS:
        value = _text(ant.get(key), CWD_CHARS if key == "resolved" else TURN_CHARS)
        if value is not None:
            out[key] = value
    return out


def _mentions(value: Any) -> list[str]:
    """At most MENTION_CAP paths, each clipped — the extension's own shape."""
    if not isinstance(value, list):
        return []
    return [_text(item, CWD_CHARS) for item in value[:MENTION_CAP]
            if isinstance(item, str)]


def _text(value: Any, limit: int) -> str | None:
    """`value` if it is a string within `limit`, clipped if longer, else None."""
    if not isinstance(value, str):
        return None
    return value[:limit] if len(value) > limit else value


def mentioned(keywords: list[str], digest: dict[str, Any]) -> list[str]:
    """Artifact keys/items the digest mentions containing any keyword."""
    hits: list[str] = []
    for ant in digest.get("antecedents", []):
        for kw in keywords:
            if kw and kw.lower() in str(ant.get("phrase", "")).lower():
                hits.append(str(ant))
    return hits
