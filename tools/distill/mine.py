"""mine — real operator phrasings out of the local pi transcript corpus.

The frozen 150 were hand-authored. Hand-authored rows carry the author's house
style, and 01 §6 names phrasing shift as a failure mode: a corpus where every
request starts with a verb and ends with a period teaches rhythm, not intent. So
this module pulls requests out of the operator's own transcripts — lowercase,
typo-ridden, comma-spliced, half a thought — and lets the teacher label them.

It obeys the ``session-insights`` skill, which is the only supported way to read
this corpus:

* ``--json`` on every call, and the payload's ``schema`` is asserted equal to
  ``pi-sessions/search@1`` — a renamed key is a stop, not a guess;
* ``--role user --root main`` is the only form that means "where did I say X": under
  ``root:"sub"`` a ``block:user`` is a subagent's prompt, i.e. text a dispatcher
  wrote, so ``root`` is re-checked on every hit;
* no recursive filesystem walk over ``~/.pi`` or ``~`` — ``search`` is the answer,
  and a walk over the transcript corpus is what this avoids.

What it does *not* do is pretend the corpus is a routing benchmark. Operator turns
are specifications, not single-op requests, so the filters below keep only short
cue-bearing sentences and drop everything that looks like pasted output. The yield
is reported, not assumed: ``utterances.py`` prints the per-seed counts and the
manifest records them.

Note the asymmetry that follows: a mined unit has to cue an op for the candidate
block around it to mean anything, so mining can produce t1 and t3 rows and cannot
produce a t2 (veiled) row. t2 coverage is the template generator's job.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, Iterator

from . import bands

CLI = pathlib.Path(os.environ.get(
    "PI_SESSIONS_CLI",
    "~/.pi/profiles/registry/session-insights/scripts/pi-sessions")).expanduser()
SCHEMA = "pi-sessions/search@1"

# Search seeds. Cue-flavoured ones find tool-shaped sentences; the generic request
# ones ("please", "can you") surface the register the operator writes in, and the
# cue filter below does the selecting either way.
SEEDS: tuple[str, ...] = (
    "grep", "which files mention", "read the file", "list the files",
    "search the web", "fetch the page", "pull up", "look up",
    "reddit", "the sub", "digest", "summarize the",
    "pick the", "choose", "which one", "extract", "keywords", "search terms",
    "extract the", "pull out",
    "please", "can you", "i want you", "give me",
)

MIN_CHARS, MAX_CHARS = 12, 90
MIN_WORDS, MAX_WORDS = 3, 16

# Request shape. A mined unit is a candidate *request* or it is nothing: the corpus
# supervises "which op carries out this request", so a heading, a report line, a
# pasted table or a mid-thought fragment is poison even when it happens to cue an op.
# Operator turns are largely specifications and pasted output, so this filter — not
# the cue filter — is the one that does the real work, and it is why the yield is
# what it is. Reported, never assumed.
_LEAD = {
    "please", "can", "could", "would", "will", "i", "lets", "how", "what", "which",
    "where", "when", "who", "why", "find", "list", "read", "open", "cat", "print",
    "show", "display", "dump", "pull", "grab", "fetch", "load", "get", "search",
    "grep", "look", "check", "digest", "summarize", "summarise", "pick", "choose",
    "select", "rank", "extract", "turn", "give", "tell", "run", "do", "does", "is",
    "are", "any", "anything", "has", "have",
}
# A unit opening on one of these is the tail of an earlier sentence or a list
# continuation, never a standalone request.
_FRAGMENT = {"or", "and", "but", "so", "then", "also", "after", "before", "plus",
             "because", "since", "while", "this", "that", "it", "there"}

# Pasted output, agent reports and code: a routing request is none of these, and a
# sentence that matches one of these markers is someone quoting the transcript back.
_REJECT = re.compile(
    r"[{}\[\]<>|`\"\u201c\u201d]|->|==|\.\.\.|\u2026|\*\*|={2,}|"
    r"\bsubagent\b|\bcommit\b|\bimplement\w*\b|\bverif\w+\b|\btest suite\b|"
    r"\brefactor\w*\b|\bfix\w*\b|\bbug\b|\bplan\b|\breview\b|\bskill\b|"
    r"\breadme\b|: |^#|\bthe user\b|\bassistant\b|\bTODO\b|\bexit code\b|\bline \d|"
    r"[\u2500-\u257f\u2705\u26a0\U0001f300-\U0001faff]",
    re.IGNORECASE)
_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_LEAD_TOKEN = re.compile(r"^[A-Za-z']+")


@dataclass
class Mined:
    """One candidate utterance plus where it came from (so a report can trace it)."""

    text: str
    band: str
    op: str
    rival: str | None
    seed: str
    session_id: str = ""
    seq: int = 0

    @property
    def family(self) -> str:
        """Split unit: the session, so paraphrases from one session stay together."""
        base = self.session_id or self.seed
        return f"t3-session:{base}" if self.rival else f"session:{base}"


@dataclass
class Report:
    searched: int = 0
    hits: int = 0
    units: int = 0
    accepted: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    per_seed: dict[str, dict[str, int]] = field(default_factory=dict)
    skipped: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"cli": _display(CLI), "searched": self.searched, "hits": self.hits,
                "units": self.units, "accepted": self.accepted,
                "rejected": self.rejected, "per_seed": self.per_seed,
                "skipped": self.skipped, "seeds": list(SEEDS)}


def _display(path: pathlib.Path) -> str:
    """`~/…` form: the manifest is a repo file and carries no home directory."""
    home = pathlib.Path(os.path.expanduser("~"))
    try:
        return "~/" + str(path.relative_to(home))
    except ValueError:
        return str(path)


def available() -> bool:
    return CLI.is_file()


def _run(args: list[str], timeout: float = 120.0) -> dict[str, Any] | None:
    """One CLI call, with the skill's exit-3 recovery applied once."""
    for attempt in (1, 2):
        try:
            out = subprocess.run([str(CLI), *args], capture_output=True, text=True,
                                 timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if out.returncode == 3 and attempt == 1:
            # index missing/stale/locked: index once, then retry the same command once
            subprocess.run([str(CLI), "index", "--json"], capture_output=True,
                           text=True, timeout=timeout, check=False)
            continue
        if out.returncode not in (0, 1):
            return None
        try:
            payload = json.loads(out.stdout)
        except ValueError:
            return None
        if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
            return None
        return payload
    return None


def search(seed: str, limit: int = 40, snippet: int = 400) -> tuple[list[dict], int]:
    """User-role, main-root hits for one seed; (items, n_total)."""
    payload = _run(["search", seed, "--role", "user", "--root", "main", "--json",
                    "--limit", str(limit), "--snippet", str(snippet), "--no-refresh"])
    if not payload:
        return [], 0
    items = [it for it in payload.get("items", [])
             if it.get("match_col") == "block:user" and it.get("root") == "main"]
    return items, int(payload.get("n_total") or 0)


def units(snippet: str) -> Iterator[str]:
    """Sentence-sized units out of one excerpt, with the match markers stripped."""
    text = snippet.replace("\u00ab", "").replace("\u00bb", "").strip(" \u2026")
    for chunk in _SPLIT.split(text):
        unit = " ".join(chunk.split()).strip(" -\u2022*,")
        if unit:
            yield unit


def accept(unit: str) -> str:
    """\"\" when a unit is usable as a routing utterance, else the reason it is not."""
    if not MIN_CHARS <= len(unit) <= MAX_CHARS:
        return "length"
    words = len(unit.split())
    if not MIN_WORDS <= words <= MAX_WORDS:
        return "words"
    if sum(ch.isdigit() for ch in unit) > 2:
        return "digits"
    if _REJECT.search(unit):
        return "pasted-or-code"
    lead = _LEAD_TOKEN.match(unit)
    first = lead.group(0).lower() if lead else ""
    if first in _FRAGMENT:
        return "sentence-fragment"
    if first not in _LEAD and not unit.endswith("?"):
        return "no-request-lead"
    if not bands.cue_hits(unit):
        return "no-op-cue"
    return ""


def mine(limit: int = 40, snippet: int = 400) -> tuple[list[Mined], Report]:
    """Mined units in seed order, deduplicated within the run."""
    report = Report()
    if not available():
        report.skipped = f"pi-sessions CLI not found at {_display(CLI)}"
        return [], report
    seen: set[str] = set()
    out: list[Mined] = []
    for seed in SEEDS:
        report.searched += 1
        items, n_total = search(seed, limit=limit, snippet=snippet)
        report.per_seed[seed] = {"n_total": n_total, "hits": len(items)}
        report.hits += len(items)
        kept = 0
        for item in items:
            for unit in units(str(item.get("snippet", ""))):
                report.units += 1
                why = accept(unit)
                if why:
                    report.rejected[why] = report.rejected.get(why, 0) + 1
                    continue
                band = bands.classify(unit)
                key = unit.lower().rstrip("?.")
                if key in seen:
                    report.rejected["duplicate"] = report.rejected.get("duplicate", 0) + 1
                    continue
                seen.add(key)
                ranked = sorted(band.hits, key=lambda op: (-len(band.hits[op]), op))
                out.append(Mined(text=unit, band=band.band, op=ranked[0],
                                 rival=band.rival, seed=seed,
                                 session_id=str(item.get("session_id", "")),
                                 seq=int(item.get("seq") or 0)))
                kept += 1
        report.per_seed[seed]["kept"] = kept
    report.accepted = len(out)
    return out, report
