"""Inspector — classify the intent, resolve level + template.

v1 classification (meta decision, one `choice`): passthrough (L0) /
one_hop slot-fill (L1) / research (L2) / reddit_digest (L2) / disk_lookup (L2).
P1 implements rule-first classification + a `decide` fallback call site;
P2 replaces the rule layer with a routed d1 choice.

The rule layer is keyword frames, and the runtime evidence says the vocabulary
was too thin to hold: `service/logs/hops.jsonl` carries 16 distinct intents and
the seven that ask for something off-disk — a record, a paper, a measurement, a
court judgment — were every one of them executed as `filesys_glob`
(docs/dataset-status.md measured this and pinned the rows). Two words were
admitted, "research" and "compare", so "search the web for X" and "find X"
reached the disk template by falling through it, and the disk plan then walked
the filesystem for a file named after the subject.

Ordered rules, first match wins — the order between them is the design:

    R1 EXPLICIT_OP        -> L0                (the caller resolved the op)
    R1.5 RESEARCH_VERB    -> L2 research       (the caller said "research")
    R2 REDDIT_ROUTE       -> L2 reddit_digest
    R3 WEB_CHANNEL        -> L2 research       (never gated)
    R4 EXTERNAL_SOURCE    -> L2 research       (gated on no local artefact)
    R5 RESEARCH_FRAME     -> L2 research       (gated on no local frame)
    DEFAULT               -> L1 disk_lookup

R1 first because R3-R5 are broad enough to claim a resolved op: "find the
background job logs" carrying params.tool=filesys_read is a call, not a
question, and a template would discard the args the caller sent.

R2 before R3/R5 because it is the more specific L2: "what does r/rust say about
async runtimes" is a reddit digest, and R5's "what does ... say" frame would
otherwise claim it.

R3 is never gated because a named channel is not a guess: "search the web for
the README example" is a web search even though its object is a file this repo
also has.

R4 and R5 are what the local gate exists for, and they take different gates
because they carry different evidence. R4 names a published artefact — a paper,
an arXiv id, a court judgment — which does not live in a working tree, so only a
*named* local object (a file-set noun, a known file class, a file extension)
refuses it; a verb alone does not, or "open the arxiv abs page" would be read as
a disk read. R5 carries only a research verb ("find", "look up", "search for",
"what does X say"), and the disk path uses the same verbs just as often ("find
the file named foo.py"), so any local frame refuses it.

The local vocabularies are kept narrow on purpose: a false positive here
re-creates the bug this module was fixed for, so the extension list holds only
suffixes a research intent does not plausibly carry and leaves out `js`, `ts`
and `cpp`, which are also how products are named — "look up the Node.js stream
docs" is exactly the phrasing that must not read as a local artefact.

Lexical only, like shape.py: no model call, no network, no added latency. The
spec's §7 scope split (quick_lookup | full_research | reddit_centric |
disk_centric) collapses to these levels today; P2 replaces this layer with a
routed choice and nothing below it has to change.
"""
from __future__ import annotations

import re
from typing import Any, Callable

from .shape import FILECLASS

EXPLICIT_OP = "EXPLICIT_OP"
REDDIT_ROUTE = "REDDIT_ROUTE"
RESEARCH_VERB = "RESEARCH_VERB"
WEB_CHANNEL = "WEB_CHANNEL"
EXTERNAL_SOURCE = "EXTERNAL_SOURCE"
RESEARCH_FRAME = "RESEARCH_FRAME"
DISK_DEFAULT = "DISK_DEFAULT"


def _clauses(*patterns: str) -> Callable[[str], bool]:
    """A rule satisfied by any single one of its clauses."""
    compiled = tuple(re.compile(p) for p in patterns)
    return lambda text: any(c.search(text) for c in compiled)


# R2 — a reddit request has its own L2 plan; it is not a search that happens to
# name one. `r/<name>` is the same route said shorter.
_REDDIT = re.compile(r"reddit|\br/")

# R1.5 — the caller asked for research by name. Deliberately narrow: the bare
# word "research" also names a directory ("research/notes.md"), so the verb form
# is what is matched, and "research the repo for" stays a disk lookup.
_RESEARCH_VERB = re.compile(
    r"^\s*research\b(?!\s+(?:the|this|that|our|my)\s+(?:repo|repository|code|codebase|files|tree|dir|directory|folder))"
    r"|\bresearch\s+(?:what|how|whether|why|which)\b"
    r"|\bresearch\s+(?:the\s+)?(?:literature|papers|state of the art)\b", re.I)


# R3 — the caller named the channel.
_WEB_CHANNEL = _clauses(
    r"\bweb[ _-]?search\b",
    r"\b(?:search(?:es|ed|ing)?|look(?:s|ed|ing)?|browse|check|google)\b"
    r".{0,40}\b(?:web|internet|online)\b",
    r"\bon\s+the\s+(?:web|internet)\b",
    r"\bgoogle\b",
    r"\bhttps?://|\bwww\.",
    r"\b[\w-]+\.(?:com|org|net|edu|gov|io|ai|dev|ca|onion)\b",
)


# R4 — an artefact that is published rather than stored.
_EXTERNAL_SOURCE = _clauses(
    r"\barxiv\b",
    r"\bpreprints?\b",
    r"\bpapers?\b",
    r"\bpublications?\b|\bpublished\b|\bpeer[-\s]?reviewed\b",
    r"\bjournals?\b|\bproceedings\b",
    r"\bdoi\b|\bcitations?\b|\bbibliograph\w*\b",
    r"\bcanlii\b|\bcase\s+law\b|\bcourts?\b|\bjudg?ments?\b",
    r"\bwikipedia\b|\bstack\s*overflow\b",
)


# R5 — a research verb. The disk path uses the same verbs, so this only fires
# when nothing in the intent names something local.
_RESEARCH_FRAME = _clauses(
    r"\bresearch\w*\b",
    r"\bcompare[sd]?\b|\bcomparing\b",
    r"\b(?:find|locate|look\s*up|look\s+for|search(?:es|ed|ing)?)\b",
    r"\bwhat\s+(?:does|do|did)\b.{0,60}\bsay\b",
    r"\bwhat(?:'s|\s+is|\s+are)\s+the\s+(?:latest|newest)\b",
    r"\bwhere\s+(?:can|could|should|do|did)\s+(?:i|we)\s+find\b",
    r"\bhow\s+(?:do|does|did)\s+(?:people|others|teams|folks|you"
    r"|other\s+(?:projects|teams|people|tools|languages|libs?|libraries))\b",
)


# A named artefact: an extension, a known file class, or a filesystem noun.
_EXTENSION = re.compile(
    r"\.(?:py|pyi|md|markdown|rst|txt|sh|bash|zsh|jsonl|json|ya?ml|toml"
    r"|ini|cfg|conf|csv|tsv|log|rs|go|rb|pl|lua|sql|xml|html?|css|scss|java|kt"
    r"|swift|ipynb|parquet|zip|tar|gz|pdf|env|lock)\b")
_FILE_CLASS = re.compile(FILECLASS)
_FILESYSTEM_NOUN = re.compile(
    r"\b(?:files?|file\s*names?|filenames?|director(?:y|ies)|dirs?|folders?"
    r"|repos?(?:itory)?|paths?|entries|logs?|fixtures?|manifests?|dotfiles?"
    r"|artifacts?|snapshots?|codebase|subdirector(?:y|ies))\b")

# A disk operation, or a corpus named as something to search inside. Weaker than
# the nouns above on its own — "open the arxiv abs page" opens a page — but it is
# the difference between "find the Node.js stream docs" and "search the docs for
# the quota wording".
_DISK_OPERATION = re.compile(
    r"\b(?:read|open|cat|print|dump|display|edit|write|save|delete|remove|move"
    r"|rename|copy|summari[sz]e|inspect|view|list|ls)\b")
_SEARCH_INSIDE = re.compile(
    r"\bsearch(?:es|ed|ing)?\b.{0,25}\b(?:docs?|documentation|code|source|repo"
    r"|files?|codebase|tree|logs?|tests?)\b")


def _strong_local(text: str) -> bool:
    return bool(_EXTENSION.search(text)
                or _FILE_CLASS.search(text)
                or _FILESYSTEM_NOUN.search(text))


def _local_frame(text: str) -> bool:
    return bool(_strong_local(text)
                or _DISK_OPERATION.search(text)
                or _SEARCH_INSIDE.search(text))


def _fire(intent: str, params: dict[str, Any]) -> tuple[str, str, str | None]:
    """(rule that decided it, level, template) — first match wins."""
    text = (intent or "").lower()
    if params.get("tool"):
        return EXPLICIT_OP, "L0", None
    if _RESEARCH_VERB.search(text):
        # R1.5 — an explicit research verb outranks the reddit route, because
        # what the caller asked for is research and the community is a *source*
        # for it, not the whole plan. This is exactly the case spec §7.6 exists
        # for: "research what r/X says about Y" builds the research plan and
        # adds the reddit cross-check as its optional step, whereas routing it
        # to `reddit_digest` alone hard-errors when the CLI is unavailable
        # (that template's error is deliberate — see its docstring).
        return RESEARCH_VERB, "L2", "research"
    if _REDDIT.search(text):
        return REDDIT_ROUTE, "L2", "reddit_digest"
    if _WEB_CHANNEL(text):
        return WEB_CHANNEL, "L2", "research"
    if _EXTERNAL_SOURCE(text) and not _strong_local(text):
        return EXTERNAL_SOURCE, "L2", "research"
    if _RESEARCH_FRAME(text) and not _local_frame(text):
        return RESEARCH_FRAME, "L2", "research"
    return DISK_DEFAULT, "L1", "disk_lookup"


def rule_for(intent: str, params: dict[str, Any] | None = None) -> str:
    """The first rule that fires on this intent."""
    return _fire(intent, params or {})[0]


def inspect(intent: str, params: dict[str, Any], digest: dict[str, Any]) -> dict[str, Any]:
    rule, level, template = _fire(intent, params)
    return {"intent": intent, "level": level, "template": template, "rule": rule}
