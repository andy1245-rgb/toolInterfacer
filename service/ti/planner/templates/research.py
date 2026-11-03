"""research — spec §7's L2 pipeline: classify → queries → search → pick →
fetch+verify → optional cross-check → digest.

Step 1 of §7 (classify: quick_lookup | full_research | reddit_centric |
disk_centric) belongs to `planner.inspector`: an intent only reaches this
template once the inspector has already called it a research intent. This file
owns steps 2-7:

    #  §7 step               op                artifact       conf
    2  extract queries       decide_extract    value_list     extract
    3  search                web_search        url_list       query coverage
    4  pick top-k            decide_pick       picked_items   choice
    5  fetch + verify        web_fetch (loop)  pages          mean of the kept
    6  reddit cross-check    reddit_digest     thread_digest  posts present?
    7  assemble              digest_assemble   digest         page coverage

Step 5 is a loop, not a batch: `over={"urls": "picked"}` runs `web_fetch` once per
picked URL, the step's `gate.post` asks the `noul` primitive whether *that page*
is relevant, a page below `policy.gate_noul` is dropped rather than fetched into
the digest, and `stop_after` ends the loop once `research.good_pages_target`
pages have verified. One verdict per page is the point of the step: a single
check over a batch can neither skip one bad page nor stop early.

Step 6 exists only when the reddit-research CLI resolves (`docs/ops.md` §6 — the
project is not vendored, so on most boxes it does not) *and* the ask names a
community, or when `research.reddit_cross_check` is "on". It is an `optional`
step, so a mount that disappears between planning and execution degrades the
digest instead of failing the run.

EVERY DEGREE IS IN conf (`service/ti/conf/defaults.json`); none is in this file:

    research.query_count[1]      queries the extractor asks for (band 3..6)
    research.backends[0]         the web_search backend the search step uses
    research.pick_top_k          candidates the pick step reduces the hits to
    research.good_pages_target   verified pages the fetch loop stops at
    research.reddit_cross_check  "off" | "auto" (ask names a community) | "on"
    research.digest.claim_chars  length of the gist lifted from each source
    policy.gate_slot             pick confidence required before fetching
    policy.gate_noul             relevance a page must clear to be kept

Shape is published here, degree is published there: changing any number above
never requires editing a step.
"""
from __future__ import annotations

import re
from typing import Any

from ...conf import load
from ...core.gates import Gate
from ...core.runner import Step

DEFAULT_QUERY_COUNT = 6
DEFAULT_SEARCH_BACKEND = "free-search"

#: What makes a research ask a *community* ask: a cross-check against what people
#: posted is only a cross-check if the caller wanted people's opinions.
_COMMUNITY = re.compile(
    r"\breddit\b|\br/[A-Za-z0-9_]+|\bcommunity\b|\bforums?\b"
    r"|\bwhat (?:do|are|did) (?:people|users|folks|others)\b|\bpeople say\b"
    r"|\bexperience with\b|\bworth it\b", re.I)
_SUBREDDIT = re.compile(r"\br/([A-Za-z0-9_]+)", re.I)


def build(intent: str, digest: dict, params: dict[str, Any] | None = None,
          ) -> tuple[list[Step], str, list[str]]:
    conf = load()
    cfg = conf["research"]
    gate_slot = float(conf["policy"]["gate_slot"])
    gate_noul = float(conf["policy"]["gate_noul"])
    assumptions: list[str] = []

    steps: list[Step] = [
        Step(id="queries", op="decide_extract",
             params={"instructions": "Extract the search queries that would answer "
                                     "this intent.",
                     "count": _query_count(cfg), "intent_context": intent}),
        Step(id="hits", op="web_search", inject={"queries": "queries"},
             params={"backend": _search_backend(cfg)}),
        Step(id="picked", op="decide_pick",
             params={"source_artifact": "hits", "intent_context": intent,
                     "instructions": "Pick the URLs worth reading for the intent.",
                     "top_k": int(cfg["pick_top_k"])}),
        Step(id="pages", op="web_fetch",
             over={"urls": "picked"}, stop_after=int(cfg["good_pages_target"]),
             # Two different questions authorize the fan-out: did the search cover
             # the queries (web_search's conf is coverage, not a decision), and was
             # the pick confident. A half-answered search must not look like a full
             # one — docs/ops.md §5 gives that number this gate as its purpose.
             gate=Gate(pre=[{"artifact": "hits", "kind": "conf_gte", "min": gate_slot},
                            {"artifact": "picked", "kind": "conf_gte", "min": gate_slot}],
                       post=[{"verifier": "is_relevant", "min": gate_noul}]),
             params={}),
    ]

    cross_check = _cross_check(intent, cfg, assumptions)
    if cross_check is not None:
        steps.append(cross_check)

    assembled = {"pages": "pages", "hits": "hits", "queries": "queries"}
    if cross_check is not None:
        assembled["threads"] = cross_check.id
    steps.append(Step(id="digest", op="digest_assemble", inject=assembled,
                      params={"intent": intent}))
    return steps, "digest", assumptions


def _query_count(cfg: dict[str, Any]) -> int:
    """Top of the configured band; the extractor clamps to its own 3..6."""
    band = cfg.get("query_count") or []
    value = band[-1] if isinstance(band, (list, tuple)) and band else band
    try:
        return int(value)
    except (TypeError, ValueError):
        return DEFAULT_QUERY_COUNT


def _search_backend(cfg: dict[str, Any]) -> str:
    backends = [str(b) for b in cfg.get("backends") or [] if str(b).strip()]
    return backends[0] if backends else DEFAULT_SEARCH_BACKEND


def _cross_check(intent: str, cfg: dict[str, Any],
                 assumptions: list[str]) -> Step | None:
    """The §7.6 step, when the ask wants it and the CLI can serve it."""
    mode = str(cfg.get("reddit_cross_check") or "auto").lower()
    if mode == "off":
        return None
    subreddit = _subreddit_of(intent)
    if mode != "on" and not (subreddit or _COMMUNITY.search(intent)):
        return None
    gap = _reddit_gap()
    if gap:
        assumptions.append(f"cross-check skipped: {gap}")
        return None
    step_params: dict[str, Any] = {"query": intent}
    if subreddit:
        step_params["subreddit"] = subreddit
    return Step(id="cross_check", op="reddit_digest", params=step_params, optional=True)


def _subreddit_of(intent: str) -> str | None:
    match = _SUBREDDIT.search(intent)
    return match.group(1) if match else None


def _reddit_gap() -> str:
    """Why the cross-check cannot run here, or "" when it can.

    Probed while the plan is built, for the same reason `disk_lookup` probes its
    directory: the plan should state whether a step is available instead of
    discovering it halfway through a run.
    """
    from ...ops.cli import CliError, ops_conf
    from ...ops.reddit.digest import resolve_project
    try:
        resolve_project(ops_conf().get("reddit") or {})
    except CliError as exc:
        return str(exc).splitlines()[0]
    return ""
