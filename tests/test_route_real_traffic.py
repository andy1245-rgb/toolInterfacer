"""Routing regression table for real traffic: 15 intents lifted from the hop log.

Every utterance below is verbatim from `service/logs/hops.jsonl` — the
`soft.intent_context` string of a hop line. The logger clamps that field to 100
characters, so the rows that end mid-phrase (`... maximum range be`,
`... Real-Wor`, `... that describe `) are truncated by the evidence itself and
are reproduced as they were logged rather than tidied; the trailing blank in the
last one is part of the string.

Nothing here is synthesised. `docs/dataset-status.md` audited that log and found
16 distinct intents, of which seven ask for something off-disk — a record, a
paper, a measurement, a court judgment — and all seven were executed as
`filesys_glob`: the disk template walked the filesystem looking for a file named
after the subject. The other eight are listings and reads whose `filesys_glob`
answer was sound, and the same audit names them as the sound rows.

So this file is a regression table for a bug the log proves happened, not a
fixture set: seven rows that must leave the disk path, eight that must stay on
it, and — below them — phrasings that never appeared in the log, because a rule
that only matches the words in it has memorised seven sentences instead of
learning what a web-research intent looks like.
"""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))

from ti.planner.inspector import inspect  # noqa: E402
from ti.service_entry import use_tool  # noqa: E402

# The seven. Between them they carry every phrasing the old rule set could not
# see: an explicit "web_search:" marker, "web search for", "search the web",
# "Search the web", two "Find <published thing>" rows and one "Find <court
# judgments>" row — and the ones starting with "Find" are the point, because
# only two words ("research", "compare") ever reached the research template.
RESEARCH_TRAFFIC = (
    'web_search: Sam Cooke "Bring It On Home to Me" 1962 mono stereo mix '
    "alternate take master take",
    "web search for Clickteam Fusion 2.5 asset extraction tools CTFAK Anaconda",
    "Find published measurements of ultra-wideband (DW3000/DWM3000) ranging "
    "accuracy and maximum range be",
    "search the web for the Vasarhelyi 2018 Science Robotics paper on optimized "
    "flocking of autonomous dr",
    'Find the arXiv ID and title for the paper "MCPTox: A Benchmark for Tool '
    "Poisoning Attack on Real-Wor",
    "Search the web for measured ESP-NOW latency and throughput benchmarks "
    "between ESP32 devices, and aca",
    "Find Canadian court judgments (canlii.org, Federal Court, provincial "
    "superior courts) that describe ",
)

# The eight sound rows, verbatim, with the audit's observation counts for the
# seven it tabulates by count (it leaves the eighth off its table).
DISK_TRAFFIC = (
    "list the files",                              # 75 hop lines
    "ls",                                          # 18
    "list all the files in the repo",              # 18
    "list the .txt files",                         # 15
    "list the .sh files in the bench directory",   # 15
    "what files are in the ghost directory",       # 18
    "list the files in the bench directory",       # 18
    "read the README.md in the repo root",
)

# The shape the route has to produce, asserted through the plan rather than
# through the level alone: a research intent must not reach a filesys step, and
# a disk intent must not reach a web step.
RESEARCH_OPS = {"web_search", "web_fetch", "reddit_digest"}
DISK_OPS = {"filesys_glob", "filesys_read", "filesys_grep"}


def _plan(intent: str) -> list[str]:
    out = use_tool({"intent": intent, "mode": "plan",
                    "digest": {"cwd": str(ROOT)}})
    return [step["op"] for step in out["steps"]]


class TheLogRowsTest(unittest.TestCase):
    def test_the_table_is_the_measured_one(self):
        self.assertEqual(len(RESEARCH_TRAFFIC), 7)
        self.assertEqual(len(DISK_TRAFFIC), 8)
        self.assertEqual(len(set(RESEARCH_TRAFFIC) | set(DISK_TRAFFIC)), 15)

    def test_the_seven_research_intents_leave_the_disk_path(self):
        for intent in RESEARCH_TRAFFIC:
            with self.subTest(intent=intent):
                meta = inspect(intent, {}, {})
                self.assertEqual(meta["level"], "L2", meta)
                self.assertEqual(meta["template"], "research", meta)
                self.assertIn(meta["rule"], ("WEB_CHANNEL", "EXTERNAL_SOURCE",
                                             "RESEARCH_FRAME"), meta)

    def test_the_eight_disk_intents_stay_on_the_disk_path(self):
        for intent in DISK_TRAFFIC:
            with self.subTest(intent=intent):
                meta = inspect(intent, {}, {})
                self.assertEqual(meta["level"], "L1", meta)
                self.assertEqual(meta["template"], "disk_lookup", meta)
                # the default, not a rule that happened to agree with it
                self.assertEqual(meta["rule"], "DISK_DEFAULT", meta)

    def test_the_plan_is_where_the_misroute_showed(self):
        # the log's evidence is the *op that ran*, so the assertion is the plan.
        # The first op is asserted by registry membership rather than by name:
        # the template used to open with the unregistered `__decide_extract__`,
        # which is why every research intent errored (docs/review B1). Pinning
        # that string here would have re-pinned the bug.
        from ti.ops.op_api import all_ops
        for intent in RESEARCH_TRAFFIC:
            with self.subTest(kind="research", intent=intent):
                ops = _plan(intent)
                self.assertIn(ops[0], all_ops(), ops)
                self.assertEqual(ops[0], "decide_extract", ops)
                self.assertIn("web_search", ops, ops)
                self.assertEqual(set(ops) & DISK_OPS, set(), ops)
        for intent in DISK_TRAFFIC:
            with self.subTest(kind="disk", intent=intent):
                ops = _plan(intent)
                self.assertEqual(ops[0], "filesys_glob", ops)
                self.assertEqual(set(ops) & RESEARCH_OPS, set(), ops)


class CollisionTest(unittest.TestCase):
    """The rows that a broadened "find / search / look up" rule could swallow.

    Each is real traffic too — the first is the live.jsonl row the audit calls
    "visibly not a glob decision", the second pins the file-open reading of a
    verb that "look up" sits next to.
    """

    def test_a_read_with_a_downstream_pick_is_a_read(self):        # "read the service logs live.jsonl and summarize the last few records"
        # ran filesys_glob -> decide_pick -> filesys_read: the glob is candidate
        # discovery and the answer is the read of the picked file. The route and
        # the shape were both right; only the label the dataset put on the hop
        # was wrong. It is still a read, so it stays on the disk path and keeps
        # the singular shape — a listing would return the candidate set as the
        # answer and never summarise anything.
        intent = ("read the service logs live.jsonl and summarize the last few "
                  "records")
        meta = inspect(intent, {}, {})
        self.assertEqual((meta["level"], meta["template"]),
                         ("L1", "disk_lookup"), meta)
        self.assertEqual(_plan(intent), ["filesys_glob", "decide_pick",
                                         "filesys_read"])

    def test_a_file_open_is_a_file_open(self):
        # "open the ops matcher implementation" names no published source and no
        # channel, and "open" is the disk path's own verb; nothing here asks a
        # question the filesystem cannot answer.
        intent = "open the ops matcher implementation"
        meta = inspect(intent, {}, {})
        self.assertEqual((meta["level"], meta["template"]),
                         ("L1", "disk_lookup"), meta)
        self.assertEqual(_plan(intent), ["filesys_glob", "decide_pick",
                                         "filesys_read"])

    def test_an_explicit_op_is_not_reclassified(self):
        # params.tool is the caller's own answer, sent with the args for it. A
        # template would discard those args, so the research reading of "find"
        # must not outrank the resolved call — this is why EXPLICIT_OP fires
        # before every rule that reads the intent text.
        meta = inspect("find the arXiv paper on tool poisoning",
                       {"tool": "filesys_read", "args": {"path": "README.md"}}, {})
        self.assertEqual((meta["level"], meta["template"]), ("L0", None), meta)


class UnseenPhrasingTest(unittest.TestCase):
    """One block per rule: what it must catch, and what it must refuse.

    None of these strings is in the log. The negatives are where the rules could
    do damage, and "find the file named foo.py" is the hard one — it carries the
    same verb as three of the seven real rows and is a disk lookup.
    """

    WEB_CHANNEL_HITS = (
        "search the web for the latest RISC-V SBC benchmarks",
        "web search: best cast iron seasoning method",
        "look up the python ssl module docs online",
        "check online for the current LIBOR replacement rate",
        "load https://docs.python.org/3/library/ssl.html",
    )
    WEB_CHANNEL_MISSES = (
        "list the files under the web app directory",
        "read the web handler module",
        "what files are in the internet cache directory",
    )

    EXTERNAL_SOURCE_HITS = (
        "find the arXiv preprint on tool poisoning",
        "look up the published throughput numbers for the DW3000",
        "pull the canlii judgment for that case",
        "what do the proceedings say about the flocking model",
    )
    EXTERNAL_SOURCE_MISSES = (
        "read the paper.md",
        "find the papers folder",
        "summarize publications/2024.pdf",
    )

    RESEARCH_FRAME_HITS = (
        "look up the Node.js stream docs",
        "what does the Go community say about generics",
        "find the current EUR/USD rate",
        "how do other projects handle graceful shutdowns",
        "compare tokio and async-std",
        "what's the latest on the d1 weights",
        "where can I find the DW3000 errata",
    )
    RESEARCH_FRAME_MISSES = (
        "find the file named foo.py",
        "find foo.py",
        "read the paper.md",
        "search the docs for the quota wording",
        "search the codebase for the retry cap",
        "compare the two config files",
        "find the background job logs",
        "what does the makefile actually do?",
        "what's in README.md",
        "",
    )

    def _assert_route(self, intents, rule, template):
        for intent in intents:
            with self.subTest(intent=intent):
                meta = inspect(intent, {}, {})
                self.assertEqual(meta["template"], template, meta)
                self.assertEqual(meta["rule"], rule, meta)

    def test_the_web_channel_rule_catches_a_named_channel(self):
        self._assert_route(self.WEB_CHANNEL_HITS, "WEB_CHANNEL", "research")

    def test_the_web_channel_rule_ignores_the_word_web_on_its_own(self):
        # "web" as a directory or a module name is not a channel; R3 is gated on
        # the channel being named, never on the noun appearing.
        self._assert_route(self.WEB_CHANNEL_MISSES, "DISK_DEFAULT", "disk_lookup")

    def test_the_published_source_rule_catches_an_off_disk_artefact(self):
        self._assert_route(self.EXTERNAL_SOURCE_HITS, "EXTERNAL_SOURCE", "research")

    def test_the_published_source_rule_yields_to_a_named_file(self):
        # a named artefact on disk (an extension, a file-set noun) refuses R4:
        # "papers" is a directory here, not a literature search.
        self._assert_route(self.EXTERNAL_SOURCE_MISSES, "DISK_DEFAULT",
                           "disk_lookup")

    def test_the_research_verb_rule_catches_an_unlocalised_question(self):
        self._assert_route(self.RESEARCH_FRAME_HITS, "RESEARCH_FRAME", "research")

    def test_the_research_verb_rule_yields_to_a_local_frame(self):
        self._assert_route(self.RESEARCH_FRAME_MISSES, "DISK_DEFAULT",
                           "disk_lookup")

    def test_the_reddit_route_outranks_the_research_verb_rule(self):
        # "what does X say" is the research frame R5 would claim, and r/rust is
        # the more specific L2 — the digest op is the plan, not a web search.
        meta = inspect("what does r/rust say about async runtimes", {}, {})
        self.assertEqual((meta["level"], meta["template"], meta["rule"]),
                         ("L2", "reddit_digest", "REDDIT_ROUTE"), meta)

    def test_the_rules_do_not_depend_on_capitalisation(self):
        for intent in ("FIND the arXiv preprint on tool poisoning",
                       "Search The Web For esp32 benchmarks"):
            with self.subTest(intent=intent):
                self.assertEqual(inspect(intent, {}, {})["template"], "research")


class ExplicitResearchVerbTest(unittest.TestCase):
    """An explicit research verb outranks the reddit route, and only that.

    "research what r/X says about Y" is research *about* a community: the plan
    is the research one and the subreddit is its optional cross-check step
    (spec §7.6). Routing it to `reddit_digest` alone turns the community into the
    whole answer, and that template hard-errors when the CLI is absent — by
    design, because an empty digest would read as a community with nothing to
    say. So the verb decides the plan and the mention decides the source.

    "what does r/rust say about X" has no research verb and stays a digest.
    "research the repo for X" names a local tree and stays a disk lookup.
    """

    def test_a_research_verb_plus_a_subreddit_is_research(self):
        for intent in (
            "research what r/MechanicalKeyboards says about hall effect switches",
            "research how people rate the DW3000 ranging accuracy",
            "research the literature on tool poisoning benchmarks",
        ):
            with self.subTest(intent=intent):
                meta = inspect(intent, {}, {})
                self.assertEqual((meta["level"], meta["template"], meta["rule"]),
                                 ("L2", "research", "RESEARCH_VERB"), meta)

    def test_a_subreddit_without_a_research_verb_is_still_a_digest(self):
        meta = inspect("what does r/rust say about async runtimes", {}, {})
        self.assertEqual((meta["template"], meta["rule"]),
                         ("reddit_digest", "REDDIT_ROUTE"), meta)

    def test_a_research_verb_over_a_local_tree_stays_a_disk_lookup(self):
        for intent in ("research the repo for the retry cap",
                       "research the codebase for handle_stream",
                       "research our files for the config loader"):
            with self.subTest(intent=intent):
                meta = inspect(intent, {}, {})
                self.assertEqual((meta["template"], meta["rule"]),
                                 ("disk_lookup", "DISK_DEFAULT"), meta)


if __name__ == "__main__":
    unittest.main()
