"""roundtrip_tasks — the fixed task list the round-trip bench runs in both arms.

One entry per task, and each entry carries the three things a fair comparison
needs:

  intent/params   what the caller says, unchanged for both arms
  native          the shortest sequence of tool calls that satisfies the same
                  check, in the order an agent model would have to issue them.
                  One entry is one model round trip: the model cannot send call
                  n+1 before it has seen the result of call n. `ask_user` is a
                  round trip that executes no tool — it is the pass that puts a
                  question to the human, and arm B pays the same pass when the
                  service returns need_input.
  check           a predicate over the artifact, not over the response body, so
                  the same code grades arm A's tool output and arm B's
                  `result`. Every expected value is recomputed here from the
                  repo or the fixture files; nothing is compared to a golden
                  string this module wrote itself.

The native policies are deliberately *minimal*: each is the shortest sequence
that produces the artifact the check asks for. That is the kindest possible
reading of arm A — a real agent's confirmation loop only adds round trips — so
a saving measured against it is a floor, not a best case.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

#: Files that make "the notes file" a one-of-three slot fill.
NOTES_DIR_FILES = {
    "notes.md": "pick me\n",
    "server-out.txt": "server: listening on 127.0.0.1:8080\n",
    "other.txt": "nothing to see\n",
}

#: Files that make "the report file" a genuinely ambiguous pick (5 candidates).
REPORT_FILES = {f"report-{name}.md": f"report {name}\n"
                for name in ("draft", "final", "old", "v1", "v2")}

#: The answer handed back when the service asks; one of the five candidates.
REPORT_ANSWER = "report-v1.md"


@dataclass(frozen=True)
class Task:
    id: str
    shape: str
    intent: str
    check: Callable[[Any, "Ctx"], tuple[bool, str]]
    native: tuple[tuple[str, Any], ...]
    params: dict[str, Any] = field(default_factory=dict)
    where: str = "repo"          # which directory the run's cwd is
    answer: str = ""             # fixed answer for an ask-back, "" = first option
    note: str = ""               # what this task is here to show


@dataclass
class Ctx:
    """Every expected value, recomputed from disk when the run starts."""

    root: Path
    notes: Path
    reports: Path
    readme: str = ""
    notes_text: str = ""
    report_text: str = ""
    walk_text: str = ""
    tests_py: list[str] = field(default_factory=list)

    def build(self) -> "Ctx":
        self.readme = (self.root / "README.md").read_text(encoding="utf-8")
        self.notes_text = (self.notes / "notes.md").read_text(encoding="utf-8")
        self.report_text = (self.reports / REPORT_ANSWER).read_text(encoding="utf-8")
        self.walk_text = (self.root / "service/ti/ops/filesys/walk.py").read_text(encoding="utf-8")
        # The stack's glob matches `*` across separators (fnmatch), so "the
        # .py files in the tests directory" is the recursive set here; the
        # expectation is recomputed with rglob, never with the code under test.
        self.tests_py = sorted(p.relative_to(self.root).as_posix()
                               for p in (self.root / "tests").rglob("*.py"))
        return self

    def cwd(self, task: Task) -> Path:
        return {"repo": self.root, "notes": self.notes, "reports": self.reports}[task.where]


# ----------------------------------------------------------------- the checks
# Each returns (correct, detail). `detail` names what was expected, so a FAIL
# line in the report says what the artifact should have been.


def _text(result: Any) -> str:
    return result if isinstance(result, str) else ""


def _check_readme(result: Any, ctx: Ctx) -> tuple[bool, str]:
    ok = _text(result) == ctx.readme
    return ok, "README.md bytes" if not ok else ""


def _check_grep_hits(result: Any, ctx: Ctx) -> tuple[bool, str]:
    """The hits must include the file that actually defines `build` in templates."""
    hits = result if isinstance(result, list) else []
    paths = [str(h.get("path", "")) for h in hits if isinstance(h, dict)]
    # grep reports paths relative to its own search root, so the hit on the
    # defining file is the basename here.
    ok = any(p.endswith("disk_lookup.py") for p in paths)
    return ok, f"a grep hit on disk_lookup.py (got {len(paths)} hits)"


def _check_listing(result: Any, ctx: Ctx) -> tuple[bool, str]:
    got = sorted(str(p) for p in result) if isinstance(result, list) else []
    ok = got == ctx.tests_py
    return ok, f"the {len(ctx.tests_py)} tests/*.py paths"


def _check_notes(result: Any, ctx: Ctx) -> tuple[bool, str]:
    ok = _text(result) == ctx.notes_text
    return ok, "the bytes of notes.md" if not ok else ""


def _check_report(result: Any, ctx: Ctx) -> tuple[bool, str]:
    ok = _text(result) == ctx.report_text
    return ok, f"the bytes of {REPORT_ANSWER}" if not ok else ""


def _check_symbol_defined(result: Any, ctx: Ctx) -> tuple[bool, str]:
    """`where is X defined` wants the defining file — a read of some other file
    is not that answer, however plausible it looked.

    A native grep reports the hit relative to its own search root, so the
    defining file is the basename there; a read answers with the file's bytes,
    so the check has to look at both shapes.
    """
    hits = result if isinstance(result, list) else []
    paths = [str(h.get("path", "")) for h in hits if isinstance(h, dict)]
    if paths:
        return (any(p.endswith("impl.py") for p in paths),
                f"a grep hit on impl.py (got {paths[:3]})")
    ok = "filesys/impl.py" in str(result)
    return ok, "a reference to service/ti/ops/filesys/impl.py"


def _pages(result: Any) -> list[dict]:
    return [p for p in result if isinstance(p, dict)] if isinstance(result, list) else []


def _check_research(result: Any, ctx: Ctx) -> tuple[bool, str]:
    """One usable source: a fetch that returned page text, or a digest that
    cites one. Both arms are held to the same bar — the artifact, not the prose."""
    if isinstance(result, dict) and "sources" in result:
        usable = [s for s in result.get("sources") or []
                  if isinstance(s, dict) and str(s.get("url", "")).startswith("http")
                  and len(str(s.get("claim", ""))) >= 40]
        return bool(usable), "a digest citing >=1 source with a claim"
    pages = [p for p in _pages(result)
             if str(p.get("url", "")).startswith("http")
             and len(str(p.get("content", ""))) >= 200]
    return bool(pages), "one fetched page with >=200 chars of text"


def _check_reddit(result: Any, ctx: Ctx) -> tuple[bool, str]:
    threads = result.get("threads") if isinstance(result, dict) else None
    ok = isinstance(threads, list) and len(threads) >= 1
    return ok, ">=1 reddit thread"


def _check_walk(result: Any, ctx: Ctx) -> tuple[bool, str]:
    """The compound ask wants the file that carries the answer; `max_depth` is
    the token it must contain, and the exact bytes are the strong form."""
    text = _text(result)
    ok = "max_depth" in text and len(text) >= 200
    return ok, "the bytes of service/ti/ops/filesys/walk.py"


WAIT_FOR_HUMAN = "ask_user"


def tasks(ctx: Ctx) -> list[Task]:
    """The list. Built from `ctx` so paths are the run's, not a recorded run's."""
    return [
        Task(
            id="l0_read",
            shape="L0 read (caller names the path)",
            intent="read the file I named",
            params={"tool": "filesys_read", "args": {"path": "README.md"}},
            native=(("filesys_read", {"path": "README.md"}),),
            check=_check_readme,
            note="the shape the layer cannot improve: one tool call either way",
        ),
        Task(
            id="l0_grep",
            shape="L0 grep (caller names the pattern)",
            intent="search the planner templates for the build entry points",
            params={"tool": "filesys_grep",
                    "args": {"pattern": r"^def build", "cwd": "service/ti/planner/templates"}},
            native=(("filesys_grep", {"pattern": r"^def build",
                                      "cwd": "service/ti/planner/templates"}),),
            check=_check_grep_hits,
            note="a resolved pattern is one round trip natively; the service adds only a hop",
        ),
        Task(
            id="l1_list",
            shape="L1 listing",
            intent="list the .py files in the tests directory",
            native=(("filesys_glob", {"pattern": "tests/*.py", "cwd": "."}),),
            check=_check_listing,
            note="listing terminates on candidates: one glob either way",
        ),
        Task(
            id="l1_read",
            shape="L1 read (slot fill over 3 files)",
            intent="read the notes file",
            where="notes",
            native=(("filesys_glob", {"pattern": "**/*", "cwd": "."}),
                    ("filesys_read", {"path": "notes.md"})),
            check=_check_notes,
            note="the service globs and picks inside one round trip",
        ),
        Task(
            id="l1_ask",
            shape="L1 ambiguous -> ask-back",
            intent="read the report file",
            where="reports",
            answer=REPORT_ANSWER,
            native=(("filesys_glob", {"pattern": "**/*", "cwd": "."}),
                    (WAIT_FOR_HUMAN, {}),
                    ("filesys_read", {"path": REPORT_ANSWER})),
            check=_check_report,
            note="five equal candidates: both arms must put a question to the human",
        ),
        Task(
            id="l1_symbol",
            shape="L1 grep intent (routed as a read)",
            intent="where is filesys_glob defined",
            native=(("filesys_grep", {"pattern": r"def run", "cwd": "service/ti/ops/filesys",
                                      "glob": "impl.py"}),),
            check=_check_symbol_defined,
            note="a grep intent the disk template answers as a read — recorded as measured",
        ),
        Task(
            id="l2_research",
            shape="L2 research (search -> fetch)",
            intent="research llama.cpp quantization options for cpu inference",
            # The native agent searches, then fetches the top hit: the shortest
            # sequence that yields one usable source, which is the check.
            native=(("web_search", {"queries": ["llama.cpp quantization cpu"]}),
                    ("web_fetch", lambda prior: {"urls": _top_url(prior)})),
            check=_check_research,
            note="the pipeline shape: several hops inside one useTool call",
        ),
        Task(
            id="l2_reddit",
            shape="L2 reddit digest",
            intent="what does r/rust say about async runtimes",
            native=(("reddit_digest", {"query": "async runtimes rust", "subreddit": "rust"}),),
            check=_check_reddit,
            note="the CLI is absent on this box: an arm-B error, not a task failure",
        ),
        Task(
            id="compound_walk",
            shape="compound human intent (find + quote)",
            intent="find the file named walk.py and tell me the max depth it enforces",
            native=(("filesys_glob", {"pattern": "**/walk.py", "cwd": "."}),
                    ("filesys_read", {"path": "service/ti/ops/filesys/walk.py"})),
            check=_check_walk,
            note="a two-clause ask a human really issues; the service answers it as one read",
        ),
    ]


def _top_url(prior: list[Any]) -> list[str]:
    """First http URL the search returned, in the order the search ranked them."""
    for step in prior:
        if isinstance(step, list):
            for item in step:
                url = item.get("url") if isinstance(item, dict) else None
                if isinstance(url, str) and url.startswith("http"):
                    return [url]
    return []


def fixtures(base: Path) -> tuple[Path, Path]:
    """The two directories the slot-fill tasks read. Deterministic content."""
    notes = base / "notes"
    reports = base / "reports"
    for directory, files in ((notes, NOTES_DIR_FILES), (reports, REPORT_FILES)):
        directory.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (directory / name).write_text(text, encoding="utf-8")
    return notes, reports


def digest_of(value: Any) -> str:
    """A short stable hash of an artifact, so two runs can be compared without
    the report carrying file bodies."""
    blob = value if isinstance(value, bytes) else repr(value).encode()
    if isinstance(value, str):
        blob = value.encode()
    return hashlib.sha1(blob).hexdigest()[:12]  # run-to-run identity, not security
