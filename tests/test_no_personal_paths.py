"""Guard: no machine-personal absolute paths in source, config, or docs.

Everything the walk reaches must stay portable: an environment variable, a
``~``-relative default, a repo-relative path, or a ``<placeholder>``. If this
test fails, the fix is to parameterise the path, not to add an exception.

What is skipped, and what is not
--------------------------------
Skipped: ``node_modules``, ``__pycache__``, ``.venv``, ``.pytest_cache`` and
``.git`` wherever they appear, the cmake trees (``build/``, ``build-vk/``,
generated output that records the source path it was configured from),
``extern/`` (vendored code) and ``service/logs/`` (runtime output). The pytest
cache is skipped because its ``nodeids`` file records whatever paths the runner
was invoked with, so leaving it in would make the guard fail depending on how
somebody last ran the suite. A ``.git`` entry is skipped whether it
is a directory (an ordinary clone) or a file (a linked worktree, whose body is
a ``gitdir:`` line plus an absolute path outside the repo): either way it is
VCS plumbing, not committed content.

``bench/results/`` **is scanned.** It was exempt once, on the grounds that the
recorded model paths were the proof of the phantom-lane bug. They are redacted
now — ``$HOME/`` for the Linux runs and ``<workspace>\\models\\<model>.gguf`` for
the Windows ones — and the basename, the part that carried the proof, is
intact. So the evidence tree is guarded like everything else, and this file is
what pins the redaction: put a real machine path back into a recorded run and
the suite goes red. This header said the opposite until the walker was
rewritten — the exemption had already been deleted and the prose had not, which
left a reader believing the largest source of recorded paths in the repo was
exempt when it was being scanned.

The walk must be able to prove it walked
----------------------------------------
``scan()`` returns the files it opened alongside the offenders it found. The
assertion that matters is not ``offenders == []`` — that is satisfied by a walk
that reads nothing, which is precisely what this file used to be: with
``SUFFIXES`` emptied and ``ROOT`` pointed at the real tree it reported ten green
tests and fifty-one subtests while opening zero files, and a typo in the walk or
a widened ``SKIP_PREFIXES`` would have stayed green forever. So the walk is
asserted to have read files and to have reached the files this guard exists to
protect, and ``WalkerTest`` runs the same walk over a temporary tree with a
planted violation in it, which a broken walk fails to report.

What the guard covers
---------------------
Five patterns, all built from parts so this file does not trip its own check:

``_HOME``
    ``/home/<name>`` in path position. The account segment is matched whether or
    not a ``/`` follows: a terminator of end-of-string, whitespace, a markdown
    backtick, a quote, ``(``, ``)``, ``.``, ``,``, ``:``, ``;``, ``\\`` or ``/``
    all count. Requiring the trailing ``/`` (as this pattern did until the
    pre-publish audit) matched only the nested form, so the bare form — which is
    how a path appears when it is quoted in prose or captured verbatim in an
    error message — passed the suite while two files still carried it.

``_USERS``
    ``C:\\Users\\<name>`` and ``C:/Users/<name>``, same terminator rule, so the
    bare form matches here too.

``_WIN_STRUCTURE``
    A drive-rooted Windows path that still carries a recorded directory tree:
    a drive letter and a ``Documents`` or ``models`` component anywhere in it.
    Naming a specific sibling directory would
    hardcode one machine's layout, so the rule flags the generic shape of "a
    captured absolute path from a Windows tree" instead.

``_HOME_TREE``
    ``~/Documents/<name>`` — a ``~``-relative path that descends into the
    personal document tree. This is the one home-relative shape the guard used
    to wave through, and it shipped: ``service/ti/conf/defaults.json`` (package
    data, inside the wheel) carried
    ``~/Documents/<private-project>/Research/<project>`` as the default for
    ``ops.reddit.cwd`` and both of its candidates, so every install was told to
    look in one operator's documents folder. "Portable" cannot mean "starts
    with a tilde": ``~/Documents/...`` names a private tree just as surely as
    ``/home/<account>/...`` does. ``~/mnt/somehost`` and
    ``~/.local/state/ti/...`` stay legal, because a mount point and an XDG
    runtime path are generic locations, not one machine's documents.

``_HOME_VAR_TREE``
    ``$HOME/Documents/<name>`` and ``${HOME}/Documents/<name>``: the
    shell-variable spelling of ``_HOME_TREE``. It is a separate rule because
    ``~/`` and ``$HOME/`` are different strings and no other pattern covered
    the second, so the same leak written with a variable would have walked
    straight past the guard. It flags nothing today — the tree holds 31
    ``$HOME/`` occurrences in 16 files and every one is either the redaction
    marker (``$HOME/``), a bracketed placeholder
    (``$HOME/Documents/<private-project>/...``) or the repo's own name
    (``$HOME/<project>``, ``$HOME/toolInterfacer/models/...``) — and it is here
    for the file that spells the documents tree with a real name in it
    tomorrow. The general ``$HOME/<name>`` form is deliberately *not* flagged:
    unlike a tilde path into a documents tree it names no private tree, and it
    is what this repo redacts its own recorded paths *to*.

What it does not cover
----------------------
- **History.** The guard cannot promise repo portability, because it never
  reads history: it scans files on disk, not the blobs in the object database.
  Measured when this walker was rewritten: the operator's Linux home path is in
  124 of the 197 commits reachable across all refs, the Windows one in 104, and
  they sit in source as well as evidence — the extension's ``src/lifecycle.ts``
  in 27 of those commits, ``bench/pyhost_qwen35.py`` in 25, ``bench/sv2lane.sh``
  in 24, ``tests/test_cascade.py`` in 16. A path deleted in a later commit is
  still in the tree of every earlier one, and only a rewrite reaches that. A
  green run here means the current tree is portable; it says nothing about what
  a clone can recover with ``git log -p``.
- **The index.** This is a working-tree walk, not an index read: an untracked
  file left in the tree is scanned (and can fail the guard), and git objects are
  never read. A file that is committed but absent from the checkout is not seen.
- **The account name is not enumerated.** Any name will do, so the guard cannot
  tell a real account from a plausible placeholder. That is deliberate — the
  repo's placeholder convention is angle-bracketed, and ``/home/<user>`` and
  ``C:\\Users\\<user>`` do not match because ``<`` is not a name character.
- **Placeholder forms are allowed.** ``/home/<user>``, ``C:\\Users\\<user>``,
  ``<workspace>\\models\\<model>.gguf``, ``~/mnt/somehost``,
  ``~/Documents/<project>/x`` and ``$HOME/Documents/<project>/x`` are portable
  by the repo's own rule and are never flagged — ``<`` is not a name character,
  so a bracketed placeholder reads as a placeholder. What is *not* portable is a
  tilde- or ``$HOME``-relative path into the documents tree with a real name in
  it; the prefix makes the path resolve, not generic.
- **Sibling project names are not matched by name.** They are only caught when
  they sit behind ``Documents/``, ``Documents\\`` or ``models\\``. A prose
  mention in provenance attribution is not a leak and stays.
- **macOS and absolute mount paths are not covered.** ``/Users/<name>`` and a
  root-anchored ``/mnt/<host>`` occur nowhere in this tree (checked before
  adding, not assumed), so no pattern was added for either — a rule for a shape
  that exists nowhere buys nothing today and would need pinning like any other.
  The only ``/mnt/`` forms here are ``~/mnt/<host>``, which is a generic mount
  point and stays legal. A green run therefore says nothing about a macOS
  checkout; that shape needs a pattern first.
- **Binary and unlisted suffixes.** Files are read when their suffix is in
  ``SUFFIXES``, which now includes ``""`` so an extensionless or dot-only name
  (``LICENSE``, ``.gitignore``, ``.gitattributes``) is read too. ``.svg`` and
  ``.png`` are read as text with ``errors="replace"`` because a recorded
  screenshot carries the path it was taken at in its metadata. Anything else —
  a ``.lock``, a ``.ipynb``, a ``.gguf`` — is not read.
- **A drive-rooted path with no ``Documents``/``models`` component** (a synthetic
  fixture such as ``C:\\fixture\\x.gguf``) is not flagged: it names no structure.
"""
import pathlib
import re
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
# Directory *names* skipped wherever they appear, and path *prefixes* skipped
# only at the repo root. `node_modules` was previously matched as a prefix, so
# only a top-level one was skipped and `pi-extension/node_modules` was scanned
# as portable source — which is how the extension's own dependency tree made
# this guard fail on files nobody wrote. `build*` are cmake trees: the CPU build
# lives at the root, the Vulkan one is out-of-tree beside it. A name is skipped
# wherever it appears including as the entry itself: in a linked worktree `.git`
# is a *file* whose body is a `gitdir:` line pointing at an absolute path
# outside the repo, and it is never committed content. `.pytest_cache` is runtime
# output whose `nodeids` cache echoes back whatever paths pytest was invoked with.
SKIP_DIR_NAMES = {"node_modules", "__pycache__", ".venv", ".pytest_cache",
                  ".git", "build", "build-vk"}
# Vendored code and runtime output. `bench/results/` is deliberately not here:
# its recorded paths are redacted to `$HOME/` and `<workspace>\models\...` and
# the evidence is scanned (see the module docstring).
SKIP_PREFIXES = {"extern", "service/logs"}
SKIP_FILES = {pathlib.Path(__file__).resolve()}
# Files the walk reads, by suffix. `""` is not a typo: pathlib reports no suffix
# for an extensionless or dot-only name, and `LICENSE`, `.gitignore` and
# `.gitattributes` are portable files this guard was silently not reading.
# Everything below the source formats is deliberate widening that catches
# nothing today -- `bench/lane.env.example` is already sanitised to
# `C:/path/to/...` and the `.jsonl`/`.txt`/`.csv`/`.cast` fixtures hold no path
# at all -- and is read so that tomorrow's file is caught instead of skipped.
# They are cheap, and measured rather than assumed: 344 files and 7.8 MB of text
# before, 358 files and 8.6 MB after (the 15 tracked files that were unscanned
# plus one pytest-cache README that is now skipped), and this file's own runtime
# goes from 0.6-0.9 s to 0.9-1.0 s (best of four, alternating, on a loaded box).
# `.svg` and `.png` are read as text with `errors="replace"` because a recorded
# screenshot carries the path it was taken at in its metadata; binary bytes
# cannot raise.
SUFFIXES = {".py", ".ts", ".json", ".sh", ".md", ".toml", ".yml", ".yaml",
            ".example", ".jsonl", ".txt", ".csv", ".cast", ".svg", ".png", ""}

# Built from parts so this file does not trip its own check. The single
# terminator expression is shared: whatever follows the account name, end of
# string and the punctuation a path-but-no-slash is quoted with must match.
_END = r"(?=$|[\s`'\"().,:;/\\])"
_NAME = r"[A-Za-z0-9_.-]+"
# The drive letter must not be the tail of a word: without the lookbehind,
# `https://host/docs/models/` is read as drive `s:` and `tried:\n` as drive `d:`.
_DRIVE = r"(?<![A-Za-z0-9_])[A-Za-z]:" + r"[\\/]+"
_HOME = re.compile("/" + "home" + r"/" + _NAME + _END)
_USERS = re.compile(_DRIVE + "Us" + "ers" + r"[\\/]+" + _NAME + _END)
_WIN_STRUCTURE = re.compile(
    _DRIVE
    + r"(?:[A-Za-z0-9_.<> -]+[\\/]+)*?"
    + r"(?:Doc" + r"uments|models)[\\/]+"
)
_HOME_TREE = re.compile("~" + r"/" + "Doc" + "uments" + r"/" + _NAME + _END)
# `$HOME/` and `${HOME}/` are the shell spellings of the same leak as `~/`, and
# neither `_HOME_TREE` nor `_HOME` covers them.
_HOME_VAR_TREE = re.compile(
    r"\$\{?" + "HO" + "ME" + r"\}?" + "/" + "Doc" + "uments" + r"/" + _NAME + _END
)
PATTERNS = (_HOME, _USERS, _WIN_STRUCTURE, _HOME_TREE, _HOME_VAR_TREE)


def _skipped(rel: pathlib.Path) -> bool:
    s = rel.as_posix()
    if any(s == d or s.startswith(d + "/") for d in SKIP_PREFIXES):
        return True
    return any(part in SKIP_DIR_NAMES for part in rel.parts)


def scan(root: pathlib.Path) -> tuple[list[str], list[str]]:
    """Walk `root`; return (offenders, files read) as repo-relative paths.

    The files are returned, not just the offenders, because a walk that opens
    nothing finds nothing: `offenders == []` on its own is satisfied by a guard
    that reads no files at all, so what was read has to be checkable.
    """
    offenders: list[str] = []
    read: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in SUFFIXES or path in SKIP_FILES:
            continue
        rel = path.relative_to(root)
        if _skipped(rel):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        read.append(rel.as_posix())
        for pattern in PATTERNS:
            for match in pattern.finditer(text):
                offenders.append(f"{rel.as_posix()}: {match.group(0)}")
    return offenders, read


class GuardPatternTest(unittest.TestCase):
    """The pattern must match the shapes it exists to catch.

    A guard that cannot fail is worse than no guard: it advertises a portability
    guarantee it does not enforce. The trailing-slash `_HOME` this file used to
    carry matched `/home/user/…` and missed `/home/user` followed by a backtick,
    a quote or end of line, and the suite stayed green with two files still
    holding the path. These cases pin both forms, and pin the near-misses so a
    later edit cannot widen the pattern into false positives.
    """

    def assertMatches(self, pattern, text):
        self.assertIsNotNone(pattern.search(text), f"{pattern.pattern!r} missed {text!r}")

    def assertNoMatch(self, pattern, text):
        self.assertIsNone(pattern.search(text), f"{pattern.pattern!r} false-positive on {text!r}")

    def test_home_matches_nested_form(self):
        self.assertMatches(_HOME, "/home/someone/project/models/d1-3B-Q4_K_M.gguf")
        self.assertMatches(_HOME, "path is /home/someone/")

    def test_home_matches_bare_form(self):
        # end of string, and every character a bare path gets quoted with
        for tail in ("", " ", "\n", "`", "'", '"', ")", "(", ".", ",", ":", ";", "\\"):
            with self.subTest(tail=tail):
                self.assertMatches(_HOME, "roots ('/home/someone" + tail)

    def test_home_ignores_words_that_only_start_with_home(self):
        for text in ("/homework/x", "/home", "/homepage", "~/mnt/somehost",
                     "/home/<user>", "/home/"):
            with self.subTest(text=text):
                self.assertNoMatch(_HOME, text)

    def test_users_matches_both_separators_and_the_bare_form(self):
        for text in (r"C:\Users\someone\Documents", "C:/Users/someone",
                     r"C:\Users\someone", "C:/Users/someone`", r"C:\Users\someone/",
                     r"('C:\Users\someone')"):
            with self.subTest(text=text):
                self.assertMatches(_USERS, text)

    def test_users_ignores_placeholders_and_urls(self):
        for text in (r"C:\Users\<user>\Documents", "C:/Users/<user>",
                     "https://host/Users/someone", r"Users\someone"):
            with self.subTest(text=text):
                self.assertNoMatch(_USERS, text)

    def test_structure_matches_a_captured_windows_prefix(self):
        for text in (
            r"C:\Users\someone\Documents\sibling\lab\models\d1.gguf",
            r"C:\Users\<user>\Documents\\sibling\\personal\\lab\\models\\d1.gguf",
            r"D:\models\Qwen3.5-4B-Q4_K_M.gguf",
            r"C:\data\Documents\x",
        ):
            with self.subTest(text=text):
                self.assertMatches(_WIN_STRUCTURE, text)

    def test_structure_ignores_placeholders_urls_and_home_relative(self):
        # `~/Documents/sibling/Research/x` is deliberately in this list: the
        # Windows-structure rule must not fire on a home-relative path. It is
        # caught by `_HOME_TREE` instead -- the two rules answer different
        # questions and neither may absorb the other.
        for text in (r"<workspace>\models\d1-3B-Q4_K_M.gguf",
                     "https://unsloth.ai/docs/models/",
                     "~/Documents/sibling/Research/x",
                     r"tried:\n  .../Documents/sibling/x",
                     r"C:\fixture\lab\d1-3B-Q4_K_M.gguf",
                     "$DELL_MOUNT/Documents/x"):
            with self.subTest(text=text):
                self.assertNoMatch(_WIN_STRUCTURE, text)

    def test_home_tree_matches_a_private_documents_tree(self):
        """The shape that shipped in `conf/defaults.json`: what the old guard
        waved through as "`~`-relative, therefore portable"."""
        for text in ("cwd: ~/Documents/some-machines-tree/Research/some-project",
                     '"~/Documents/sibling/Research/x"',
                     "~/Documents/sibling`",
                     "~/Documents/proj",
                     "~/Documents/a-private-tree"):
            with self.subTest(text=text):
                self.assertMatches(_HOME_TREE, text)

    def test_home_tree_ignores_generic_home_locations_and_placeholders(self):
        for text in ("~/Documents/<project>/x", "~/mnt/somehost",
                     "~/.local/state/ti/http.token", "~/redditScrapingSearching",
                     "~/Documents", "~/.pi/profiles/registry/x",
                     "Documents/x"):
            with self.subTest(text=text):
                self.assertNoMatch(_HOME_TREE, text)

    def test_home_var_tree_matches_the_shell_spelling(self):
        """The `$HOME` form of the leak `_HOME_TREE` pins: same tree, same
        account, different prefix, and no other pattern saw it."""
        for text in ('cwd: $HOME/Documents/some-machines-tree/Research/proj',
                     "HOME_CWD=${HOME}/Documents/sibling/Research/x",
                     '"$HOME/Documents/sibling"',
                     "$HOME/Documents/a-private-tree\n",
                     "tried:\n  $HOME/Documents/proj`"):
            with self.subTest(text=text):
                self.assertMatches(_HOME_VAR_TREE, text)

    def test_home_var_tree_ignores_the_redaction_marker_and_placeholders(self):
        # `$HOME/` is what this repo redacts its own recorded paths to, and
        # `$HOME/...` is how it writes "some path under home" in prose. Neither
        # may be flagged, nor may the repo's own name or a bracketed
        # placeholder: a rule that fires on the sanitised form would make the
        # guard fail on its own evidence.
        for text in ("$HOME/", "every home path is `$HOME/`",
                     "$HOME/...", "path: $HOME/<project>",
                     "$HOME/Documents/<private-project>/<project>",
                     "$HOME/toolInterfacer/models/d1-3B-Q4_K_M.gguf",
                     "${HOME}/<project>", "$HOME/mnt/somehost",
                     "HOME=Documents/x", "$HOMEDIR/Documents/x",
                     'SRC=${TI_LLAMA_SRC:-$HOME/toolInterfacer/extern/llama.cpp}'):
            with self.subTest(text=text):
                self.assertNoMatch(_HOME_VAR_TREE, text)


class WalkerTest(unittest.TestCase):
    """The walk itself must be able to fail.

    `offenders == []` is satisfied by a walker that reads nothing, and that was
    not hypothetical here: with `SUFFIXES` emptied and `ROOT` pointed at the
    real tree this file reported ten passed and fifty-one subtests passed while
    opening zero files. These cases run the production `scan()` over a
    temporary tree with one known-bad file in it, so a walk that stops reading,
    or stops counting, fails here first.
    """

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = pathlib.Path(tmp.name)
        (self.root / "src").mkdir()
        (self.root / "docs").mkdir()
        # the shape that shipped: a home path with a real account in it, in a
        # file type the guard reads
        (self.root / "src" / "conf.json").write_text(
            '{"cwd": "/home/someone/Research/some-project"}\n', encoding="utf-8")
        # portable files: read, counted, and not reported
        (self.root / "docs" / "note.md").write_text("a portable note\n", encoding="utf-8")
        (self.root / "docs" / "budget.csv").write_text("name,count\n", encoding="utf-8")

    def test_planted_violation_is_reported(self):
        offenders, read = scan(self.root)
        self.assertEqual(sorted(read),
                         ["docs/budget.csv", "docs/note.md", "src/conf.json"])
        self.assertEqual([o.split(":")[0] for o in offenders], ["src/conf.json"],
                         f"the planted violation was not reported: {offenders}")

    def test_walk_disabled_reads_nothing_so_the_count_is_what_catches_it(self):
        """The counterfactual, kept as a test rather than done once by hand:
        empty the suffix filter and the walker still reports no offenders, so an
        empty offender list is identical in both states. Only what was read can
        tell them apart, which is why the real tree asserts on it."""
        saved = globals()["SUFFIXES"]
        globals()["SUFFIXES"] = set()
        try:
            offenders, read = scan(self.root)
        finally:
            globals()["SUFFIXES"] = saved
        self.assertEqual((offenders, read), ([], []),
                         "this case is meant to show the walk reading nothing")
        self.assertNotEqual(scan(self.root)[1], [],
                            "with the filter restored the walk must read files again")


class NoPersonalPathsTest(unittest.TestCase):
    # Files this guard exists to protect. A count above zero is a weak floor on
    # its own -- a widened skip prefix or a typo in the suffix filter can leave
    # a handful of files and still pass -- so the walk has to reach these.
    REQUIRED = ("README.md", "pyproject.toml", "CONTRIBUTING.md",
                "service/ti/conf/defaults.json")

    def test_no_personal_absolute_paths(self):
        offenders, read = scan(ROOT)
        self.assertGreater(
            len(read), 0,
            "the walk read no files, so an empty offender list would prove "
            "nothing; check SUFFIXES, SKIP_PREFIXES and SKIP_DIR_NAMES",
        )
        for required in self.REQUIRED:
            self.assertIn(required, read, "the walk never reached a file this guard exists to protect")
        self.assertEqual(
            offenders, [],
            "machine-personal absolute paths in portable files "
            "(use an env var / ~-relative / <placeholder>):\n" + "\n".join(offenders[:20]),
        )


if __name__ == "__main__":
    unittest.main()
