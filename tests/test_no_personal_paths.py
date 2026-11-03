"""Guard: no machine-personal absolute paths in source, config, or docs.

Recorded evidence under ``bench/results/`` legitimately contains the machine's
model paths (they are the proof of the phantom-lane bug), ``extern/`` is vendored
code, ``service/logs/`` is runtime output, and the cmake build trees
(``build/``, ``build-vk/``) are generated output that records the source path it
was configured from — all of them are skipped.

Everything else must stay portable: an environment variable, a ``~``-relative
default, a repo-relative path, or ``<placeholder>``. If this test fails, the fix
is to parameterise the path, not to add an exception.

What the guard covers
---------------------
Three patterns, all built from parts so this file does not trip its own check:

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

What it does not cover
----------------------
- **History.** It scans the working tree, not committed blobs. A path removed in
  a later commit is still in the trees of earlier ones; only a rewrite touches
  that.
- **The account name is not enumerated.** Any name will do, so the guard cannot
  tell a real account from a plausible placeholder. That is deliberate — the
  repo's placeholder convention is angle-bracketed, and ``/home/<user>`` and
  ``C:\\Users\\<user>`` do not match because ``<`` is not a name character.
- **Placeholder forms are allowed.** ``/home/<user>``, ``C:\\Users\\<user>``,
  ``<workspace>\\models\\<model>.gguf`` and anything ``~``-relative
  (``~/mnt/somehost``) are portable by the repo's own rule and are never flagged.
- **Sibling project names are not matched by name.** They are only caught when
  they sit behind a drive-rooted ``Documents\\`` or ``models\\``. A prose mention
  in provenance attribution is not a leak and stays.
- **Only files whose suffix is listed** (``SUFFIXES``) are read. A binary, a
  ``.lock``, a ``.png`` or an extensionless file is not scanned.
- **A drive-rooted path with no ``Documents``/``models`` component** (a synthetic
  fixture such as ``C:\\fixture\\x.gguf``) is not flagged: it names no structure.
"""
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
# Directory *names* skipped wherever they appear, and path *prefixes* skipped
# only at the repo root. `node_modules` was previously matched as a prefix, so
# only a top-level one was skipped and `pi-extension/node_modules` was scanned
# as portable source — which is how the extension's own dependency tree made
# this guard fail on files nobody wrote. `build*` are cmake trees: the CPU build
# lives at the root, the Vulkan one is out-of-tree beside it.
SKIP_DIR_NAMES = {"node_modules", "__pycache__", ".venv", ".git", "build",
                  "build-vk"}
# `bench/results/` used to be exempt because the recorded model paths were the
# proof of the phantom-lane bug. Both prefixes are redacted now -- `$HOME/` for
# the Linux runs and `<workspace>\models\<model>.gguf` for the Windows ones --
# and the basename, the part that proved the point, is intact, so the exemption
# is gone and the evidence is still guarded.
SKIP_PREFIXES = {"extern", "service/logs"}
SKIP_FILES = {pathlib.Path(__file__).resolve()}
SUFFIXES = {".py", ".ts", ".json", ".sh", ".md", ".toml", ".yml", ".yaml"}

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
PATTERNS = (_HOME, _USERS, _WIN_STRUCTURE)


def _skipped(rel: pathlib.Path) -> bool:
    s = rel.as_posix()
    if any(s == d or s.startswith(d + "/") for d in SKIP_PREFIXES):
        return True
    return any(part in SKIP_DIR_NAMES for part in rel.parts[:-1])


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
        for text in (r"<workspace>\models\d1-3B-Q4_K_M.gguf",
                     "https://unsloth.ai/docs/models/",
                     "~/Documents/sibling/Research/x",
                     r"tried:\n  .../Documents/sibling/x",
                     r"C:\fixture\lab\d1-3B-Q4_K_M.gguf",
                     "$DELL_MOUNT/Documents/x"):
            with self.subTest(text=text):
                self.assertNoMatch(_WIN_STRUCTURE, text)


class NoPersonalPathsTest(unittest.TestCase):
    def test_no_personal_absolute_paths(self):
        offenders: list[str] = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.suffix not in SUFFIXES or path in SKIP_FILES:
                continue
            rel = path.relative_to(ROOT)
            if _skipped(rel):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for pattern in PATTERNS:
                for match in pattern.finditer(text):
                    offenders.append(f"{rel}: {match.group(0)}")
        self.assertEqual(
            offenders, [],
            "machine-personal absolute paths in portable files "
            "(use an env var / ~-relative / <placeholder>):\n" + "\n".join(offenders[:20]),
        )


if __name__ == "__main__":
    unittest.main()
