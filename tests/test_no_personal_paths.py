"""Guard: no machine-personal absolute paths in source, config, or docs.

Recorded evidence under ``bench/results/`` legitimately contains the machine's
model paths (they are the proof of the phantom-lane bug), ``extern/`` is vendored
code, and ``service/logs/`` is runtime output — all three are skipped.

Everything else must stay portable: an environment variable, a ``~``-relative
default, a repo-relative path, or ``<placeholder>``. If this test fails, the fix
is to parameterise the path, not to add an exception.
"""
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SKIP_DIRS = {"bench/results", "extern", "service/logs", ".git", "node_modules",
             "__pycache__", ".venv"}
SKIP_FILES = {pathlib.Path(__file__).resolve()}
SUFFIXES = {".py", ".ts", ".json", ".sh", ".md", ".toml", ".yml", ".yaml"}

# Built from parts so this file does not trip its own check.
_HOME = re.compile("/" + "home" + r"/[A-Za-z0-9_.-]+/")
_USERS = re.compile(r"[A-Za-z]:[\\/]+" + "Us" + r"ers[\\/]+[A-Za-z0-9_.-]+")


def _skipped(rel: pathlib.Path) -> bool:
    s = rel.as_posix()
    return any(s == d or s.startswith(d + "/") for d in SKIP_DIRS)


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
            for pattern in (_HOME, _USERS):
                for match in pattern.finditer(text):
                    offenders.append(f"{rel}: {match.group(0)}")
        self.assertEqual(
            offenders, [],
            "machine-personal absolute paths in portable files "
            "(use an env var / ~-relative / <placeholder>):\n" + "\n".join(offenders[:20]),
        )


if __name__ == "__main__":
    unittest.main()
