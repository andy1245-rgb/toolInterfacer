"""Guard: no machine-personal absolute paths in source, config, or docs.

Recorded evidence under ``bench/results/`` legitimately contains the machine's
model paths (they are the proof of the phantom-lane bug), ``extern/`` is vendored
code, ``service/logs/`` is runtime output, and the cmake build trees
(``build/``, ``build-vk/``) are generated output that records the source path it
was configured from — all of them are skipped.

Everything else must stay portable: an environment variable, a ``~``-relative
default, a repo-relative path, or ``<placeholder>``. If this test fails, the fix
is to parameterise the path, not to add an exception.
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
SKIP_PREFIXES = {"bench/results", "extern", "service/logs"}
SKIP_FILES = {pathlib.Path(__file__).resolve()}
SUFFIXES = {".py", ".ts", ".json", ".sh", ".md", ".toml", ".yml", ".yaml"}

# Built from parts so this file does not trip its own check.
_HOME = re.compile("/" + "home" + r"/[A-Za-z0-9_.-]+/")
_USERS = re.compile(r"[A-Za-z]:[\\/]+" + "Us" + r"ers[\\/]+[A-Za-z0-9_.-]+")


def _skipped(rel: pathlib.Path) -> bool:
    s = rel.as_posix()
    if any(s == d or s.startswith(d + "/") for d in SKIP_PREFIXES):
        return True
    return any(part in SKIP_DIR_NAMES for part in rel.parts[:-1])


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
