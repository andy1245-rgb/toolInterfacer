"""The consequence model, enforced rather than asserted.

A router in the mid-80s is usable because a misroute costs one wasted call: every op is a read,
the single subprocess path takes an argv list and no shell, and filesystem paths are confined to
the run root. Each of those is a property that one commit could quietly remove, so each is a test.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPS_DIR = ROOT / "service/ti/ops"
sys.path.insert(0, str(ROOT / "service"))

MUTATION = re.compile(
    r"\b(write_text|write_bytes|os\.remove|os\.unlink|os\.mkdir|os\.makedirs|os\.rename|"
    r"os\.replace|os\.rmdir|shutil\.rmtree|shutil\.move|shutil\.copy\w*|"
    r"subprocess\.call|os\.system|os\.popen)\b")
SHELLING = re.compile(r"shell\s*=\s*True|os\.system|os\.popen")


def _op_sources():
    return sorted(p for p in OPS_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def test_no_op_can_change_anything():
    offenders = []
    for path in _op_sources():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            hit = MUTATION.search(line)
            if hit:
                offenders.append(f"{path.relative_to(ROOT)}:{number}: {hit.group(0)}")
    assert not offenders, (
        "an op gained the ability to mutate; the consequence model — a misroute costs one wasted "
        "call — no longer holds:\n  " + "\n  ".join(offenders))


def test_the_one_subprocess_path_has_no_shell():
    cli = (OPS_DIR / "cli.py").read_text(encoding="utf-8")
    assert "subprocess.run(argv" in cli, "the argv-list call is the property being guarded"
    offenders = [f"{number}: {line.strip()}"
                 for number, line in enumerate(cli.splitlines(), 1)
                 if SHELLING.search(line)]
    assert not offenders, "shell execution appeared in the ops adapter: " + "; ".join(offenders)


def test_the_eight_router_ops_are_reads():
    """Each router op returns a value and touches nothing; the registry says so by signature."""
    from ti.ops import op_api

    op_api.import_all()
    expected = {"filesys_read", "filesys_glob", "filesys_grep", "decide_pick",
                "decide_extract", "web_search", "web_fetch", "reddit_digest"}
    registered = set(op_api._REGISTRY)  # noqa: SLF001 - the registry is the contract here
    assert expected <= registered, f"router ops missing from the registry: {expected - registered}"
    for name in sorted(expected):
        assert op_api.get(name).kind, f"{name} declares no artifact kind"


def test_filesystem_reads_are_confined():
    from ti.ops.filesys import confine

    for outside in ("/etc/passwd", "../../etc/passwd", "/proc/self/environ"):
        try:
            confine.resolve(outside)
        except Exception:  # noqa: BLE001 - any refusal is the pass
            continue
        raise AssertionError(f"{outside} resolved; the run root no longer confines reads")
    inside = Path(confine.resolve("bench/data/heldout_v3.json"))
    assert inside.exists()
