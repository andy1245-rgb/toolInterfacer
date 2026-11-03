"""Stats — per-split counts, label distribution, band coverage.

Band coverage is reported against the t1/t2/t3 bands ``utterances_v2`` declares:
t1 unambiguous, t2 veiled, t3 boundary pairs. A split that is missing a band
outright is a hard problem for evaluation — the split cannot tell you whether the
model generalises past keyword matching — so absent bands are called out rather
than printed as a zero somewhere in a table.

Only ``utterances_v2`` rows carry a band. Every other source reports under
``band: "-"`` and is excluded from band coverage, so the percentages are over the
subset that actually declares a band and cannot be diluted by v1 rows.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from . import criteria as C

BANDS = ("t1", "t2", "t3")


def collect(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    per_split: dict[str, Any] = {}
    for name in ("train", "val", "test"):
        rows = list(splits.get(name, []))
        per_split[name] = {
            "n": len(rows),
            "by_kind": dict(Counter(r.get("kind") for r in rows).most_common()),
            "by_source": dict(Counter(r.get("source") for r in rows).most_common()),
            "by_label": dict(Counter(r.get("label") for r in rows).most_common()),
            "by_criteria": dict(Counter(r.get("criteria_id") for r in rows).most_common()),
            "by_label_source": dict(Counter(r.get("label_source") for r in rows).most_common()),
            "bands": {b: sum(1 for r in rows if r.get("band") == b) for b in BANDS},
        }
    totals = {
        "n": sum(s["n"] for s in per_split.values()),
        "by_label": dict(Counter(r.get("label") for rows in splits.values()
                                 for r in rows).most_common()),
        "bands": {b: sum(s["bands"][b] for s in per_split.values()) for b in BANDS},
    }
    return {"splits": per_split, "totals": totals,
            "band_gaps": _band_gaps(per_split)}


def render(summary: Mapping[str, Any], ingests: Mapping[str, Any] | None = None) -> str:
    lines: list[str] = []
    totals = summary["totals"]
    lines.append(f"records: {totals['n']}")
    if ingests:
        lines.extend(_render_ingest(ingests))

    lines.append("")
    lines.append(f"{'split':<7}{'n':>6}{'train/val/test %':>18}")
    for name, s in summary["splits"].items():
        share = (100.0 * s["n"] / totals["n"]) if totals["n"] else 0.0
        lines.append(f"{name:<7}{s['n']:>6}{share:>17.1f}%")

    lines.append("")
    lines.append("per split, by source:")
    for name, s in summary["splits"].items():
        body = ", ".join(f"{k}={v}" for k, v in s["by_source"].items()) or "-"
        lines.append(f"  {name:<6} {body}")

    lines.append("")
    lines.append("per split, label source (gold = vetted file, observed = executed op):")
    for name, s in summary["splits"].items():
        body = ", ".join(f"{k}={v}" for k, v in s["by_label_source"].items()) or "-"
        lines.append(f"  {name:<6} {body}")

    lines.append("")
    lines.append("label distribution (all splits):")
    for label, n in totals["by_label"].items():
        cells = " ".join(f"{name[:2]}={summary['splits'][name]['by_label'].get(label, 0):>3}"
                         for name in ("train", "val", "test"))
        lines.append(f"  {str(label):<32}{n:>5}   {cells}")

    lines.append("")
    lines.append("band coverage (utterances_v2 bands only):")
    for name, s in summary["splits"].items():
        cell = " ".join(f"{b}={s['bands'][b]}" for b in BANDS)
        lines.append(f"  {name:<6} {cell}")
    for gap in summary["band_gaps"]:
        lines.append(f"  ! {gap}")
    return "\n".join(lines)


def _band_gaps(per_split: Mapping[str, Any]) -> list[str]:
    gaps: list[str] = []
    for name, s in per_split.items():
        if not s["n"]:
            gaps.append(f"{name} is empty")
            continue
        missing = [b for b in BANDS if not s["bands"][b]]
        if missing and sum(s["bands"].values()):
            gaps.append(f"{name} has no {', '.join(missing)} rows — band coverage incomplete")
    return gaps


def _render_ingest(ingests: Mapping[str, Any]) -> list[str]:
    lines = ["", "ingest: per source, and what the rule rejected"]
    sources = ingests.get("sources") or {}
    for name, info in sources.items():
        if not info.get("present"):
            lines.append(f"  {name:<22} absent")
            continue
        extra = ""
        if info.get("per_op"):
            extra = "  " + ", ".join(
                f"{op} {v['hops']} hops/{v['with_intent']} with intent/{v['records']} rec"
                if isinstance(v, dict) else f"{op}:{v}"
                for op, v in info["per_op"].items())
        lines.append(f"  {name:<22} {info.get('records', 0):>5} records{extra}")
    for reason, n in (ingests.get("skipped") or {}).items():
        lines.append(f"  rejected: {reason}: {n}")
    return lines


def render_ingest(ingests: Mapping[str, Any]) -> str:
    """The ingest section on its own, for the ``ingest`` verb's output."""
    return "\n".join(_render_ingest(ingests)).lstrip()
