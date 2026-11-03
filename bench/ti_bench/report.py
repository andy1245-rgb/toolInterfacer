"""report — the report file: written once, and checked against its own contract.

No-clobber is enforced by the filesystem, not by a check-then-write. Two runs
racing for the same path would otherwise both pass an ``exists()`` test and the
second would silently replace a measurement with another measurement of the same
name. ``open(..., "x")`` lets exactly one win.

``validate`` is the same contract the selftest and the tests assert, so a report
that is internally inconsistent (counts that do not add up, a band table that
does not sum to n, a percentile ordering that is impossible) fails at write time
instead of being read as a number months later.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import stats
from .scoring import ABSTAIN, ERROR, IN_PROCESS, MODEL, SCHEMA

TOP_KEYS = ("schema", "generated", "arm", "set", "guard", "warmup", "counts", "accuracy",
            "per_band", "latency", "calibration", "calibration_n", "per_row")
ROW_KEYS = ("id", "uid", "idx", "band", "utterance", "expected", "chosen", "correct",
            "confidence", "latency_ms", "real_call", "source", "error", "extra")
SOURCES = (MODEL, IN_PROCESS, ABSTAIN, ERROR)
ACCURACY_TOL = 5e-4  # accuracy is stored rounded to 4 dp


class ReportExistsError(FileExistsError):
    """The output path already holds a report and --force was not given."""


def write_report(path: str | Path, report: dict, force: bool = False) -> Path:
    """Write ``report`` to ``path`` atomically; refuse an existing path unless forced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(report, indent=2) + "\n"
    try:
        with open(path, "w" if force else "x", encoding="utf-8") as fh:
            fh.write(blob)
    except FileExistsError as e:
        raise ReportExistsError(
            f"{path} already exists — refusing to overwrite a measurement "
            f"(pass --force, or --out somewhere else)") from e
    return path


def read_report(path: str | Path) -> dict:
    """Load a report and refuse one written by something else."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a {SCHEMA} report (schema={data.get('schema')!r})")
    problems = validate(data)
    if problems:
        raise ValueError(f"{path}: malformed report: " + "; ".join(problems[:5]))
    return data


def validate(report: dict) -> list[str]:
    """Every way ``report`` breaks its own contract; empty means it is well formed."""
    problems: list[str] = []
    missing = [k for k in TOP_KEYS if k not in report]
    if missing:
        return [f"missing key(s): {', '.join(missing)}"]
    if report["schema"] != SCHEMA:
        problems.append(f"schema is {report['schema']!r}, expected {SCHEMA!r}")

    rows = report["per_row"]
    if not isinstance(rows, list) or not rows:
        return problems + ["per_row is empty"]
    counts, accuracy, latency = report["counts"], report["accuracy"], report["latency"]
    n = len(rows)
    if counts.get("n") != n:
        problems.append(f"counts.n={counts.get('n')} != {n} rows")
    if accuracy.get("n") != n:
        problems.append(f"accuracy.n={accuracy.get('n')} != {n} rows")
    correct = sum(1 for r in rows if r.get("correct"))
    if counts.get("correct") != correct or accuracy.get("correct") != correct:
        problems.append(f"correct count {correct} disagrees with counts/accuracy")
    if counts.get("wrong") != n - correct:
        problems.append("counts.wrong != n - correct")
    if accuracy.get("accuracy") is not None and abs(accuracy["accuracy"] - correct / n) > ACCURACY_TOL:
        problems.append(f"accuracy {accuracy.get('accuracy')} != {correct}/{n}")
    lo, hi = accuracy["wilson95"]["low"], accuracy["wilson95"]["high"]
    if not 0.0 <= lo <= accuracy["accuracy"] <= hi <= 1.0:
        problems.append(f"wilson interval [{lo}, {hi}] does not bracket the accuracy")

    for band, block in report["per_band"].items():
        members = [r for r in rows if r.get("band") == band]
        if block.get("n") != len(members):
            problems.append(f"per_band[{band}].n={block.get('n')} != {len(members)} rows")
        got = sum(1 for r in members if r.get("correct"))
        if block.get("correct") != got:
            problems.append(f"per_band[{band}].correct={block.get('correct')} != {got}")
    if sum(b.get("n") or 0 for b in report["per_band"].values()) != n:
        problems.append("per_band counts do not sum to n")

    used = [r for r in rows if r.get("latency_ms") is not None and r.get("source") != ERROR]
    if latency.get("n_used") != len(used):
        problems.append(f"latency.n_used={latency.get('n_used')} != {len(used)} timed rows")
    if used:
        p50, p90, p99 = latency["p50"], latency["p90"], latency["p99"]
        if not p50 <= p90 <= p99:
            problems.append(f"percentiles out of order: p50={p50} p90={p90} p99={p99}")
        if min(latency["min"], p50) < 0:
            problems.append("negative latency")
        recomputed = stats.percentile([r["latency_ms"] for r in used], 0.50)
        if abs(recomputed - p50) > 1e-2:
            problems.append(f"p50={p50} != recomputed {recomputed:.3f}")

    with_conf = [r for r in rows if r.get("confidence") is not None]
    if report.get("calibration_n") != len(with_conf):
        problems.append(f"calibration_n={report.get('calibration_n')} != {len(with_conf)}")
    if sum(b.get("n") or 0 for b in report["calibration"]) != len(with_conf):
        problems.append("calibration bands do not cover every answered row")
    for band in report["calibration"]:
        if band["n"] and not 0.0 <= band["mean_conf"] <= 1.0:
            problems.append(f"calibration band {band['lo']}-{band['hi']} has mean_conf "
                            f"{band['mean_conf']}")

    warm = report["warmup"]
    if warm.get("calls_discarded") != len(warm.get("latency_ms") or []):
        problems.append("warmup.calls_discarded != number of warm-up samples")

    for row in rows:
        absent = [k for k in ROW_KEYS if k not in row]
        if absent:
            problems.append(f"row {row.get('id')} missing key(s): {', '.join(absent)}")
            continue
        if row["source"] not in SOURCES:
            problems.append(f"row {row['id']}: source {row['source']!r} not in {SOURCES}")
        if row["correct"] and row["chosen"] != row["expected"]:
            problems.append(f"row {row['id']}: correct but {row['chosen']!r} != "
                            f"{row['expected']!r}")
        if row["source"] in (ABSTAIN, ERROR) and row["real_call"]:
            problems.append(f"row {row['id']}: an {row['source']} row claims a real call")
        if row["source"] == MODEL and not row["real_call"]:
            problems.append(f"row {row['id']}: a model row without a real call")
        if row["source"] in (ABSTAIN, ERROR) and row["confidence"] is not None:
            problems.append(f"row {row['id']}: an {row['source']} row carries a confidence")
    return problems


def summary_line(report: dict) -> str:
    """One-line digest, for the run banner and for `compare` headers."""
    acc = report["accuracy"]
    lat = report["latency"]
    return (f"arm={report['arm']['name']} backend={report['arm']['backend']} "
            f"set={report['set']['name']} n={acc['n']} "
            f"acc={acc['correct']}/{acc['n']} ({_pct(acc['accuracy'])}) "
            f"wilson95=[{_pct(acc['wilson95']['low'])},{_pct(acc['wilson95']['high'])}] "
            f"p50={lat['p50']}ms p90={lat['p90']}ms p99={lat['p99']}ms")


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"
