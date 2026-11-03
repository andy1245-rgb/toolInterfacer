"""selftest — the whole path, offline: no network, no model server, no service.

This is the CI-able smoke test. It scores the spec's offline arms over the first
rows of the frozen v2 set, checks each report against the shape the harness
promises, proves the no-clobber rule on a real file, then diffs the two reports so
``compare`` runs too. Any failure exits non-zero and says which step it was.

It runs in a temp directory, so it never writes into ``bench/results``.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from . import arms, compare, report as report_mod, scoring, sets

LIMIT = 12  # stride rows are enough to exercise every aggregate; the point is the path
OFFLINE_ARMS = ("fixture", "heuristic")


def _spread(rows: list[dict], k: int) -> list[dict]:
    """Every k-th row, so a small sample still covers all three bands."""
    if len(rows) <= k:
        return rows
    step = len(rows) / k
    return [rows[int(i * step)] for i in range(k)]


def main(argv: list[str] | None = None) -> int:
    try:
        return _run()
    except AssertionError as e:
        print(f"selftest FAILED: {e}", file=sys.stderr)
        return 1


def _run() -> int:
    spec = arms.load_arms()
    chosen = [name for name in OFFLINE_ARMS if name in spec]
    assert chosen, f"no offline arm in the spec (looked for {OFFLINE_ARMS})"
    for name in chosen:
        assert spec[name].offline, f"arm {name!r} is not offline; selftest must not need a server"
    print(f"[ok] spec: {len(spec)} arm(s), offline: {', '.join(chosen)}")

    rows = _spread(sets.load("v2"), LIMIT)
    assert len(rows) == LIMIT, f"frozen set v2 is shorter than {LIMIT} rows"
    print(f"[ok] set v2: {len(rows)} rows, bands {sets.bands(rows)}")

    reports = {}
    for name in chosen:
        rep = scoring.score(spec[name], rows, warmup=0, stream=sys.stderr)
        problems = report_mod.validate(rep)
        assert not problems, f"arm {name!r} produced a malformed report: {problems[:3]}"
        counts = rep["counts"]
        assert counts["real_calls"] == 0, f"arm {name!r} claimed a real call while offline"
        print(f"[ok] scored {name}: {counts['correct']}/{counts['n']} correct, "
              f"{counts['abstain']} abstain, {counts['error']} error, "
              f"p50={rep['latency']['p50']}ms")
        reports[name] = rep

    with tempfile.TemporaryDirectory(prefix="ti-bench-selftest-") as tmp:
        paths = {}
        for name, rep in reports.items():
            path = report_mod.write_report(Path(tmp) / f"{name}.json", rep)
            assert path.read_text().startswith("{"), "report file is not JSON"
            try:
                report_mod.write_report(path, rep)
            except report_mod.ReportExistsError:
                pass
            else:
                raise AssertionError(f"writing {path} twice did not hit the no-clobber rule")
            report_mod.write_report(path, rep, force=True)  # --force is the only way past it
            paths[name] = path
        print(f"[ok] report write: {len(paths)} file(s), no-clobber refused and --force replaced")

        if len(chosen) >= 2:
            a, b = (report_mod.read_report(paths[name]) for name in chosen[:2])
            data = compare.compare(a, b)
            mc = data["mcnemar"]
            assert 0.0 <= mc["p_value"] <= 1.0, f"p-value out of range: {mc['p_value']}"
            assert mc["discordant"] == mc["a_only"] + mc["b_only"], "discordant counts disagree"
            assert mc["both_correct"] + mc["both_wrong"] + mc["discordant"] == data["n"], \
                "McNemar cells do not sum to n"
            assert data["latency"]["n_pairs"] > 0, "compare found no paired latency rows"
            lines = compare.render(data)
            assert lines, "compare rendered nothing"
            print(f"[ok] compare {chosen[0]} vs {chosen[1]}: discordant {mc['discordant']} "
                  f"p={mc['p_value']:.6f}, {data['latency']['n_pairs']} paired latency rows")
        else:
            print("[--] compare skipped: only one offline arm in the spec")

    print("selftest PASSED — harness path is offline-clean and internally consistent")
    return 0
