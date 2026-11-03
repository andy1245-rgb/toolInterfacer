#!/usr/bin/env python3
"""run.py — one command to score a decide arm and compare two of them.

    python3 bench/run.py list
    python3 bench/run.py run --arm d1 --set v2 --out bench/results/d1-v2.json
    python3 bench/run.py compare bench/results/d1-v2.json bench/results/pair-v2.json
    python3 bench/run.py selftest

Every arm is built with ``make_backend`` directly, so neither the decision cache
nor the ``heuristic_micro`` pre-gate can answer a row and put a 0 ms sample into
a latency distribution; the arms themselves come from ``bench/arms.json``; and any
arm that needs a live llama-server asserts the served model through
``/props`` before a single probe byte is sent — the guard
``docs/model-sweep-consolidated.md`` (CORRECTION) exists because a lane without it
measured a stale server and published it under another model's name.

Exit codes: 0 ok · 1 refused (an existing output file, a malformed report) ·
2 usage or I/O · 3 the endpoint guard refused the run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ti_bench import arms as arms_mod  # noqa: E402
from ti_bench import compare as compare_mod  # noqa: E402
from ti_bench import guard as guard_mod  # noqa: E402
from ti_bench import report as report_mod  # noqa: E402
from ti_bench import scoring, selftest as selftest_mod, sets as sets_mod  # noqa: E402


def _rel(path: Path | str) -> str:
    """Path relative to the repo when it is inside it (reports stay portable)."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _default_out(arm: arms_mod.Arm, set_name: str) -> Path:
    return ROOT / "bench/results" / f"{arm.name}-{set_name}.json"


def cmd_list(args: argparse.Namespace) -> int:
    spec = arms_mod.load_arms(args.arms)
    offline = [a for a in spec.values() if a.offline]
    spec_path = Path(args.arms) if args.arms else arms_mod.DEFAULT_SPEC
    print(f"# {len(spec)} arm(s) from {_rel(spec_path)} "
          f"({len(offline)} offline, {len(spec) - len(offline)} need an endpoint)")
    print(f"{'arm':<14}{'backend':<17}{'needs':<26}{'realcall':<10}guard")
    for arm in spec.values():
        print(f"{arm.name:<14}{arm.backend:<17}{arm.where:<26}"
              f"{'yes' if arm.real_call else 'no':<10}{_guard_desc(arm)}")
    uncovered = [name for name in arms_mod.registered_backends()
                 if name not in {a.backend for a in spec.values()}]
    if uncovered:
        print(f"\n# registered backends with no arm in the spec: {', '.join(uncovered)}")
    else:
        print("\n# every registered backend has an arm in the spec")
    return 0


def _guard_desc(arm: arms_mod.Arm) -> str:
    parts = []
    if arm.needs_llama:
        parts.append(f"/props {arm.expect_model or '(needs --expect-model)'}")
    if arm.needs_hosted:
        parts.append(f"env {guard_mod.api_key_env()}")
    return " + ".join(parts) or "none (in-process)"


def cmd_run(args: argparse.Namespace) -> int:
    arm = arms_mod.get_arm(args.arm, args.arms)
    rows = sets_mod.load(args.set)
    out = Path(args.out) if args.out else _default_out(arm, args.set)
    if out.exists() and not args.force:
        print(f"refusing to overwrite {_rel(out)} — pass --force or --out elsewhere",
              file=sys.stderr)
        return 1
    try:
        rep = scoring.score(arm, rows, limit=args.limit, warmup=args.warmup,
                            expect_model=args.expect_model, base_url=args.base_url,
                            allow_unasserted=args.allow_unasserted_model)
    except guard_mod.GuardError as e:
        print(f"ABORT: {e}\nno probe traffic was sent", file=sys.stderr)
        return 3
    except arms_mod.ArmSpecError as e:
        print(f"ABORT (arm {args.arm!r}): {e}", file=sys.stderr)
        return 2
    problems = report_mod.validate(rep)
    if problems:
        print("ABORT: the run produced a malformed report, nothing written:\n  "
              + "\n  ".join(problems), file=sys.stderr)
        return 1
    report_mod.write_report(out, rep, force=args.force)
    _print_report(rep, out)
    return 0


def _print_report(rep: dict, out: Path) -> None:
    print(report_mod.summary_line(rep))
    guard_block = rep["guard"]
    if guard_block["checked"]:
        print(f"guard: {guard_block['source']} asserts {guard_block['expect_model']} "
              f"— served {guard_block['served_basename']}")
    elif guard_block["required"]:
        print(f"guard: no /props assertion ({', '.join(guard_block['required'])} only)")
    for band, block in rep["per_band"].items():
        print(f"  {band:<5}{block['correct']:>4}/{block['n']:<4} = "
              f"{block['accuracy'] * 100:5.1f}%  "
              f"[{block['wilson95']['low'] * 100:.1f}-{block['wilson95']['high'] * 100:.1f}]")
    lat = rep["latency"]
    print(f"latency p50={lat['p50']} p90={lat['p90']} p99={lat['p99']} ms over "
          f"n={lat['n_used']} rows "
          f"({rep['warmup']['calls_discarded']} warm-up call(s) discarded, "
          f"{lat['error_rows_excluded']} error row(s) excluded)")
    print(f"calibration (n={rep['calibration_n']} answered rows)")
    for band in rep["calibration"]:
        if not band["n"]:
            continue
        print(f"  conf {band['lo']:.1f}-{band['hi']:.1f}  n={band['n']:<4} "
              f"accuracy={band['accuracy'] * 100:5.1f}%  mean_conf={band['mean_conf']:.3f}")
    counts = rep["counts"]
    print(f"counts: {counts['correct']} correct, {counts['wrong']} wrong, "
          f"{counts['abstain']} abstain, {counts['error']} error, "
          f"{counts['real_calls']} real call(s)")
    print(f"wrote {_rel(out)}")


def cmd_compare(args: argparse.Namespace) -> int:
    try:
        a = report_mod.read_report(args.file_a)
        b = report_mod.read_report(args.file_b)
        data = compare_mod.compare(a, b)
    except (OSError, ValueError) as e:  # CompareError and a malformed report are ValueErrors
        print(f"ABORT: {e}", file=sys.stderr)
        return 2
    print("\n".join(compare_mod.render(data)))
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    return selftest_mod.main()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    def spec_flag(p: argparse.ArgumentParser) -> None:
        p.add_argument("--arms", default=None,
                       help="arm spec JSON (default: bench/arms.json)")

    p_list = sub.add_parser("list", help="the arms in the spec and what each one needs")
    spec_flag(p_list)
    p_list.set_defaults(func=cmd_list)

    p_run = sub.add_parser("run", help="score one arm over one frozen set")
    p_run.add_argument("--arm", required=True, help="arm name from the spec")
    p_run.add_argument("--set", default="v2", choices=sorted(sets_mod.SETS),
                       help="frozen set (default %(default)s)")
    p_run.add_argument("--out", default="", help="report path (default "
                                                "bench/results/<arm>-<set>.json)")
    p_run.add_argument("--force", action="store_true", help="replace an existing report")
    p_run.add_argument("--limit", type=int, default=0, help="score only the first N rows")
    p_run.add_argument("--warmup", type=int, default=None,
                       help="throwaway calls before the scored pass (default: 1 for an arm "
                            "that calls out, 0 for an in-process arm)")
    p_run.add_argument("--expect-model", default="",
                       help="gguf basename the live server must serve; overrides the arm spec")
    p_run.add_argument("--base-url", default="",
                       help="repoint the arm's endpoint (a tunnel or another port)")
    p_run.add_argument("--allow-unasserted-model", action="store_true",
                       help="probe a live server without a /props assertion (loudly warned)")
    spec_flag(p_run)
    p_run.set_defaults(func=cmd_run)

    p_cmp = sub.add_parser("compare", help="McNemar exact + latency delta for two reports")
    p_cmp.add_argument("file_a", help="report A")
    p_cmp.add_argument("file_b", help="report B")
    p_cmp.set_defaults(func=cmd_compare)

    p_self = sub.add_parser("selftest", help="the whole path offline (CI smoke test)")
    p_self.set_defaults(func=cmd_selftest)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
