"""run — one CLI over the four seams: any dataset, any predictor, one set of metrics.

    python3 -m bench.harness.run \\
        --dataset tool_routing \\
        --arm floor_a=recorded|arm=floor_a \\
        --arm grep_only=recorded|arm=grep_only \\
        --reference floor_a --aa

    python3 -m bench.harness.run --dataset tool_routing \\
        --predictor 'fake|accuracy=0.8|seed=7' --aa --order rev

An arm is a name and a predictor spec. The same dataset is scored by every arm, in the same
session, with the same option order (and that order, plus its sha256, goes into the artifact).

Without ``--preregister`` the run prints numbers and **no verdict**: the conditions that decide
whether a delta counts are supposed to exist before the run does.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import controls, metrics
from .protocol import PredictorError, build, option_order_sha, score
from .rows import Dataset, DatasetError, load as load_dataset
from .predictors import *  # noqa: F401,F403 - importing the package registers every predictor


def parse_arms(args) -> list[tuple[str, str]]:
    arms: list[tuple[str, str]] = []
    if args.predictor:
        arms.append(("arm", args.predictor))
    for item in args.arm or []:
        if "=" not in item:
            raise SystemExit(f"--arm {item!r} must be name=spec")
        name, spec = item.split("=", 1)
        arms.append((name.strip(), spec.strip()))
    if args.arms:
        for item in args.arms.split(";"):
            if not item.strip():
                continue
            if "=" not in item:
                raise SystemExit(f"--arms entry {item!r} must be name=spec")
            name, spec = item.split("=", 1)
            arms.append((name.strip(), spec.strip()))
    if not arms:
        raise SystemExit("no arms: give --predictor SPEC or --arm name=SPEC")
    names = [n for n, _ in arms]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate arm names: {names}")
    return arms


def _fmt_ci(acc: dict) -> str:
    lo, hi = acc["wilson95"]
    return f"{100 * acc['accuracy']:5.1f}% [{100 * lo:.1f},{100 * hi:.1f}]"


def summarise_arm(recs, reference=None) -> dict:
    out = {"accuracy": metrics.accuracy(recs),
           "latency": metrics.latency(recs),
           "calibration": metrics.calibration(recs),
           "coverage_at_precision90": metrics.coverage_at_precision(recs, 0.9),
           "order": metrics.order_summary(recs),
           "groups": metrics.group_report(recs),
           "confusion": metrics.confusion(recs)}
    if reference is not None:
        out["paired_vs_reference"] = metrics.paired(reference, recs)
        out["ledger_vs_reference"] = metrics.ledger(reference, recs)
    return out


def print_arm_table(arms: dict, reference: str) -> None:
    print(f"\n{'arm':<14}{'n':>5}{'correct':>8}  {'accuracy (95% Wilson)':<24}"
          f"{'p50 ms':>8}{'p95 ms':>8}  {'ECE/Brier/NLL':<22}")
    for name, recs in arms.items():
        acc = metrics.accuracy(recs)
        lat = metrics.latency(recs)
        cal = metrics.calibration(recs)
        cal_txt = (f"{cal['ece']:.3f}/{cal['brier']:.3f}/{cal['nll']:.3f}"
                   if cal else "no probabilities")
        mark = "*" if name == reference else " "
        print(f"{mark}{name:<13}{acc['n']:>5}{acc['correct']:>8}  {_fmt_ci(acc):<24}"
              f"{lat['p50']:>8.0f}{lat['p95']:>8.0f}  {cal_txt:<22}")


def print_ledger(reference: str, name: str, led: dict) -> None:
    print(f"\nper-label ledger: {reference} -> {name}")
    print(f"  {'label':<18}{'n':>5}{'base':>6}{'treat':>7}{'fix':>5}{'brk':>5}{'net':>5}")
    for label, v in led["by_label"].items():
        print(f"  {label:<18}{v['n']:>5}{v['base']:>6}{v['treat']:>7}"
              f"{v['fixed']:>5}{v['broke']:>5}{v['net']:>+5}")
    print(f"  {'TOTAL':<18}{sum(v['n'] for v in led['by_label'].values()):>5}"
          f"{'':>6}{'':>7}{led['fixed']:>5}{led['broke']:>5}{led['net']:>+5}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="tool_routing", help="name or name|key=value|...")
    ap.add_argument("--predictor", default="", help="one arm, named 'arm'")
    ap.add_argument("--arm", action="append", metavar="NAME=SPEC")
    ap.add_argument("--arms", default="", help="name=spec;name=spec")
    ap.add_argument("--reference", default="", help="the arm every other is paired against")
    ap.add_argument("--aa", action="store_true", help="score the reference arm twice: the floor")
    ap.add_argument("--order", default="", help="rev | shuffle:<seed>")
    ap.add_argument("--paraphrase", default="", help="{option: wording} JSON")
    ap.add_argument("--preregister", default="", help="conditions JSON/YAML, read before the run")
    ap.add_argument("--out", default="", help="directory for summary.json and rows.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="score at most N rows (smoke test)")
    ap.add_argument("--cv", type=int, default=0, help="grouped CV folds for fitted predictors")
    args = ap.parse_args(argv)

    dataset: Dataset = load_dataset(args.dataset)
    if args.limit:
        dataset = Dataset(dataset.name, dataset.rows[: args.limit], dataset.options,
                          dataset.source)
    if args.paraphrase:
        mapping = controls.load_paraphrase(args.paraphrase)
        dataset = controls.apply_paraphrase(dataset, mapping)
    prereg = controls.load_preregister(args.preregister) if args.preregister else None

    arms_specs = parse_arms(args)
    reference = args.reference or arms_specs[0][0]
    if reference not in {n for n, _ in arms_specs}:
        raise SystemExit(f"--reference {reference!r} is not one of the arms")

    order_mode = args.order
    order_for = controls.options_for(order_mode)
    order_sha = option_order_sha(controls.permute(dataset.options, order_mode))

    print(f"dataset   {dataset.name}  rows {len(dataset)}  options {len(dataset.options)}  "
          f"groups {len(dataset.groups())}  order_sha {order_sha}"
          + (f"  order={order_mode}" if order_mode else ""))
    if args.paraphrase:
        print(f"paraphrase {args.paraphrase} over {len(mapping)} option(s)")

    records: dict[str, list] = {}
    for name, spec in arms_specs:
        try:
            predictor = build(spec)
        except PredictorError as exc:
            raise SystemExit(f"arm {name}: {exc}")
        records[name] = score(name, list(dataset.rows), predictor, options_for=order_for)

    summary = {"dataset": {"name": dataset.name, "rows": len(dataset), "source": dataset.source,
                           "options": list(dataset.options), "order_sha": order_sha,
                           "order_mode": order_mode or "frozen"},
               "reference": reference, "arms": {}}
    print_arm_table(records, reference)

    for name, recs in records.items():
        if name == reference:
            continue
        paired = metrics.paired(records[reference], recs)
        led = metrics.ledger(records[reference], recs)
        print(f"\npair: {reference} -> {name}")
        print(f"  fixed {paired['fixed']}  broke {paired['broke']}  "
              f"delta {paired['treatment_correct'] - paired['reference_correct']:+d} rows  "
              f"exact McNemar p = {paired['p']:.4f}")
        print_ledger(reference, name, led)

    for name, recs in records.items():
        cap = metrics.coverage_at_precision(recs, 0.9)
        if cap["coverage"] is not None:
            thr = ("no confidence reaches it" if cap["threshold"] is None
                   else f"threshold {cap['threshold']:.3f}")
            print(f"\nreject curve {name}: coverage@precision=0.90 = {cap['coverage']:.3f} "
                  f"({cap['n_accepted']}/{len(recs)} rows, {thr})")

    floor = None
    if args.aa:
        ref_spec = dict(arms_specs)[reference]
        again = score(f"{reference}~aa", list(dataset.rows), build(ref_spec),
                      options_for=order_for)
        floor = metrics.a_a_flips(records[reference], again)
        print(f"\nA/A floor  {reference} run twice: {floor['choice_flips']} choice flips in "
              f"{floor['n']} rows ({100 * floor['rate']:.2f}%)")
        if ref_spec.startswith("recorded"):
            print("           note: a replayed arm is a file, so this is a determinism check of "
                  "the harness, not a floor for the served model")

    if args.cv:
        for name, spec in arms_specs:
            try:
                predictor = build(spec)
            except PredictorError:
                continue
            if not hasattr(predictor, "fit"):
                print(f"\ngrouped CV  {name}: skipped (this predictor has no fit())")
                continue
            cv = metrics.grouped_cv(list(dataset.rows), (lambda s=spec: build(s)),
                                    folds=args.cv)
            acc = cv["oof_accuracy"]
            print(f"\ngrouped CV  {name}: {args.cv} folds over {len(cv['fold_of'])} groups, "
                  f"out-of-fold {acc['correct']}/{acc['n']} = {100 * acc['accuracy']:.1f}%")
            for f in cv["per_fold"]:
                a = "—" if f["accuracy"] is None else f"{100 * f['accuracy']:.1f}%"
                print(f"  fold {f['fold']}: train {f['n_train']:>4}  test {f['n_test']:>4}  {a}")
            summary["arms"][name] = {"grouped_cv": {
                "folds": args.cv, "fold_of": cv["fold_of"], "per_fold": cv["per_fold"],
                "oof_accuracy": acc}}

    for name, recs in records.items():
        summary["arms"].setdefault(name, {}).update(summarise_arm(
            recs, reference=records[reference] if name != reference else None))

    # ---- the verdict, which needs a declaration made before the run
    if prereg is None:
        print("\nno --preregister file: numbers reported, no verdict. Conditions belong in the "
              "file before the run, not in the reading of it.")
    else:
        target = prereg.get("arm") or next(n for n, _ in arms_specs if n != reference)
        led = summary["arms"][target]["ledger_vs_reference"]
        edited = set(prereg.get("edited_labels") or [])
        ctx = {
            "mcnemar_p": summary["arms"][target]["paired_vs_reference"]["p"],
            "delta": summary["arms"][target]["paired_vs_reference"]["treatment_correct"]
            - summary["arms"][target]["paired_vs_reference"]["reference_correct"],
            "delta_rows": summary["arms"][target]["paired_vs_reference"]
            ["treatment_correct"] - summary["arms"][target]["paired_vs_reference"]
            ["reference_correct"],
            "accuracy": summary["arms"][target]["accuracy"]["accuracy"],
            "coverage": summary["arms"][target]["coverage_at_precision90"]["coverage"],
            "precision": summary["arms"][target]["coverage_at_precision90"]["precision"],
            "ledger": led["by_label"],
            "ledger_net": led["net"],
            "ledger_untouched_net": metrics.ledger_net(
                led, [l for l in led["by_label"] if l not in edited]) if edited else led["net"],
            "aa_flips": (floor or {}).get("choice_flips", 0),
        }
        verdict = controls.check(prereg, ctx)
        print(f"\npre-registered conditions (arm {target} vs {reference}):")
        for c in verdict["conditions"]:
            label = c.get("name") or c["expr"]
            print(f"  {'PASS' if c['passed'] else 'FAIL'}  {label:<42} "
                  f"[{c['value']} {c['op']} {c['target']}]")
        summary["prereg"] = {"path": args.preregister, "arm": target,
                             "conditions": verdict["conditions"], "passed": verdict["passed"]}
        if floor is not None:
            gate = controls.gate_verdict(ctx["delta"], floor["choice_flips"])
            summary["prereg"]["floor_gate"] = gate
            print(f"  {'PASS' if gate['clears'] else 'FAIL'}  clears 3x the A/A floor"
                  f"{'':<27}[{gate['delta']} rows vs {gate['needed']:.0f} needed]")
            verdict["passed"] = verdict["passed"] and gate["clears"]
            summary["prereg"]["passed"] = verdict["passed"]
        print(f"\nVERDICT: {'PASS' if verdict['passed'] else 'FAIL'}")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "rows.jsonl").open("w", encoding="utf-8") as fh:
            for name, recs in records.items():
                for r in recs:
                    fh.write(json.dumps(r.as_dict(), sort_keys=True) + "\n")
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n",
                                          encoding="utf-8")
        print(f"\nwrote {out / 'summary.json'} and {out / 'rows.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
