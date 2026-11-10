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
from .protocol import PredictorError, build, honours_order, option_order_sha, score
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
    ap.add_argument("--cv-seeds", default="0", help="comma-separated fold seeds (default 0)")
    ap.add_argument("--split", default="",
                    help="score only rows whose meta.split matches this value")
    ap.add_argument("--fit-on", default="",
                    help="fit on rows whose meta.split matches this value (refused with --cv)")
    args = ap.parse_args(argv)

    dataset: Dataset = load_dataset(args.dataset)
    if args.limit:
        dataset = Dataset(dataset.name, dataset.rows[: args.limit], dataset.options,
                          dataset.source)
    if args.paraphrase:
        mapping = controls.load_paraphrase(args.paraphrase)
        dataset = controls.apply_paraphrase(dataset, mapping)
    prereg = controls.load_preregister(args.preregister) if args.preregister else None

    rows = list(dataset.rows)
    all_rows = rows
    if args.split:
        rows = [r for r in rows if str(r.meta.get("split", "")) == args.split]
        if not rows:
            raise SystemExit(f"--split {args.split!r} matched no row")
        dataset = Dataset(dataset.name, rows, dataset.options, dataset.source)
    fit_rows = None
    if args.fit_on:
        if args.cv:
            raise SystemExit(
                "--fit-on and --cv are mutually exclusive: --fit-on answers 'fit on these rows, "
                "score those', which is the clean protocol; --cv answers 'fit on every fold but "
                "one'. Doing both would fit on the scoring rows inside each fold, which is the "
                "leak this flag exists to close.")
        fit_rows = [r for r in all_rows if str(r.meta.get("split", "")) == args.fit_on]
        if not fit_rows:
            raise SystemExit(f"--fit-on {args.fit_on!r} matched no row")

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
    cv_runs: dict[str, dict[int, dict]] = {}
    cv_seeds = ([int(s) for s in str(args.cv_seeds).split(",") if s.strip()]
                if args.cv else [0])
    for name, spec in arms_specs:
        try:
            predictor = build(spec)
        except PredictorError as exc:
            raise SystemExit(f"arm {name}: {exc}")
        if order_mode and not honours_order(predictor):
            raise SystemExit(
                f"arm {name}: --order {order_mode} cannot be honoured by {getattr(predictor, 'name', spec)}."
                f" It replays a file recorded under one option order, so the run would record a "
                f"permutation the model never saw next to an order_sha claiming otherwise. "
                f"Drop --order, or score a predictor that takes the option set at call time.")
        if args.cv and hasattr(predictor, "fit"):
            # the out-of-fold records *are* the arm's records: that is what lets a paired
            # test run between a fitted arm and a replayed one
            cv_runs[name] = {}
            for seed in cv_seeds:
                cv = metrics.grouped_cv(rows, (lambda s=spec: build(s)), folds=args.cv,
                                        seed=seed, options_for=order_for)
                for rec in cv["oof"]:
                    rec.arm = name
                cv_runs[name][seed] = cv
            records[name] = cv_runs[name][cv_seeds[0]]["oof"]
            continue
        if fit_rows is not None and hasattr(predictor, "fit"):
            try:
                predictor.fit(fit_rows)
            except PredictorError as exc:
                raise SystemExit(f"arm {name}: {exc}")
        records[name] = score(name, rows, predictor, options_for=order_for)

    summary = {"dataset": {"name": dataset.name, "rows": len(dataset), "source": dataset.source,
                           "options": list(dataset.options), "order_sha": order_sha,
                           "order_mode": order_mode or "frozen",
                           "split": args.split or "all",
                           "fit_on": args.fit_on or "none",
                           "cv": args.cv, "cv_seeds": cv_seeds},
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

    if cv_runs:
        for name, runs in cv_runs.items():
            first = runs[cv_seeds[0]]
            print(f"\ngrouped CV  {name}: {args.cv} folds over {first['n_groups']} groups / "
                  f"{first['n_labels']} labels, stratified by label")
            for seed, cv in runs.items():
                acc = cv["oof_accuracy"]
                folds = " ".join("—" if f["accuracy"] is None else f"{100 * f['accuracy']:.0f}"
                                 for f in cv["per_fold"])
                print(f"  seed {seed}: out-of-fold {acc['correct']}/{acc['n']} = "
                      f"{100 * acc['accuracy']:.1f}%   folds {folds}")
                missing = {f["fold"]: f["train_missing_labels"] for f in cv["per_fold"]
                           if f.get("train_missing_labels")}
                if missing:
                    print(f"    WARNING fold(s) {sorted(missing)} hold out a label the training "
                          f"side never sees: {missing} — those rows cannot be predicted")
            print(f"  the arm's records are seed {cv_seeds[0]}'s out-of-fold rows")
            summary["arms"].setdefault(name, {})["grouped_cv"] = {
                "folds": args.cv, "seeds": cv_seeds,
                "n_groups": first["n_groups"], "n_labels": first["n_labels"],
                "per_seed": {str(s): {"fold_of": cv["fold_of"], "per_fold": cv["per_fold"],
                                      "oof_accuracy": cv["oof_accuracy"]}
                             for s, cv in runs.items()}}

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
        paired = summary["arms"][target]["paired_vs_reference"]
        delta_by_seed: dict[str, int] = {}
        if target in cv_runs:
            for seed, cv in cv_runs[target].items():
                p = metrics.paired(records[reference], cv["oof"])
                delta_by_seed[str(seed)] = p["treatment_correct"] - p["reference_correct"]
        else:
            delta_by_seed[str(cv_seeds[0])] = (paired["treatment_correct"]
                                               - paired["reference_correct"])
        ctx = {
            "mcnemar_p": paired["p"],
            "delta": paired["treatment_correct"] - paired["reference_correct"],
            "delta_rows": paired["treatment_correct"] - paired["reference_correct"],
            "accuracy": summary["arms"][target]["accuracy"]["accuracy"],
            "coverage": summary["arms"][target]["coverage_at_precision90"]["coverage"],
            "precision": summary["arms"][target]["coverage_at_precision90"]["precision"],
            "ledger": led["by_label"],
            "ledger_net": led["net"],
            "min_label_net": min((v["net"] for v in led["by_label"].values()), default=0),
            "seeds_positive": sum(1 for d in delta_by_seed.values() if d > 0),
            "seeds_total": len(delta_by_seed),
            "min_seed_delta": min(delta_by_seed.values(), default=0),
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
                             "conditions": verdict["conditions"], "passed": verdict["passed"],
                             "delta_by_seed": delta_by_seed}
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
