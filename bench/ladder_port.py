"""Port the localAlexa ladder's winners onto this harness, and check they still hold.

Four arms, each a harness predictor spec rather than a bespoke script:

| arm | spec | ladder (grouped) |
|---|---|---:|
| ``r0-word-svc`` | ``sklearn_text|vectorizer=tfidf_word|clf=linsvc|C=1`` | 89.79% |
| ``r2-word-logreg-C100`` | ``sklearn_text|vectorizer=tfidf_word|clf=logreg|C=100`` | 90.84% |
| ``r3-setfit`` | ``torch_text`` + ``trainer=localalexa.harness_adapters:train_setfit`` | 93.98% |
| ``r5-ensemble-setfit-z`` | ``ensemble|mode=mean|standardise=true`` over r2 + r3 | 95.03% |

Two things had to be expressible for "the same grouped folds" to mean anything, and
both are options on the harness rather than forks of it:

* the fold map. ``--fold-map`` takes the assignment the ladder wrote into its
  artifact, so this is not "folds that look similar" but the same 110 phrasing
  families in the same five folds;
* the training pool. ``--fit-pool localalexa_refs`` trains on the 200 references
  minus any that are a literal slot fill of a held-out phrasing family. Training on
  the other folds of the 382 instead is a different, easier experiment.

    python ladder_port.py                  # all four arms
    python ladder_port.py --skip-torch     # the two sklearn arms, ~10 s
    python ladder_port.py --verify         # exit non-zero if any arm drifts > 1 row

Needs a python with scikit-learn, and torch/setfit plus the ``localalexa`` package
installed for the two torch arms (``/home/andrew/localAlexa/.venv-embed/bin/python``
on this box).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from harness.metrics import grouped_cv  # noqa: E402
from harness.protocol import build, score  # noqa: E402
from harness.rows import load as load_dataset  # noqa: E402

LADDER_ROOT = Path("/home/andrew/localAlexa")
LADDER_ARTIFACT = LADDER_ROOT / "results" / "ladder" / "rung5-ensemble.json"
LADDER_LINEAR_ARTIFACT = LADDER_ROOT / "results" / "ladder" / "rung4-crossencoder.json"
DATASET = "localalexa_intents|group_by=phrasing"
FIT_POOL = "localalexa_refs"
TOLERANCE_ROWS = 1
CHECKPOINT = BENCH_ROOT / "results" / "ladder-port-2026-11-10" / "setfit.pt"

# What the ladder measured, in the arms' own names. Grouped = 5 folds over the 110
# phrasing families; rowwise = fit on all 200 references, score all 382 rows.
LADDER = {
    "r0-word-svc": {"grouped": 0.8979, "rowwise": 0.9319},
    "r2-word-logreg-C100": {"grouped": 0.9084, "rowwise": 0.9346},
    "r3-setfit": {"grouped": 0.9398, "rowwise": 0.9424},
    "r5-ensemble-setfit-z": {"grouped": 0.9503, "rowwise": 0.9686},
}

SKLEARN_ARMS = {
    "r0-word-svc": "sklearn_text|vectorizer=tfidf_word|clf=linsvc|C=1",
    "r2-word-logreg-C100": "sklearn_text|vectorizer=tfidf_word|clf=logreg|C=100",
}


def torch_arms() -> dict[str, str]:
    from localalexa.harness_adapters import arm_specs

    return arm_specs(str(CHECKPOINT))


def fold_map() -> dict[str, int]:
    """The ladder's own five-fold assignment, read from the artifact that carries it."""
    payload = json.loads(LADDER_ARTIFACT.read_text(encoding="utf-8"))
    assignment = payload["fold_assignments"]
    key = "5" if "5" in assignment else next(iter(assignment))
    return {group: int(fold) for group, fold in assignment[key]["family_fold"].items()}


def ladder_rows(arm: str, protocol: str) -> int:
    return round(LADDER[arm][protocol] * 382)


def run_arm(name: str, spec: str, dataset, refs, mapping: dict[str, int]) -> dict:
    cv = grouped_cv(list(dataset.rows), (lambda s=spec: build(s)), fold_of=mapping,
                    fit_pool=list(refs.rows))
    grouped = cv["oof_accuracy"]
    rowwise_predictor = build(spec)
    rowwise_predictor.fit(list(refs.rows))
    rowwise = score(name, list(dataset.rows), rowwise_predictor)
    latencies = [record.latency_ms for record in cv["oof"]]
    return {
        "arm": name,
        "spec": spec,
        "grouped": {
            "correct": grouped["correct"],
            "n": grouped["n"],
            "accuracy": grouped["accuracy"],
            "ladder_correct": ladder_rows(name, "grouped"),
            "delta_rows": grouped["correct"] - ladder_rows(name, "grouped"),
            "per_fold": [(f["fold"], f["correct"], f["n_train"], f["fit_pool_dropped"])
                         for f in cv["per_fold"]],
            "within_tolerance": abs(grouped["correct"] - ladder_rows(name, "grouped"))
            <= TOLERANCE_ROWS,
        },
        "rowwise": {
            "correct": sum(1 for r in rowwise if r.correct),
            "n": len(rowwise),
            "accuracy": sum(1 for r in rowwise if r.correct) / len(rowwise),
            "ladder_correct": ladder_rows(name, "rowwise"),
            "delta_rows": sum(1 for r in rowwise if r.correct) - ladder_rows(name, "rowwise"),
            "within_tolerance": abs(sum(1 for r in rowwise if r.correct)
                                    - ladder_rows(name, "rowwise")) <= TOLERANCE_ROWS,
        },
        "latency_ms_p50": round(statistics.median(latencies), 2),
        "fold_map": cv["fold_map"],
        "fit_pool_rows": cv["fit_pool"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arms", default="", help="comma list of arm names (default: all four)")
    parser.add_argument("--skip-torch", action="store_true", help="only the two sklearn arms")
    parser.add_argument("--out", default=str(BENCH_ROOT / "results" / "ladder-port-2026-11-10"))
    parser.add_argument("--json", action="store_true", help="print the payload instead of a table")
    args = parser.parse_args(argv)

    specs = dict(SKLEARN_ARMS)
    if not args.skip_torch:
        try:
            specs.update(torch_arms())
        except ImportError as exc:
            print(f"torch arms unavailable ({exc}); running the sklearn arms only", file=sys.stderr)
    if args.arms:
        wanted = [name.strip() for name in args.arms.split(",") if name.strip()]
        unknown = [name for name in wanted if name not in specs]
        if unknown:
            raise SystemExit(f"unknown arms {unknown}; known: {sorted(specs)}")
        specs = {name: specs[name] for name in wanted}

    dataset = load_dataset(DATASET)
    refs = load_dataset(FIT_POOL)
    mapping = fold_map()
    print(f"dataset   {dataset.name}  rows {len(dataset)}  groups {len(dataset.groups())}")
    print(f"fit pool  {refs.name}  rows {len(refs)}  "
          f"leaking {sum(1 for r in refs.rows if r.meta.get('leaks_into'))}")
    print(f"fold map  {LADDER_ARTIFACT.name}  {len(mapping)} groups  "
          f"{len(set(mapping.values()))} folds")

    results = [run_arm(name, spec, dataset, refs, mapping) for name, spec in specs.items()]

    header = (f"\n{'arm':24s} {'grouped':>9s} {'ladder':>9s} {'Δrows':>6s} "
              f"{'rowwise':>9s} {'ladder':>9s} {'Δrows':>6s} {'p50 ms':>7s}")
    print(header)
    print("-" * len(header))
    for result in results:
        g, r = result["grouped"], result["rowwise"]
        print(f"{result['arm']:24s} {g['correct']:5d}/{g['n']:<3d} {g['ladder_correct']:5d}/382 "
              f"{g['delta_rows']:+6d} {r['correct']:5d}/{r['n']:<3d} {r['ladder_correct']:5d}/382 "
              f"{r['delta_rows']:+6d} {result['latency_ms_p50']:7.2f}")

    failing = [r["arm"] for r in results
               if not (r["grouped"]["within_tolerance"] and r["rowwise"]["within_tolerance"])]
    payload = {
        "dataset": DATASET,
        "fit_pool": FIT_POOL,
        "tolerance_rows": TOLERANCE_ROWS,
        "fold_map": mapping,
        "arms": results,
        "all_within_tolerance": not failing,
        "outside_tolerance": failing,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nevery arm within {TOLERANCE_ROWS} row: {not failing}"
          + (f" — outside: {failing}" if failing else ""))
    print(f"wrote {out_dir / 'summary.json'}")
    if args.json:
        print(json.dumps(payload, indent=2))
    return 0 if not failing else 1


if __name__ == "__main__":
    raise SystemExit(main())
