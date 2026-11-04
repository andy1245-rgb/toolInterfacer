#!/usr/bin/env python3
"""distill_transfer — what the corpus-trained classifier does on the frozen 150.

The corpus is the training set; `tests/data/utterances_v2.json` is the only eval
set that shares its 8-op taxonomy (`docs/distill-corpus-certification-2026-11-03.md`
§1). So this is a **transfer** measurement: the model is fitted on
`data/distill/train.jsonl`, then frozen and scored on all 150 v2 rows. Nothing in
this file fits on v2 — the `Predictor.frozen` flag makes `eval_harness` skip its
refit loop entirely, exactly as it does for the recorded 3B, so both signals are
scored once, on the same rows, with the same cluster-bootstrap interval.

Three things this instrument refuses to do:

* **a pooled headline.** `--by-band` is the report, not an option. t1 is 64 of the
  150 rows and its utterances name the operation.
* **an unpaired comparison.** Every "beats" claim is an exact McNemar over the
  rows the two signals disagree on, because at n=150 a two-point gap is noise.
* **a straight face about the split.** v2's family grouping is the dataset's
  declared t3 boundary pairs only, so 95 of 150 rows are atomic — the interval
  below is honest about dependence, but the *number* is optimistic by an unknown
  amount because 63% of rows cannot be family-held-out at all.

Usage::

    python3 bench/distill_transfer.py                # the shipped model
    python3 bench/distill_transfer.py --sweep        # every declared candidate
    python3 bench/distill_transfer.py --json bench/results/distill_transfer.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT / "tools"), str(ROOT / "bench")):
    if p not in sys.path:
        sys.path.insert(0, p)

import eval_harness as H  # noqa: E402
from distill import clf  # noqa: E402
from ti_bench.stats import mcnemar_exact  # noqa: E402

D1_RUN = ROOT / "bench/results/p1-diagnose/baseline_live.json"

# Declared before the v2 measurement, so the choice is not made on the eval set.
# C1 is the family-CV winner and the degenerate reference: unbounded Adagrad,
# 100% in-sample, eight seeds that disagree out of distribution.
CANDIDATES: dict[str, dict] = {
    "C1-ens-raw-nodecay": {"learner": "ensemble", "mode": "raw"},
    "C2-ens-both-nodecay": {"learner": "ensemble", "mode": "both"},
    "C3-ens-abstract-nodecay": {"learner": "ensemble", "mode": "abstract"},
    "C4-ens-both-decay85": {"learner": "ensemble", "mode": "both", "decay": 0.85},
    "C5-ridge-raw": {"learner": "ridge", "mode": "raw", "lam": 1.0},
    "C6-ridge-both": {"learner": "ridge", "mode": "both", "lam": 1.0},
    "C7-ridge-abstract": {"learner": "ridge", "mode": "abstract", "lam": 1.0},
}
SHIPPED = "C6-ridge-both"


# --------------------------------------------------------------------------- #
# predictors — all frozen, none of them fits on v2
# --------------------------------------------------------------------------- #

class Distill(H.Predictor):
    """The corpus-trained classifier, frozen."""

    frozen = True

    def __init__(self, name: str, config: dict) -> None:
        self.name = name
        self.config = config
        self.model = None
        self.conf: dict[str, float] = {}

    def fit(self, train):
        """Fit on the corpus, and refuse if the scored set overlaps it.

        The harness hands `fit` the rows being scored — v2. Fitting on them would
        be the one thing this measurement cannot survive, so the rows are fitted
        from `data/distill/train.jsonl` and the argument is used only as the
        disjointness check.
        """
        corpus = clf.load_split("train")
        overlap = {r["utterance"] for r in corpus} & {e.text for e in train}
        if overlap:
            raise SystemExit(f"{self.name}: {len(overlap)} scored rows are in the "
                             "training corpus — refusing to report a number")
        self.model = clf.DistillClassifier.from_config(self.config).fit(corpus)

    def predict(self, text: str) -> str | None:
        label, conf = self.model.predict(text)
        self.conf[text] = conf
        return label


class Recorded(H.Predictor):
    """d1-3B's answers from the archived per-row run — no live calls."""

    frozen = True

    def __init__(self, path: Path, name: str = "d1_3B") -> None:
        self.path = path
        self.name = name
        self._choice: dict[str, str | None] = {}

    def fit(self, train):
        rows = json.loads(self.path.read_text())["per_row"]
        if len(rows) != len(train):
            raise SystemExit(f"{self.path.name}: {len(rows)} rows vs dataset {len(train)}")
        for i, (row, ex) in enumerate(zip(rows, train)):
            if row["utterance"] != ex.text:
                raise SystemExit(f"{self.path.name}: row {i} is not this dataset")
        self._choice = {r["utterance"]: r.get("choice") for r in rows}

    def predict(self, text):
        return self._choice.get(text)


class Rules(H.Predictor):
    """`bench/local_clf.py`'s keyword cue table: frozen, and trained on nothing.

    The bar worth clearing is not the 3B. A hand-written cue table already scores
    78.7% on these 150 rows with no training data at all, so a classifier that
    costs a corpus and a fit has to beat *that* to be worth shipping.
    """

    frozen = True
    name = "rules"

    def __init__(self) -> None:
        self._fn = None

    def fit(self, train):
        import local_clf as lc
        self._fn = lc.rules_predict

    def predict(self, text):
        return self._fn(text)[0]


# --------------------------------------------------------------------------- #
# paired analysis
# --------------------------------------------------------------------------- #

def paired(a: H.Result, b: H.Result, ds: H.Dataset) -> dict:
    """Rows a fixes, rows a breaks, and the exact McNemar p over the discordant set."""
    fixed = sum(1 for x, y in zip(a.outcomes, b.outcomes) if x > y)
    broke = sum(1 for x, y in zip(a.outcomes, b.outcomes) if y > x)
    return {"a": a.predictor, "b": b.predictor, "fixed": fixed, "broke": broke,
            "net": fixed - broke, "p": round(mcnemar_exact(fixed, broke), 4)}


def per_band(res: H.Result, ds: H.Dataset) -> dict:
    hits: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for o, e in zip(res.outcomes, ds.examples):
        band = e.meta.get("band") or "?"
        hits[band][0] += int(o)
        hits[band][1] += 1
    return {b: {"correct": c, "n": n, "acc": round(c / n, 4)}
            for b, (c, n) in sorted(hits.items())}


def per_class(res: H.Result, ds: H.Dataset) -> dict:
    hits: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for o, e in zip(res.outcomes, ds.examples):
        hits[e.label][0] += int(o)
        hits[e.label][1] += 1
    return {lab: {"correct": c, "n": n, "acc": round(c / n, 4)}
            for lab, (c, n) in sorted(hits.items())}


def confusion(res: H.Result, ds: H.Dataset) -> dict:
    out: dict[str, dict[str, int]] = collections.defaultdict(dict)
    for pred, e in zip(res.preds_per_seed[0], ds.examples):
        out[e.label][pred or "(abstain)"] = out[e.label].get(pred or "(abstain)", 0) + 1
    return {k: dict(sorted(v.items(), key=lambda kv: -kv[1])) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# corpus-side criteria, so the choice among candidates is visible
# --------------------------------------------------------------------------- #

def corpus_scores(config: dict) -> dict:
    """Family-grouped CV over train.jsonl plus the certified val split."""
    from distill import train as T  # noqa: PLC0415 — tools/ is on the path above

    rows = clf.load_split("train")
    cv = T.grouped_cv(rows, config)
    model = clf.DistillClassifier.from_config(config).fit(rows)
    val = T.score(model, clf.load_split("val"))
    return {"cv": cv, "val": {b: val["bands"][b] for b in ("t1", "t2", "t3")},
            "val_pooled": val["bands"]["all"]}


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def render(rows: list[H.Result], ds: H.Dataset, analyses: dict) -> str:
    by = {r.predictor: r for r in rows}
    lines = [H.render_table(rows, "v2 — frozen transfer, family-grouped split"), ""]

    lines += ["## per band (the headline, per the certification's own rule)", "",
              "| signal | t1 | t2 | t3 |", "|---|---:|---:|---:|"]
    bands = {r.predictor: per_band(r, ds) for r in rows}
    for name, b in bands.items():
        cells = " | ".join(f"{b[k]['correct']}/{b[k]['n']} = {b[k]['acc'] * 100:.1f}%"
                           if k in b else "—" for k in ("t1", "t2", "t3"))
        lines.append(f"| `{name}` | {cells} |")

    lines += ["", "## per class", "", "| op | n | " +
              " | ".join(f"`{n}`" for n in by) + " |",
              "|---|---:|" + "---:|" * len(by)]
    per = {name: per_class(r, ds) for name, r in by.items()}
    for lab in ds.labels:
        cells = " | ".join(f"{per[n][lab]['acc'] * 100:.0f}%" for n in by)
        n = per[next(iter(by))][lab]["n"]
        lines.append(f"| `{lab}` | {n} | {cells} |")

    if analyses:
        lines += ["", "## paired tests (exact McNemar over discordant rows)", "",
                  "| a | b | a fixes | a breaks | net | p |", "|---|---|---:|---:|---:|---:|"]
        for a in analyses.values():
            verdict = "no evidence" if a["p"] >= 0.05 else "significant"
            lines.append(f"| `{a['a']}` | `{a['b']}` | {a['fixed']} | {a['broke']} | "
                         f"{a['net']:+d} | {a['p']:.4f} ({verdict}) |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep", action="store_true",
                    help="measure every declared candidate, not just the shipped one")
    ap.add_argument("--json", default="", help="write every number here")
    ap.add_argument("--no-corpus-cv", action="store_true",
                    help="skip the family-CV/val columns (slow)")
    args = ap.parse_args(argv)

    ds = H.load_v2(grouping="boundary")
    chosen = CANDIDATES if args.sweep else {SHIPPED: CANDIDATES[SHIPPED]}
    preds: list[H.Predictor] = [Distill(n, c) for n, c in chosen.items()]
    preds.append(Recorded(D1_RUN))
    try:
        import local_clf  # noqa: F401 — the cue table's home, and a numpy check
        preds.append(Rules())
    except ImportError:
        print("# rules table unavailable (bench/local_clf.py needs numpy) — "
              "published 78.7% referenced only\n")

    rows = H.measure(ds, preds, split="group")
    by = {r.predictor: r for r in rows}
    analyses = {}
    for name in chosen:
        if name in by and "d1_3B" in by:
            analyses[f"{name}~d1"] = paired(by[name], by["d1_3B"], ds)
        if name in by and "rules" in by:
            analyses[f"{name}~rules"] = paired(by[name], by["rules"], ds)

    print(render(rows, ds, analyses))
    print(f"\nv2 group report: {ds.group_report()}")
    print("63% of v2 rows are atomic, so the group split cannot hold out a family "
          "for them and every number above is optimistic by an unknown amount.")

    payload = {"dataset": ds.name, "group_report": ds.group_report(),
               "rows": [r.row() for r in rows],
               "per_band": {n: per_band(r, ds) for n, r in by.items()},
               "per_class": {n: per_class(r, ds) for n, r in by.items()},
               "confusion": {n: confusion(r, ds) for n, r in by.items()},
               "paired": analyses}

    if not args.no_corpus_cv:
        print("\n".join(["", "## corpus-side criteria (train.jsonl only — no v2 in these)", "",
                        "| candidate | family-CV | sd | val t1 | val t2 | val t3 | val pooled |",
                        "|---|---:|---:|---:|---:|---:|---:|"]))
        payload["corpus_scores"] = {}
        for name, cfg in chosen.items():
            s = corpus_scores(cfg)
            payload["corpus_scores"][name] = s
            v = s["val"]
            print(f"| `{name}` | {s['cv']['mean'] * 100:.1f}% | "
                  f"{s['cv']['sd'] * 100:.1f} | {v['t1']['acc'] * 100:.0f}% | "
                  f"{v['t2']['acc'] * 100:.0f}% | {v['t3']['acc'] * 100:.0f}% | "
                  f"{s['val_pooled']['acc'] * 100:.1f}% |", flush=True)

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
