#!/usr/bin/env python3
"""eval_signals — every local signal this repo publishes, re-measured through
`eval_harness` under an explicit split.

Nothing here is tuned. Each signal is the one the repo already ships, wired to
the harness's `Predictor` interface at the configuration the repo uses, so the
numbers are comparable with the recorded ones and with each other.

| name | what it is | source |
|---|---|---|
| `rules` | keyword cue table, weighted match | `local_clf.RULES` |
| `lr` | tf-idf word+char features, softmax regression | `local_clf.SoftmaxRegression` |
| `1nn` | char-ngram cosine 1-nearest-neighbour | `local_clf.NearestNeighbour` |
| `d1_3B` | d1-3B-Q4_K_M via `/v1/systemone`, recorded per-row runs | `bench/results/dell/` |
| `classical_svm` | tf-idf(word 1-2) + linear SVM | `ti.decide.backends.classical` |
| `classical_refs` | the same model fitted on the 200 registry references | recorded protocol |

`rules` is `frozen`: its table is fixed, nothing is fitted, so a re-split cannot
change its predictions. It is therefore the one published signal whose accuracy
is *not* inflated by row-wise CV — and the only one that survives the
family-held-out split unchanged. Every other signal here fits on the training
fold, so the split decides its number.

Usage:
    python3 bench/eval_signals.py                      # every table, ~5 seeds
    python3 bench/eval_signals.py --json out.json --only v2
    python3 bench/eval_signals.py --d1-source live      # probe the shared server
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT / "service"), str(ROOT / "bench")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import eval_harness as H  # noqa: E402
import local_clf as lc  # noqa: E402
from live_probe import INSTRUCTIONS, OP_DESCRIPTIONS  # noqa: E402
from ti.conf import load  # noqa: E402
from ti.decide.backends.classical import ClassicalBackend, load_corpus  # noqa: E402

D1_DIR = ROOT / "bench/results/dell"
# Fixed a priori, mid-range of local_clf.GRID. No tuning: this phase measures.
LR_MIN_DF = 1
LR_L2 = 1e-3
KNN_MIN_DF = 2
# the label `classical` emits as `choice=None` (its learned abstain class)
NONE = "__none__"


class Rules(H.Predictor):
    name = "rules"
    frozen = True          # a cue table with no fitting cannot be split-sensitive

    def fit(self, train):
        pass

    def predict(self, text):
        return lc.rules_predict(text)[0]


class LinearSoftmax(H.Predictor):
    name = "lr"

    def __init__(self) -> None:
        self.feat = None
        self.model = None

    def fit(self, train):
        texts = [e.text for e in train]
        self.feat = lc.Featurizer(min_df=LR_MIN_DF).fit(texts)
        X = self.feat.transform(texts)
        self.model = lc.SoftmaxRegression(l2=LR_L2).fit(
            X, [lc.LABEL_INDEX[e.label] for e in train])

    def predict(self, text):
        p = self.model.predict_proba(self.feat.transform([text]))[0]
        return lc.LABELS[int(np.argmax(p))]


class CharOneNN(H.Predictor):
    name = "1nn"

    def __init__(self) -> None:
        self.model = None

    def fit(self, train):
        self.model = lc.NearestNeighbour(min_df=KNN_MIN_DF).fit(
            [e.text for e in train], [e.label for e in train])

    def predict(self, text):
        p = self.model.predict_proba([text])[0]
        return lc.LABELS[int(np.argmax(p))]


class D1Recorded(H.Predictor):
    """d1-3B from a recorded per-row run. Frozen: the model cannot be refitted,
    so its predictions are identical under every split and the only thing a split
    can change is the interval around its number."""

    frozen = True

    def __init__(self, path: Path, name: str) -> None:
        self.path = path
        self.name = name
        self._choice: dict[str, str | None] = {}

    def fit(self, train):
        """Align the recording to the rows by order, refusing a mismatch.

        The recorded files carry `per_row` in dataset order; pairing them to the
        wrong rows would produce a plausible-looking accuracy for the wrong
        model, so this asserts both the length and every utterance.
        """
        rows = json.loads(self.path.read_text())["per_row"]
        if len(rows) != len(train):
            raise SystemExit(f"{self.path.name}: {len(rows)} rows vs dataset {len(train)}")
        for i, (row, ex) in enumerate(zip(rows, train)):
            if row["utterance"] != ex.text:
                raise SystemExit(
                    f"{self.path.name}: row {i} is {row['utterance']!r}, dataset has "
                    f"{ex.text!r} — recording is not aligned with this dataset")
        self._choice = {row["utterance"]: row.get("choice") for row in rows}

    def predict(self, text):
        return self._choice.get(text)


class ClassicalSVM(H.Predictor):
    """The `classical` backend fitted per training fold.

    `abstain` is the label the backend encodes as `choice=None` (its learned
    `__none__` class). Without it, every *correct* abstention would be scored as
    a miss — which is worth exactly 15 of the 356 rows the 382 protocol gets
    right, so it is not a rounding detail.
    """

    name = "classical_svm"

    def __init__(self, criteria: dict[str, str], abstain: str | None = None) -> None:
        self.criteria = criteria
        self.abstain = abstain
        self.backend = None
        self.cfg = dict(load()["decide"]["classical"])

    def fit(self, train):
        self.backend = ClassicalBackend(
            config=dict(self.cfg, corpus_path="(eval-harness)"),
            corpus=[(e.text, e.label) for e in train])

    def predict(self, text):
        choice = self.backend.choice(text, INSTRUCTIONS, self.criteria).choice
        return self.abstain if choice is None else choice


class ClassicalReferences(H.Predictor):
    """The recorded 93.19% protocol: fit on the registry's 200 references, score
    the frozen set. No holdout within the scored set, so a re-split cannot move
    it; the interval says how much of it is the set's own sampling noise."""

    frozen = True

    def __init__(self, name: str, registry_path: Path, criteria,
                 abstain: str | None = None) -> None:
        self.name = name
        self.corpus = load_corpus(registry_path)
        self.criteria = criteria
        self.abstain = abstain

    def fit(self, train):
        cfg = dict(load()["decide"]["classical"])
        self.backend = ClassicalBackend(config=cfg, corpus=self.corpus)

    def predict(self, text):
        choice = self.backend.choice(text, INSTRUCTIONS, self.criteria).choice
        return self.abstain if choice is None else choice


# --------------------------------------------------------------------------- #
# paired diagnostics over two already-measured signals
# --------------------------------------------------------------------------- #

def union_ceiling(a: H.Result, b: H.Result, ds: H.Dataset) -> H.Result:
    """Rows where *either* signal is right — the ceiling for any rule that picks
    between them row by row, whatever the rule is. No per-class map, threshold
    or selector can beat this, which makes it the honest cap on the repo's
    `nested_selection_acc`."""
    merged = [max(x, y) for x, y in zip(a.outcomes, b.outcomes)]
    acc = sum(merged) / len(merged)
    lo, hi = H.cluster_bootstrap_ci(merged, ds.groups)
    return H.Result(
        predictor=f"oracle union({a.predictor}|{b.predictor})", split=a.split,
        n=ds.n, n_groups=a.n_groups, k=a.k, seeds=a.seeds, acc_mean=round(acc, 4),
        acc_sd=0.0, acc_min=round(acc, 4), acc_max=round(acc, 4), ci_lo=lo, ci_hi=hi,
        per_seed=[round(acc, 4)], dataset=ds.name,
        atomic_row_fraction=a.atomic_row_fraction,
        note="upper bound: picks the right signal per row, not a deployable rule")


def band_breakdown(res: H.Result, ds: H.Dataset) -> dict:
    out: dict[str, list[float]] = {}
    for o, e in zip(res.outcomes, ds.examples):
        out.setdefault(e.meta.get("band") or "?", []).append(o)
    return {k: {"n": len(v), "acc": round(sum(v) / len(v), 4)}
            for k, v in sorted(out.items())}


def mcnemar_exact(a: H.Result, b: H.Result) -> dict:
    """Paired sign test on the seed-mean outcomes: does a differ from b?"""
    win_a = sum(1 for x, y in zip(a.outcomes, b.outcomes) if x > y)
    win_b = sum(1 for x, y in zip(a.outcomes, b.outcomes) if y > x)
    n = win_a + win_b
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p": 1.0}
    # exact two-sided binomial, p=0.5
    from math import comb
    k = min(win_a, win_b)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {"a_only": win_a, "b_only": win_b, "p": round(min(1.0, 2 * tail), 4)}


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def gap_block(group_rows: list[H.Result], row_rows: list[H.Result],
              label: str) -> str:
    """Side-by-side: the same signal under a group split and a row split."""
    g_by = {r.predictor: r for r in group_rows}
    r_by = {r.predictor: r for r in row_rows}
    lines = [f"| signal | row-wise | {label} | gap |", "|---|---:|---:|---:|"]
    for name in g_by:
        g, rw = g_by.get(name), r_by.get(name)
        if not g or not rw:
            continue
        lines.append(f"| `{name}` | {rw.acc_mean * 100:.1f}% | "
                     f"{g.acc_mean * 100:.1f}% | "
                     f"{(rw.acc_mean - g.acc_mean) * 100:+.1f} pts |")
    return "\n".join(lines)


def ci_inflation(rows: list[H.Result]) -> str:
    """The published intervals are Wilson, which assumes rows are independent
    draws. Under grouping they are not, so the cluster bootstrap is wider. This
    table is the size of that correction on the same outcomes."""
    lines = ["| signal | split | acc | Wilson 95% (published form) | "
             "cluster-bootstrap 95% | width ratio |",
             "|---|---|---:|---|---|---:|"]
    for r in rows:
        k = round(r.acc_mean * r.n)
        lo, hi = lc.wilson(k, r.n)
        w_w = hi - lo
        w_c = r.ci_hi - r.ci_lo
        lines.append(
            f"| `{r.predictor}` | {r.split} | {r.acc_mean * 100:.1f}% | "
            f"[{lo * 100:.1f}, {hi * 100:.1f}] | "
            f"[{r.ci_lo * 100:.1f}, {r.ci_hi * 100:.1f}] | "
            f"{w_c / w_w:.2f}× |" if w_w > 0 else "")
    return "\n".join(line for line in lines if line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="", help="write every Result to this path")
    ap.add_argument("--only", default="", choices=["", "v2", "382"])
    ap.add_argument("--seeds", type=int, default=len(H.DEFAULT_SEEDS))
    ap.add_argument("--k", type=int, default=H.DEFAULT_K)
    ap.add_argument("--boot", type=int, default=H.DEFAULT_BOOT)
    args = ap.parse_args()
    seeds = tuple(range(args.seeds))
    out: list[H.Result] = []
    blocks: dict[str, list[H.Result]] = {}
    payload: dict = {}

    def run_block(ds: H.Dataset, preds: list[H.Predictor], key: str,
                  split: str) -> list[H.Result]:
        rows = H.measure(ds, preds, split=split, k=args.k, seeds=seeds,
                         n_boot=args.boot)
        out.extend(rows)
        blocks[key] = rows
        payload[key] = {"group_report": ds.group_report(),
                        "rows": [r.row() for r in rows]}
        print(H.render_table(rows, f"{ds.name} — split={split}"))
        print()
        return rows

    # ---------------- 150 rows, 8-way ----------------
    if args.only in ("", "v2"):
        d1_runs = sorted(D1_DIR.glob("v2-d1-3B-Q4-systemone-props-asserted*.json"))
        print(f"# d1 recorded runs: {[p.name for p in d1_runs]}\n")
        for grouping, split in (("boundary", "group"), ("row", "row"),
                                ("surface", "group")):
            ds = H.load_v2(grouping=grouping)
            preds: list[H.Predictor] = [Rules(), LinearSoftmax(), CharOneNN()]
            for p in d1_runs:
                tag = p.stem.split("props-asserted")[-1].lstrip("-") or "run1"
                preds.append(D1Recorded(p, f"d1_3B[{tag}]"))
            preds.append(ClassicalSVM(dict(OP_DESCRIPTIONS)))
            rows = run_block(ds, preds, f"v2/{grouping}/{split}", split)
            for r in rows:
                if r.predictor.startswith("d1_3B["):
                    payload.setdefault("v2_bands", {})[
                        f"{r.predictor}/{r.split}"] = band_breakdown(r, ds)

        grp = blocks["v2/boundary/group"]
        print("## The gap the split makes (150 rows, declared t3 families only)\n")
        print(gap_block(grp, blocks["v2/row/row"], "family-held-out"))
        print()
        print("## Same, with t1/t2 near-duplicate clusters added to the grouping\n")
        print(gap_block(blocks["v2/surface/group"], blocks["v2/row/row"],
                        "grouped (surface)"))
        print()

        print("## Are the intervals honest? Wilson (rows independent) vs "
              "cluster bootstrap (groups independent)\n")
        print(ci_inflation(grp))
        print()

        # oracle ceiling + paired test for the two signals behind `nested_selection_acc`
        g_by = {r.predictor: r for r in grp}
        r_by = {r.predictor: r for r in blocks["v2/row/row"]}
        d1_name = next(n for n in g_by if n.startswith("d1_3B["))
        for split, by, grouping in (("group", g_by, "boundary"),
                                    ("row", r_by, "row")):
            ds = H.load_v2(grouping=grouping)
            u = union_ceiling(by["lr"], by[d1_name], ds)
            m = mcnemar_exact(by["lr"], by[d1_name])
            print(f"- {split:5s}: oracle union(lr|d1) = {u.acc_mean * 100:.1f}% "
                  f"[{u.ci_lo * 100:.1f}–{u.ci_hi * 100:.1f}%] — the cap on any "
                  f"per-row selector. published `nested_selection_acc` 84.7%")
            print(f"          lr vs d1 paired: lr-only {m['a_only']}, "
                  f"d1-only {m['b_only']}, exact two-sided p={m['p']}")
            payload[f"v2_union/{split}"] = u.row()
        print()

    # ---------------- 382 rows, 18-way ----------------
    if args.only in ("", "382"):
        criteria = None
        for grouping, split in (("template", "group"), ("row", "row"),
                                ("topic", "group")):
            ds = H.load_382(grouping=grouping)
            if criteria is None:
                criteria = {lab: lab for lab in ds.labels}
            preds: list[H.Predictor] = [
                ClassicalSVM(criteria, abstain=NONE),
                ClassicalReferences("classical_refs(93.19% protocol)",
                                    ROOT / "bench/registry_382.json", criteria,
                                    abstain=NONE)]
            run_block(ds, preds, f"382/{grouping}/{split}", split)

        print("## The gap the split makes (382 rows, reconstructed templates)\n")
        print(gap_block(blocks["382/template/group"], blocks["382/row/row"],
                        "template-held-out"))
        print()
        print("## Same, holding out the shipped 7 topic families instead\n")
        print(gap_block(blocks["382/topic/group"], blocks["382/row/row"],
                        "topic-held-out"))
        print()
        print("## Are the intervals honest? (382, template grouping)\n")
        print(ci_inflation(blocks["382/template/group"]))
        print()

    if args.json:
        H.dump_json(out, args.json)
        print(f"wrote {args.json}")
    Path("/tmp/eval_payload.json").write_text(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
