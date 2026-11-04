#!/usr/bin/env python3
"""eval_harness — the measurement instrument for local decision quality.

**The rule this module exists to enforce: a number without a split method is not
a number.** Every `Result` carries its split, its group structure, its n, its
seed count and a confidence interval, because the repo has already published one
figure (93.19%) that reads as generalisation and is really template reuse — the
same protocol is 66.23% with whole phrasing families held out. A bare accuracy
cannot distinguish those two, so this module will not produce one.

Why group-aware, and why repeated
---------------------------------
A random-row split puts paraphrases of one phrasing on both sides of the split,
so the test fold is scored on shapes the model was trained on. That measures
memorisation. Splitting by *family* (the phrasing a row is an instance of) costs
whatever the family-aware model really loses. `DEFAULT_SEEDS` splits are taken
and reported as a distribution rather than a point, because a single fold
assignment is one draw from a wide distribution at these n.

The confidence interval is a **cluster bootstrap**: whole groups are resampled
with replacement, so the interval respects the dependence between rows of one
family. Bootstrapping rows instead would narrow the interval by exactly the
amount the grouping says it should not be narrow.

Deduplication note: the two datasets in this repo resolve their families by
different mechanisms, and the harness reproduces each existing implementation
rather than inventing a third. `tests/test_eval_harness.py` asserts agreement
with `bench/local_clf.py::build_groups` and
`bench/classical_baseline.py::template_key`.

Interface for a later agent
---------------------------
    from eval_harness import Predictor, load_v2, measure, render_table

    class MyPredictor(Predictor):
        name = "mine"
        def fit(self, train):            # train: list[Example]
            ...
        def predict(self, text):          # -> str | None
            ...

    ds = load_v2()                                  # family = declared t3 pairs
    rows = measure(ds, [MyPredictor()], split="group")
    print(render_table(rows))

    train, test = load_v3(split="group")            # 8-way, family split by construction
    print(render_table(score_split(train, test, [MyPredictor()])))

A frozen predictor (a served model, a recorded run: nothing to fit) sets
`frozen = True`; the harness then skips the CV loop and scores it once, still
with a cluster-bootstrap CI. Anything registered through this interface lands in
the same table, under the same split, on the same rows — which is the only way
two signals are comparable.

Splits
------
`split="group"`  — whole families held out (the honest split).
`split="row"`    — stratified per label, groups ignored (the mirage split).
Both are reported side by side; the gap is the quantity of interest.

`load_v3` is the one dataset where the split is not reconstructed but *declared*: its
generator assigned every family to train or test before it wrote an utterance, so
`load_v3(split="group")` returns `(train, test)` and `score_split` fits on one and
scores the other. That is the instrument `utterances_v2.json` cannot be — see
`bench/data/heldout_v3_NOTES.md` for what it can and cannot measure.

Stdlib only: no numpy, no service imports. The predictors live in
`bench/eval_signals.py`, which does import both.
"""
from __future__ import annotations

import json
import math
import random
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_K = 5
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_BOOT = 2000
CI_LEVEL = 0.95

V2_PATH = ROOT / "tests/data/utterances_v2.json"
R382_PATH = ROOT / "tests/data/routing_382.json"
REGISTRY_382_PATH = ROOT / "bench/registry_382.json"
V3_PATH = ROOT / "bench/data/heldout_v3.json"
V3_FAMILIES_PATH = ROOT / "bench/data/heldout_v3_families.json"


# --------------------------------------------------------------------------- #
# data model
# --------------------------------------------------------------------------- #

@dataclass
class Example:
    text: str
    label: str
    group: str
    meta: dict = field(default_factory=dict)


@dataclass
class Dataset:
    name: str
    examples: list[Example]
    labels: tuple[str, ...]
    note: str = ""

    @property
    def n(self) -> int:
        return len(self.examples)

    @property
    def groups(self) -> list[str]:
        return [e.group for e in self.examples]

    def group_report(self) -> dict:
        """How much of the set is actually grouped, and how much is atomic.

        `singleton_fraction` is the number that decides whether a group split
        means anything at all on this dataset: a group that holds one row is a
        row split wearing a family split's clothes.
        """
        sizes: dict[str, int] = {}
        for g in self.groups:
            sizes[g] = sizes.get(g, 0) + 1
        hist: dict[int, int] = {}
        for s in sizes.values():
            hist[s] = hist.get(s, 0) + 1
        singles = hist.get(1, 0)
        return {
            "n": self.n,
            "groups": len(sizes),
            "largest_group": max(sizes.values()) if sizes else 0,
            "singleton_groups": singles,
            "singleton_row_fraction": round(singles / self.n, 4) if self.n else 0.0,
            "size_histogram": {str(k): hist[k] for k in sorted(hist)},
        }


# --------------------------------------------------------------------------- #
# family construction — reproduces the repo's existing implementations
# --------------------------------------------------------------------------- #

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _char_shingles(text: str, n: int = 3) -> set[str]:
    s = " " + _NON_ALNUM.sub(" ", text.lower()).strip() + " "
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def v2_boundary_groups(rows: list[dict]) -> list[str]:
    """Declared v2 families only: t3 rows grouped by (label, tier_target).

    `tier_target` is the dataset's own record of which confusable op a t3 row was
    written against, so `(label, tier_target)` is a family the *dataset*
    declares rather than one this file infers. t1/t2 rows carry no such field and
    get one group each.

    This is the honest ceiling on v2: it holds out both directions of one
    confusable boundary at a time, and leaves 95 of 150 rows atomic.
    """
    out = []
    for r in rows:
        if r.get("band") == "t3" and r.get("tier_target"):
            a, b = sorted((r["label"], r["tier_target"]))
            out.append(f"t3|{a}~{b}")
        else:
            out.append(f"t12|{r['label']}|{len(out)}")
    return out


def v2_surface_groups(rows: list[dict], dup_threshold: float = 0.25) -> list[str]:
    """Declared t3 families + single-link near-duplicate clusters inside t1/t2.

    Reproduces `bench/local_clf.py::build_groups`: char-3gram Jaccard at 0.25,
    one label at a time, union-find. Reported so the reader can see that
    clustering t1/t2 recovers almost nothing — 95 rows resolve to 91 groups —
    which is the evidence that v2's t1/t2 rows are not paraphrases of each other
    and therefore cannot be family-split.
    """
    groups = v2_boundary_groups(rows)
    by_label: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        if r.get("band") != "t3":
            by_label.setdefault(r["label"], []).append(i)

    for label, idxs in by_label.items():
        parent = {i: i for i in idxs}

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        shingles = {i: _char_shingles(rows[i]["utterance"]) for i in idxs}
        for pos, a in enumerate(idxs):
            for b in idxs[pos + 1:]:
                if _jaccard(shingles[a], shingles[b]) >= dup_threshold:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[rb] = ra
        for i in idxs:
            groups[i] = f"t12|{label}|{find(i)}"
    return groups


def r382_template_groups(rows: list[dict], registry: dict) -> list[str]:
    """Reconstruct the generating template for a 382 row.

    Reproduces `bench/classical_baseline.py::template_key`: `gen_data.py` built
    each row by substituting slots into a registry template, so a row still
    carries that template's literal prefix and suffix. This yields the 110
    groups the repo's 66.23% figure was measured on (109 templates + the
    template-less `__none__` rows, which fall into one per-label bucket).
    """
    templates = {t["id"]: t["templates"] for t in registry["tools"]}
    out = []
    for r in rows:
        key = None
        for template in templates.get(r["tool"], []):
            head, _, tail = template.partition("{")
            suffix = tail.split("}")[-1] if "}" in tail else ""
            if r["text"].startswith(head) and (not suffix or r["text"].endswith(suffix)):
                key = f"{r['tool']}::{template}"
                break
        out.append(key or f"{r['tool']}::unmatched")
    return out


def r382_topic_groups(rows: list[dict]) -> list[str]:
    """The `family` column as shipped: 7 topic buckets, not phrasing families.

    Coarser than the template split and therefore a weaker test — holding out
    `timers` removes every timer phrasing at once. Included because it is the
    only family column either dataset actually carries.
    """
    return [r["family"] for r in rows]


def _row_groups(n: int) -> list[str]:
    return [f"row|{i}" for i in range(n)]


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #

def load_v2(path: Path | str = V2_PATH, grouping: str = "boundary",
            name: str | None = None) -> Dataset:
    """150 rows, 8-way, `utterances_v2.json`.

    `grouping`: `"boundary"` (default, declared t3 pairs), `"surface"`
    (boundary + char-3gram clustering), `"row"` (every row its own group).
    """
    rows = json.loads(Path(path).read_text())
    if grouping == "boundary":
        groups = v2_boundary_groups(rows)
    elif grouping == "surface":
        groups = v2_surface_groups(rows)
    elif grouping == "row":
        groups = _row_groups(len(rows))
    else:
        raise ValueError(f"unknown v2 grouping {grouping!r}")
    examples = [
        Example(text=r["utterance"], label=r["label"], group=g,
                meta={"band": r.get("band"), "tier_target": r.get("tier_target")})
        for r, g in zip(rows, groups)
    ]
    labels = tuple(sorted({r["label"] for r in rows}))
    return Dataset(name=name or f"v2 (group={grouping})", examples=examples,
                   labels=labels,
                   note="8-way hand-authored, t1/t2/t3 bands; no family field")


def load_v3(path: Path | str = V3_PATH, split: str | None = "group",
            name: str | None = None) -> Dataset | tuple[Dataset, Dataset]:
    """The generated 8-way set, whose families the *generator* fixed before writing it.

    Unlike `load_v2`, this set carries a `family` column on every row and a `split`
    column that was assigned from the frame inventory before any utterance existed
    (`bench/gen_heldout_v3.py`). There is nothing to reconstruct and nothing to
    cluster: the family is the authored frame, the split is the generator's decision,
    and this function's job is to return both halves and refuse a crossing.

    `split="group"` returns `(train, test)` as declared, whole families per side.
    `split="row"`  returns the same row counts repartitioned per label with families
    ignored — the mirage side of the comparison, on the identical data.
    `split=None`    returns the single unsplit `Dataset`, for scoring a frozen model.
    """
    rows = json.loads(Path(path).read_text())
    examples = [Example(text=r["utterance"], label=r["label"], group=r["family"],
                        meta={"band": r.get("band"), "tier_target": r.get("tier_target"),
                              "split": r.get("split"), "id": r.get("id")})
                for r in rows]
    labels = tuple(sorted({r["label"] for r in rows}))
    base = name or "v3 (generated, family-split)"
    ds = Dataset(name=base, examples=examples, labels=labels,
                 note="8-way generated from authored frames; family and split declared "
                      "by the generator, not inferred here")
    if split is None:
        return ds
    if split == "group":
        train = Dataset(name=base + " train", examples=[], labels=labels)
        test = Dataset(name=base + " test", examples=[], labels=labels)
        for ex in examples:
            side = ex.meta.get("split")
            if side not in ("train", "test"):
                raise ValueError(f"row {ex.meta.get('id')} carries no train/test split")
            (train if side == "train" else test).examples.append(ex)
        assert_no_family_crossing(train, test)
        return train, test
    if split == "row":
        return _repartition_by_row(examples, labels, base)
    raise ValueError(f"unknown v3 split {split!r} (expected 'group', 'row' or None)")


def _repartition_by_row(examples: list[Example], labels: tuple[str, ...], base: str,
                        seed: int = 0) -> tuple[Dataset, Dataset]:
    """The declared split's sizes, refilled row-wise — families deliberately crossed."""
    rng = random.Random(seed)
    declared_test: dict[str, int] = {}
    for ex in examples:
        if ex.meta.get("split") == "test":
            declared_test[ex.label] = declared_test.get(ex.label, 0) + 1
    train: list[Example] = []
    test: list[Example] = []
    for lab in labels:
        pool = [ex for ex in examples if ex.label == lab]
        rng.shuffle(pool)
        cut = declared_test.get(lab, 0)
        test += pool[:cut]
        train += pool[cut:]
    order = {ex.meta.get("id"): i for i, ex in enumerate(examples)}
    for side in (train, test):
        side.sort(key=lambda e: order.get(e.meta.get("id"), 0))
    return (Dataset(name=base + " train (row split)", examples=train, labels=labels),
            Dataset(name=base + " test (row split)", examples=test, labels=labels))


def assert_no_family_crossing(train: Dataset, test: Dataset) -> None:
    """The guarantee the whole instrument rests on. Raises with the offending families."""
    sides: dict[str, set[str]] = {}
    for group in train.groups:
        sides.setdefault(group, set()).add("train")
    for group in test.groups:
        sides.setdefault(group, set()).add("test")
    crossing = sorted(g for g, s in sides.items() if len(s) > 1)
    if crossing:
        raise ValueError(f"families on both sides of the split: {crossing}")


def score_split(train: Dataset, test: Dataset, predictors: list[Predictor], *,
                n_boot: int = DEFAULT_BOOT, boot_seed: int = 0) -> list[Result]:
    """Fit each predictor on `train`, score it on the held-out families in `test`.

    One draw, not a seed distribution: the split is declared by the data, so there is
    no fold assignment to average over. The interval is still a cluster bootstrap over
    the test families, because rows of one family are one observation.
    """
    import copy

    results = []
    for predictor in predictors:
        p = copy.deepcopy(predictor)
        p.fit(train.examples)
        outcomes = [1.0 if p.predict(e.text) == e.label else 0.0 for e in test.examples]
        acc = sum(outcomes) / len(outcomes) if outcomes else 0.0
        lo, hi = cluster_bootstrap_ci(outcomes, test.groups, n_boot=n_boot, seed=boot_seed)
        results.append(Result(
            predictor=p.name, split="group", n=test.n, n_groups=len(set(test.groups)),
            k=1, seeds=1, acc_mean=round(acc, 4), acc_sd=0.0, acc_min=round(acc, 4),
            acc_max=round(acc, 4), ci_lo=lo, ci_hi=hi, per_seed=[round(acc, 4)],
            dataset=test.name,
            atomic_row_fraction=test.group_report()["singleton_row_fraction"],
            note="fitted on the train families, scored on the test families",
            outcomes=outcomes, preds_per_seed=[[]], folds_per_seed=[[-1] * test.n]))
    return results


def load_382(path: Path | str = R382_PATH, grouping: str = "template",
             registry_path: Path | str = REGISTRY_382_PATH,
             name: str | None = None) -> Dataset:
    """382 rows, 18-way, `routing_382.json`.

    `grouping`: `"template"` (default, reconstructed generating template),
    `"topic"` (the shipped `family` column), `"row"`.
    """
    rows = json.loads(Path(path).read_text())
    if grouping == "template":
        groups = r382_template_groups(rows, json.loads(Path(registry_path).read_text()))
    elif grouping == "topic":
        groups = r382_topic_groups(rows)
    elif grouping == "row":
        groups = _row_groups(len(rows))
    else:
        raise ValueError(f"unknown 382 grouping {grouping!r}")
    examples = [Example(text=r["text"], label=r["tool"], group=g,
                        meta={"family": r.get("family")})
                for r, g in zip(rows, groups)]
    labels = tuple(sorted({r["tool"] for r in rows}))
    return Dataset(name=name or f"382 (group={grouping})", examples=examples,
                   labels=labels,
                   note="18-way template-generated home-assistant commands")


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #

def group_folds(groups: list[str], labels: list[str], k: int, seed: int = 0) -> list[int]:
    """Place whole groups into k folds, stratified by label, deterministically.

    Groups are ordered largest-first (so a big boundary family cannot land
    entirely in one fold) with a seeded shuffle breaking ties — that tie-break is
    what makes two seeds two genuinely different splits. Each placement then
    minimises (a) how much of this group's labels the fold already holds, then
    (b) how full the fold is.
    """
    members: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        members.setdefault(g, []).append(i)
    uniq = sorted(members)
    rng = random.Random(seed)
    rng.shuffle(uniq)
    rank = {g: i for i, g in enumerate(uniq)}
    order = sorted(members, key=lambda g: (-len(members[g]), rank[g]))
    k = max(1, min(k, len(order)))

    fold_label: list[dict[str, int]] = [dict() for _ in range(k)]
    fold_size = [0] * k
    assign = [0] * len(groups)
    for g in order:
        idxs = members[g]
        counts: dict[str, int] = {}
        for i in idxs:
            counts[labels[i]] = counts.get(labels[i], 0) + 1
        best_f, best_key = 0, None
        for f in range(k):
            held = sum(fold_label[f].get(lab, 0) for lab in counts)
            key = (held, fold_size[f], f)
            if best_key is None or key < best_key:
                best_f, best_key = f, key
        for i in idxs:
            assign[i] = best_f
        fold_size[best_f] += len(idxs)
        for lab, c in counts.items():
            fold_label[best_f][lab] = fold_label[best_f].get(lab, 0) + c
    return assign


def row_folds(labels: list[str], k: int, seed: int = 0) -> list[int]:
    """Stratified per label, groups ignored. The split that inflates scores."""
    rng = random.Random(seed)
    assign = [0] * len(labels)
    per_label: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        per_label.setdefault(lab, []).append(i)
    for _, idxs in per_label.items():
        shuffled = idxs[:]
        rng.shuffle(shuffled)
        for j, i in enumerate(shuffled):
            assign[i] = j % k
    return assign


def folds_for(ds: Dataset, split: str, k: int, seed: int) -> list[int]:
    if split == "group":
        return group_folds(ds.groups, [e.label for e in ds.examples], k, seed)
    if split == "row":
        return row_folds([e.label for e in ds.examples], k, seed)
    raise ValueError(f"unknown split {split!r} (expected 'group' or 'row')")


# --------------------------------------------------------------------------- #
# predictor interface
# --------------------------------------------------------------------------- #

class Predictor:
    """Uniform interface. Subclass, set `name`, implement `fit`/`predict`.

    `predict` returns a label or None (abstain). An abstain counts as wrong: a
    signal that declines to answer has still failed to route the row.

    `frozen = True` for a signal with nothing to fit (a served model, a recorded
    run). The harness then skips the retraining loop — resplitting would only
    re-score the same fixed predictions — and scores it once, still with a
    cluster-bootstrap CI over groups.
    """

    name: str = "unnamed"
    frozen: bool = False

    def fit(self, train: list[Example]) -> None:
        raise NotImplementedError

    def predict(self, text: str) -> str | None:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# confidence intervals
# --------------------------------------------------------------------------- #

def cluster_bootstrap_ci(outcomes: list[float], groups: list[str],
                         n_boot: int = DEFAULT_BOOT, level: float = CI_LEVEL,
                         seed: int = 0) -> tuple[float, float]:
    """Percentile cluster bootstrap: resample whole groups with replacement.

    `outcomes[i]` is row i's correctness (1.0/0.0, or its mean across seeds).
    Resampling groups rather than rows is the whole point — rows of one family
    are not independent draws, and a row bootstrap would return an interval that
    pretends they are.
    """
    by_group: dict[str, list[float]] = {}
    for o, g in zip(outcomes, groups):
        by_group.setdefault(g, []).append(o)
    keys = sorted(by_group)
    if not keys:
        return (0.0, 1.0)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        pick = [keys[rng.randrange(len(keys))] for _ in keys]
        total = hits = 0
        for g in pick:
            vals = by_group[g]
            hits += sum(vals)
            total += len(vals)
        means.append(hits / total if total else 0.0)
    means.sort()
    lo = means[int(math.floor((1 - level) / 2 * (len(means) - 1)))]
    hi = means[int(math.ceil((1 + level) / 2 * (len(means) - 1)))]
    return (round(lo, 4), round(hi, 4))


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #

@dataclass
class Result:
    predictor: str
    split: str
    n: int
    n_groups: int
    k: int
    seeds: int
    acc_mean: float
    acc_sd: float
    acc_min: float
    acc_max: float
    ci_lo: float
    ci_hi: float
    per_seed: list[float] = field(default_factory=list)
    dataset: str = ""
    atomic_row_fraction: float = 0.0
    note: str = ""
    # kept off the printed table but exported, so callers can pair two signals
    # row by row (union ceilings, per-class expert selection, McNemar)
    outcomes: list[float] = field(default_factory=list)
    preds_per_seed: list[list] = field(default_factory=list)
    folds_per_seed: list[list[int]] = field(default_factory=list)

    def row(self) -> dict:
        return {
            "predictor": self.predictor, "split": self.split, "n": self.n,
            "groups": self.n_groups, "k": self.k, "seeds": self.seeds,
            "acc_mean": round(self.acc_mean, 4), "acc_sd": round(self.acc_sd, 4),
            "ci95": [self.ci_lo, self.ci_hi], "per_seed": [round(s, 4) for s in self.per_seed],
            "dataset": self.dataset, "note": self.note,
        }


def _measure_frozen(ds: Dataset, predictor: Predictor) -> tuple[list[float], list[list]]:
    predictor.fit(ds.examples)
    preds = [predictor.predict(e.text) for e in ds.examples]
    outcomes = [1.0 if p == e.label else 0.0 for p, e in zip(preds, ds.examples)]
    return outcomes, [preds]


def _measure_cv(ds: Dataset, predictor: Predictor, split: str, k: int,
                seeds: tuple[int, ...]) -> tuple[list[float], list[float], list[list], list[list[int]]]:
    """Returns (seed-mean per-row outcome, per-seed accuracies, per-seed
    predictions, per-seed fold assignment)."""
    import copy

    n = ds.n
    texts = [e.text for e in ds.examples]
    labels = [e.label for e in ds.examples]
    accumulated = [0.0] * n
    per_seed: list[float] = []
    preds_per_seed: list[list] = []
    folds_per_seed: list[list[int]] = []

    for seed in seeds:
        fold_of = folds_for(ds, split, k, seed)
        correct = [0.0] * n
        preds: list = [None] * n
        for f in range(k):
            train = [e for i, e in enumerate(ds.examples) if fold_of[i] != f]
            p = copy.deepcopy(predictor)
            p.fit(train)
            for i in range(n):
                if fold_of[i] == f:
                    preds[i] = p.predict(texts[i])
                    correct[i] = 1.0 if preds[i] == labels[i] else 0.0
        per_seed.append(round(sum(correct) / n, 4))
        preds_per_seed.append(preds)
        folds_per_seed.append(fold_of)
        for i in range(n):
            accumulated[i] += correct[i]

    return [a / len(seeds) for a in accumulated], per_seed, preds_per_seed, folds_per_seed


def measure(ds: Dataset, predictors: list[Predictor], *, split: str = "group",
            k: int = DEFAULT_K, seeds: tuple[int, ...] = DEFAULT_SEEDS,
            n_boot: int = DEFAULT_BOOT, boot_seed: int = 0) -> list[Result]:
    """Score every predictor on one dataset under one split.

    `split` is required in spirit: it has no default that hides the choice, and
    every Result records it. Call `measure` twice — once with `"group"`, once
    with `"row"` — to see the gap the split makes.
    """
    if split not in ("group", "row"):
        raise ValueError(f"unknown split {split!r} (expected 'group' or 'row')")
    groups = ds.groups
    rep = ds.group_report()
    results = []

    for predictor in predictors:
        folds_per_seed: list[list[int]] = []
        if predictor.frozen:
            outcomes, preds_per_seed = _measure_frozen(ds, predictor)
            per_seed = [round(sum(outcomes) / ds.n, 4)]
            folds_per_seed = [[-1] * ds.n]
        else:
            outcomes, per_seed, preds_per_seed, folds_per_seed = _measure_cv(
                ds, predictor, split, k, seeds)
        acc = sum(outcomes) / ds.n
        lo, hi = cluster_bootstrap_ci(outcomes, groups, n_boot=n_boot, seed=boot_seed)
        sd = statistics.pstdev(per_seed) if len(per_seed) > 1 else 0.0
        results.append(Result(
            predictor=predictor.name, split=split, n=ds.n,
            n_groups=rep["groups"], k=1 if predictor.frozen else k,
            seeds=len(per_seed), acc_mean=round(acc, 4), acc_sd=round(sd, 4),
            acc_min=round(min(per_seed), 4), acc_max=round(max(per_seed), 4),
            ci_lo=lo, ci_hi=hi, per_seed=per_seed, dataset=ds.name,
            atomic_row_fraction=rep["singleton_row_fraction"],
            note="frozen: no refit, split does not change predictions"
                 if predictor.frozen else "",
            outcomes=outcomes, preds_per_seed=preds_per_seed,
            folds_per_seed=folds_per_seed,
        ))
    return results


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def render_table(results: list[Result], title: str = "") -> str:
    head = ("| signal | split | n | groups | atomic rows | acc | 95% CI | "
            "seeds (min–max) |")
    rule = "|---|---|---:|---:|---:|---:|---|---|"
    lines = [f"## {title}" if title else "", head, rule]
    for r in results:
        ci = f"{r.ci_lo * 100:.1f}–{r.ci_hi * 100:.1f}%"
        spread = (f"{r.acc_min * 100:.1f}–{r.acc_max * 100:.1f}%"
                  if r.seeds > 1 else "—")
        atomic = f"{r.atomic_row_fraction * 100:.0f}%"
        lines.append(f"| `{r.predictor}` | {r.split} | {r.n} | {r.n_groups} | "
                     f"{atomic} | **{r.acc_mean * 100:.1f}%** | {ci} | {spread} |")
    return "\n".join(line for line in lines if line != "")


def dump_json(results: list[Result], path: Path | str) -> None:
    Path(path).write_text(json.dumps([r.row() | {"dataset": r.dataset}
                                      for r in results], indent=2) + "\n")
