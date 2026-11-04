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

Training-corpus overlap
-----------------------
`bench/data/heldout_v3.json` shares its authored frame inventory with
`data/distill/`, so a model fitted on that corpus has already seen the skeletons of
most of the rows here. A predictor declares what it was fitted on:

    class Mine(Predictor):
        name = "mine"
        trained_on = ("distill",)          # -> data/distill/train.jsonl + val.jsonl

and every public route to an accuracy (`measure`, `score_split`) then joins the
scored rows against that corpus's `family` column and drops what overlaps, recording
how many rows and families it dropped in `Result.overlap` and in `render_table`'s
output. There is no mode that includes an overlapping row silently: `drop` reports,
`raise` refuses. `scoreable()` is the query — "which of these rows may this predictor
legitimately be scored on" — and it is what both entry points call.

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
from functools import lru_cache
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
DISTILL_DIR = ROOT / "data/distill"


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
    # `family_kind = "frame"` means `Example.group` is a training corpus's own
    # `family` value, so it is a join key rather than a name invented here. Only
    # a dataset that declares it can be checked for overlap; see `scoreable`.
    family_kind: str = ""

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
                      "by the generator, not inferred here",
                 family_kind="frame")
    if split is None:
        return ds
    if split == "group":
        train = Dataset(name=base + " train", examples=[], labels=labels,
                        family_kind="frame")
        test = Dataset(name=base + " test", examples=[], labels=labels,
                       family_kind="frame")
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
    return (Dataset(name=base + " train (row split)", examples=train, labels=labels,
                    family_kind="frame"),
            Dataset(name=base + " test (row split)", examples=test, labels=labels,
                    family_kind="frame"))


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
                n_boot: int = DEFAULT_BOOT, boot_seed: int = 0,
                on_contamination: str = "drop") -> list[Result]:
    """Fit each predictor on `train`, score it on the held-out families in `test`.

    One draw, not a seed distribution: the split is declared by the data, so there is
    no fold assignment to average over. The interval is still a cluster bootstrap over
    the test families, because rows of one family are one observation.

    The scored side goes through `scoreable` first, so a predictor that declares a
    training corpus is never scored on a test row that corpus contains. The training
    side is not filtered: those rows are legitimately training data either way.
    """
    import copy

    results = []
    for predictor in predictors:
        scored, overlap = scoreable(test, predictor,
                                    on_contamination=on_contamination)
        p = copy.deepcopy(predictor)
        p.fit(train.examples)
        outcomes = [1.0 if p.predict(e.text) == e.label else 0.0 for e in scored.examples]
        acc = sum(outcomes) / len(outcomes) if outcomes else 0.0
        lo, hi = cluster_bootstrap_ci(outcomes, scored.groups, n_boot=n_boot, seed=boot_seed)
        note = "fitted on the train families, scored on the test families"
        if overlap is not None and overlap.excluded_rows:
            note += f"; {overlap.note()}"
        results.append(Result(
            predictor=p.name, split="group", n=scored.n,
            n_groups=len(set(scored.groups)),
            k=1, seeds=1, acc_mean=round(acc, 4), acc_sd=0.0, acc_min=round(acc, 4),
            acc_max=round(acc, 4), ci_lo=lo, ci_hi=hi, per_seed=[round(acc, 4)],
            dataset=test.name,
            atomic_row_fraction=scored.group_report()["singleton_row_fraction"],
            note=note,
            outcomes=outcomes, preds_per_seed=[[]], folds_per_seed=[[-1] * scored.n],
            overlap=overlap))
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
    # The committed corpora this signal was fitted on, as keys of TRAINING_CORPORA.
    # Empty means "declares nothing", and the harness has nothing to check — every
    # predictor in `bench/eval_signals.py` is fitted per fold on the dataset it is
    # scored on, so none of them declares a corpus. A predictor fitted on
    # `data/distill/` says so here and is scored on the rows it has not seen.
    trained_on: tuple[str, ...] = ()

    def fit(self, train: list[Example]) -> None:
        raise NotImplementedError

    def predict(self, text: str) -> str | None:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# training-corpus overlap — the caveat, as a guard
# --------------------------------------------------------------------------- #

TRAINING_CORPORA: dict[str, tuple[str, ...]] = {
    "distill": ("train.jsonl", "val.jsonl"),
    "distill:train": ("train.jsonl",),
    "distill:val": ("val.jsonl",),
}


class OverlapError(RuntimeError):
    """A predictor would be scored on rows its own training corpus is made of."""


def corpus_files(*declared: str) -> tuple[str, ...]:
    """Resolve `trained_on` keys to the corpus files they name."""
    out: list[str] = []
    for key in declared:
        try:
            files = TRAINING_CORPORA[key]
        except KeyError:
            raise ValueError(f"unknown training corpus {key!r} "
                             f"(known: {sorted(TRAINING_CORPORA)})") from None
        for name in files:
            if name not in out:
                out.append(name)
    return tuple(out)


@lru_cache(maxsize=None)
def corpus_families(files: tuple[str, ...]) -> frozenset[str]:
    """Every `family` value the named corpus files carry.

    The distill files and the v3 generator write the family id with the same grammar
    (`tools/distill/templates.py::_row`), which is what makes this a mechanical join
    rather than a similarity guess.
    """
    families: set[str] = set()
    for name in files:
        path = DISTILL_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"training corpus file missing: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                families.add(json.loads(line)["family"])
    return frozenset(families)


@dataclass
class Overlap:
    """What the join removed, and from where. Carried by every affected `Result`."""

    declared: tuple[str, ...]
    files: tuple[str, ...]
    corpus_families: int
    overlapping_families: tuple[str, ...]
    excluded_rows: int
    excluded_by_split: dict[str, int]
    kept_rows: int

    def note(self) -> str:
        return (f"excluded {self.excluded_rows} of "
                f"{self.excluded_rows + self.kept_rows} rows in "
                f"{len(self.overlapping_families)} families that {'+'.join(self.files)} "
                f"already trains on")

    def row(self) -> dict:
        return {"trained_on": list(self.declared), "files": list(self.files),
                "corpus_families": self.corpus_families,
                "overlapping_families": len(self.overlapping_families),
                "excluded_rows": self.excluded_rows,
                "excluded_by_split": dict(self.excluded_by_split),
                "kept_rows": self.kept_rows}


def scoreable(ds: Dataset, predictor: Predictor, *,
              on_contamination: str = "drop") -> tuple[Dataset, Overlap | None]:
    """The rows of `ds` this predictor may legitimately be scored on.

    Returns `(dataset, overlap)`. `overlap` is None only for a predictor that
    declares no training corpus — nothing to check. Otherwise it records the join
    even when it removed nothing, so "checked and clean" is distinguishable from
    "not checked".

    `on_contamination` has two settings and no third: `"drop"` removes the rows the
    predictor's own corpus trains on and reports the count, `"raise"` refuses the
    measurement. There is deliberately no way to ask for the contaminated rows —
    silent inclusion is the one outcome this function exists to make impossible.
    """
    if on_contamination not in ("drop", "raise"):
        raise ValueError(f"unknown on_contamination {on_contamination!r} "
                         "(expected 'drop' or 'raise')")
    declared = tuple(getattr(predictor, "trained_on", ()) or ())
    if not declared:
        return ds, None
    files = corpus_files(*declared)
    families = corpus_families(files)
    if ds.family_kind != "frame":
        raise ValueError(
            f"predictor {predictor.name!r} declares training corpora {declared} but "
            f"dataset {ds.name!r} carries no family join key, so its rows cannot be "
            "shown clean. Score it on a dataset whose groups are corpus family ids "
            "(`load_v3`), or remove the declaration if the overlap truly is zero.")

    hit = [ex for ex in ds.examples if ex.group in families]
    by_split: dict[str, int] = {}
    for ex in hit:
        split = ex.meta.get("split") or "?"
        by_split[split] = by_split.get(split, 0) + 1
    overlap = Overlap(declared=declared, files=files,
                      corpus_families=len(families),
                      overlapping_families=tuple(sorted({ex.group for ex in hit})),
                      excluded_rows=len(hit), excluded_by_split=by_split,
                      kept_rows=ds.n - len(hit))

    if hit and on_contamination == "raise":
        raise OverlapError(
            f"{overlap.note()}; scoring {predictor.name!r} here would measure "
            f"{len(overlap.overlapping_families)} phrasings it was fitted on. "
            "Pass on_contamination='drop' to score the rest, or use a predictor "
            "that does not train on that corpus.")
    if not hit:
        return ds, overlap
    if overlap.kept_rows == 0:
        raise OverlapError(f"{overlap.note()}; nothing left to score")

    kept = [ex for ex in ds.examples if ex.group not in families]
    return (Dataset(name=ds.name, examples=kept, labels=ds.labels,
                    note=ds.note, family_kind=ds.family_kind),
            overlap)


def _refuse_unscored_overlap(ds: Dataset, predictor: Predictor) -> None:
    """Second gate, on the private engines.

    `measure` and `score_split` already dropped the overlap, so this never fires on
    that path. It exists for the caller that reaches past them into `_measure_cv` or
    `_measure_frozen` directly: those two are where outcomes are computed, so they
    refuse contaminated rows rather than trust their caller.
    """
    if not getattr(predictor, "trained_on", ()) or ds.family_kind != "frame":
        return
    families = corpus_families(corpus_files(*predictor.trained_on))
    leaked = sorted({ex.group for ex in ds.examples if ex.group in families})
    if leaked:
        raise OverlapError(
            f"dataset {ds.name!r} hands {len(leaked)} families to {predictor.name!r} "
            f"that its training corpus contains; go through measure/score_split, "
            "which drop them and report the count.")


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
    # None only when the predictor declared no training corpus; otherwise the join,
    # including the rows it removed. A published number travels with this.
    overlap: Overlap | None = None

    def row(self) -> dict:
        return {
            "predictor": self.predictor, "split": self.split, "n": self.n,
            "groups": self.n_groups, "k": self.k, "seeds": self.seeds,
            "acc_mean": round(self.acc_mean, 4), "acc_sd": round(self.acc_sd, 4),
            "ci95": [self.ci_lo, self.ci_hi], "per_seed": [round(s, 4) for s in self.per_seed],
            "dataset": self.dataset, "note": self.note,
            "overlap": self.overlap.row() if self.overlap else None,
        }


def _measure_frozen(ds: Dataset, predictor: Predictor) -> tuple[list[float], list[list]]:
    _refuse_unscored_overlap(ds, predictor)
    predictor.fit(ds.examples)
    preds = [predictor.predict(e.text) for e in ds.examples]
    outcomes = [1.0 if p == e.label else 0.0 for p, e in zip(preds, ds.examples)]
    return outcomes, [preds]


def _measure_cv(ds: Dataset, predictor: Predictor, split: str, k: int,
                seeds: tuple[int, ...]) -> tuple[list[float], list[float], list[list], list[list[int]]]:
    """Returns (seed-mean per-row outcome, per-seed accuracies, per-seed
    predictions, per-seed fold assignment)."""
    import copy

    _refuse_unscored_overlap(ds, predictor)
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
            n_boot: int = DEFAULT_BOOT, boot_seed: int = 0,
            on_contamination: str = "drop") -> list[Result]:
    """Score every predictor on one dataset under one split.

    `split` is required in spirit: it has no default that hides the choice, and
    every Result records it. Call `measure` twice — once with `"group"`, once
    with `"row"` — to see the gap the split makes.

    The dataset goes through `scoreable` per predictor, so a predictor that declares
    a training corpus is scored only on rows that corpus does not contain, and the
    count removed is on the Result and in the table.
    """
    if split not in ("group", "row"):
        raise ValueError(f"unknown split {split!r} (expected 'group' or 'row')")
    results = []

    for predictor in predictors:
        scored, overlap = scoreable(ds, predictor, on_contamination=on_contamination)
        rep = scored.group_report()
        folds_per_seed: list[list[int]] = []
        if predictor.frozen:
            outcomes, preds_per_seed = _measure_frozen(scored, predictor)
            per_seed = [round(sum(outcomes) / scored.n, 4)]
            folds_per_seed = [[-1] * scored.n]
        else:
            outcomes, per_seed, preds_per_seed, folds_per_seed = _measure_cv(
                scored, predictor, split, k, seeds)
        acc = sum(outcomes) / scored.n
        lo, hi = cluster_bootstrap_ci(outcomes, scored.groups,
                                      n_boot=n_boot, seed=boot_seed)
        sd = statistics.pstdev(per_seed) if len(per_seed) > 1 else 0.0
        note = ("frozen: no refit, split does not change predictions"
                if predictor.frozen else "")
        if overlap is not None and overlap.excluded_rows:
            note = "; ".join(filter(None, [note, overlap.note()]))
        results.append(Result(
            predictor=predictor.name, split=split, n=scored.n,
            n_groups=rep["groups"], k=1 if predictor.frozen else k,
            seeds=len(per_seed), acc_mean=round(acc, 4), acc_sd=round(sd, 4),
            acc_min=round(min(per_seed), 4), acc_max=round(max(per_seed), 4),
            ci_lo=lo, ci_hi=hi, per_seed=per_seed, dataset=ds.name,
            atomic_row_fraction=rep["singleton_row_fraction"],
            note=note,
            outcomes=outcomes, preds_per_seed=preds_per_seed,
            folds_per_seed=folds_per_seed, overlap=overlap,
        ))
    return results


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def render_table(results: list[Result], title: str = "") -> str:
    head = ("| signal | split | n | groups | atomic rows | dropped | acc | 95% CI | "
            "seeds (min–max) |")
    rule = "|---|---|---:|---:|---:|---:|---:|---|---|"
    lines = [f"## {title}" if title else "", head, rule]
    for r in results:
        ci = f"{r.ci_lo * 100:.1f}–{r.ci_hi * 100:.1f}%"
        spread = (f"{r.acc_min * 100:.1f}–{r.acc_max * 100:.1f}%"
                  if r.seeds > 1 else "—")
        atomic = f"{r.atomic_row_fraction * 100:.0f}%"
        if r.overlap is None:
            dropped = "—"
        elif r.overlap.excluded_rows:
            dropped = f"**{r.overlap.excluded_rows}**"
        else:
            dropped = "0"
        lines.append(f"| `{r.predictor}` | {r.split} | {r.n} | {r.n_groups} | "
                     f"{atomic} | {dropped} | **{r.acc_mean * 100:.1f}%** | {ci} | "
                     f"{spread} |")
    footnotes = [f"> `{r.predictor}`: {r.overlap.note()} — scored on {r.n} rows."
                 for r in results if r.overlap is not None and r.overlap.excluded_rows]
    if footnotes:
        lines += [""] + footnotes
    return "\n".join(line for line in lines if line != "")


def dump_json(results: list[Result], path: Path | str) -> None:
    Path(path).write_text(json.dumps([r.row() | {"dataset": r.dataset}
                                      for r in results], indent=2) + "\n")
