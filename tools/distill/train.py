"""train — fit the op classifier on the distilled corpus and report it per band.

The corpus certification (`docs/distill-corpus-certification-2026-11-03.md`) rules
out a pooled headline: t1 is 134 rows whose utterance names the operation, the
teacher agrees with all 134, and any model that learns "fetch" → `web_fetch`
collects 33% of the corpus for free. So every number this script prints is split
by band first, and the pooled figure is printed last, labelled as the inflated
one.

Two protocols, and neither of them touches the eval set:

* `val` — the certified family-disjoint split. 81 rows (t1 32 / t2 21 / t3 28),
  one row worth 1.2 points, so a per-band cell here is a hint rather than an
  estimate. Reported because it is the split the artefact ships with.
* `--cv` — family-grouped 5-fold CV over all 329 train rows, three seeds. This is
  where the configuration was chosen and the only in-corpus number with enough
  rows per band to carry a claim.

`utterances_v2` appears nowhere in this file. The transfer measurement lives in
`bench/distill_transfer.py`, which reads the fitted model and never fits on v2.

Usage::

    python3 tools/distill/train.py                 # fit, report, write the recipe
    python3 tools/distill/train.py --cv            # add the grouped-CV estimate
    python3 tools/distill/train.py --check         # does the recipe reproduce?
    python3 tools/distill/train.py --artefact var/distill/clf.json

**No tuned-on-val claim.** The configuration in `clf.DEFAULT` was selected by
`--cv` over `train.jsonl`. During development the val split was *viewed* for a
handful of feature-set variants before that rule existed; the document says so,
and the numbers worth quoting are the CV estimate here and the v2 transfer.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import random
import statistics
import sys

if __package__ in (None, ""):  # run as a script: put tools/ on the path for `distill.*`
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from distill import clf  # noqa: E402
from distill.frames import SLOTS  # noqa: E402

RECIPE_PATH = clf.CORPUS_DIR / "clf_recipe.json"

# Longest value first: `service/ti/decide/resolve.py` must not be read as `dir`
# `service/ti` plus a suffix, or two different objects share one key.
_OBJECT_VALUES = tuple(sorted(
    ((kind, value.lower()) for kind, values in SLOTS.items() for value in values),
    key=lambda kv: -len(kv[1])))
CV_FOLDS = 5
CV_SEEDS = (0, 1, 2)


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #

def score(model: clf.DistillClassifier, rows: list[dict]) -> dict:
    """Per-band, per-class and pooled accuracy on `rows`."""
    bands: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    cells: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    misses = []
    for r in rows:
        pred, conf = model.predict(r["utterance"])
        hit = int(pred == r["label"])
        for key in (r["band"], "all"):
            bands[key][0] += hit
            bands[key][1] += 1
        cells[f"{r['band']}/{r['label']}"][0] += hit
        cells[f"{r['band']}/{r['label']}"][1] += 1
        if not hit:
            misses.append({"id": r["id"], "band": r["band"], "label": r["label"],
                           "pred": pred, "conf": round(conf, 4),
                           "tier_target": r.get("tier_target"),
                           "utterance": r["utterance"]})
    return {
        "bands": {b: {"correct": c, "n": n, "acc": round(c / n, 4) if n else None}
                  for b, (c, n) in sorted(bands.items())},
        "cells": {k: {"correct": c, "n": n, "acc": round(c / n, 4) if n else None}
                  for k, (c, n) in sorted(cells.items())},
        "per_class": _per_class(rows, misses),
        "misses": misses,
    }


def _per_class(rows: list[dict], misses: list[dict]) -> dict:
    wrong = collections.Counter(m["label"] for m in misses)
    total = collections.Counter(r["label"] for r in rows)
    return {lab: {"n": total[lab], "correct": total[lab] - wrong[lab],
                  "acc": round((total[lab] - wrong[lab]) / total[lab], 4)}
            for lab in sorted(total)}


# --------------------------------------------------------------------------- #
# family-grouped CV within train.jsonl
# --------------------------------------------------------------------------- #

def group_folds(groups: list[str], labels: list[str], k: int, seed: int) -> list[int]:
    """Place whole families into k folds, stratifying by label.

    Same shape as `bench/eval_harness.py::group_folds` and deliberately not
    imported from there: this script must run from `tools/` with nothing on the
    path but the stdlib, and the corpus's families are already disjoint by
    construction, so the only thing needed here is the placement rule.
    """
    members: dict[str, list[int]] = collections.defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)
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
        counts = collections.Counter(labels[i] for i in idxs)
        best_f, best_key = 0, None
        for f in range(k):
            key = (sum(fold_label[f].get(lab, 0) for lab in counts), fold_size[f], f)
            if best_key is None or key < best_key:
                best_f, best_key = f, key
        for i in idxs:
            assign[i] = best_f
        fold_size[best_f] += len(idxs)
        for lab, c in counts.items():
            fold_label[best_f][lab] = fold_label[best_f].get(lab, 0) + c
    return assign


def grouped_cv(rows: list[dict], config: dict | None = None,
               k: int = CV_FOLDS, seeds: tuple[int, ...] = CV_SEEDS,
               unit: str = "family") -> dict:
    """Held-out CV over the corpus: the in-corpus estimate worth quoting.

    `unit="family"` holds out whole frames (the corpus's own split unit).
    `unit="object"` holds out whole slot values, so an unseen noun has to be
    recognised from its frame — the transfer condition, as far as the corpus can
    express it. See `object_key`.
    """
    texts = [r["utterance"] for r in rows]
    labels = [r["label"] for r in rows]
    if unit == "family":
        groups = [r["family"] for r in rows]
    elif unit == "object":
        groups = [object_key(t) or f"__row{i}" for i, t in enumerate(texts)]
    else:
        raise ValueError(f"unknown CV unit {unit!r} (expected 'family' or 'object')")
    per_seed: list[float] = []
    band_hits: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for seed in seeds:
        fold = group_folds(groups, labels, k, seed)
        correct = 0
        for f in range(k):
            tr = [i for i in range(len(rows)) if fold[i] != f]
            te = [i for i in range(len(rows)) if fold[i] == f]
            model = clf.DistillClassifier.from_config(config or {}).fit(
                [rows[i] for i in tr])
            for i in te:
                pred, _ = model.predict(texts[i])
                hit = int(pred == labels[i])
                correct += hit
                band_hits[rows[i]["band"]][0] += hit
                band_hits[rows[i]["band"]][1] += 1
        per_seed.append(round(correct / len(rows), 4))
    draws = len(seeds)
    return {
        "unit": unit, "k": k, "seeds": list(seeds), "per_seed": per_seed,
        "mean": round(statistics.fmean(per_seed), 4),
        "sd": round(statistics.pstdev(per_seed), 4) if len(per_seed) > 1 else 0.0,
        "bands": {b: {"correct": c // draws, "n": n // draws,
                      "acc": round(c / n, 4)}
                  for b, (c, n) in sorted(band_hits.items())},
    }


def object_key(text: str) -> str | None:
    """The slot value an utterance names, as `type:value` — or None.

    Used only to build an **object-held-out** split: every row naming the same
    object goes to the same fold, so a fold can test generalisation to a noun the
    training folds never saw. The inventory is the corpus's own (`frames.SLOTS`),
    which is also the mechanism a memorising model exploits — a frame's slot
    values recur across families, so family-held-out CV leaves every noun in the
    training fold and cannot see the memorisation at all.
    """
    low = text.lower()
    for kind, value in _OBJECT_VALUES:
        if value in low:
            return f"{kind}:{value}"
    return None


def object_folds(rows: list[dict], k: int = CV_FOLDS, seeds: tuple[int, ...] = CV_SEEDS,
                 config: dict | None = None) -> dict:
    """Family-grouped CV, then the same CV with the split unit replaced by object.

    Reported side by side because they disagree, and the disagreement is the
    point: a split that leaves the nouns in place rewards exactly the model that
    will not transfer (`docs/distill-classifier-2026-11-04.md` §6).
    """
    return grouped_cv(rows, config, k=k, seeds=seeds, unit="object")


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def render(result: dict, cv: dict | None, ocv: dict | None,
           model: clf.DistillClassifier) -> str:
    lines = ["## val — the certified family-disjoint split", "",
             "| band | correct | n | accuracy |", "|---|---:|---:|---:|"]
    for band in ("t1", "t2", "t3"):
        b = result["bands"].get(band, {"correct": 0, "n": 0, "acc": None})
        lines.append(f"| {band} | {b['correct']} | {b['n']} | "
                     f"{b['acc'] * 100:.1f}% |")
    a = result["bands"]["all"]
    lines.append(f"| **pooled** | {a['correct']} | {a['n']} | {a['acc'] * 100:.1f}% |")
    lines += ["", "The pooled row is the inflated one: t1 is 40% of this split and "
              "its utterances name the operation.", ""]

    if cv:
        lines += ["## family-grouped 5-fold CV over train.jsonl — the quoteable "
                  "in-corpus number", "",
                  "| band | correct (per draw) | n | accuracy |", "|---|---:|---:|---:|"]
        for band in ("t1", "t2", "t3"):
            b = cv["bands"].get(band, {"correct": 0, "n": 0, "acc": None})
            lines.append(f"| {band} | {b['correct']} | {b['n']} | "
                         f"{b['acc'] * 100:.1f}% |")
        lines.append(f"\nacross {cv['k']} folds x {len(cv['seeds'])} seeds: "
                     f"**{cv['mean'] * 100:.1f}%** (sd {cv['sd'] * 100:.1f}) per seed "
                     f"{[round(p * 100, 1) for p in cv['per_seed']]}")

    if ocv:
        lines += ["", "## object-held-out CV — the same folds, with the slot values "
                  "held out instead of the frames", "",
                  "| band | correct (per draw) | n | accuracy |", "|---|---:|---:|---:|"]
        for band in ("t1", "t2", "t3"):
            b = ocv["bands"].get(band, {"correct": 0, "n": 0, "acc": None})
            lines.append(f"| {band} | {b['correct']} | {b['n']} | "
                         f"{b['acc'] * 100:.1f}% |")
        lines.append(f"\nacross {ocv['k']} folds x {len(ocv['seeds'])} seeds: "
                     f"**{ocv['mean'] * 100:.1f}%** (sd {ocv['sd'] * 100:.1f}). Higher "
                     "than the family split is not good news: it says an unseen noun "
                     "inside a seen frame is the easy case (doc §6).")

    lines += ["", "## per class (val)", "", "| op | correct | n | accuracy |",
              "|---|---:|---:|---:|"]
    for lab, d in result["per_class"].items():
        lines.append(f"| `{lab}` | {d['correct']} | {d['n']} | {d['acc'] * 100:.1f}% |")

    if result["misses"]:
        lines += ["", f"## the {len(result['misses'])} val misses", "",
                  "| band | label | predicted | conf | utterance |", "|---|---|---|---:|---|"]
        for m in result["misses"]:
            lines.append(f"| {m['band']} | `{m['label']}` | `{m['pred']}` | "
                         f"{m['conf']:.2f} | {m['utterance']} |")
    return "\n".join(lines)


def recipe(model: clf.DistillClassifier, val: dict, cv: dict | None = None,
           ocv: dict | None = None) -> dict:
    out = {
        "format": clf.FORMAT,
        "produced_by": "tools/distill/train.py",
        "corpus": {
            "train": {"rows": len(clf.load_split("train")),
                      "sha": sha256_file(clf.CORPUS_DIR / "train.jsonl")},
            "val": {"rows": len(clf.load_split("val")),
                    "sha": sha256_file(clf.CORPUS_DIR / "val.jsonl")},
        },
        "label": "label — the row's authored `intended` op; never target.tool",
        "config": model.config(),
        "artefact_sha256": model.artefact_sha(),
        "artefact_note": ("the artefact itself is not committed — it is ~3 MB of "
                          "float text. This sha is over its canonical JSON "
                          "(`DistillClassifier.to_artefact`, sort_keys, no spaces); "
                          "`train.py --check` fails if a re-fit does not reproduce it"),
        "accuracy": {"val_by_band": {b: val["bands"][b] for b in ("t1", "t2", "t3")},
                     "val_pooled": val["bands"]["all"],
                     "val_per_class": val["per_class"]},
    }
    if cv:
        out["accuracy"]["grouped_cv"] = cv
    if ocv:
        out["accuracy"]["object_held_out_cv"] = ocv
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cv", action="store_true",
                    help="also run family-grouped CV over train.jsonl (slow, ~1 min)")
    ap.add_argument("--object-cv", action="store_true",
                    help="also run object-held-out CV (the transfer-shaped split, §6)")
    ap.add_argument("--check", action="store_true",
                    help="re-fit and assert the committed recipe's sha still reproduces")
    ap.add_argument("--write", action="store_true",
                    help="write data/distill/clf_recipe.json from this fit")
    ap.add_argument("--artefact", default="",
                    help="write the fitted artefact JSON here (git-ignored default: var/)")
    ap.add_argument("--json", action="store_true", help="machine-readable dump")
    args = ap.parse_args(argv)

    train_rows = clf.load_split("train")
    val_rows = clf.load_split("val")
    model = clf.train_default()
    val = score(model, val_rows)
    cv = grouped_cv(train_rows) if args.cv else None
    ocv = (grouped_cv(train_rows, unit="object")
           if args.object_cv or args.write or args.check else None)

    if args.check:
        committed = json.loads(RECIPE_PATH.read_text())
        got = model.artefact_sha()
        want = committed["artefact_sha256"]
        status = "reproduces" if got == want else "DOES NOT REPRODUCE"
        print(f"artefact sha {got} vs committed {want}: {status}")
        if cv:
            print(f"grouped CV {cv['mean']:.3f} (sd {cv['sd']:.3f})")
        if ocv:
            print(f"object-held-out CV {ocv['mean']:.3f} (sd {ocv['sd']:.3f})")
        return 0 if got == want else 1

    if args.artefact:
        path = pathlib.Path(args.artefact)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(model.to_artefact()))
        print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
        return 0

    if args.json:
        print(json.dumps({"val": val, "cv": cv, "object_cv": ocv,
                          "recipe": recipe(model, val, cv, ocv)}, indent=2))
        return 0

    print(render(val, cv, ocv, model))
    print(f"\nartefact sha256 {model.artefact_sha()[:16]}… "
          f"({model.n_train} train rows, {model.learner} learner)")

    built = recipe(model, val, cv, ocv)
    if args.write:
        RECIPE_PATH.write_text(json.dumps(built, indent=2, sort_keys=True) + "\n")
        print(f"wrote {RECIPE_PATH.relative_to(clf.ROOT)}")
    elif RECIPE_PATH.exists():
        old = json.loads(RECIPE_PATH.read_text())
        if old.get("artefact_sha256") != built["artefact_sha256"]:
            print(f"\nNOTE: {RECIPE_PATH.name} records a different sha "
                  f"({old.get('artefact_sha256', '?')[:16]}…) — re-run with "
                  "--write to refresh it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
