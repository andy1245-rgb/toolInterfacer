"""certify — freeze the distilled label store into a training set, with the audit attached.

The label store in ``var/distill/`` is the most valuable asset this repo has for
raising local routing accuracy, and it is git-ignored, so nothing about it is
checkable by a reader and nothing stops the next run from quietly changing it.
This module is the answer to both: it reads the store once, audits it against the
frozen eval sets, and writes a frozen, committed artefact whose manifest states
every number and whose exclusion list states every removal and why.

Four audits, each of which has already mattered once in this repo:

**contamination.** ``tests/data/utterances_v2.json`` (150 rows) is the eval set
for the same 8-op taxonomy, so a distilled row that restates one of its rows makes
the resulting accuracy meaningless. The generator refuses collisions at generation
and label time (``corpus.assert_not_frozen``), and this audit re-runs the check
exhaustively over every pair rather than trusting the manifest's own claim.
``tests/data/routing_382.json`` is checked too, but it is an *imported* benchmark
over a disjoint 18-class taxonomy — it cannot be contaminated by these labels in
the sense that matters, and no classifier trained here can be scored on it.

**label quality.** ``intended`` is the label these frames were authored to carry
(see ``tests/data/utterances_v2_NOTES.md``: "the label is the semantically correct
one, ``tier_target`` is the confusable rival"). ``target.tool`` is what the hosted
teacher answered, and it is *not* a label: it disagrees with ``intended`` on 36
rows, and in several frames it gives opposite answers to two rows that differ only
in a slot value — at 0.9 confidence both times. That is teacher noise, so the
frozen set carries ``intended`` and the teacher's answer only as a diagnostic.

**split leakage.** The corpus ships a train/val split. If a family crosses it, the
corpus's own val number is contaminated and so is any checkpoint selection done on
it. ``family`` is the frame, and ``corpus.assign_split`` hashes it, so this should
be clean by construction — asserted here rather than assumed.

**count reconciliation.** The repo quotes several row counts (267 labelled, 414
candidates, 412 labels). The manifest records the exact arithmetic so a reader does
not have to guess which number was stale.

Usage::

    python3 tools/distill/certify.py                       # store from $TI_DISTILL_DIR or var/distill
    python3 tools/distill/certify.py --data-dir var/distill --out data/distill
    python3 tools/distill/certify.py --check               # audit only, write nothing
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

if __package__ in (None, ""):  # run as a script: put tools/ on the path for `distill.*`
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from distill import corpus, similarity  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_IN = "var/distill"
DEFAULT_OUT = "data/distill"
ENV_IN = "TI_DISTILL_DIR"

ROUTING_PATH = ROOT / "tests" / "data" / "routing_382.json"

# A row that is a *strict token-subset* of an eval row this close by character is
# that eval request with a clause deleted ("list every file ending in .toml" vs
# "list every file ending in .toml anywhere in the repo"). The leak guard's 0.80
# ratio bar is calibrated for paraphrase, which lengthens; deletion shortens, so
# subsumption needs its own, lower bar. 0.70 catches the two observed cases and
# leaves the generic short phrases ("which one should we ship?", 0.56) in.
SUBSUME_RATIO = 0.70

EXCLUSION_RULES = {
    "leak_rule": "trips the repo's own leakage guard against a frozen eval row "
                 "(normalised-exact OR jaccard>=0.70 OR ratio>=0.80)",
    "eval_subsumption": f"strict token-subset of a frozen eval row at "
                        f"SequenceMatcher ratio>={SUBSUME_RATIO}",
    "label_off_ballot": "`intended` is not among the row's own candidate keys — a "
                        "labelling impossibility",
    "no_label": "no `intended` label at all",
}


# ---------------------------------------------------------------- io


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha12(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(body, encoding="utf-8")


# ---------------------------------------------------------------- audits


def eval_sets() -> dict[str, tuple[pathlib.Path, list[str], str]]:
    """The two frozen sets, with the text field each one spells its row with."""
    v2 = json.loads(corpus.FROZEN_PATH.read_text(encoding="utf-8"))
    out = {"utterances_v2": (corpus.FROZEN_PATH, [r["utterance"] for r in v2], "utterance")}
    if ROUTING_PATH.exists():
        r382 = json.loads(ROUTING_PATH.read_text(encoding="utf-8"))
        out["routing_382"] = (ROUTING_PATH, [r["text"] for r in r382], "text")
    return out


@dataclass(frozen=True)
class Refs:
    """One eval set, normalised once: the sweep and the nearest-neighbour pass
    both read it, and normalising per pair was the whole cost of the audit."""
    texts: list[str]
    norms: list[str]
    tokens: list[set[str]]

    @classmethod
    def of(cls, texts: list[str]) -> "Refs":
        norms = [similarity.normalize(t) for t in texts]
        return cls(texts, norms, [set(n.split()) for n in norms])

    def nearest(self, text: str) -> tuple[similarity.Comparison, int]:
        """Nearest reference by Jaccard, with the exact character ratio."""
        norm = similarity.normalize(text)
        mine = set(norm.split())
        best_j, best_i = 0.0, -1
        for index, theirs in enumerate(self.tokens):
            if not mine or not theirs:
                continue
            shared = len(mine & theirs) / len(mine | theirs)
            if shared > best_j:
                best_j, best_i = shared, index
        if best_i < 0:
            return similarity.Comparison(0.0, 0.0, 0.0), -1
        return similarity.Comparison(1.0 if norm == self.norms[best_i] else 0.0, best_j,
                                     similarity.ratio_norm(norm, self.norms[best_i])), best_i


def ratio_ceiling(norm_a: str, norm_b: str) -> float:
    """The largest ``SequenceMatcher.ratio`` two strings of these lengths can score.

    ratio <= 2*min/(len_a+len_b), because a match block cannot be longer than the
    shorter string. Skipping pairs whose ceiling is already under the threshold
    makes an all-pairs ratio sweep both exhaustive and cheap.
    """
    la, lb = len(norm_a), len(norm_b)
    return 0.0 if not la or not lb else 2.0 * min(la, lb) / (la + lb)


def sweep(rows: list[dict[str, Any]], refs: Refs) -> tuple[list[dict[str, Any]], float, float]:
    """Every (row, reference) pair, pruned only where the ratio bound proves a miss.

    Returns the per-row hits plus the global maximum of each measure, so the
    contamination claim rests on the worst pair that exists rather than on the
    nearest neighbour of each row — those are different quantities, and only the
    first one is a guarantee.
    """
    hits: list[dict[str, Any]] = []
    max_ratio = 0.0
    max_jaccard = 0.0
    for row in rows:
        n = similarity.normalize(row["utterance"])
        mine = set(n.split())
        for index, reference in enumerate(refs.texts):
            theirs = refs.tokens[index]
            if not mine or not theirs:
                continue
            jaccard = len(mine & theirs) / len(mine | theirs)
            max_jaccard = max(max_jaccard, jaccard)
            ceiling = ratio_ceiling(n, refs.norms[index])
            if jaccard < similarity.LEAK_JACCARD and ceiling < similarity.LEAK_RATIO:
                continue
            ratio = similarity.ratio_norm(n, refs.norms[index])
            max_ratio = max(max_ratio, ratio)
            if (n == refs.norms[index] or jaccard >= similarity.LEAK_JACCARD
                    or ratio >= similarity.LEAK_RATIO):
                hits.append({"id": row.get("id"), "utterance": row["utterance"],
                             "eval_row": reference, "jaccard": round(jaccard, 3),
                             "ratio": round(ratio, 3),
                             "rule": "exact" if n == refs.norms[index]
                                     else "jaccard" if jaccard >= similarity.LEAK_JACCARD
                                     else "ratio"})
    return hits, max_ratio, max_jaccard


def subsumes(row_text: str, ref_text: str) -> bool:
    """Whether one token set is a *strict* subset of the other (a deletion)."""
    mine = set(similarity.normalize(row_text).split())
    theirs = set(similarity.normalize(ref_text).split())
    return bool(mine and theirs) and (mine < theirs or theirs < mine)


def audit_contamination(rows: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {"eval_sets": {}, "method": similarity.METHOD,
                              "subsume_ratio": SUBSUME_RATIO,
                              "sweep": "all (row, eval-row) pairs, exhaustive; ratio "
                                       "skipped only where 2*min/(sum length) proves it "
                                       "cannot reach the threshold"}
    for name, (path, texts, _field) in eval_sets().items():
        refs = Refs.of(texts)
        hits, max_ratio, max_jaccard = sweep(rows, refs)
        subsumed: list[dict[str, Any]] = []
        closest_rows: list[tuple[float, str, str, str | None]] = []
        corpus_ref = similarity.Corpus(texts, label=name)
        for row in rows:
            comparison, index = refs.nearest(row["utterance"])
            if index >= 0 and subsumes(row["utterance"], refs.texts[index]) \
                    and comparison.ratio >= SUBSUME_RATIO:
                subsumed.append({"id": row.get("id"), "utterance": row["utterance"],
                                 "eval_row": refs.texts[index],
                                 "jaccard": round(comparison.jaccard, 3),
                                 "ratio": round(comparison.ratio, 3)})
            # The repo's own guard, as an independent second opinion.
            guarded, hit = corpus_ref.match(row["utterance"])
            if guarded.leaks and not any(h["id"] == row.get("id") for h in hits):
                hits.append({"id": row.get("id"), "utterance": row["utterance"],
                             "eval_row": hit, "jaccard": round(guarded.jaccard, 3),
                             "ratio": round(guarded.ratio, 3), "rule": "corpus.match"})
            closest_rows.append((round(comparison.ratio, 3), row.get("id"),
                                 row["utterance"], refs.texts[index] if index >= 0 else None))
        closest_rows.sort(reverse=True)
        norm_refs = set(refs.norms)
        report["eval_sets"][name] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha": sha12(path),
            "rows": len(texts),
            "exact": sum(1 for row in rows
                         if similarity.normalize(row["utterance"]) in norm_refs),
            "leak_rule_hits": hits,
            "subsumption_hits": subsumed,
            "max_ratio": round(max_ratio, 4),
            "max_jaccard": round(max_jaccard, 4),
            "closest": [{"ratio": r, "id": i, "utterance": u, "eval_row": e}
                        for r, i, u, e in closest_rows[:5]],
        }
    report["ok"] = all(not s["leak_rule_hits"] for s in report["eval_sets"].values())
    return report


def audit_labels(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bands = sorted({r["band"] for r in rows})
    ops = sorted({r["intended"] for r in rows if r.get("intended")})
    cells = {band: {op: sum(1 for r in rows if r["band"] == band and r["intended"] == op)
                    for op in ops} for band in bands}
    per_band: dict[str, Any] = {}
    for band in bands:
        subset = [r for r in rows if r["band"] == band]
        agree = sum(1 for r in subset if r["target"]["tool"] == r["intended"])
        confs = [r["target"]["conf"] for r in subset]
        per_band[band] = {
            "rows": len(subset),
            "teacher_agrees": agree,
            "teacher_agreement": round(agree / len(subset), 3),
            "mean_confidence": round(sum(confs) / len(confs), 3),
            "conf_saturated_0.95_plus": sum(1 for c in confs if c >= 0.95),
            "label_method": dict(collections.Counter(r["label_method"] for r in subset)),
        }
    disagree = [r for r in rows if r["target"]["tool"] != r["intended"]]
    by_family: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by_family[r["family"]].append(r)
    multi = {f: v for f, v in by_family.items() if len(v) > 1}
    unstable = {f: v for f, v in multi.items()
                if len({r["target"]["tool"] for r in v}) > 1}
    t3 = [r for r in rows if r["band"] == "t3"]
    return {
        "rows": len(rows),
        "label_method": dict(collections.Counter(r["label_method"] for r in rows)),
        "per_band": per_band,
        "cells": cells,
        "thin_cells": [f"{b}/{o}" for b, m in cells.items() for o, n in m.items() if n < 15],
        "intended_distribution": dict(collections.Counter(r["intended"] for r in rows)),
        "label_off_ballot": [r["id"] for r in rows if r["intended"] not in r["candidates"]],
        "teacher_off_ballot": [r["id"] for r in rows if r["target"]["tool"] not in r["candidates"]],
        "teacher_disagreements": {
            "rows": len(disagree),
            "ids": [r["id"] for r in disagree],
            "by_band": dict(collections.Counter(r["band"] for r in disagree)),
            "pairs": {f"{intended}->{choice}": n for (intended, choice), n in
                      collections.Counter((x["intended"], x["target"]["tool"])
                                          for x in disagree).items()},
        },
        "frame_instability": {
            "multi_row_families": len(multi),
            "unstable_families": len(unstable),
            "unstable_ids": sorted(r["id"] for v in unstable.values() for r in v),
            "detail": {f: [{"id": r["id"], "choice": r["target"]["tool"],
                            "conf": r["target"]["conf"], "utterance": r["utterance"]}
                           for r in v] for f, v in sorted(unstable.items())},
        },
        "t3_decoy": {
            "rows": len(t3),
            "teacher_picked_label": sum(1 for r in t3 if r["target"]["tool"] == r["intended"]),
            "teacher_picked_decoy": sum(1 for r in t3 if r["target"]["tool"] == r["tier_target"]),
            "teacher_picked_other": sum(1 for r in t3
                                        if r["target"]["tool"] not in (r["intended"], r["tier_target"])),
        },
    }


def audit_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, set[str]] = collections.defaultdict(set)
    family_sizes = collections.Counter(r["family"] for r in rows)
    for r in rows:
        by_family[r["family"]].add(r["split"])
    crossing = sorted(f for f, s in by_family.items() if len(s) > 1)
    return {
        "splits_before_exclusion": dict(collections.Counter(r["split"] for r in rows)),
        "families": len(by_family),
        "families_multi_row": sum(1 for n in family_sizes.values() if n > 1),
        "families_crossing_split": crossing,
        "leaks": bool(crossing),
        "per_band": {b: dict(collections.Counter(r["split"] for r in rows if r["band"] == b))
                     for b in sorted({r["band"] for r in rows})},
        "split_unit": "family (the frame); see corpus.assign_split",
    }


# ---------------------------------------------------------------- exclusions


def decide(rows: list[dict[str, Any]], contamination: dict[str, Any]
           ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split the store into (retained, excluded); every exclusion names its rule."""
    excluded: dict[str, dict[str, Any]] = {}
    for name, block in contamination["eval_sets"].items():
        for hit in block["leak_rule_hits"]:
            excluded.setdefault(hit["id"], {"id": hit["id"], "utterance": hit["utterance"],
                                            "rule": "leak_rule", "eval_set": name,
                                            "detail": hit})
        for hit in block["subsumption_hits"]:
            excluded.setdefault(hit["id"], {"id": hit["id"], "utterance": hit["utterance"],
                                            "rule": "eval_subsumption", "eval_set": name,
                                            "detail": hit})
    for row in rows:
        if not row.get("intended"):
            excluded.setdefault(row["id"], {"id": row["id"], "utterance": row["utterance"],
                                            "rule": "no_label", "detail": {}})
        elif row["intended"] not in row["candidates"]:
            excluded.setdefault(row["id"], {"id": row["id"], "utterance": row["utterance"],
                                            "rule": "label_off_ballot",
                                            "detail": {"intended": row["intended"],
                                                       "candidates": sorted(row["candidates"])}})
    retained = [r for r in rows if r["id"] not in excluded]
    for record in excluded.values():
        record["why"] = EXCLUSION_RULES[record["rule"]]
    return retained, [excluded[k] for k in sorted(excluded)]


def freeze(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The training row: what a trainer needs, and nothing that invites a mistake.

    ``label`` is ``intended`` — the authored answer. The teacher's choice is
    deliberately absent: it is a measurement of the teacher, it disagrees on 36
    rows, and shipping it beside the label is how a trainer ends up fitting noise.
    """
    return [{"id": r["id"], "utterance": r["utterance"], "label": r["intended"],
             "band": r["band"], "split": r["split"], "family": r["family"],
             "tier_target": r["tier_target"], "candidates": sorted(r["candidates"])}
            for r in rows]


def ballot(rows: list[dict[str, Any]]) -> dict[str, str]:
    """The op descriptions, once. Every row's ballot draws from these 8 keys."""
    out: dict[str, str] = {}
    for row in rows:
        out.update(row["candidates"])
    return dict(sorted(out.items()))


# ---------------------------------------------------------------- main


def certify(data_dir: pathlib.Path, out_dir: pathlib.Path, write: bool = True) -> dict[str, Any]:
    labels_path = data_dir / "labels.jsonl"
    candidates_path = data_dir / "candidates.jsonl"
    rows = read_jsonl(labels_path)
    candidates = read_jsonl(candidates_path)
    if not rows:
        raise SystemExit(f"no labels at {labels_path} — set --data-dir or ${ENV_IN}")

    contamination = audit_contamination(rows)
    lab = audit_labels(rows)
    split = audit_split(rows)
    retained, excluded = decide(rows, contamination)

    excluded_ids = {e["id"] for e in excluded}
    train = freeze([r for r in retained if r["split"] == "train"])
    val = freeze([r for r in retained if r["split"] == "val"])

    candidate_ids = {r["id"] for r in candidates}
    label_ids = {r["id"] for r in rows}
    rejects = read_jsonl(data_dir / "rejects.jsonl")
    dropped = read_jsonl(data_dir / "dropped.jsonl")

    counts = {
        "candidates": len(candidates),
        "labels": len(rows),
        "candidates_never_labelled": sorted(candidate_ids - label_ids),
        "excluded": len(excluded),
        "retained": len(retained),
        "train": len(train),
        "val": len(val),
        "reject_attempts": len(rejects),
        "reject_ids": len({r["id"] for r in rejects}),
        "reject_ids_never_labelled": sorted({r["id"] for r in rejects} - label_ids),
        "generator_drops_logged": dict(collections.Counter(d["kind"] for d in dropped)),
    }

    manifest: dict[str, Any] = {
        "generated_by": "tools/distill/certify.py",
        "source": {"dir": str(data_dir.name), "labels_sha": sha12(labels_path),
                   "candidates_sha": sha12(candidates_path) if candidates_path.exists() else None},
        "schema": {
            "label": "`intended` — the frame's authored answer; see "
                     "tests/data/utterances_v2_NOTES.md",
            "not_label": "the hosted teacher's answer is absent from the frozen rows by design "
                         "(36 rows disagree); see diagnostics.jsonl",
            "fields": ["id", "utterance", "label", "band", "split", "family",
                       "tier_target", "candidates"],
            "tier_target": "decoy op for t3 rows (null elsewhere); never the label",
        },
        "counts": counts,
        "contamination": contamination,
        "labels": lab,
        "split": split,
        "exclusions": {"rules": EXCLUSION_RULES, "rows": excluded},
        "verdict": {
            "leak_rule_clean": contamination["ok"],
            "borderline_removed": len(excluded),
            "contaminated_after_exclusion": bool(
                contamination["eval_sets"]["utterances_v2"]["leak_rule_hits"]),
            "trainable": len(train) > 0,
            "eval_set_for_this_corpus": "tests/data/utterances_v2.json",
            "routing_382_relation": "disjoint taxonomy (18 imported classes vs this 8); "
                                    "cannot be trained on from this corpus and is not "
                                    "contaminated by it",
        },
    }

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(out_dir / "train.jsonl", train)
        write_jsonl(out_dir / "val.jsonl", val)
        write_jsonl(out_dir / "excluded.jsonl", excluded)
        write_jsonl(out_dir / "diagnostics.jsonl", [
            {"id": r["id"], "utterance": r["utterance"], "label": r["intended"],
             "teacher_choice": r["target"]["tool"], "teacher_confidence": r["target"]["conf"],
             "family": r["family"], "band": r["band"], "split": r["split"]}
            for r in rows if r["id"] not in excluded_ids
            and (r["target"]["tool"] != r["intended"]
                 or r["family"] in lab["frame_instability"]["detail"])])
        (out_dir / "ballot.json").write_text(
            json.dumps(ballot(rows), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest["files"] = {name: {"sha": sha12(out_dir / name),
                                    "rows": sum(1 for _ in (out_dir / name).open(encoding="utf-8"))
                                    if name.endswith(".jsonl") else None}
                             for name in ("train.jsonl", "val.jsonl", "excluded.jsonl",
                                          "diagnostics.jsonl", "ballot.json")}
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def report(manifest: dict[str, Any]) -> None:
    c = manifest["contamination"]
    print("contamination")
    for name, block in c["eval_sets"].items():
        print(f"  vs {name:14s} {block['rows']:4d} rows  exact={block['exact']:3d}  "
              f"leak-rule={len(block['leak_rule_hits']):3d}  "
              f"subsumption={len(block['subsumption_hits']):3d}  "
              f"max ratio={block['max_ratio']:.3f}  max jaccard={block['max_jaccard']:.3f}")
        for hit in block["subsumption_hits"]:
            print(f"      SUBSET {hit['id']} ratio={hit['ratio']:.2f} {hit['utterance']!r}"
                  f"\n             ~ {hit['eval_row']!r}")
    print(f"  verdict: {'CLEAN' if c['ok'] else 'CONTAMINATED'}")
    counts = manifest["counts"]
    print("\ncounts")
    for key in ("candidates", "labels", "excluded", "retained", "train", "val"):
        print(f"  {key:22s} {counts[key]}")
    print(f"  never labelled         {counts['candidates_never_labelled']}")
    print(f"  reject attempts        {counts['reject_attempts']} "
          f"({counts['reject_ids']} ids, {len(counts['reject_ids_never_labelled'])} never recovered)")
    lab = manifest["labels"]
    print("\nlabels")
    for band, block in lab["per_band"].items():
        print(f"  {band}: n={block['rows']:3d} teacher-agree={block['teacher_agreement']:.1%} "
              f"meanconf={block['mean_confidence']:.3f} saturated={block['conf_saturated_0.95_plus']}")
    print(f"  label off ballot: {len(lab['label_off_ballot'])}  "
          f"teacher off ballot: {len(lab['teacher_off_ballot'])}")
    print(f"  teacher disagreements: {lab['teacher_disagreements']['rows']}")
    print(f"  unstable frames: {lab['frame_instability']['unstable_families']} "
          f"of {lab['frame_instability']['multi_row_families']} multi-row families")
    print(f"  t3 decoy: label={lab['t3_decoy']['teacher_picked_label']} "
          f"decoy={lab['t3_decoy']['teacher_picked_decoy']} "
          f"other={lab['t3_decoy']['teacher_picked_other']}")
    split = manifest["split"]
    print(f"\nsplit (source, pre-exclusion)  {split['splits_before_exclusion']}  families crossing: "
          f"{len(split['families_crossing_split'])} -> "
          f"{'LEAK' if split['leaks'] else 'clean'}")
    print("\nexclusions")
    for row in manifest["exclusions"]["rows"]:
        print(f"  {row['id']} [{row['rule']}] {row['utterance']!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=os.environ.get(ENV_IN) or DEFAULT_IN,
                        help=f"label store to read (default: ${ENV_IN} or {DEFAULT_IN})")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"frozen artefact directory (default: {DEFAULT_OUT})")
    parser.add_argument("--check", action="store_true",
                        help="audit and print, write nothing")
    args = parser.parse_args(argv)
    manifest = certify(pathlib.Path(args.data_dir), pathlib.Path(args.out),
                       write=not args.check)
    report(manifest)
    return 0 if manifest["verdict"]["leak_rule_clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
