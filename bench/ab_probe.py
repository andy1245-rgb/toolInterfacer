#!/usr/bin/env python3
"""ab_probe — old-vs-new A/B over utterances_v2, end-to-end through use_tool().

One lane per process, because the lane is chosen by which `ti` package is
imported: `--service-root` is put at sys.path[0] *before* any ti import, so the
same script file drives the pre-fix library and the current one. Calls are
in-process on purpose — no HTTP, no extension — which is what leaves the
extension-side capture (service/logs/live.jsonl) untouched.

Usage (two lanes, then the cross-lane report):
  PYTHONPATH=<root>/service python3 bench/ab_probe.py --service-root <old-root> \
      --out bench/results/ab-probe-old-<date>.json
  PYTHONPATH=<root>/service python3 bench/ab_probe.py --service-root <new-root> \
      --out bench/results/ab-probe-new-<date>.json
  python3 bench/ab_probe.py --report-only \
      --out bench/results/ab-probe-new-<date>.json \
      --compare bench/results/ab-probe-old-<date>.json

The listing diagnostic is measurement only: it reports whether a filesys_glob
result respects the extension/filename constraint the utterance named, and never
changes what the plan does. That defect is sized, not fixed.

A lane's display name is its `lane` field; a name containing "baseline" reads as
`old`, anything else as `new`, so the report's sides are stable either way.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_guard import ModelAssertionError, preflight  # noqa: E402

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_UTTERANCES = ROOT / "tests/data/utterances_v2.json"
DEFAULT_RESULTS = ROOT / "bench/results"
GLOB_LABEL = "filesys_glob"


def _portable(value: Any) -> Any:
    """`$HOME/...` for every home path in the recorded document.

    A lane's identity is its root, its model path, and — inside an assumption —
    the cwd a listing fell back to. Recorded verbatim they put a machine-personal
    path into `bench/results/`, which `tests/test_no_personal_paths.py` scans; the
    convention there is that the prefix is `$HOME/` and the directory basename is
    what identifies the lane, so the rewrite is prefix-only. Nothing measured
    moves.
    """
    if isinstance(value, dict):
        return {key: _portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable(item) for item in value]
    if isinstance(value, str):
        return value.replace(str(Path.home()), "$HOME")
    return value


# ---------------------------------------------------------------- constraints

_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
# An extension named as a token of its own: ".toml", "*.sh". The lookbehind
# keeps "config/permissions.yaml" (a filename in a sentence) out of the set.
_EXT_TOKEN = re.compile(r"(?<![\w.])\.([a-z][a-z0-9+]{0,5})(?![\w])", re.I)
_NAMED_FILE = re.compile(r"\b(dockerfile|docker-compose|readme|manifest|makefile|"
                         r"license|changelog|gemfile|procfile)\b", re.I)
_PREFIX = re.compile(r"\b(?:start|starts|starting|begin|begins|beginning)\s+with\s+"
                     r"['\"`]?([\w.\-]+)", re.I)
_SUBSTRING = re.compile(r"\b(?:have|has|with|containing)\s+['\"`]([\w.\-]+)['\"`]"
                        r"\s+in\s+(?:their|its|the)\s+name", re.I)
_DOTFILE = re.compile(r"\bdot\s*-?\s*files?\b", re.I)


def constraint_of(utterance: str) -> dict[str, str] | None:
    """The name constraint the utterance states, as {kind, value, evidence}."""
    text = _URL.sub(" ", utterance or "")
    match = _EXT_TOKEN.search(text)
    if match:
        return {"kind": "ext", "value": match.group(1).lower(),
                "evidence": match.group(0)}
    match = _PREFIX.search(text)
    if match:
        return {"kind": "prefix", "value": match.group(1), "evidence": match.group(0)}
    match = _SUBSTRING.search(text)
    if match:
        return {"kind": "substring", "value": match.group(1), "evidence": match.group(0)}
    if _DOTFILE.search(text):
        return {"kind": "dotfile", "value": ".", "evidence": _DOTFILE.search(text).group(0)}
    match = _NAMED_FILE.search(text)
    if match:
        return {"kind": "name", "value": match.group(1).lower(), "evidence": match.group(0)}
    return None


def _satisfies(path: str, constraint: dict[str, str]) -> bool:
    base = str(path).replace("\\", "/").rsplit("/", 1)[-1].lower()
    kind, value = constraint["kind"], constraint["value"].lower()
    if kind == "ext":
        return base.endswith("." + value)
    if kind == "prefix":
        return base.startswith(value)
    if kind == "substring":
        return value in base
    if kind == "dotfile":
        return base.startswith(".")
    return value in base


def listing_diagnostic(result: Any, utterance: str) -> dict[str, Any]:
    """Whether the returned paths respect the constraint the utterance named.

    `listing_respected` is None when nothing was judged: no constraint was named,
    the plan returned no path list at all (the read shape returns one file's
    content), or the list was empty.
    """
    returned_list = isinstance(result, list)
    paths = [str(p) for p in result] if returned_list else []
    constraint = constraint_of(utterance)
    out: dict[str, Any] = {"listing_constraint": constraint,
                           "listing_respected": None,
                           "returned_list": returned_list,
                           "n_returned": len(paths) if returned_list else None,
                           "n_violations": None}
    if constraint is None or not paths:
        return out
    violations = [p for p in paths if not _satisfies(p, constraint)]
    out["listing_respected"] = not violations
    out["n_violations"] = len(violations)
    out["violation_samples"] = violations[:5]
    return out


# ---------------------------------------------------------------- lane plumbing

def lane_name(service_root: Path, explicit: str) -> str:
    if explicit:
        return explicit
    name = service_root.name
    return "old" if "baseline" in name.lower() else "new"


def side_of(lane: str) -> str:
    return "old" if "baseline" in lane.lower() or lane.lower() == "old" else "new"


def service_head(service_root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(service_root), *args],
                              capture_output=True, text=True, timeout=20).stdout.strip()

    try:
        return {"commit": git("rev-parse", "--short", "HEAD"),
                "subject": git("log", "-1", "--pretty=%s"),
                "dirty": bool(git("status", "--porcelain"))}
    except Exception as exc:  # noqa: BLE001 — a non-repo root is not a failure
        return {"commit": "", "subject": "", "dirty": None, "error": str(exc)}


def guard_model(base_url: str, expect_model: str) -> dict[str, Any]:
    """Abort before any row runs when the decision model is not reachable."""
    url = base_url.rstrip("/")
    try:
        with urllib.request.urlopen(url + "/health", timeout=10) as resp:
            health = json.loads(resp.read())
        if health.get("status") != "ok":
            raise SystemExit(f"ABORT: decision model unhealthy at {url}: {health}")
        with urllib.request.urlopen(url + "/props", timeout=10) as resp:
            props = json.loads(resp.read())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"ABORT: decision model is not reachable at {url}: {exc} — "
                         f"refusing to run a lane with no model")
    try:
        served, warning = preflight(props, expect_model, source=url + "/props")
    except ModelAssertionError as exc:
        raise SystemExit(f"ABORT: {exc}")
    if warning:
        print(warning, file=sys.stderr)
    return served


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def run_row(idx: int, row: dict[str, Any], cwd: str, use_tool: Any) -> dict[str, Any]:
    utterance = str(row["utterance"])
    label = str(row["label"])
    t0 = time.perf_counter()
    try:
        out = use_tool({"intent": utterance, "digest": {"cwd": cwd}})
        if not isinstance(out, dict):
            out = {"status": "bad_response", "error": f"use_tool returned {type(out).__name__}",
                   "hops": []}
    except Exception as exc:  # noqa: BLE001 — a raising row is data, not a crash
        out = {"status": "exception", "error": f"{type(exc).__name__}: {exc}", "hops": []}
    wall_ms = (time.perf_counter() - t0) * 1000.0
    hops = out.get("hops") or []
    result = out.get("result")
    rec: dict[str, Any] = {
        "idx": idx,
        "utterance": utterance,
        "label": label,
        "band": row.get("band"),
        "tier_target": row.get("tier_target"),
        "status": out.get("status"),
        "degraded": bool(out.get("degraded")),
        "confidence": _num(out.get("confidence")),
        "n_hops": len(hops),
        "hop_ops": [str(h.get("op")) for h in hops],
        "wall_ms": round(wall_ms, 1),
        "svc_wall_ms": _num(out.get("wall_ms")),
        "result_type": "none" if result is None else type(result).__name__,
        "result_len": len(result) if hasattr(result, "__len__") else None,
        "error": out.get("error"),
        "assumptions": [str(a) for a in (out.get("assumptions") or [])],
    }
    if label == GLOB_LABEL:
        rec.update(listing_diagnostic(result, utterance))
    return rec


# ------------------------------------------------------------------ aggregates

def _group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    walls = [r["wall_ms"] for r in rows]
    confs = [r["confidence"] for r in rows if r["confidence"] is not None]
    globs = [r for r in rows if r["label"] == GLOB_LABEL]
    return {
        "n": len(rows),
        "wall_mean": round(statistics.fmean(walls), 1) if walls else None,
        "wall_p50": round(statistics.median(walls), 1) if walls else None,
        "conf_mean": round(statistics.fmean(confs), 4) if confs else None,
        "degraded_rate": round(sum(1 for r in rows if r["degraded"]) / len(rows), 4)
        if rows else None,
        "hops_mean": round(statistics.fmean(r["n_hops"] for r in rows), 3) if rows else None,
        "listing": {
            "n_glob": len(globs),
            "respected": sum(1 for r in globs if r["listing_respected"] is True),
            "violated": sum(1 for r in globs if r["listing_respected"] is False),
            "no_constraint": sum(1 for r in globs if r["listing_constraint"] is None),
            "no_list": sum(1 for r in globs if r["listing_constraint"] is not None
                           and not r["returned_list"]),
            "empty_list": sum(1 for r in globs if r["listing_constraint"] is not None
                              and r["returned_list"] and not r["n_returned"]),
        },
    }


def aggregate(per_row: list[dict[str, Any]]) -> dict[str, Any]:
    def by(key: str) -> dict[str, Any]:
        keys = sorted({str(r.get(key)) for r in per_row})
        return {k: _group([r for r in per_row if str(r.get(key)) == k]) for k in keys}

    return {"by_label": by("label"), "by_band": by("band"), "all": _group(per_row)}


# ------------------------------------------------------------------- reporting

def _pct(rate: float | None) -> str:
    return "   -  " if rate is None else f"{rate:5.1%} "


def _num_cell(value: float | None, width: int = 7, prec: int = 1) -> str:
    return " " * (width - 1) + "-" if value is None else f"{value:>{width}.{prec}f}"


def _listing_cell(group: dict[str, Any]) -> str:
    listing = group["listing"]
    if not listing["n_glob"]:
        return "    -   "
    unjudged = listing["no_constraint"] + listing["no_list"] + listing["empty_list"]
    return f"{listing['respected']}/{listing['violated']}/{unjudged}".ljust(8)


def print_table(title: str, groups: dict[str, dict[str, Any]], lanes: list[str]) -> None:
    hdr = (f"{'':<16}{'lane':<6}{'n':>4}{'hops':>7}{'conf':>8}{'deg':>8}"
           f"{'wall_mean':>10}{'wall_p50':>10}  lst ok/bad/none")
    print(f"\n== {title} ==")
    print(hdr)
    print("-" * len(hdr))
    for key in groups[lanes[0]]:
        for lane in lanes:
            group = groups[lane].get(key)
            if group is None:
                continue
            print(f"{key:<16}{lane:<6}{group['n']:>4}{_num_cell(group['hops_mean'], 7, 2)}"
                  f"{_num_cell(group['conf_mean'], 8, 3)}{_pct(group['degraded_rate']):>8}"
                  f"{_num_cell(group['wall_mean'], 10)}{_num_cell(group['wall_p50'], 10)}"
                  f"  {_listing_cell(group)}")
        print()


def print_lane(data: dict[str, Any]) -> None:
    agg = data["aggregate"]
    served = data.get("model") or {}
    print(f"\nlane={data['lane']} service_root={data['service_root']} "
          f"head={data['service_head'].get('commit')} rows={data['n_rows']}")
    print(f"model={served.get('basename') or served.get('path')} "
          f"({served.get('ftype')}) utterances={Path(data['utterances']).name}")
    print_table("by label", {data["lane"]: agg["by_label"]}, [data["lane"]])
    print_table("by band", {data["lane"]: agg["by_band"]}, [data["lane"]])
    allg = agg["all"]
    print(f"\n== all rows ==")
    print(f"n={allg['n']} hops_mean={allg['hops_mean']} conf_mean={allg['conf_mean']} "
          f"degraded_rate={allg['degraded_rate']} wall_mean={allg['wall_mean']}ms "
          f"wall_p50={allg['wall_p50']}ms "
          f"| listing ok/bad={allg['listing']['respected']}/{allg['listing']['violated']} "
          f"(no constraint {allg['listing']['no_constraint']}, "
          f"no list {allg['listing']['no_list']}, "
          f"empty {allg['listing']['empty_list']})")
    statuses: dict[str, int] = {}
    for row in data["per_row"]:
        statuses[str(row["status"])] = statuses.get(str(row["status"]), 0) + 1
    print("status: " + ", ".join(f"{k}={v}" for k, v in sorted(statuses.items())))


def _plan_key(row: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (tuple(row["hop_ops"]), row["n_hops"], row["status"])


def compare(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    old_rows = {r["idx"]: r for r in old["per_row"]}
    new_rows = {r["idx"]: r for r in new["per_row"]}
    common = sorted(set(old_rows) & set(new_rows))
    plan_changes, conf_moves = [], []
    for idx in common:
        o, n = old_rows[idx], new_rows[idx]
        if _plan_key(o) != _plan_key(n):
            plan_changes.append({
                "idx": idx, "label": n["label"], "band": n["band"],
                "utterance": n["utterance"],
                "old_hop_ops": o["hop_ops"], "new_hop_ops": n["hop_ops"],
                "old_n_hops": o["n_hops"], "new_n_hops": n["n_hops"],
                "old_status": o["status"], "new_status": n["status"],
                "old_conf": o["confidence"], "new_conf": n["confidence"],
            })
        elif o["confidence"] is not None and n["confidence"] is not None \
                and abs(o["confidence"] - n["confidence"]) > 1e-6:
            conf_moves.append({"idx": idx, "label": n["label"], "band": n["band"],
                               "utterance": n["utterance"],
                               "old_conf": o["confidence"], "new_conf": n["confidence"]})

    glob_rows = [idx for idx in common if new_rows[idx]["label"] == GLOB_LABEL]
    hop_pairs = [(old_rows[i]["n_hops"], new_rows[i]["n_hops"]) for i in glob_rows]
    single_hop = [i for i in glob_rows
                  if new_rows[i]["n_hops"] == 1 and "decide_pick" not in new_rows[i]["hop_ops"]]
    dropped_pick = [i for i in glob_rows
                    if "decide_pick" in old_rows[i]["hop_ops"]
                    and "decide_pick" not in new_rows[i]["hop_ops"]]
    quality = {
        "old_violated": [i for i in glob_rows if old_rows[i].get("listing_respected") is False],
        "new_violated": [i for i in glob_rows if new_rows[i].get("listing_respected") is False],
        "new_respected": [i for i in glob_rows if new_rows[i].get("listing_respected") is True],
        "new_unjudged": [i for i in glob_rows if new_rows[i].get("listing_respected") is None],
        "old_returned_list": [i for i in glob_rows if old_rows[i].get("returned_list")],
        "new_returned_list": [i for i in glob_rows if new_rows[i].get("returned_list")],
        "rows": [{"idx": i, "utterance": new_rows[i]["utterance"],
                  "constraint": new_rows[i].get("listing_constraint"),
                  "old_n_returned": old_rows[i].get("n_returned"),
                  "new_n_returned": new_rows[i].get("n_returned"),
                  "old_respected": old_rows[i].get("listing_respected"),
                  "new_respected": new_rows[i].get("listing_respected"),
                  "new_violations": new_rows[i].get("n_violations")}
                 for i in glob_rows],
    }
    return {
        "n_common": len(common),
        "plan_changes": plan_changes,
        "collateral": [c for c in plan_changes if c["label"] != GLOB_LABEL],
        "expected_changes": [c for c in plan_changes if c["label"] == GLOB_LABEL],
        "conf_moves": conf_moves,
        "glob_rows": glob_rows,
        "glob_hops": {
            "old_mean": round(statistics.fmean(a for a, _ in hop_pairs), 3) if hop_pairs else None,
            "new_mean": round(statistics.fmean(b for _, b in hop_pairs), 3) if hop_pairs else None,
            "old_sum": sum(a for a, _ in hop_pairs),
            "new_sum": sum(b for _, b in hop_pairs),
            "old_wall_mean": round(statistics.fmean(old_rows[i]["wall_ms"] for i in glob_rows), 1)
            if glob_rows else None,
            "new_wall_mean": round(statistics.fmean(new_rows[i]["wall_ms"] for i in glob_rows), 1)
            if glob_rows else None,
            "old_wall_p50": round(statistics.median(old_rows[i]["wall_ms"] for i in glob_rows), 1)
            if glob_rows else None,
            "new_wall_p50": round(statistics.median(new_rows[i]["wall_ms"] for i in glob_rows), 1)
            if glob_rows else None,
            "single_hop_new": single_hop,
            "decide_pick_dropped": dropped_pick,
            "old_conf_mean": round(statistics.fmean(
                old_rows[i]["confidence"] for i in glob_rows
                if old_rows[i]["confidence"] is not None), 4) if glob_rows else None,
            "new_conf_mean": round(statistics.fmean(
                new_rows[i]["confidence"] for i in glob_rows
                if new_rows[i]["confidence"] is not None), 4) if glob_rows else None,
        },
        "quality": quality,
    }


def print_compare(old: dict[str, Any], new: dict[str, Any]) -> None:
    old_lane = old["lane"] if side_of(old["lane"]) == "old" else new["lane"]
    new_lane = new["lane"] if side_of(new["lane"]) == "new" else old["lane"]
    if old_lane == new_lane:  # both lanes recorded the same label
        old_lane, new_lane = f"{old_lane}-old", f"{new_lane}-new"
    sides = {old_lane: old, new_lane: new}
    lanes = [old_lane, new_lane]

    print(f"\n{'=' * 78}")
    print(f"A/B: old={old_lane} ({old['service_head'].get('commit')}) "
          f"vs new={new_lane} ({new['service_head'].get('commit')})")
    print("=" * 78)
    print_table("by label", {ln: sides[ln]["aggregate"]["by_label"] for ln in lanes}, lanes)
    print_table("by band", {ln: sides[ln]["aggregate"]["by_band"] for ln in lanes}, lanes)
    for ln in lanes:
        allg = sides[ln]["aggregate"]["all"]
        print(f"== all: {ln:<6} n={allg['n']} hops_mean={allg['hops_mean']} "
              f"conf_mean={allg['conf_mean']} degraded_rate={allg['degraded_rate']} "
              f"wall_mean={allg['wall_mean']}ms wall_p50={allg['wall_p50']}ms "
              f"listing ok/bad={allg['listing']['respected']}/{allg['listing']['violated']}")
    print()

    report = compare(old, new)
    print("-- A. SURGICAL: plan changes outside the filesys_glob label --")
    print(f"   rows compared: {report['n_common']}  "
          f"filesys_glob plan changes: {len(report['expected_changes'])}  "
          f"collateral plan changes: {len(report['collateral'])}")
    if report["collateral"]:
        for c in report["collateral"]:
            print(f"   {c['idx']:>3} {c['label']:<14} {c['band']:<3} "
                  f"{c['old_n_hops']}->{c['new_n_hops']} hops  "
                  f"{'->'.join(c['old_hop_ops']) or '-'} => {'->'.join(c['new_hop_ops']) or '-'}"
                  f"  status {c['old_status']}->{c['new_status']}")
            print(f"       {c['utterance']}")
    else:
        print("   none — every changed plan sits on a filesys_glob-labeled row")
    if report["conf_moves"]:
        print(f"   confidence moved without a plan change on {len(report['conf_moves'])} row(s): "
              + ", ".join(f"{m['idx']}({m['label']} {m['old_conf']:.2f}->{m['new_conf']:.2f})"
                          for m in report["conf_moves"][:8])
              + (" ..." if len(report["conf_moves"]) > 8 else ""))

    hops = report["glob_hops"]
    print("\n-- B. WIN on filesys_glob rows --")
    print(f"   hops  old mean={hops['old_mean']} (sum {hops['old_sum']})  "
          f"new mean={hops['new_mean']} (sum {hops['new_sum']})")
    print(f"   wall  old mean={hops['old_wall_mean']}ms p50={hops['old_wall_p50']}ms  "
          f"new mean={hops['new_wall_mean']}ms p50={hops['new_wall_p50']}ms   [indicative only]")
    print(f"   conf  old mean={hops['old_conf_mean']}  new mean={hops['new_conf_mean']}")
    print(f"   single hop, no decide_pick, new: {len(hops['single_hop_new'])}/"
          f"{len(report['glob_rows'])} rows {hops['single_hop_new']}")
    print(f"   decide_pick dropped (old had it, new does not): "
          f"{len(hops['decide_pick_dropped'])} rows {hops['decide_pick_dropped']}")

    quality = report["quality"]
    print("\n-- C. QUALITY GAP: constraint stated vs constraint respected --")
    print(f"   old violated {len(quality['old_violated'])}  "
          f"new violated {len(quality['new_violated'])}  "
          f"new respected {len(quality['new_respected'])}  "
          f"new unjudged {len(quality['new_unjudged'])} of {len(report['glob_rows'])} glob rows")
    print(f"   rows returning a path list: old {len(quality['old_returned_list'])}  "
          f"new {len(quality['new_returned_list'])}")
    print(f"   new rows violating the named constraint: {quality['new_violated']}")
    print(f"\n   {'idx':>4} {'constraint':<16}{'old n':>7}{'new n':>7}  old/new respected   utterance")
    for row in quality["rows"]:
        c = row["constraint"]
        shown = "-" if c is None else f"{c['kind']}={c['value']}"
        print(f"   {row['idx']:>4} {shown:<16}{str(row['old_n_returned']):>7}"
              f"{str(row['new_n_returned']):>7}  "
              f"{str(row['old_respected']):<5}/{str(row['new_respected']):<5} "
              f"{row['utterance'][:44]}")


# ------------------------------------------------------------------------- cli

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--service-root", default=str(ROOT),
                    help="repo whose service/ is imported as `ti` (sys.path[0]); "
                         "default: the repo this script lives in")
    ap.add_argument("--utterances", default=str(DEFAULT_UTTERANCES))
    ap.add_argument("--limit", type=int, default=0, help="run only the first N rows (0 = all)")
    ap.add_argument("--only-idx", default="",
                    help="comma-separated row indices to run (a targeted re-check)")
    ap.add_argument("--cwd", default="",
                    help="digest cwd every row is run against (default: the service root) — "
                         "set it to another tree to hold the filesystem frame constant")
    ap.add_argument("--out", default="",
                    help="result JSON path (default bench/results/ab-probe-<lane>-<date>.json); "
                         "an existing file is never overwritten")
    ap.add_argument("--lane", default="", help="lane label recorded in the JSON")
    ap.add_argument("--compare", default="", help="another lane's JSON to report against")
    ap.add_argument("--report-only", action="store_true",
                    help="skip the run; print the cross-lane report from --out and --compare")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL,
                    help="decision model the lane asserts before running")
    ap.add_argument("--expect-model", default="",
                    help="gguf basename /props must serve; a mismatch aborts before any row")
    return ap


def free_path(path: Path) -> Path:
    """`path`, or the first `-2`, `-3`, ... variant that does not exist yet."""
    if not path.exists():
        return path
    for n in range(2, 1000):
        candidate = path.with_name(f"{path.stem}-{n}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise SystemExit(f"refusing to overwrite {path} and no free suffix below 1000")


def main() -> None:
    args = build_parser().parse_args()
    service_root = Path(args.service_root).expanduser().resolve()

    if args.report_only:
        if not (args.out and args.compare):
            raise SystemExit("--report-only needs --out (this lane) and --compare (the other)")
        print_compare(json.loads(Path(args.compare).read_text()),
                      json.loads(Path(args.out).read_text()))
        return

    if not (service_root / "service" / "ti").is_dir():
        raise SystemExit(f"ABORT: {service_root} has no service/ti — not a library root")

    sys.path.insert(0, str(service_root / "service"))
    from ti import service_entry  # noqa: E402  — after sys.path[0] is the lane root
    from ti.conf import load as load_conf  # noqa: E402

    served = guard_model(args.base_url, args.expect_model)
    configured = str(load_conf().get("decide", {}).get("d1_systemone", {}).get("base_url", ""))
    if configured and configured.rstrip("/") != args.base_url.rstrip("/"):
        print(f"note: conf serves decide from {configured}, asserted {args.base_url}",
              file=sys.stderr)

    indexed = list(enumerate(json.loads(Path(args.utterances).read_text())))
    if args.limit:
        indexed = indexed[: args.limit]
    if args.only_idx:
        wanted = {int(t) for t in args.only_idx.replace(" ", "").split(",") if t}
        indexed = [(idx, row) for idx, row in indexed if idx in wanted]
    cwd = args.cwd or str(service_root)
    lane = lane_name(service_root, args.lane)

    print(f"lane={lane} service_root={service_root} cwd={cwd} rows={len(indexed)} "
          f"model={served.get('basename')} base_url={args.base_url}", file=sys.stderr)
    per_row: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    for idx, row in indexed:
        rec = run_row(idx, row, cwd, service_entry.use_tool)
        per_row.append(rec)
        conf = "  -  " if rec["confidence"] is None else f"{rec['confidence']:.3f}"
        print(f"[{len(per_row):>3}/{len(indexed)}] {rec['idx']:>3} {rec['label']:<14} "
              f"{str(rec['status']):<9} hops={rec['n_hops']} conf={conf} "
              f"{rec['wall_ms']:>8.0f}ms  {rec['utterance'][:46]}", file=sys.stderr)

    out_path = Path(args.out) if args.out else \
        DEFAULT_RESULTS / f"ab-probe-{lane}-{time.strftime('%Y%m%d')}.json"
    out_path = free_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "harness": "bench/ab_probe.py",
        "lane": lane,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "service_root": str(service_root),
        "service_head": service_head(service_root),
        "model": served,
        "base_url": args.base_url,
        "cwd": cwd,
        "utterances": str(args.utterances),
        "n_rows": len(per_row),
        "run_seconds": round(time.perf_counter() - t_start, 1),
        "in_process": True,
        "aggregate": aggregate(per_row),
        "per_row": per_row,
    }
    out_path.write_text(json.dumps(_portable(document), indent=2) + "\n")
    print(f"\nwrote {out_path}", file=sys.stderr)

    print_lane(document)
    if args.compare:
        print_compare(json.loads(Path(args.compare).read_text()), document)


if __name__ == "__main__":
    main()
