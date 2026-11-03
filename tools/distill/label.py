#!/usr/bin/env python3
"""label — attach teacher targets to candidate rows, resumably and under a budget.

Four rules shape this file, and each one is a way the corpus could be ruined:

**A row is written only if it is schema-valid.** The teacher's reply has to name one
of the offered candidates and a confidence in 0..1. Anything else — a prose answer, a
truncated object, an invented tool name — is counted and dropped, never repaired.
A guessed target is worse than a missing row: the missing row shows up in the count.

**A 429 is not a row.** The backend already retries 429/5xx with backoff; a call that
still fails after that is *deferred*, not recorded, and one deferred pass runs at the
end of the run. A row that never succeeds stays absent, so `--resume` picks it up.

**Money is bounded.** The first ``--gate`` rows (stratified, so the sample has one of
each band) are labelled, the measured spend is projected across the remainder, and if
the projection exceeds ``--budget`` the run stops there with exit 3. Measuring 20 and
then extrapolating is cheaper than discovering the price at row 400.

**The eval set is off-limits.** Every row is re-checked against the frozen
150-utterance set immediately before it is written, so a hand-edited candidate file
cannot smuggle eval content in. That check runs here as well as at generation
because this is the file the training step reads.

Persist-incrementally is the same rule as resumable: each row is appended and fsynced
as it arrives (the run's value is upfront token spend, so a crash must not take it),
and ``--resume`` skips every row id already present after validating it.

    export TI_OPENROUTER_API_KEY="$(cat $TI_OPENROUTER_KEY_FILE)"
    python3 tools/distill/label.py --in var/distill/candidates.jsonl \\
        --out var/distill/labels.jsonl --resume --gate 20 --budget 2.0
    python3 tools/distill/label.py ... --dry-run        # fixture teacher, no key
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import random
import sys
import time
from typing import Any

if __package__ in (None, ""):  # run as a script: put tools/ on the path for `distill.*`
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from distill import catalog, corpus, cost, teacher  # noqa: E402

STOP_BUDGET = 3
DEFAULT_BUDGET_USD = 2.0


class _Lock:
    """One writer per output file.

    Two labellers appending to the same file interleave rows silently and leave a
    corpus with duplicate ids and no way to tell which run wrote what — and the run
    that created it is the one that spends money. The lock is a pidfile, O_EXCL:
    if it exists and the pid is alive the second run refuses; a lock left by a dead
    process is stale and is taken over.
    """

    def __init__(self, out_path: pathlib.Path) -> None:
        self.path = out_path.with_suffix(".lock")
        self.held = False

    def __enter__(self) -> "_Lock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                pid = corpus.read_json(self.path, {}) or {}
                other = int(pid.get("pid") or 0) if isinstance(pid, dict) else 0
                if other and _alive(other):
                    raise SystemExit(
                        f"{self.path.name}: another label run (pid {other}) is writing "
                        f"{self.path.with_suffix('')} — wait for it or stop it first") from None
                self.path.unlink(missing_ok=True)  # stale lock from a dead run
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"pid": os.getpid(), "started_at": corpus.now()}, fh)
            self.held = True
            return self
        raise SystemExit(f"{self.path.name}: could not take the lock")

    def __exit__(self, *exc: Any) -> None:
        if self.held:
            self.path.unlink(missing_ok=True)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _input_sha(rows: list[dict[str, Any]]) -> str:
    blob = "\n".join(f"{r['id']}\t{r['utterance']}" for r in rows)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _prompt_chars(row: dict[str, Any]) -> int:
    return (len(str(row.get("state", ""))) + len(str(row.get("instructions", "")))
            + sum(len(k) + len(v) for k, v in row.get("candidates", {}).items()))


def _label_one(teacher_impl: Any, row: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    """One unlabelled row in, one labelled row out. Raises TeacherError when unusable."""
    answer = teacher_impl.choose(row["state"], row["instructions"], row["candidates"])
    usd, method = (0.0, cost.FIXTURE) if dry_run else cost.row_cost(
        answer.usage, _prompt_chars(row), len(answer.raw))
    labelled = dict(row)
    labelled["target"] = answer.target()
    labelled["teacher"] = answer.teacher()
    labelled["cost_usd"] = round(usd, 8)
    labelled["label_method"] = method
    labelled["ts"] = corpus.now()
    corpus.assert_row(labelled)
    return labelled


def _stratified(rows: list[dict[str, Any]], count: int, seed: str) -> list[dict[str, Any]]:
    """`count` rows spread across bands, then across intended ops, deterministically."""
    if count >= len(rows):
        return list(rows)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault((row["band"], str(row["intended"])), []).append(row)
    rng = random.Random(seed)
    order = sorted(buckets)
    rng.shuffle(order)
    picked: list[dict[str, Any]] = []
    while len(picked) < count and any(buckets[k] for k in order):
        for key in order:
            if buckets[key] and len(picked) < count:
                picked.append(buckets[key].pop(0))
    return picked


def _interleaved(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Round-robin across (band, op) pairs, corpus order inside each.

    Row ids are allocated band by band and op by op, so labelling in file order and
    stopping early — a --limit, a crash, a budget or wall-clock stop — would leave a
    checkpoint holding the alphabetically early ops of one band. Interleaving over
    the pairs means a partial run is balanced on both axes, so a corpus sliced at
    row 200 is still a corpus you can compute a per-tool number on (03 §3).
    """
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault((row["band"], str(row["intended"])), []).append(row)
    keys = sorted(buckets)
    out: list[dict[str, Any]] = []
    while any(buckets[k] for k in keys):
        for key in keys:
            if buckets[key]:
                out.append(buckets[key].pop(0))
    return out


def _report_gate(projection: dict[str, Any], budget: float, budget_note: str) -> None:
    print("-- cost gate " + "-" * 48)
    print(f"  labelled so far  {projection['labelled']} rows, "
          f"${projection['spent_usd']:.6f} ({budget_note})")
    print(f"  per row          ${projection['per_row_usd']:.8f}")
    print(f"  remaining        {projection['remaining']} rows")
    print(f"  projected total  ${projection['projected_total_usd']:.4f} "
          f"(rest ${projection['projected_rest_usd']:.4f})")
    print(f"  budget           ${budget:.2f}")
    verdict = ("STOP — projection exceeds the budget; rows so far are kept and "
               "--resume continues after a budget decision"
               if projection["projected_total_usd"] > budget else
               "PROCEED — projection is inside the budget")
    print(f"  verdict          {verdict}")
    print("-" * 62)


def label_all(*, rows: list[dict[str, Any]], out_path: pathlib.Path,
              dry_run: bool, gate: int, budget: float, min_interval: float,
              limit: int | None = None, resume: bool = True,
              seed: str = "distill", max_seconds: float | None = None,
              progress: int = 0) -> dict[str, Any]:
    """Label in order, returning the run report. Appends one fsynced row at a time.

    ``max_seconds`` bounds the wall clock, which on a slow upstream is the real
    budget: a pinned provider can answer in one second or in one minute, and a run
    that cannot be stopped cleanly cannot be resumed cheaply either. Hitting the cap
    leaves the checkpoint at whatever has been written, which is exactly what
    ``--resume`` continues from.
    """
    reference = corpus.frozen()
    stale = sorted({str(r.get("catalog_sha")) for r in rows} -
                   {catalog.catalog_sha()})
    if stale:
        # 01 §1: a candidate block is rendered against one catalog version, and
        # mixed versions silently teach two formats. Refuse rather than relabel.
        raise SystemExit(
            f"candidate rows were built against catalog {stale}, this tree is "
            f"{catalog.catalog_sha()} — re-run utterances.py to re-derive them")
    for row in rows:  # both guards run on every input row, resumed or not
        corpus.assert_row(row)
        if row["target"] is None:
            corpus.assert_not_frozen(row["utterance"], reference)

    meta_path = out_path.with_suffix(".meta.json")
    sha = _input_sha(rows)
    done: dict[str, dict[str, Any]] = {}
    if out_path.exists():
        if not resume:
            raise SystemExit(f"{out_path} exists — pass --resume to continue it, or "
                             "pick another --out")
        for row in corpus.read_jsonl(out_path):
            corpus.assert_labelled(row)
            corpus.assert_not_frozen(row["utterance"], reference)
            done[row["id"]] = row
        meta = corpus.read_json(meta_path, {}) or {}
        if meta.get("input_sha") and meta["input_sha"] != sha:
            raise SystemExit(
                f"{out_path} was labelled from a different candidate set "
                f"({meta['input_sha']} != {sha}) — a resume would mix two corpora; "
                "start a new --out or pass a matching --in")
    corpus.write_json(meta_path, {**(corpus.read_json(meta_path, {}) or {}),
                                  "input_sha": sha, "input_rows": len(rows),
                                  "started_at": (corpus.read_json(meta_path, {}) or {}).get(
                                      "started_at") or corpus.now(),
                                  "last_run_at": corpus.now(),
                                  "teacher": "fixture" if dry_run else "openrouter"})

    pending = [r for r in rows if r["id"] not in done]
    pending = _interleaved(pending)
    if limit is not None:
        pending = pending[:limit]

    impl = teacher.make_teacher(dry_run)
    run: dict[str, Any] = {"labelled": 0, "refused": 0, "refused_reasons": {},
                           "deferred": [], "gate": None, "stopped": False,
                           "timed_out": False, "resumed_rows": len(done)}
    started = time.time()

    def out_of_time() -> bool:
        return max_seconds is not None and (time.time() - started) >= max_seconds

    def attempt(row: dict[str, Any]) -> bool:
        try:
            labelled = _label_one(impl, row, dry_run)
        except teacher.TeacherError as exc:
            key = "rate-limited" if teacher.is_transient(exc) else "schema-invalid"
            run["refused"] += 1
            run["refused_reasons"][key] = run["refused_reasons"].get(key, 0) + 1
            if key == "rate-limited":
                run["deferred"].append(row)
            else:
                corpus.append_jsonl(out_path.with_name("rejects.jsonl"),
                                    {"id": row["id"], "utterance": row["utterance"],
                                     "ts": corpus.now(), "reason": str(exc)[:400]})
            return False
        corpus.append_jsonl(out_path, labelled)
        run["labelled"] += 1
        if progress and run["labelled"] % progress == 0:
            rate = run["labelled"] / max(time.time() - started, 1e-9)
            print(f"  ... {run['labelled']} labelled, "
                  f"{run['refused']} refused, {len(run['deferred'])} deferred, "
                  f"{rate:.2f} rows/s", flush=True)
        if min_interval and not dry_run:
            time.sleep(min_interval)
        return True

    def finish() -> dict[str, Any]:
        """Totals over the whole checkpoint, however the run ended."""
        rows_now = corpus.read_jsonl(out_path)
        run["total_rows"] = len(rows_now)
        run["cost"] = cost.summarise(rows_now)
        run["deferred"] = [r["id"] for r in run["deferred"]]
        run["elapsed_s"] = round(time.time() - started, 1)
        return run

    # 1) the gate: label a stratified slice, then size the whole run against it
    gated: list[dict[str, Any]] = []
    if gate > 0 and pending:
        gated = _stratified(pending, gate, seed)
        gated_ids = {r["id"] for r in gated}
        for row in gated:
            if out_of_time():
                run["timed_out"] = True
                break
            attempt(row)
        done_rows = corpus.read_jsonl(out_path)
        projection = cost.project(
            labelled=len(done_rows), spent=cost.summarise(done_rows)["usd"],
            remaining=len([r for r in pending if r["id"] not in gated_ids]))
        summary = cost.summarise(done_rows)
        run["gate"] = {**projection, "method": summary["method"],
                       "budget": budget, "note": cost.price_note()}
        _report_gate(projection, budget, summary["method"])
        if projection["projected_total_usd"] > budget:
            run["stopped"] = True
            return finish()
        pending = [r for r in pending if r["id"] not in gated_ids]

    # 2) the remainder, in corpus order
    for row in pending:
        if out_of_time():
            run["timed_out"] = True
            break
        attempt(row)

    # 3) one deferred pass: a 429 that survived the backend's own retries gets a
    #    slower second chance before it is left for --resume
    if run["deferred"] and not run["timed_out"]:
        if min_interval:
            time.sleep(min(min_interval * 4, 5.0))
        run["deferred_retried"] = len(run["deferred"])
        retry, run["deferred"] = run["deferred"], []
        for row in retry:
            attempt(row)

    return finish()


def _print_run(run: dict[str, Any], out_path: pathlib.Path) -> None:
    print(f"labelled {run['labelled']} this run -> {out_path} "
          f"({run['total_rows']} rows total, {run['resumed_rows']} resumed)")
    if run["refused"]:
        print(f"  refused   {run['refused']} {run['refused_reasons']} "
              "(never recorded — a 429 is not a row)")
    if run["deferred"]:
        print(f"  deferred  {len(run['deferred'])} rows still failing; --resume retries them")
    if run.get("timed_out"):
        print(f"  timed out at --max-seconds; {len(corpus.read_jsonl(out_path))} rows on "
              "disk — --resume continues from there")
    summary = run.get("cost") or {}
    print(f"  cost      ${summary.get('usd', 0.0):.6f} over {summary.get('rows', 0)} rows "
          f"({summary.get('method', 'none')})")
    print(f"  elapsed   {run['elapsed_s']}s")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="source", default=str(corpus.OUT_DIR / "candidates.jsonl"),
                    help="candidate rows (JSONL) from utterances.py")
    ap.add_argument("--out", dest="target", default=str(corpus.OUT_DIR / "labels.jsonl"),
                    help="labelled rows (JSONL), appended one per row")
    ap.add_argument("--resume", action="store_true",
                    help="continue an existing --out, skipping rows already labelled")
    ap.add_argument("--dry-run", action="store_true",
                    help="fixture teacher: no network, no key, no cost")
    ap.add_argument("--gate", type=int, default=20,
                    help="rows to label before projecting the full run (0 disables)")
    ap.add_argument("--budget", type=float, default=DEFAULT_BUDGET_USD,
                    help="USD ceiling for the projected full run")
    ap.add_argument("--limit", type=int, default=None, help="label at most N rows")
    ap.add_argument("--min-interval", type=float, default=0.2,
                    help="seconds between calls (be a good citizen)")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="wall-clock cap; the checkpoint stays resumable")
    ap.add_argument("--progress", type=int, default=25,
                    help="print a progress line every N rows (0 disables)")
    args = ap.parse_args(argv)

    src = pathlib.Path(args.source).expanduser()
    if not src.exists():
        raise SystemExit(f"no candidate rows at {src} — run utterances.py first")
    rows = corpus.read_jsonl(src)

    out_path = pathlib.Path(args.target).expanduser()
    with _Lock(out_path):
        run = label_all(rows=rows, out_path=out_path,
                        dry_run=args.dry_run, gate=args.gate, budget=args.budget,
                        min_interval=args.min_interval, limit=args.limit,
                        resume=args.resume, max_seconds=args.max_seconds,
                        progress=args.progress)
    _print_run(run, out_path)
    return STOP_BUDGET if run.get("stopped") else 0


if __name__ == "__main__":
    raise SystemExit(main())
