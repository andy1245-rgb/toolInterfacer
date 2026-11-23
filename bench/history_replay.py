#!/usr/bin/env python3
"""history_replay — real routing requests from the pi session corpus, replayed offline.

The shadow has no traffic because nothing routes through the service: the operator's assistant does
not ask this repository which tool to use, it just uses one. So the real traffic exists, but as a
transcript rather than as requests to the router. This reads those transcripts, pairs each user
request with the tool the agent actually called and how that call ended, and asks the shadow's own
logic what it would have picked.

**Collection only.** Nothing is executed, no tool is run, no model is called, nothing is written
except aggregate counts. The corpus is the operator's real traffic, so by the data rule
(`docs/harness-spec-2026-11-10.md`) only counts and rates leave this script — never a request, never
a tool argument, never a path.

    python3 bench/history_replay.py                       # default corpus, aggregates to stdout
    python3 bench/history_replay.py --max-sessions 200 --json /tmp/replay.json

What a "pair" is, and what it is not:

    pair    = the most recent user message before an assistant tool call, the tool that call
              invoked, and the tool result's own `isError` flag.
    not     = a routing decision. The agent may have had context the text does not carry — it had
              just read a file, or the previous call failed. The comparison below is therefore
              "would a text classifier, given the user's request and the session's tool set, have
              picked the tool the agent picked", which is a lower bound on a router's difficulty
              and not a measurement of one.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = Path("~/.pi/agent/sessions").expanduser()
MAX_REQUEST_CHARS = 600
# Agent-management tools are routing noise: they are called because the conversation is about the
# agent, not because the request is a file or web task.
META_TOOLS = {"forge_run", "forge_result", "forge_resume", "todo", "note", "board", "ask",
              "job_start", "job_wait", "job_status", "job_kill", "job_list", "job_tail",
              "profile_info", "profile_skills", "herdr_name", "useTool", "look_at",
              "oh_my_pi_subagent", "oh_my_pi_delegate_task", "interactive_bash"}


@dataclass(frozen=True)
class Pair:
    session: str
    request: str
    tool: str
    is_error: bool
    duration_ms: float


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    out = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            out.append(str(part.get("text") or ""))
    return " ".join(out)


def pairs_from_session(path: Path) -> list[Pair]:
    """One pass over a transcript: user text -> tool call -> tool result."""
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (json.JSONDecodeError, OSError):
        return []
    results: dict[str, dict] = {}
    for row in rows:
        if row.get("type") != "message":
            continue
        message = row.get("message") or {}
        if message.get("role") == "toolResult":
            results[str(message.get("toolCallId") or "")] = message

    out: list[Pair] = []
    pending_user = ""
    for row in rows:
        if row.get("type") != "message":
            continue
        message = row.get("message") or {}
        role = message.get("role")
        if role == "user":
            text = _text_of(message.get("content")).strip()
            if text:
                pending_user = text[:MAX_REQUEST_CHARS]
            continue
        if role != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "toolCall":
                continue
            result = results.get(str(part.get("id") or ""))
            if result is None or not pending_user:
                continue
            out.append(Pair(session=path.stem, request=pending_user,
                            tool=str(part.get("name") or ""),
                            is_error=bool(result.get("isError")),
                            duration_ms=float(result.get("durationMs") or 0.0)))
    return out


def walk(corpus: Path, max_sessions: int, max_pairs: int) -> tuple[list[Pair], int]:
    pairs: list[Pair] = []
    seen = 0
    for path in sorted(corpus.rglob("*.jsonl")):
        seen += 1
        if seen > max_sessions or len(pairs) >= max_pairs:
            break
        pairs.extend(pairs_from_session(path))
    return pairs[:max_pairs], seen


def report(pairs: list[Pair], sessions: int, top: int = 12) -> dict:
    tools = Counter(p.tool for p in pairs)
    errors = Counter(p.tool for p in pairs if p.is_error)
    per_session = defaultdict(int)
    for p in pairs:
        per_session[p.session] += 1
    by_tool = {}
    for tool, n in tools.most_common(top):
        durations = sorted(p.duration_ms for p in pairs if p.tool == tool)
        by_tool[tool] = {
            "pairs": n,
            "error_rate": round(errors[tool] / n, 3) if n else 0.0,
            "p50_ms": round(durations[len(durations) // 2], 1) if durations else None,
            "share": round(n / len(pairs), 3) if pairs else 0.0,
        }
    majority_tool, majority_n = tools.most_common(1)[0] if tools else ("", 0)
    return {
        "majority_tool": majority_tool,
        "majority_baseline": round(majority_n / len(pairs), 3) if pairs else 0.0,
        "sessions_scanned": sessions,
        "pairs": len(pairs),
        "sessions_with_pairs": len(per_session),
        "distinct_tools": len(tools),
        "error_rate": round(sum(1 for p in pairs if p.is_error) / len(pairs), 3) if pairs else 0.0,
        "tools": by_tool,
        "meta_tool_share": round(sum(1 for p in pairs if p.tool in META_TOOLS) / len(pairs), 3)
                          if pairs else 0.0,
    }


def shadow_comparison(pairs: list[Pair]) -> dict:
    """What the shipped shadow member would have said, on this vocabulary.

    The shipped member is the eight-op classical router. The corpus's tool vocabulary is whatever
    the assistant uses. This reports the overlap per session — which is the same `vocab_match`
    field the live shadow log records — and the answer rate that follows from it.
    """
    sys.path.insert(0, str(ROOT / "service"))
    from ti.decide.backends.classical import ClassicalBackend  # noqa: PLC0415

    backend = ClassicalBackend()
    fitted = set(backend.fitted_labels())
    by_session: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        by_session[pair.session].add(pair.tool)

    zero_overlap = 0
    answered = agreed = 0
    for row in pairs:
        options = sorted(by_session[row.session])
        if not options:
            continue
        if not (fitted & set(options)):
            zero_overlap += 1
            continue
        answer = backend.choice(row.request, "", {o: o for o in options})
        if getattr(answer, "choice", None) is None:
            continue
        answered += 1
        agreed += 1 if answer.choice == row.tool else 0
    return {
        "member": "classical (the shipped eight-op router)",
        "fitted_labels": sorted(fitted),
        "pairs_with_no_vocabulary_overlap": zero_overlap,
        "pairs_answered": answered,
        "answer_rate": round(answered / len(pairs), 3) if pairs else 0.0,
        "agreement_when_answered": round(agreed / answered, 3) if answered else None,
    }


def vocabulary_matched_member(pairs: list[Pair], folds: int = 5) -> dict:
    """A member fitted on THIS corpus, scored on sessions it never saw.

    Folds are by session, because two requests in one session share context and a split that lets
    them straddle the fold boundary would measure how well the model memorised a conversation.
    """
    try:
        sys.path.insert(0, str(ROOT))
        from bench.harness import metrics
        from bench.harness.protocol import build
        from bench.harness.rows import Row
    except Exception as exc:  # noqa: BLE001 - scikit-learn may be absent
        return {"skipped": f"{type(exc).__name__}: {exc}"}

    by_tool = Counter(p.tool for p in pairs)
    usable_tools = {tool for tool, n in by_tool.items() if n >= 5}
    keep = [p for p in pairs if p.tool in usable_tools]
    if not keep:
        return {"skipped": "no tool has five examples"}
    options = tuple(sorted({p.tool for p in keep}))
    rows = [Row(id=f"h{i}", text=p.request, label=p.tool, group=p.session, options=options,
                meta={"legacy_tool": p.tool}) for i, p in enumerate(keep)]
    sessions = len({r.group for r in rows})
    if sessions < folds:
        return {"skipped": f"{sessions} sessions is too few for {folds} folds"}

    spec = "sklearn_text|vectorizer=tfidf_word_char|clf=logreg"
    cv = metrics.grouped_cv(rows, (lambda: build(spec)), folds=folds, seed=0, stratify=True)
    acc = cv["oof_accuracy"]
    cap = metrics.coverage_at_precision(cv["oof"], 0.9)
    return {
        "member": spec,
        "options": list(options),
        "pairs": len(rows),
        "sessions": sessions,
        "held_out_session_accuracy": round(acc["accuracy"], 3),
        "wilson95": [round(x, 3) for x in acc["wilson95"]],
        "coverage_at_precision90": round(cap["coverage"], 3) if cap["coverage"] is not None else None,
        "per_fold": [round(f["accuracy"], 3) if f["accuracy"] is not None else None
                     for f in cv["per_fold"]],
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sessions", default=str(DEFAULT_CORPUS))
    ap.add_argument("--max-sessions", type=int, default=400)
    ap.add_argument("--max-pairs", type=int, default=8000)
    ap.add_argument("--json", default="", help="write the aggregate report here")
    ap.add_argument("--no-models", action="store_true", help="counts only, skip both comparisons")
    args = ap.parse_args(argv)

    corpus = Path(args.sessions).expanduser()
    if not corpus.exists():
        raise SystemExit(f"{corpus} does not exist")
    pairs, scanned = walk(corpus, args.max_sessions, args.max_pairs)
    if not pairs:
        raise SystemExit("no usable request/outcome pairs found")
    out = report(pairs, scanned)
    print(f"corpus {corpus}\nfiles scanned {scanned}  pairs {len(pairs)}  "
          f"sessions with pairs {out['sessions_with_pairs']}  tools {out['distinct_tools']}  "
          f"error rate {100 * out['error_rate']:.1f}%  meta-tool share "
          f"{100 * out['meta_tool_share']:.1f}%\n")
    print(f"{'tool':<22}{'pairs':>7}{'share':>8}{'err%':>7}{'p50 ms':>9}")
    for tool, row in out["tools"].items():
        print(f"{tool:<22}{row['pairs']:>7}{100 * row['share']:>7.1f}%"
              f"{100 * row['error_rate']:>7.1f}{row['p50_ms'] if row['p50_ms'] is not None else '-':>9}")
    if not args.no_models:
        out["shipped_member"] = shadow_comparison(pairs)
        out["vocabulary_matched"] = vocabulary_matched_member(pairs)
        shipped = out["shipped_member"]
        print(f"\nthe shipped shadow member ({shipped['member']}):")
        print(f"  pairs where its eight ops overlap the session's tools at all: "
              f"{len(pairs) - shipped['pairs_with_no_vocabulary_overlap']}/{len(pairs)}")
        print(f"  answered {shipped['pairs_answered']}  "
              f"({100 * shipped['answer_rate']:.1f}% of pairs)"
              + (f"  agreement when answered {100 * shipped['agreement_when_answered']:.1f}%"
                 if shipped["agreement_when_answered"] is not None else ""))
        matched = out["vocabulary_matched"]
        if "skipped" in matched:
            print(f"\na member fitted on this vocabulary: skipped ({matched['skipped']})")
        else:
            print(f"\na member fitted on this corpus ({matched['member']}), "
                  f"held out by session:")
            print(f"  {matched['options']}")
            print(f"  {matched['pairs']} pairs over {matched['sessions']} sessions, "
                  f"accuracy {100 * matched['held_out_session_accuracy']:.1f}% "
                  f"{matched['wilson95']}, per fold {matched['per_fold']}")
            if matched["coverage_at_precision90"] is not None:
                print(f"  coverage at 90% precision: {100 * matched['coverage_at_precision90']:.1f}%")
            base = out["majority_baseline"]
            verdict = ("worse than always answering %r" % out["majority_tool"]
                       if matched["held_out_session_accuracy"] < base else "beats the constant")
            print(f"  the constant answer {out['majority_tool']!r} scores "
                  f"{100 * base:.1f}% on the same pairs — {verdict}")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n",
                                   encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
