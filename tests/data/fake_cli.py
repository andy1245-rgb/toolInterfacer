#!/usr/bin/env python3
"""A stand-in for the three skill CLIs, for ops tests. No network, no state.

It reproduces the machine contract the real CLIs expose — the ``--- NAME ---``
footer with ``RESULT_FILE``/``RUN_DIR``, the ``--print-path-only`` bare path, the
``--json`` payload-on-stdout — and writes a document whose *shape* matches the
real one, so the ops parse it exactly as they would parse a live artifact.

Which CLI it is impersonating is read off the argv the op built, the same way a
reader tells them apart: ``--max-chars`` is web-fetch, ``--limit`` is
reddit-research, anything else is a search.

Knobs, all of them things the request itself carries rather than flags the ops
would pass in production:

    ``FAKE_CLI_OUT``      directory the artifact is written to (required)
    ``FAKE_CLI_REPLAY``   serve this recorded document instead of a synthetic one
    target text           a marker inside the query/URL picks the failure mode:
                          boom (exit 3, no artifact), timeout (sleeps),
                          nofile (footer names a file never written), badjson
                          (artifact is not JSON), empty (valid empty run),
                          notext / degraded (fetch only), pathonly / jsonout
                          (alternate advert convention)

``_argv`` is echoed into every document so a test can assert on the argv the op
actually built.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

OUT = Path(os.environ.get("FAKE_CLI_OUT") or ".")

MARKERS = ("boom", "timeout", "nofile", "badjson", "empty", "notext",
           "degraded", "pathonly", "jsonout")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fake-cli", add_help=True)
    p.add_argument("words", nargs="*", help="positionals the op passed")
    # flags the ops pass in production, accepted and ignored
    p.add_argument("--no-preview", action="store_true")
    p.add_argument("--json", dest="json_output", action="store_true")
    p.add_argument("--print-path-only", action="store_true")
    p.add_argument("--max-chars", type=int)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--no-brave", action="store_true")
    p.add_argument("--brave", action="store_true")
    p.add_argument("--provider")
    p.add_argument("--limit", type=int)
    p.add_argument("--subreddit")
    p.add_argument("--dev-mode", choices=("auto", "live", "replay"))
    return p


def scenario_of(raw: list[str]) -> str:
    if "--max-chars" in raw:
        return "fetch"
    if "--limit" in raw:
        return "reddit"
    return "search"


def target_of(words: list[str]) -> str:
    if len(words) > 1 and words[0] == "search":
        return words[1]
    return words[0] if words else ""


def marker(target: str) -> str | None:
    lowered = target.lower()
    for name in MARKERS:
        if name in lowered:
            return name
    return None


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "empty"


def search_doc(target: str, provider: str | None, argv: list[str]) -> dict:
    results = [
        {"title": "A frag", "url": "https://example.test/a#frag",
         "snippet": "s1", "source_count": 1, "sources": ["s1"]},
        {"title": "A dup", "url": "https://EXAMPLE.test/a",
         "snippet": "s2", "source_count": 1, "sources": ["s2"]},
        {"title": "Other " + target, "url": "https://other.test/" + slug(target),
         "snippet": "s3", "source_count": 1, "sources": ["s3"]},
    ]
    if marker(target) == "empty":
        results = []
    common = {"query": target, "ranAt": "2026-01-01T00:00:00+00:00", "_argv": argv}
    if provider:
        return {**common, "execMode": "single", "providers": [provider], "errors": [],
                "manifests": [{"provider": provider, "query": target,
                               "results": [{k: r[k] for k in ("title", "url", "snippet")}
                                           for r in results]}]}
    return {**common, "mode": "free", "braveTrigger": "none", "sourceErrors": [],
            "results": results}


def fetch_doc(target: str, argv: list[str]) -> tuple[dict, int]:
    mode = marker(target)
    if mode == "notext":
        return {"url": target, "error": "All fallbacks failed",
                "startedAt": "2026-01-01T00:00:00Z",
                "fallbackTried": [{"method": "curl", "error": "connection refused"}],
                "_argv": argv}, 1
    doc = {"url": target, "ranAt": "2026-01-01T00:00:00+00:00", "source": "curl",
           "contentLength": 600, "extractionMethod": "regex", "extractionQuality": None,
           "metadata": {"title": None}, "text": "content:" + target,
           "textLength": len("content:" + target), "truncated": False, "cached": False,
           "_argv": argv}
    if mode == "degraded":
        doc["degraded"] = True
        doc["degradedReason"] = "meta description only"
        doc["source"] = "python-stdlib"
    return doc, 0


def reddit_posts(target: str) -> dict:
    posts = [{"id": f"p{i}", "title": f"post {i} about {target}",
              "subreddit": "MechanicalKeyboards", "author": f"user{i}",
              "permalink": f"/r/MechanicalKeyboards/comments/p{i}/",
              "url": f"https://www.reddit.com/r/MechanicalKeyboards/comments/p{i}/",
              "score": 100 - i, "num_comments": i, "created_utc": 1700000000 + i,
              "selftext": f"body {i}", "unknown_field": "dropped"}
             for i in range(7)]
    return {"posts": posts, "subreddits": [{"name": "MechanicalKeyboards"}]}


def emit(doc: dict, exit_code: int, title: str, extra: list[tuple[str, str]],
         args: argparse.Namespace) -> int:
    path = OUT / "combined.json"
    mode = marker(target_of(args.words))
    path.write_text("{not json at all" if mode == "badjson"
                    else json.dumps(doc, indent=2), encoding="utf-8")
    if args.print_path_only or mode == "pathonly":
        print(path)
        return exit_code
    if args.json_output or mode == "jsonout":
        print(json.dumps(doc, indent=2))
        return exit_code
    sys.stdout.write(f"\n--- {title} ---\n")
    advertised = OUT / "never-written.json" if mode == "nofile" else path
    sys.stdout.write(f"RESULT_FILE: {advertised}\n")
    sys.stdout.write(f"RUN_DIR: {path.parent}\n")
    for key, value in extra:
        sys.stdout.write(f"{key}: {value}\n")
    return exit_code


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw)
    scenario = scenario_of(raw)
    target = target_of(args.words)
    replay = os.environ.get("FAKE_CLI_REPLAY")

    if marker(target) == "timeout":
        time.sleep(30)
        return 9

    if marker(target) == "boom":
        print(f"error: {scenario} failed", file=sys.stderr)
        return 3

    if scenario == "reddit":
        posts_path = OUT / f"raw-{slug(target)}.json"
        posts_path.write_text(json.dumps(reddit_posts(target), indent=2), encoding="utf-8")
        summary = {"query": target, "sort": "relevance", "t": "all",
                   "subreddit": args.subreddit or "MechanicalKeyboards",
                   "fetched_at": "2026-01-01T00:00:00Z", "subreddit_results": 1,
                   "posts_found": 7, "posts_new": 7, "raw_json": str(posts_path),
                   "next_after": None}
        print(json.dumps(summary, indent=2))
        return 0

    if replay:
        doc = json.loads(Path(replay).read_text(encoding="utf-8"))
        doc["_argv"] = raw
        return emit(doc, 0, "REPLAY", [("MODE", "replay")], args)

    if scenario == "fetch":
        doc, code = fetch_doc(target, raw)
        return emit(doc, code, "WEBFETCH", [("METHOD", doc.get("source", "error"))], args)

    doc = search_doc(target, args.provider, raw)
    items = doc.get("results") or (doc.get("manifests") or [{}])[0].get("results") or []
    return emit(doc, 0, "WEBSEARCH",
                [("MODE", "single"), ("TOTAL_RESULTS", str(len(items)))], args)


if __name__ == "__main__":
    sys.exit(main())
