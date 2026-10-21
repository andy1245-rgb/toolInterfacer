# Ops: the CLI contract `web_search` / `web_fetch` / `reddit_digest` rely on

Three ops shell out to a skill CLI instead of doing work in-process:
`web_search` (`url_list`), `web_fetch` (`pages`), `reddit_digest`
(`thread_digest`). All three go through one adapter, `service/ti/ops/cli.py`, so
the subprocess rules live in exactly one place.

The ops assume nothing about the CLIs beyond what is written here. Where a rule
below was read off the live source rather than executed, it says so.

## 1. Invocation

- **An argv list, never a shell string.** A query or URL is one element of the
  list, so `;`, `&&` and `$(...)` are data. `--` precedes the positional so a
  query that starts with `-` is not read as a flag.
- **Launcher.** `cli.launcher(script)` returns the argv prefix for a configured
  script: an executable script with a sane shebang execs as itself; anything
  else runs under the service's own interpreter. A CRLF-terminated shebang also
  falls back to the interpreter — the kernel would otherwise look for
  `python3\r` and exit 127 (see §7).
- **Timeout.** `ops.cli_timeout_s` (default 60) bounds every invocation. A
  timeout raises `CliTimeout`; the process is killed, not orphaned.
- **Dev-mode hook.** When `ops.dev_mode` is set, `--dev-mode <value>` is inserted
  after the launcher. It is a devskill-wrapper-only flag: pointing it at a live
  CLI is a config error, not something the adapter hides.

## 2. Locating the result document

The three CLIs advertise their artifact in three different ways, and the adapter
accepts all three — a caller that parses only one silently gets nothing from the
other two:

| convention | how it looks | used by |
|---|---|---|
| machine footer | `--- WEBFETCH ---` then `RESULT_FILE: <path>`, `RUN_DIR: <dir>` | the default call shape |
| bare path | a single line holding the artifact path | `--print-path-only` |
| payload on stdout | the JSON document itself, no path, no footer | `--json` |

`result_path()` handles the first two plus a `RUN_DIR`-only footer (it then
expects `combined.json` inside). `read_result()` falls back to parsing stdout as
JSON when no path was advertised. A path that was advertised but is not on disk
raises `CliMissingResult`; an advertised file that is not JSON raises
`CliUnparseable`.

The ops invoke with `--no-preview`, which suppresses the human preview but
**keeps the footer** — verified against both live search/fetch scripts. That is
also why `--json` is not used by default: it would trade the artifact path for
the payload when the same file is already there.

## 3. Exit codes

`run_result(argv, timeout, tolerate=…)` raises `CliFailed` for any exit code not
in `tolerate`. Which codes still carry a usable document is per-CLI knowledge
that belongs to the op:

| CLI | exit 0 | exit 1 | exit 2 | op's `tolerate` |
|---|---|---|---|---|
| `free-search` | pile built (empty or not) | every source errored | usage | `(1,)` — the artifact is still valid |
| `web-search` | ≥1 provider manifest | zero manifests | usage / config | `(1,)` |
| `web-fetch` | page text extracted | no text extracted (artifact says why) | usage | `(1,)` — that artifact *is* the failure report |
| `reddit-research` | summary printed | gated / network failure | usage | none |

Exit code alone is never a health signal: a successful empty run and a run where
every source broke can share a code, and web-fetch's `--batch` mode returns 0
even when every URL failed. Every op therefore reads the document.

## 4. Config

```json
"ops": {
  "cli_timeout_s": 60,
  "web_search":          {"script": "…/web-search/scripts/free-search",
                          "backend": "free-search", "brave": "auto"},
  "web_search_provider": {"script": "…/web-search/scripts/web-search",
                          "provider": "duckduckgo"},
  "web_fetch":           {"script": "…/web-fetch/scripts/web-fetch",
                          "max_chars": 20000},
  "reddit":              {"cwd": "~/…/redditScrapingSearching",
                          "cwd_candidates": ["/home/andrew/mnt/dell321/…/redditScrapingSearching"],
                          "cli": [".venv/bin/python", "-m", "reddit_research.cli"]}
}
```

- `web_search.brave` ∈ `off | auto | forced` → `--no-brave` / nothing /
  `--brave`. `auto` is the CLI's own default; `off` is the strict zero-payment
  path.
- `web_search.backend` `free-search` (default) or `web-search`; the `backend`
  op param overrides it, and the provider path reads `web_search_provider`.
- `web_fetch.max_chars` mirrors the CLI's `--max-chars` (20000 default).
- `ops.dev_mode` (absent in live defaults) is the replay hook for the devskill
  wrappers.

`TI_CONF` selects a whole alternate config file; `TI_CONF_OVERLAY` deep-merges a
partial one on top (dicts recurse, lists replace). The overlay is how tests and
a dev bench repoint `ops.*` without copying the live file.

## 5. Op shapes

### `web_search` — `url_list`

`{"queries": [str]}` (a bare string is accepted — the runner unwraps a 1-element
`picked_items` artifact) → one invocation per query → merged, deduped list:

```json
{"title": "…", "url": "…", "snippet": "…",
 "sources": ["ddg_lite"], "source_count": 1, "queries": ["python 3 stable version"]}
```

Dedupe key is the normalized URL: fragment dropped, scheme and host lowercased,
query string kept. Merging unions `sources` and `queries`, which is the point —
knowing a URL was hit by two sources beats keeping two copies of it.

conf is **coverage**, not decision confidence: 1.0 every query answered, 0.5
some, 0.0 none. It is what `policy.gate_slot` compares against before the fetch
fan-out, so a half-answered search neither authorizes a full fetch nor looks
like total failure. A per-query failure counts as unanswered; if *every* query
fails the op raises `CliFailed` with each query's error, because an empty
`url_list` and "nothing worked" must not look alike.

### `web_fetch` — `pages`

`{"urls": [str]}` (bare string accepted) → one invocation per URL, in input
order:

```json
{"url": "…", "content": "page text", "method": "curl|nodriver|python-stdlib|null",
 "error": null}
```

One URL per process on purpose: `--batch` prints no footer, mints one run
directory per URL and exits 0 no matter what, so it would cost the
artifact-to-URL correlation this op exists to keep.

A failed page keeps its entry with `content: ""` and its `error` — "this page
failed" and "this URL was never fetched" must not look alike, and a consumer
that wants to skip failures skips on empty content. The CLI marks total failure
by *omitting* `text`, not by setting `error`; `degraded` (a meta-description
shell instead of the page) is passed through. conf is the mean over URLs of 1.0
/ 0.5 (degraded) / 0.0.

### `reddit_digest` — `thread_digest`

`{"query": str, "subreddit"?: str, "thread_cap"?: int}` →
`{"query", "subreddit", "fetched_at", "posts": [ … ]}`, posts capped at
`thread_cap` (`ops.reddit.thread_cap`, default 5). conf 1.0 with ≥1 post.

**Currently unavailable.** See §7.

## 6. reddit-research: the op that cannot run yet

The skill's project lives on an external mount
(`/home/andrew/mnt/dell321/…`) that is not mounted, and `ops.reddit.cwd` points
at a path that does not exist either. The op probes `cwd` then every entry in
`cwd_candidates`, and when none is a directory it raises `CliMissingBinary`
naming each path it tried. It does not hang, does not return an empty digest,
and does not pretend a no-op was a success — an unmounted project and "the query
found nothing" must not look alike.

The adapter itself is written against the SKILL.md contract, not against an
observed binary:

```
cd <project> && <ops.reddit.cli…> search "<query>" [--subreddit N] [--limit N] --json
```

That CLI prints no machine footer — its machine-readable handle is the
`raw_json` *field* inside the summary dict the `--json` mode prints, a third
advert convention alongside the two in §2. The op reads the summary off stdout,
follows `raw_json`, and projects each `posts[]` entry onto a known-key list
(`POST_FIELDS`); absent keys stay `None` rather than becoming invented values.
That projection is **inferred, not observed** — the first session with the mount
available should record a real response (`devskills/reddit-research/fixtures/`)
and confirm or correct the field list.

## 7. Mismatches found between the live CLIs and what was assumed

1. **CRLF shebang.** `~/.pi/profiles/registry/web-search/scripts/free-search` is
   stored with `#!/usr/bin/env python3\r\n`. Executing it directly dies with
   `exit 127: /usr/bin/env: 'python3\r': No such file or directory` and produces
   no artifact, so the failure looks like "every query failed" rather than "the
   interpreter line is broken". `web-fetch` is LF and unaffected. The adapter
   detects a CRLF shebang and runs the script under the service interpreter;
   `launcher()` is covered by a regression test.
2. **`ops.reddit.cwd` disagrees with the skill's documented location**, and
   neither exists on this box. Both are carried (`cwd` + `cwd_candidates`) so the
   error message lists the real candidates instead of guessing.
3. **web-fetch batch mode is a trap**, as predicted: no footer, one run dir per
   URL, exit 0 on total failure. Not used — see §5.
4. Everything else matched: `--no-preview` keeps the footer in both scripts, the
   footer key names and formats in `_print_machine_footer` are exactly
   `RESULT_FILE:` / `RUN_DIR:`, and `--max-chars` / `--no-brave` / `--brave` /
   `--provider` all exist as the ops pass them (checked against `--help` and the
   source, not just the research notes).

## 8. Tests

`tests/test_ops_web.py` runs the real adapter as a subprocess against
`tests/data/fake_cli.py`, which reproduces the footer / path-only / JSON-stdout
conventions and the payload shapes of all three CLIs. Failure modes are selected
by a marker inside the URL or query (`boom`, `timeout`, `nofile`, `badjson`,
`empty`, `notext`, `degraded`, `pathonly`, `jsonout`), so no flags leak into the
production argv just to make testing possible.

```
python3 -m unittest discover -s tests -p 'test_*.py'
```

Two tests replay the recorded devskills fixtures as a read-only cross-check of
the parsers (`TI_DEVSKILLS_FIXTURES`, default
`~/toolInterfacerExtras/devskills`); they skip when that tree is absent. Nothing
in the suite touches the network or a live skill CLI.
