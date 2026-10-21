"""ops.cli — the one subprocess adapter every CLI-backed op goes through.

Three ops shell out to a skill CLI and turn what it wrote into an artifact.
The mechanics are identical in all three, so they live here once: build an argv
LIST (never a shell string, so a query or URL can never be re-read as a flag or
a command separator), run it under ``ops.cli_timeout_s``, capture both streams,
locate the result document, parse it.

Result location covers every convention the reachable CLIs actually use — the
round-1 survey found three, and a caller that parses only one silently gets
nothing from the other two:

    --- WEBFETCH ---                 the machine footer; RESULT_FILE is the doc,
    RESULT_FILE: /run/combined.json  RUN_DIR its directory
    RUN_DIR: /run

    /run/combined.json               ``--print-path-only``: a bare path on stdout

    {"query": ..., "results": [...]}  ``--json``: the document *is* stdout

Failure is typed and never swallowed: ``CliMissingBinary``, ``CliTimeout``,
``CliFailed`` (non-zero exit; the run rides along so a caller whose CLI writes a
usable document on a documented non-zero code can still read it),
``CliUnparseable`` and ``CliMissingResult``.

stdlib only; no network; knows nothing about searching, fetching or reddit.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_TIMEOUT_S = 60.0


class CliError(RuntimeError):
    """Base for every CLI failure; carries the run when there was one."""

    def __init__(self, message: str, run: "CliRun | None" = None) -> None:
        super().__init__(message)
        self.run = run


class CliMissingBinary(CliError):
    """The configured script / entry point is not on disk."""


class CliTimeout(CliError):
    """The CLI did not finish inside ``ops.cli_timeout_s``."""


class CliFailed(CliError):
    """The CLI exited non-zero."""


class CliUnparseable(CliError):
    """No result document could be located, or it is not JSON."""


class CliMissingResult(CliError):
    """A result path was advertised but the file is not there."""


@dataclass
class CliRun:
    argv: list[str]
    exit_code: int
    stdout: str
    stderr: str

    @property
    def program(self) -> str:
        return Path(self.argv[0]).name or self.argv[0]

    def tail(self, limit: int = 400) -> str:
        """The most informative stream tail available, for error messages."""
        for stream in (self.stderr, self.stdout):
            text = " ".join(stream.split()).strip()
            if text:
                return text[-limit:]
        return ""


def ops_conf() -> dict[str, Any]:
    """The live ``ops`` conf block (resolved at call time, like every other op)."""
    from ..conf import load
    return load().get("ops") or {}


def timeout_from_conf(cfg: dict[str, Any] | None = None) -> float:
    ops = ops_conf() if cfg is None else cfg
    return float(ops.get("cli_timeout_s", DEFAULT_TIMEOUT_S))


def launcher(script: str) -> list[str]:
    """argv prefix that runs a configured CLI script.

    A non-executable script, or one whose shebang is CRLF-terminated (the
    registry's ``free-search`` is stored that way, and the kernel then looks for
    ``python3\\r`` and exits 127), runs under this interpreter rather than
    failing with EACCES or a mojibake ENOENT. Everything else execs itself.
    """
    path = Path(os.path.expanduser(str(script)))
    if not path.is_file():
        raise CliMissingBinary(f"CLI not found: {path}")
    shebang = _shebang(path)
    if os.access(path, os.X_OK) and "\r" not in shebang:
        return [str(path)]
    if "python" in shebang or path.suffix in (".py", ".pyw"):
        return [sys.executable, str(path)]
    raise CliMissingBinary(
        f"CLI is not executable and has no python shebang to fall back on: {path}")


def cli_argv(script: str, args: list[str], cfg: dict[str, Any] | None = None) -> list[str]:
    """Full argv for a CLI script plus ``ops.dev_mode``, when conf sets it.

    ``--dev-mode`` is a devskills-wrapper-only flag; setting it while pointing
    at a live CLI is a config error, not something to paper over here.
    """
    ops = ops_conf() if cfg is None else cfg
    mode_args: list[str] = []
    mode = ops.get("dev_mode")
    if mode:
        mode_args = ["--dev-mode", str(mode)]
    return launcher(script) + mode_args + [str(a) for a in args]


def run_cli(argv: list[str], timeout_s: float, cwd: str | None = None) -> CliRun:
    """Run an argv list under a timeout; raise on a missing binary or a timeout.

    A non-zero exit is *returned*, not raised: which exit codes still carry a
    usable document is per-CLI knowledge that belongs to the op, not here.
    """
    try:
        proc = subprocess.run(argv, capture_output=True, text=True,
                              encoding="utf-8", errors="replace",
                              timeout=timeout_s, cwd=cwd)
    except subprocess.TimeoutExpired as exc:
        run = CliRun(argv=list(argv), exit_code=-1,
                     stdout=_text(exc.stdout), stderr=_text(exc.stderr))
        raise CliTimeout(
            f"{run.program} timed out after {timeout_s:g}s; raise ops.cli_timeout_s "
            f"or narrow the request", run) from exc
    except FileNotFoundError as exc:
        raise CliMissingBinary(f"CLI not found: {argv[0]}") from exc
    except OSError as exc:
        raise CliMissingBinary(f"cannot execute {argv[0]}: {exc}") from exc
    return CliRun(argv=list(argv), exit_code=proc.returncode,
                  stdout=proc.stdout or "", stderr=proc.stderr or "")


def result_path(run: CliRun) -> Path | None:
    """Locate the document the CLI advertised, or None if it advertised none."""
    advertised = _footer_value(run.stdout, "RESULT_FILE")
    if advertised:
        return Path(os.path.expanduser(advertised))
    run_dir = _footer_value(run.stdout, "RUN_DIR")
    if run_dir:
        return Path(os.path.expanduser(run_dir)) / "combined.json"
    bare = run.stdout.strip()
    if bare and "\n" not in bare and not bare.startswith("{"):
        candidate = Path(os.path.expanduser(bare))
        if candidate.is_file():
            return candidate
    return None


def read_result(run: CliRun) -> dict[str, Any]:
    """Read and parse the CLI's document from a path or from stdout."""
    path = result_path(run)
    if path is not None:
        if not path.is_file():
            raise CliMissingResult(f"{run.program}: result file missing: {path}", run)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise CliMissingResult(f"{run.program}: cannot read {path}: {exc}", run) from exc
        try:
            return json.loads(text)
        except ValueError as exc:
            raise CliUnparseable(f"{run.program}: {path} is not JSON: {exc}", run) from exc
    body = run.stdout.strip()
    if body:
        try:
            doc = json.loads(body)
        except ValueError as exc:
            raise CliUnparseable(
                f"{run.program}: no result path and stdout is not JSON "
                f"(exit {run.exit_code}): {run.tail()}", run) from exc
        if isinstance(doc, dict):
            return doc
        raise CliUnparseable(
            f"{run.program}: stdout JSON is {type(doc).__name__}, expected an object", run)
    raise CliUnparseable(
        f"{run.program}: no result path on stdout and nothing to parse "
        f"(exit {run.exit_code}): {run.tail()}", run)


def run_result(argv: list[str], timeout_s: float, cwd: str | None = None,
               tolerate: tuple[int, ...] = ()) -> tuple[CliRun, dict[str, Any]]:
    """Run, enforce the exit code, and return ``(run, document)``.

    ``tolerate`` lists exit codes whose document is still worth reading — e.g.
    web-fetch exits 1 for a page it could not extract while still writing the
    artifact that says why.
    """
    run = run_cli(argv, timeout_s, cwd=cwd)
    if run.exit_code != 0 and run.exit_code not in tolerate:
        raise CliFailed(f"{run.program}: exit {run.exit_code}: {run.tail()}", run)
    return run, read_result(run)


def _footer_value(stdout: str, key: str) -> str | None:
    for line in stdout.splitlines():
        text = line.strip()
        if text.startswith(key + ":"):
            value = text.partition(":")[2].strip()
            if value:
                return value
    return None


def _shebang(path: Path) -> str:
    try:
        with path.open("rb") as fh:
            first = fh.readline(256)
    except OSError:
        return ""
    return first.decode("latin-1") if first.startswith(b"#!") else ""


def _text(stream: Any) -> str:
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return str(stream)
