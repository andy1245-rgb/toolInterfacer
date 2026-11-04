"""Path resolution for the dataset pipeline.

Every path is either an environment variable or a `~`-rooted state path; nothing here is
machine-specific. The default log directory is the one the service and the extension write
to (`conf.hops_log.path` and the extension's capture path both point inside it), so the
pipeline reads the same files they write — and so that neither writer puts a stray tree in
somebody's working directory (`docs/logging-2026-11-04.md`).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

LOGS_ENV = "TI_DATASET_LOGS"
OUT_ENV = "TI_DATASET_OUT"

DEFAULT_LOGS = "~/.local/state/ti/logs"
DEFAULT_OUT = "var/dataset"


def repo_root() -> Path:
    return ROOT


def logs_dir(override: str | os.PathLike[str] | None = None) -> Path:
    """Directory holding hops.jsonl / live.jsonl."""
    raw = override or os.environ.get(LOGS_ENV) or DEFAULT_LOGS
    return _resolve(raw)


def out_dir(override: str | os.PathLike[str] | None = None) -> Path:
    """Directory the derived corpus is written to (never tracked by git)."""
    raw = override or os.environ.get(OUT_ENV) or DEFAULT_OUT
    return _resolve(raw)


def _resolve(raw: str | os.PathLike[str]) -> Path:
    path = Path(os.path.expanduser(str(raw)))
    return path if path.is_absolute() else (ROOT / path)
