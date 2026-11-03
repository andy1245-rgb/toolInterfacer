"""Path resolution for the dataset pipeline.

Every path is either an environment variable or a repo-relative default; nothing
here is machine-specific. The default log directory mirrors ``conf.hops_log.path``
so the pipeline reads the same file the service writes.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

LOGS_ENV = "TI_DATASET_LOGS"
OUT_ENV = "TI_DATASET_OUT"

DEFAULT_LOGS = "service/logs"
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
