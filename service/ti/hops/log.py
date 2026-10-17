"""Hop log — one JSONL record per executed hop; the finetune dataset feeder.

Record shape: ts, plan_meta, step, op, op_params, conf, artifact_summary, ms.
Enabled/path from conf.hops_log.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any


class HopLog:
    def __init__(self) -> None:
        from ..conf import load
        cfg = load()["hops_log"]
        self.enabled = bool(cfg["enabled"])
        self.path = os.path.expanduser(cfg["path"])

    def append(self, record: dict[str, Any]) -> None:
        if not self.enabled:
            return
        rec = {"ts": time.time(), **record}
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
