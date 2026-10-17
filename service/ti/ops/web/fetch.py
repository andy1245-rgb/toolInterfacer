"""web.fetch — fetch one URL/page batch through the web-fetch skill script.

Replayable: in fixture/bench mode the "fetch" is a recorded page payload.
"""
from __future__ import annotations

from typing import Any

from ..op_api import op


@op("web_fetch", kind="pages", required=["urls"], optional=["budget_notes"])
def run(params: dict[str, Any], store: Any) -> tuple[str, Any, float]:
    _ = params, store
    raise NotImplementedError("P4")
