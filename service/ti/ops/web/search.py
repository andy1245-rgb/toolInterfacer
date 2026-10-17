"""web.search — hits a configured web-search CLI backend, merges + dedupes URLs.

P0: implementation lands with L2 wiring in P4 (CLI arg contract mined from the
web-search skill scripts). P1 exercises this op only via replay fixtures.
"""
from __future__ import annotations

from typing import Any

from ..op_api import op


@op("web_search", kind="url_list", required=["queries"], optional=["backend"])
def run(params: dict[str, Any], store: Any) -> tuple[str, Any, float]:
    _ = params, store
    raise NotImplementedError("P4")
