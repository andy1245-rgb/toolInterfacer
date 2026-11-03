"""Artifacts — typed results each step publishes.

The runner holds one store per plan run. Steps address other steps' artifacts
by key; gates read them; the final assembly reads them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Artifact:
    key: str                    # e.g. "search_hits", "pages:0", "picked"
    src_step: str
    kind: str                   # free-form: "url_list", "page", "file_content", ...
    data: Any                   # structured payload; steps agree on shapes per op
    # A measurement when the producing op made one, None when nothing was
    # measured. Defaulting to 1.0 made "no measurement" indistinguishable from
    # "certain", so a conf_gte gate passed every artifact whose op forgot a conf.
    conf: float | None = None


class ArtifactStore:
    def __init__(self) -> None:
        self._items: dict[str, Artifact] = {}

    def put(self, key: str, src_step: str, kind: str, data: Any,
            conf: float | None = None) -> Artifact:
        art = Artifact(key=key, src_step=src_step, kind=kind, data=data, conf=conf)
        self._items[key] = art
        return art

    def get(self, key: str) -> Artifact:
        return self._items[key]

    def try_get(self, key: str) -> Artifact | None:
        return self._items.get(key)

    def all(self) -> dict[str, Artifact]:
        return dict(self._items)
