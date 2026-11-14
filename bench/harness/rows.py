"""rows — the dataset seam.

A ``Row`` is the only thing a predictor ever sees, and ``options`` lives on the row
rather than in a global, which is what lets a 17-way intent task and an 8-way routing
task be scored by the same code.

Three adapters ship: ``tool_routing`` (this repo's 415 heldout-v3 rows), ``localalexa_intents``
(the 382-row, 17-intent task) and ``generic_csv`` (any corpus, no code change).

``Row.__post_init__`` refuses a label that is not in the row's own option set. That single
check is the class of bug that put an 18-class corpus behind an 8-op router and made every
prediction abstain: a dataset whose labels and options disagree cannot be scored, so it
fails at load instead of scoring 0.0 in silence.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

ROOT = Path(__file__).resolve().parents[2]


class DatasetError(ValueError):
    """A dataset that cannot be scored honestly (bad label, mixed options, duplicate id)."""


@dataclass(frozen=True)
class Row:
    id: str
    text: str
    label: str
    group: str
    options: tuple[str, ...]
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise DatasetError("row has no id")
        if not self.text.strip():
            raise DatasetError(f"{self.id}: empty text")
        if not self.options:
            raise DatasetError(f"{self.id}: no options")
        if self.label not in self.options:
            raise DatasetError(
                f"{self.id}: label {self.label!r} is not in its option set "
                f"({len(self.options)} options) — the dataset and the decision space disagree")
        if not self.group:
            raise DatasetError(f"{self.id}: empty group (grouped CV and per-group reporting need it)")


@dataclass(frozen=True)
class Dataset:
    """Rows, the option set, and the provenance the artifact is allowed to carry.

    ``authored`` says the rows were written here, for this repo. ``generic_csv`` cannot say that
    — it is whatever path the operator points at — and a dataset that is not authored is a door
    into a public repository: the artifact may record its path, its row count and a sha256 of the
    file, and nothing about its rows.
    """

    name: str
    rows: tuple[Row, ...]
    options: tuple[str, ...]
    source: str = ""
    authored: bool = False
    sha256: str = ""

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[Row]:
        return iter(self.rows)

    def by_id(self) -> dict[str, Row]:
        return {r.id: r for r in self.rows}

    def groups(self) -> dict[str, list[Row]]:
        out: dict[str, list[Row]] = {}
        for r in self.rows:
            out.setdefault(r.group, []).append(r)
        return out

    def labels(self) -> tuple[str, ...]:
        seen = {r.label for r in self.rows}
        return tuple(o for o in self.options if o in seen)

    def slice(self, name: str, values: Iterable[str]) -> "Dataset":
        """A subset by ``meta[name]`` — the seam the v3 train/test split uses."""
        keep = set(values)
        rows = tuple(r for r in self.rows if r.meta.get(name) in keep)
        return Dataset(self.name, rows, self.options, self.source, self.authored, self.sha256)


def _validate(rows: list[Row], name: str) -> tuple[Row, ...]:
    if not rows:
        raise DatasetError(f"{name}: no rows")
    ids = [r.id for r in rows]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})[:5]
        raise DatasetError(f"{name}: duplicate ids {dupes}")
    options = rows[0].options
    for r in rows:
        if r.options != options:
            raise DatasetError(
                f"{name}: {r.id} has a different option tuple than {rows[0].id} — the option "
                f"set and its order are part of the instrument, so one dataset has one of each")
    return tuple(rows)


def file_sha256(path: str | Path) -> str:
    """The dataset's identity in the artifact: a hash of the file, never the file.

    Read in chunks so a large corpus does not have to fit in memory twice; keyed on bytes so the
    hash does not move when the loader changes.
    """
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_REGISTRY: dict[str, object] = {}


def register_dataset(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def available() -> list[str]:
    return sorted(_REGISTRY)


def load(spec: str, **overrides) -> Dataset:
    """Build a dataset from ``name`` or ``name|key=value|...``."""
    name, kw = split_spec(spec)
    kw.update({k: v for k, v in overrides.items() if v is not None})
    if name not in _REGISTRY:
        raise DatasetError(f"unknown dataset {name!r}; known: {', '.join(available())}")
    return _REGISTRY[name](**kw)


def split_spec(spec: str) -> tuple[str, dict[str, str]]:
    parts = [p for p in str(spec).split("|") if p != ""]
    name, rest = parts[0], parts[1:]
    kw: dict[str, str] = {}
    for p in rest:
        if "=" not in p:
            raise DatasetError(f"bad spec segment {p!r}; expected key=value")
        k, v = p.split("=", 1)
        kw[k.strip()] = v.strip()
    return name.strip(), kw


# ------------------------------------------------------------------ adapters

# The eight ops, in the order every historical lane was scored with. Imported rather than
# retyped: ti_bench.prompts is the frozen instrument and the dataset must not drift from it.
def _tool_options() -> tuple[str, ...]:
    from bench.ti_bench.prompts import OP_DESCRIPTIONS
    return tuple(OP_DESCRIPTIONS)


def _tool_descriptions() -> dict[str, str]:
    from bench.ti_bench.prompts import OP_DESCRIPTIONS
    return dict(OP_DESCRIPTIONS)


def _stratum(label: str, split: str) -> str:
    """The legacy stratum names, so a replay lines up with the p3-transfer artifacts."""
    targets = {"filesys_grep", "decide_extract"}
    return ("targets" if label in targets else "guard") + "-" + (split or "all")


@register_dataset("tool_routing")
def tool_routing(path: str | None = None) -> Dataset:
    """This repo's 415 heldout-v3 rows; ``group`` is the authored family."""
    src = Path(path) if path else ROOT / "bench/data/heldout_v3.json"
    if not src.exists():
        raise DatasetError(f"tool_routing: {src} is missing")
    raw = json.loads(src.read_text(encoding="utf-8"))
    options = _tool_options()
    descriptions = _tool_descriptions()
    rows = [
        Row(id=r["id"], text=r["utterance"], label=r["label"], group=r["family"],
            options=options,
            meta={"band": r.get("band"), "split": r.get("split"),
                  "tier_target": r.get("tier_target"),
                  "stratum": _stratum(r["label"], r.get("split")),
                  "descriptions": descriptions})
        for r in raw
    ]
    return Dataset("tool_routing", _validate(rows, "tool_routing"), options, str(src),
                   authored=True, sha256=file_sha256(src))


def _localalexa_root(path: str | None) -> Path:
    """Where the sibling project lives: an explicit path, the env var, or a sibling checkout."""
    if path:
        return Path(path)
    env = os.environ.get("LOCALALEXA_ROOT", "").strip()
    candidates = ([Path(env)] if env else []) + [ROOT.parent / "localAlexa"]
    for cand in candidates:
        if (cand / "data/routing_382.json").exists():
            return cand
    raise DatasetError(
        "localalexa_intents: no localAlexa checkout found; pass path=/path/to/localAlexa "
        "or set LOCALALEXA_ROOT")


@register_dataset("localalexa_intents")
def localalexa_intents(path: str | None = None, options_from: str = "registry",
                       group_by: str = "family") -> Dataset:
    """The 382-row, 17-intent task.

    ``group_by`` decides what a fold may not share, and it is the difference between a
    measurement and a wall on this corpus:

    ``family``
        the seven-value ``family`` column in the file — *domains* (timers, lights,
        knowledge). Five folds over seven domains withholds a whole domain each, so every
        fold contains intents the training side has never seen: measured 0/382 out of fold,
        which is the grouping failing rather than the arm.
    ``phrasing``
        the template that generated the row — 110 phrasing families. This is the grouping
        the localAlexa ladder used, and the one that leaves a router something to learn from
        while still holding the wording out.

    The row's own ``family`` column stays in ``meta`` either way, so per-domain reporting
    works under both.
    """
    root = _localalexa_root(path)
    rows_path = root / "data/routing_382.json"
    if not rows_path.exists():
        raise DatasetError(f"localalexa_intents: {rows_path} is missing")
    raw = json.loads(rows_path.read_text(encoding="utf-8"))
    if options_from == "registry":
        reg = json.loads((root / "data/registry.json").read_text(encoding="utf-8"))
        tools = reg["tools"]
        options = tuple(t["id"] if isinstance(t, dict) else str(t) for t in tools)
        # the abstain route is a real label in this corpus (25 of 382 rows), so it is an
        # option like any other; leaving it out is what makes a dataset unscorable
        none_route = reg.get("none_route") or {}
        if isinstance(none_route, dict) and none_route.get("id"):
            options = options + (none_route["id"],)
    else:
        options = tuple(sorted({r["tool"] for r in raw}))
    if group_by not in ("family", "phrasing"):
        raise DatasetError(
            f"localalexa_intents: unknown group_by {group_by!r}; known: family, phrasing")
    index = None
    if group_by == "phrasing":
        index = _localalexa_template_index(root)
    rows = []
    for i, r in enumerate(raw):
        group = r.get("family") or "unknown"
        if index is not None:
            hit = index.get(r["text"])
            group = hit[1] if hit else "__chitchat__"
        rows.append(Row(id=f"a382-{i:03d}", text=r["text"], label=r["tool"], group=group,
                        options=options, meta={"family": r.get("family")}))
    return Dataset("localalexa_intents", _validate(rows, "localalexa_intents"), options,
                   str(rows_path))


def _localalexa_expand(tool: dict, template: str) -> list[str]:
    """Every slot fill of one template, in slot-vocabulary product order."""
    keys = list(tool.get("slots", {}))
    if not keys:
        return [template]
    out = []
    for combo in itertools.product(*[tool["slots"][k] for k in keys]):
        text = template
        for key, value in zip(keys, combo):
            text = text.replace("{%s}" % key, value)
        out.append(text)
    return out


def _localalexa_leaks(root: Path) -> dict[str, list[str]]:
    """Every utterance -> *all* phrasing families that render it.

    ``_localalexa_template_index`` keeps the first hit because that is what the phrasing
    group is; the contamination filter needs every hit, because a reference is dropped when
    *any* held-out family can render it, whichever tool the hit came from.
    """
    reg = json.loads((root / "data/registry.json").read_text(encoding="utf-8"))
    leaks: dict[str, list[str]] = {}
    for tool in reg["tools"]:
        for template in tool.get("templates", ()):
            for text in _localalexa_expand(tool, template):
                families = leaks.setdefault(text, [])
                if template not in families:
                    families.append(template)
    return leaks


def _localalexa_template_index(root: Path) -> dict[str, tuple[str, str]]:
    """Every renderable utterance -> (tool id, template). The phrasing-family key."""
    reg = json.loads((root / "data/registry.json").read_text(encoding="utf-8"))
    index: dict[str, tuple[str, str]] = {}
    for tool in reg["tools"]:
        for template in tool.get("templates", ()):
            for text in _localalexa_expand(tool, template):
                index.setdefault(text, (tool["id"], template))
    return index


@register_dataset("localalexa_refs")
def localalexa_refs(path: str | None = None) -> Dataset:
    """The 200 hand-written references this router trains on, as a *fit pool*.

    Not a scoring set: these are the rows ``--fit-pool`` supplies so a grouped CV can hold
    the phrasing families out of the *scored* rows while still training on the reference
    pool, which is the protocol the localAlexa ladder measured. Training on the other folds
    of the 382 instead gives the model more rows and easier ones — those rows are slot fills
    of the very templates being held out — so the two are different experiments.

    ``meta.leaks_into`` carries the phrasing families this reference is a *literal slot fill*
    of. A fold that holds such a family out drops the reference with it, which is the same
    contamination filter ``localalexa`` applies; without it a reference can hand the model the
    exact wording it is about to be scored on.
    """
    root = _localalexa_root(path)
    reg = json.loads((root / "data/registry.json").read_text(encoding="utf-8"))
    options = tuple(t["id"] for t in reg["tools"])
    none_route = reg.get("none_route") or {}
    if isinstance(none_route, dict) and none_route.get("id"):
        options = options + (none_route["id"],)
    leaks = _localalexa_leaks(root)
    rows = []
    for tool in reg["tools"]:
        for j, reference in enumerate(tool.get("references", ())):
            rows.append(Row(
                id=f"ref-{tool['id']}-{j:02d}", text=reference, label=tool["id"],
                group="refs", options=options,
                meta={"leaks_into": sorted(leaks.get(reference, []))}))
    for j, reference in enumerate(none_route.get("references", ())):
        rows.append(Row(
            id=f"ref-none-{j:02d}", text=reference, label=none_route.get("id", "__none__"),
            group="refs", options=options, meta={"leaks_into": []}))
    rows_path = root / "data/registry.json"
    return Dataset("localalexa_refs", _validate(rows, "localalexa_refs"), options,
                   str(rows_path))


@register_dataset("generic_csv")
def generic_csv(path: str, text_col: str = "text", label_col: str = "label",
                group_col: str = "", options: str = "") -> Dataset:
    """Any corpus with a text column and a label column; options derived unless given.

    ``options='a,b,c'`` fixes the set and its order; otherwise the sorted label set is used.
    Reordering an existing dataset through this flag is the order control's dataset form.
    """
    src = Path(path)
    if not src.exists():
        raise DatasetError(f"generic_csv: {src} is missing")
    with src.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for col in (text_col, label_col):
            if col not in (reader.fieldnames or []):
                raise DatasetError(f"generic_csv: {src} has no column {col!r} "
                                   f"(columns: {reader.fieldnames})")
        raw = list(reader)
    if options:
        opts = tuple(o.strip() for o in options.split(",") if o.strip())
    else:
        opts = tuple(sorted({r[label_col] for r in raw}))
    rows = [
        Row(id=f"{src.stem}-{i:04d}", text=r[text_col], label=r[label_col],
            group=(r.get(group_col) if group_col else "") or "all", options=opts)
        for i, r in enumerate(raw)
    ]
    return Dataset("generic_csv", _validate(rows, "generic_csv"), opts, str(src),
                   authored=False, sha256=file_sha256(src))
