"""controls — the four things that make a number believable, and the rules that make them hard
to forget.

* ``--order rev|shuffle:<seed>``  permutes the option order, text unchanged. Mandatory for any
  claim about wording: reversing this repo's eight descriptions cost 9 rows and moved 34
  argmaxes against an A/A floor of one.
* ``--aa``                        scores the reference arm twice and reports the flip count.
  A delta smaller than three times that floor is not a result.
* ``--paraphrase <json>``         a second wording of the same edit, so "the clarification
  worked" is distinguishable from "any rewording worked".
* ``--preregister <json|yaml>``   declares the arms, the metric and the pass conditions before
  the run; the run prints passed/failed per condition and refuses a verdict without it.

Conditions are decoded, never ``eval``'d: an expression is ``PATH OP VALUE`` where PATH is a
dotted lookup into the run's own numbers. That is enough for "mcnemar_p < 0.05",
"ledger.web_search.broke <= 3" and "ledger_untouched_net >= 0", and it cannot execute anything.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .rows import Dataset, DatasetError, Row, split_spec

OPS = ("<=", ">=", "==", "!=", "<", ">")


# ------------------------------------------------------------------ option order

def permute(options: tuple[str, ...], mode: str = "") -> tuple[str, ...]:
    """``""`` unchanged, ``rev`` reversed, ``shuffle:<seed>`` a seeded permutation.

    The shuffle is hash-ordered, not ``random.shuffle``: the same seed over the same option
    set gives the same order on any Python, so a run can be reproduced from its artifact.
    """
    if not mode:
        return options
    if mode == "rev":
        return tuple(reversed(options))
    if mode.startswith("shuffle:"):
        seed = mode.split(":", 1)[1] or "0"
        return tuple(sorted(options, key=lambda o: hashlib.sha256(
            f"{seed}|{o}".encode()).hexdigest()))
    raise ValueError(f"unknown order mode {mode!r}; use rev or shuffle:<seed>")


def options_for(mode: str):
    """A per-row option-order function for ``protocol.score``."""
    if not mode:
        return None
    return lambda row: permute(tuple(row.options), mode)


# ------------------------------------------------------------------ paraphrase

def load_paraphrase(path: str | Path) -> dict[str, str]:
    """``{option: alternate wording}``. A JSON object; YAML too when PyYAML is importable."""
    p = Path(path)
    if not p.exists():
        raise DatasetError(f"paraphrase: {p} is missing")
    text = p.read_text(encoding="utf-8")
    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml  # noqa: PLC0415
        except ImportError as exc:
            raise DatasetError(
                f"paraphrase: {p.name} is YAML and PyYAML is not installed; use JSON "
                f"(the harness core is stdlib only)") from exc
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    if not isinstance(raw, dict):
        raise DatasetError("paraphrase: expected an object of {option: wording}")
    return {str(k): str(v) for k, v in raw.items()}


def apply_paraphrase(ds: Dataset, mapping: dict[str, str]) -> Dataset:
    """Rewrite the *descriptions* the predictor reads, never the option names or their order.

    An option name is a label; the description is the instrument. Rewording a label would move
    rows between classes and make the comparison meaningless, which is exactly the mistake the
    control exists to prevent.
    """
    unknown = sorted(set(mapping) - set(ds.options))
    if unknown:
        raise DatasetError(f"paraphrase names options not in this dataset: {unknown}")
    rows = []
    for r in ds.rows:
        descriptions = dict((r.meta or {}).get("descriptions") or {})
        descriptions.update({k: v for k, v in mapping.items() if k in r.options})
        meta = dict(r.meta or {})
        meta["descriptions"] = descriptions
        meta["paraphrased"] = sorted(mapping)
        rows.append(Row(id=r.id, text=r.text, label=r.label, group=r.group,
                        options=r.options, meta=meta))
    return Dataset(ds.name, tuple(rows), ds.options, ds.source)


# ------------------------------------------------------------------ pre-registration

def load_preregister(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise DatasetError(f"preregister: {p} is missing")
    text = p.read_text(encoding="utf-8")
    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml  # noqa: PLC0415
        except ImportError as exc:
            raise DatasetError(
                f"preregister: {p.name} is YAML and PyYAML is not installed; use JSON") from exc
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    if not isinstance(raw, dict):
        raise DatasetError("preregister: expected an object")
    conds = raw.get("conditions") or []
    if not isinstance(conds, list) or not conds:
        raise DatasetError("preregister: needs a non-empty conditions list")
    for c in conds:
        if not isinstance(c, dict) or "expr" not in c:
            raise DatasetError(f"preregister: every condition needs an expr; got {c!r}")
    return raw


def resolve(ctx: dict, path: str):
    """Dotted lookup: ``ledger.web_search.broke`` -> ctx["ledger"]["web_search"]["broke"]."""
    cur = ctx
    for part in path.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                raise KeyError(f"{path!r}: no {part!r} in {sorted(cur)[:8]}")
            cur = cur[part]
        elif isinstance(cur, (list, tuple)):
            cur = cur[int(part)]
        else:
            raise KeyError(f"{path!r}: cannot descend into {type(cur).__name__}")
    return cur


def evaluate_condition(expr: str, ctx: dict) -> dict:
    """Parse and evaluate one ``PATH OP VALUE`` condition. No eval, no attribute access."""
    text = str(expr).strip()
    op = next((o for o in OPS if o in text), None)
    if op is None:
        raise DatasetError(f"condition {expr!r} has no comparison operator")
    lhs, rhs = text.split(op, 1)
    value = resolve(ctx, lhs.strip())
    want_txt = rhs.strip()
    try:
        want = float(want_txt)
        got = float(value)
        ok = {"<=": got <= want, ">=": got >= want, "==": got == want,
              "!=": got != want, "<": got < want, ">": got > want}[op]
    except (TypeError, ValueError):
        ok = {"==": value == want_txt, "!=": value != want_txt}.get(
            op, False)
    return {"expr": text, "value": value, "op": op, "target": want_txt, "passed": bool(ok)}


def check(prereg: dict, ctx: dict) -> dict:
    """Evaluate every condition, and report them in order. `passed` is the conjunction."""
    results = []
    for c in prereg.get("conditions") or []:
        r = evaluate_condition(c["expr"], ctx)
        if c.get("name"):
            r["name"] = c["name"]
        results.append(r)
    return {"conditions": results, "passed": all(r["passed"] for r in results)}


def gate_verdict(delta_rows: float, floor_rows: float, factor: float = 3.0) -> dict:
    """The floor rule: a delta under ``factor`` times the A/A floor is not a result."""
    need = abs(floor_rows) * factor
    ok = abs(delta_rows) > need
    return {"delta": delta_rows, "floor": floor_rows, "needed": need, "clears": bool(ok)}
