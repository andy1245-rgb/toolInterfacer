"""shadow — compute what a different router would have said, without changing what is served.

The operator's decision, 2026-11-10: do not switch the fusion on; **shadow** it. The served
answer, the routing tier that produced it, the cache, the pre-gate and `last_source` are all
untouched — this module reads the answer on the way past, works out what the fused policy *would*
have said on the same request, and appends both to a log beside the downstream outcome.

Why this and not a fourth arm on the 415 authored rows: those rows are the corpus the fusion was
tuned against. Real traffic is not, and a shadow run answers the only question that matters for a
shipping decision — *on requests people actually made, which policy was right, and what did the
wrong one cost?*

Three properties, in the order they matter:

1. **No behaviour change.** `choice()` returns the inner backend's answer object, unchanged. The
   counterfactual is computed after the answer exists and can only fail into the log.
2. **No risk.** The counterfactual is the cheap member's opinion (a local classifier, ~0.1 ms)
   blended with the answer already in hand. It never calls a model a second time, never escalates,
   never retries, and its exceptions are recorded rather than raised.
3. **No blind spot.** Every decision is logged with both sides, and the ops layer appends the
   outcome under the same id: which op ran, whether it errored, and how many times the step was
   attempted — so "the shadow was right and the served answer was wrong" is a number, not an
   impression.

Config, off by default (absent means off):

    "shadow": {
      "enabled": false,
      "member": "classical",
      "weights": 0.5,
      "protect": ["web_search", "web_fetch"],
      "log_path": ""
    }

`log_path` empty means `$TI_SHADOW_LOG`, else `~/.local/state/ti/shadow.jsonl` — never a path
inside the repository, because the log contains traffic.
"""
from __future__ import annotations

import contextvars
import json
import os
import time
from pathlib import Path
from typing import Any

from .primitives import ChoiceAnswer, DecideBackend, make_backend

DEFAULT_PROTECT = ("web_search", "web_fetch")
ENV_PATH = "TI_SHADOW_LOG"
DEFAULT_PATH = "~/.local/state/ti/shadow.jsonl"
_MAX_TRACKED = 4096

_current: contextvars.ContextVar[str] = contextvars.ContextVar("ti_shadow_decision", default="")
_ATTEMPTS: dict[str, int] = {}


def log_path(cfg: dict[str, Any] | None = None) -> Path:
    """Where the log goes: config, then env, then the state directory. Never the repo."""
    raw = str((cfg or {}).get("log_path") or os.environ.get(ENV_PATH) or DEFAULT_PATH)
    return Path(raw).expanduser()


def _block() -> dict[str, Any]:
    from ..conf import load
    return dict((load().get("decide") or {}).get("shadow") or {})


def enabled(cfg: dict[str, Any] | None = None) -> bool:
    return bool((cfg if cfg is not None else _block()).get("enabled"))


def _append(path: Path, record: dict[str, Any]) -> None:
    """One JSON line, appended. A logging failure must never reach a decision."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    except Exception:  # noqa: BLE001 - a full disk must not become a failed request
        pass


def _dist(answer: ChoiceAnswer | None, offered: list[str]) -> dict[str, float] | None:
    probs = dict(getattr(answer, "probabilities", None) or {})
    kept = {o: float(probs[o]) for o in offered if o in probs}
    total = sum(kept.values())
    if not total > 0.0:
        return None
    return {o: v / total for o, v in kept.items()}


def _argmax(known: list[str], values: dict[str, float]) -> str:
    return min(known, key=lambda option: (-values[option], known.index(option), option))


class ShadowBackend(DecideBackend):
    """Serves the inner backend's answer and logs what the fused policy would have said."""

    def __init__(self, inner: DecideBackend, config: dict[str, Any] | None = None,
                 member: DecideBackend | None = None) -> None:
        cfg = dict(config if config is not None else _block())
        self.cfg = cfg
        self.inner = inner
        self.name = inner.name
        self.weights = float(cfg.get("weights", 0.5))
        protect = cfg.get("protect")
        self.protect = tuple(protect) if protect is not None else DEFAULT_PROTECT
        self.path = log_path(cfg)
        self.member = member or make_backend(str(cfg.get("member") or "classical"))

    # -- the decision ----------------------------------------------------

    def choice(self, state: Any, instructions: str,
               criteria: dict[str, str]) -> ChoiceAnswer:
        t0 = time.perf_counter()
        answer = self.inner.choice(state, instructions, criteria)
        shadow_ms = (time.perf_counter() - t0) * 1000.0
        decision_id = f"{int(time.time() * 1000):x}-{os.getpid():x}-{id(answer):x}"
        record = {
            "kind": "decision",
            "id": decision_id,
            "ts": round(time.time(), 3),
            "served": getattr(answer, "choice", None),
            "served_conf": round(float(getattr(answer, "confidence", 0.0) or 0.0), 6),
            "source": str(getattr(self.inner, "last_source", "") or ""),
            "n_options": len(criteria),
            "options": list(criteria),
        }
        try:
            record.update(self._counterfactual(state, instructions, criteria, answer))
        except Exception as exc:  # noqa: BLE001 - the shadow never breaks a request
            record["shadow_error"] = f"{type(exc).__name__}: {exc}"
        record["overhead_ms"] = round(shadow_ms - float(getattr(answer, "ms", 0.0) or 0.0), 3)
        _append(self.path, record)
        _current.set(decision_id)
        return answer

    def _vocab_match(self, offered: list[str]) -> float | None:
        """How much of the offered option set this member was even fitted to name.

        `classical` is an op router; `decide_pick` offers candidate ids. A member whose vocabulary
        matches none of the options cannot answer, and its abstention then says nothing about the
        policy being shadowed — this field is what makes that visible per record instead of leaving
        a reader to infer it from a uniform distribution.
        """
        labels = getattr(self.member, "fitted_labels", None)
        if not callable(labels) or not offered:
            return None
        try:
            fitted = set(labels())
        except Exception:  # noqa: BLE001 - a member that cannot describe itself is not fatal
            return None
        return round(len([o for o in offered if o in fitted]) / len(offered), 3)

    def _counterfactual(self, state: Any, instructions: str, criteria: dict[str, str],
                        answer: ChoiceAnswer) -> dict[str, Any]:
        offered = list(criteria)
        base_choice = getattr(answer, "choice", None)
        out: dict[str, Any] = {"protect": list(self.protect)}
        if base_choice is None or not offered:
            out.update({"fused": None, "why": "the served answer is not a choice"})
            return out
        if base_choice in self.protect:
            out.update({"member": None, "fused": base_choice, "why": "protected"})
            return out
        out["vocab_match"] = self._vocab_match(offered)
        member = self.member.choice(state, instructions, criteria)
        out["member"] = getattr(member, "choice", None)
        out["member_conf"] = round(float(getattr(member, "confidence", 0.0) or 0.0), 6)
        out["member_answered"] = out["member"] is not None
        base_dist = _dist(answer, offered)
        member_dist = _dist(member, offered)
        if out["member"] is None:
            # an abstaining member has no opinion, and averaging a uniform distribution into the
            # served answer would silently rewrite it towards the middle — on live traffic the
            # only choice question asked is `decide_pick`, whose options are candidate ids, so
            # this is the common case and it must read as itself
            out.update({"fused": base_choice, "why": "member abstained"})
            return out
        if base_dist is None or member_dist is None:
            out.update({"fused": base_choice, "why": "one side carries no distribution"})
            return out
        blended = {o: self.weights * base_dist.get(o, 0.0)
                      + (1.0 - self.weights) * member_dist.get(o, 0.0) for o in offered}
        total = sum(blended.values())
        fused = _argmax(offered, blended) if total > 0 else base_choice
        out.update({"fused": fused, "why": "mean",
                    "agrees": fused == base_choice,
                    "member_agrees_with_served": out["member"] == base_choice})
        return out

    # -- pass-through ----------------------------------------------------

    def noul(self, state: Any, question: str) -> float:
        return self.inner.noul(state, question)

    def score(self, state: Any, question: str, levels: list[str]) -> float:
        return self.inner.score(state, question, levels)

    def health(self) -> bool:
        return self.inner.health()

    @property
    def last_source(self) -> str:
        return str(getattr(self.inner, "last_source", "") or "")

    def __getattr__(self, name: str):
        """Anything the inner backend exposes, the shadow exposes: a wrapper must be invisible.

        `resolve()` returns this object now, and callers legitimately reach through it for
        `.cache`, `.gate`, `.primary` and the rest. Without this, enabling a logging feature
        changed the shape of what `resolve()` returns — thirteen tests said so.
        """
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)


def wrap(backend: DecideBackend, cfg: dict[str, Any] | None = None) -> DecideBackend:
    """Wrap the resolved backend when shadowing is on; otherwise return it untouched."""
    cfg = _block() if cfg is None else dict(cfg)
    if not cfg.get("enabled"):
        return backend
    try:
        return ShadowBackend(backend, config=cfg)
    except Exception:  # noqa: BLE001 - an unusable shadow must not take the route down
        return backend


def note_outcome(*, op: str, ok: bool, ms: float = 0.0, step: str = "",
                 error: str | None = None, cfg: dict[str, Any] | None = None) -> None:
    """Append the downstream outcome under the decision that caused it.

    Called by the runner after an op has run (or failed). A no-op when no decision is in flight,
    which is the normal case for a request that never asked for one. `attempt` counts outcomes
    per decision, so a retried step is visible as attempt 2 rather than as a second row the reader
    has to notice.
    """
    decision = _current.get()
    if not decision:
        return
    block = cfg if cfg is not None else _block()
    if not block.get("enabled"):
        return
    if len(_ATTEMPTS) > _MAX_TRACKED:
        _ATTEMPTS.clear()  # a long-running worker must not grow this without bound
    _ATTEMPTS[decision] = _ATTEMPTS.get(decision, 0) + 1
    record: dict[str, Any] = {"kind": "outcome", "id": decision, "ts": round(time.time(), 3),
                              "op": op, "step": step, "ok": bool(ok), "ms": round(ms, 3),
                              "attempt": _ATTEMPTS[decision]}
    if error:
        record["error"] = str(error)[:500]
    _append(log_path(block), record)


def attempts(decision_id: str) -> int:
    """How many outcomes this decision has produced so far. 0 means the first one is landing."""
    return _ATTEMPTS.get(decision_id, 0)


def clear() -> None:
    """Drop the in-flight decision id (tests, and any request boundary that reuses a context)."""
    _current.set("")
