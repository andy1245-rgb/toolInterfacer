"""resolve — pick the backend serving a decision kind, plus the `choice` fast path.

Precedence for the ``choice`` kind, outermost first:

  1. decide.cascade.enabled is authoritative. True means the cascade serves every
     choice decision whatever backends_in_use.choice names (the flag alone
     switches it on, so the two keys cannot drift out of sync); false leaves the
     backend map in charge, exactly as before.
  2. decide.backends_in_use.choice (or ``default``) names the backend that
     answers the decision.
  3. decide.routing wraps that backend in FastPathBackend, which answers — in
     this order — from a regex pre-gate, then the decision cache, then the
     backend itself. ``heuristic_micro_then_d1`` (the shipped value) puts
     heuristic_micro in front: it commits a hop for free when the state and the
     criteria agree on a URL fetch or a reddit digest, and abstains (None) for
     everything else, which is most questions — it only knows those two criteria
     keys. Any other value, ``direct`` included, leaves the gate off.

The cache is not part of that chain: decide.cache.enabled (on by default) decides
whether it wraps whatever rules 1-3 selected, and the gate sits in front of the
cascade too when both are on. On the CPU-only box a /v1/systemone call costs
seconds while a gate hit or a cache hit costs microseconds; that asymmetry is the
whole point of both.
"""
from __future__ import annotations

from typing import Any, Callable

from . import cache as cache_mod
from .primitives import ChoiceAnswer, DecideBackend, make_backend

# decide.routing values that put a pre-gate backend in front of the configured one.
PRE_GATES = {"heuristic_micro_then_d1": "heuristic_micro"}


def resolve(kind: str) -> DecideBackend:
    """Pick the backend serving a decision kind; see the module docstring."""
    if kind != "choice":
        return _mapped(kind)
    primary = make_backend("cascade") if cascade_enabled() else _mapped(kind)
    return FastPathBackend(primary, gate=pre_gate(),
                           cache=cache_mod.shared() if cache_mod.enabled() else None)


def _mapped(kind: str) -> DecideBackend:
    mapping = load_backends_in_use()
    name = mapping.get(kind) or mapping.get("default")
    if not name:
        raise KeyError(f"no decide backend configured for kind {kind!r}")
    return make_backend(name)


class FastPathBackend(DecideBackend):
    """`choice` served by pre-gate → cache → the configured backend, in that order.

    Built by resolve() around the backend the config selected and never
    registered: it is not a backend of its own, so nothing can select it by name
    and the primitive wall stays primitives. `primary` is the wrapped backend,
    `name` is that backend's name — a log line reads the same either way.

    `last_source` says which of the three answered the most recent call:

      * "pre_gate" — heuristic_micro committed; no backend call, no cache traffic.
      * "cache"    — a stored answer for the same (state, instructions, criteria),
                     returned as a copy, so it is indistinguishable from a fresh one.
      * "primary"  — the backend answered, and the answer was stored first.

    The gate goes before the cache because it is a regex over the state: cheaper
    than the digest a lookup needs, and a question it can settle needs no table
    entry. Only `choice` is cached or gated; the other primitives pass straight
    through.
    """

    def __init__(self, primary: DecideBackend, *, gate: DecideBackend | None = None,
                 cache: cache_mod.DecisionCache | None = None) -> None:
        self._primary = primary
        self._gate = gate
        self._cache = cache
        self.name = primary.name
        self._last_source = ""

    @property
    def primary(self) -> DecideBackend:
        """The backend the config selected (tests and bench unwrap the fast path here)."""
        return self._primary

    @property
    def gate(self) -> DecideBackend | None:
        """The pre-gate backend, or None when decide.routing asks for a direct route."""
        return self._gate

    @property
    def cache(self) -> cache_mod.DecisionCache | None:
        """The cache this fast path reads, or None when decide.cache.enabled is off."""
        return self._cache

    @property
    def last_source(self) -> str:
        """Which of the three answered the most recent choice() — logging only."""
        return self._last_source

    def choice(self, state: Any, instructions: str,
               criteria: dict[str, str]) -> ChoiceAnswer:
        if self._gate is not None:
            gated = self._gate.choice(state, instructions, criteria)
            if gated is not None and gated.choice is not None:
                self._last_source = "pre_gate"
                return gated
        key = ""
        if self._cache is not None:
            key = cache_mod.choice_key(state, instructions, criteria)
            hit = self._cache.get(key)
            if hit is not None:
                self._last_source = "cache"
                return hit
        answer = self._primary.choice(state, instructions, criteria)
        if self._cache is not None and answer is not None and answer.choice is not None:
            self._cache.put(key, answer)
        self._last_source = "primary"
        return answer

    def noul(self, state: Any, question: str) -> float:
        return self._primary.noul(state, question)

    def health(self) -> bool:
        return self._primary.health()


def pre_gate() -> DecideBackend | None:
    """The pre-gate decide.routing names, or None when it asks for a direct route."""
    name = PRE_GATES.get(routing_name())
    return make_backend(name) if name else None


def routing_name() -> str:
    """decide.routing as configured — the fast-path policy name."""
    from ..conf import load
    return str(load()["decide"].get("routing") or "")


def cascade_enabled() -> bool:
    """Whether decide.cascade.enabled is on (same key/default cascade.py reads)."""
    from .cascade import cascade_cfg
    return bool(cascade_cfg().get("enabled", False))


def load_backends_in_use() -> dict[str, str]:
    from ..conf import load
    return dict(load()["decide"]["backends_in_use"])


def decide_factory() -> Callable[[str], DecideBackend]:
    """The runner-facing factory: kind name → backend."""
    return resolve


def as_state(x: Any) -> str:
    return x if isinstance(x, str) else json_str(x)


def json_str(x: Any) -> str:
    import json
    return json.dumps(x, default=str)
