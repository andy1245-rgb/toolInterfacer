"""pair — a local decision first, a hosted one only when the local one cannot commit.

Registered as an ordinary DecideBackend ("pair"), so the primitive wall in
primitives.py is unchanged: callers ask for a backend name and get a
ChoiceAnswer. Both stages are built by name, straight through
primitives.make_backend and never through resolve(), so a pair never asks for the
"choice" backend it is itself serving and cannot recurse into itself.

Config (decide.pair in conf/defaults.json):

    "pair": {"stage1": "d1_systemone", "stage2": "openrouter",
             "tau": 0.90, "escalate_on_error": true}

``tau`` is on the common p_max axis (conf_scale), not on either backend's native
confidence, and that is the only axis the two formulas are comparable on: the
local stage reports ``(p_max - 1/K) / (1 - 1/K)`` while a hosted chat completion
reports the single number it asserted, so one numeric threshold read against
either as-is selects a different peak per stage. ``tau`` is therefore compared
against stage1's p_max — the peak of its probability vector, exact when the
vector is in hand. It is never applied to stage2: stage2 answers one *choice*,
and the answer handed back carries its own confidence, which is the number the
slot gate (policy.gate_slot via core/gates.check_pre) judges.

Policy, in this order:

  1. p_max(stage1) >= tau  -> stage1's answer, unchanged, and stage2 is not
                              called at all            (last_reason: confident)
  2. p_max(stage1) <  tau  -> stage2's answer          (low_confidence)
  3. stage1 raises and escalate_on_error is true (the default, and the whole
     point of a hard fallback) -> stage2's answer      (stage1_error)
  4. stage1 raises and escalate_on_error is false -> the error propagates
  5. stage1 answered below tau AND stage2 raises -> stage1's answer, as-is, and
     the error is *not* propagated                     (stage2_error)

Rule 5 is deliberate. The caller asked for a decision and stage1 had one; a
transport failure on the escalation must not turn a usable degraded decision into
a hard error. stage1's own confidence travels with the answer, so a degraded
answer still asks back downstream when it should — the gate, not this module,
decides whether it is good enough.

A stage that produces nothing (None, an answer with no choice, or one with no
probabilities) produced no decision and is treated as a stage error: None must
never reach ops/decisions.run_pick. When neither stage produces one, choice()
raises.

``noul`` has no p_max — one probability for one question is not a distribution
over candidates — so tau does not apply to it and it never escalates on a
threshold: stage1 answers, and stage2 answers only when stage1 raised.

Service requirement: stage2 is the openrouter backend, which reads its key per
call from the environment variable named by decide.openrouter.key_env
(TI_OPENROUTER_API_KEY). The service therefore needs TI_OPENROUTER_API_KEY in its
environment. If stage2 is unavailable — no key, transport failure, 429/5xx after
its own retries — every escalation raises OpenRouterError and rule 5 turns that
into stage1's answer, so the pair degrades to the local model instead of failing
the run.

Observability, because a measurement workstream reads it: ``last_stage``
("stage1" | "stage2" | "" when neither answered), ``last_escalated``,
``last_reason`` (confident | low_confidence | stage1_error | stage2_error) and
``last_stage1_conf`` (stage1's p_max, 0.0 when it did not answer) live on the
instance rather than in ChoiceAnswer — that dataclass is the primitive every
backend shares and this module does not get to grow it. One DEBUG line per call
carries the same values. The fields describe the most recent call on *this*
instance: ops/decisions builds a backend per hop, and a pre-gate or cache hit in
resolve.FastPathBackend answers without calling choice() at all, so a bench holds
its own instance instead of reading a shared one.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

from . import conf_scale
from .primitives import ChoiceAnswer, DecideBackend, make_backend, register

log = logging.getLogger(__name__)

STAGE1 = "stage1"
STAGE2 = "stage2"

CONFIDENT = "confident"
LOW_CONFIDENCE = "low_confidence"
STAGE1_ERROR = "stage1_error"
STAGE2_ERROR = "stage2_error"


def pair_cfg() -> dict[str, Any]:
    """The decide.pair block, copied (load() hands out the shared cache)."""
    from ..conf import load
    return dict((load().get("decide") or {}).get("pair") or {})


@register("pair")
class PairBackend(DecideBackend):
    """Local stage1 with a hosted stage2 behind it; see the module docstring.

    Stages may be injected (tests, bench) or built from decide.pair.stage1 /
    .stage2 by name; tau and escalate_on_error come from `cfg` or, when it is
    None, from decide.pair.
    """

    name = "pair"

    def __init__(self, stages: Sequence[DecideBackend] | None = None,
                 cfg: dict[str, Any] | None = None) -> None:
        cfg = dict(cfg) if cfg is not None else pair_cfg()
        self.tau = float(cfg.get("tau", 0.90))
        self.escalate_on_error = bool(cfg.get("escalate_on_error", True))
        self._stages = list(stages) if stages is not None else _configured_stages(cfg)
        if len(self._stages) != 2:
            raise ValueError(f"pair: need 2 stage backends, got {len(self._stages)}")
        self._last_stage = ""
        self._last_escalated = False
        self._last_reason = ""
        self._last_stage1_conf = 0.0

    @property
    def last_stage(self) -> str:
        """Which stage answered the most recent call; "" when neither did (logging only)."""
        return self._last_stage

    @property
    def last_escalated(self) -> bool:
        """Whether stage2 was called on the most recent one."""
        return self._last_escalated

    @property
    def last_reason(self) -> str:
        """Why the most recent call ended where it did; see the module docstring."""
        return self._last_reason

    @property
    def last_stage1_conf(self) -> float:
        """stage1's p_max on the most recent call, 0.0 when it answered nothing."""
        return self._last_stage1_conf

    def choice(self, state: Any, instructions: str, criteria: dict[str, str]) -> ChoiceAnswer:
        self._reset()
        # A stage is a network call: any failure to answer is a fallback trigger, not
        # just the transport's own error class, so the catch below is deliberately broad.
        try:
            first = _require_choice(self._stages[0].choice(state, instructions, criteria), STAGE1)
            c1 = _common(first)
        except Exception as exc:
            if not self.escalate_on_error:
                self._record("", STAGE1_ERROR, escalated=False, detail=f"stage1: {exc}")
                raise
            return self._escalate(state, instructions, criteria, stage1=None,
                                  reason=STAGE1_ERROR, detail=f"stage1: {exc}")
        self._last_stage1_conf = c1
        if c1 >= self.tau:
            self._record(STAGE1, CONFIDENT, escalated=False)
            return first
        return self._escalate(state, instructions, criteria, stage1=first,
                              reason=LOW_CONFIDENCE)

    def noul(self, state: Any, question: str) -> float:
        """stage1, then stage2 only when stage1 raised; no threshold either way."""
        self._reset()
        try:
            value = self._stages[0].noul(state, question)
        except Exception as exc:
            try:
                value = self._stages[1].noul(state, question)
            except Exception as exc2:
                self._record("", STAGE2_ERROR, escalated=True,
                             detail=f"stage1: {exc}; stage2: {exc2}")
                raise
            self._record(STAGE2, STAGE1_ERROR, escalated=True, detail=f"stage1: {exc}")
            return value
        self._record(STAGE1, CONFIDENT, escalated=False)
        return value

    def health(self) -> bool:
        return all(stage.health() for stage in self._stages)

    def _escalate(self, state: Any, instructions: str, criteria: dict[str, str], *,
                  stage1: ChoiceAnswer | None, reason: str,
                  detail: str = "") -> ChoiceAnswer:
        """Call stage2, falling back to `stage1`'s answer when there is one (rule 5)."""
        try:
            second = _require_choice(self._stages[1].choice(state, instructions, criteria),
                                     STAGE2)
        except Exception as exc:
            both = "; ".join(part for part in (detail, f"stage2: {exc}") if part)
            if stage1 is None:
                self._record("", STAGE2_ERROR, escalated=True, detail=both)
                raise RuntimeError("pair: no stage produced a choice") from exc
            self._record(STAGE1, STAGE2_ERROR, escalated=True, detail=both)
            return stage1
        self._record(STAGE2, reason, escalated=True, detail=detail)
        return second

    def _reset(self) -> None:
        """Clear the previous call's record, so nothing stale survives a path change."""
        self._last_stage = ""
        self._last_escalated = False
        self._last_reason = ""
        self._last_stage1_conf = 0.0

    def _record(self, stage: str, reason: str, *, escalated: bool,
                detail: str = "") -> None:
        self._last_stage = stage
        self._last_escalated = escalated
        self._last_reason = reason
        log.debug("pair: stage=%s escalated=%s reason=%s stage1_p_max=%.3f%s",
                  stage or "-", escalated, reason, self._last_stage1_conf,
                  f" ({detail})" if detail else "")


def _configured_stages(cfg: dict[str, Any]) -> list[DecideBackend]:
    stage1 = cfg.get("stage1")
    if not stage1:
        raise ValueError("pair: decide.pair.stage1 is not configured")
    stage2 = cfg.get("stage2")
    if not stage2:
        raise ValueError("pair: decide.pair.stage2 is not configured — a pair needs both")
    return [make_backend(str(name)) for name in (stage1, stage2)]


def _require_choice(answer: ChoiceAnswer | None, stage: str) -> ChoiceAnswer:
    """The answer, or ValueError when the stage produced no usable decision."""
    if answer is None or not answer.choice or not answer.probabilities:
        raise ValueError(f"pair: {stage} produced no decision")
    return answer


def _common(answer: ChoiceAnswer) -> float:
    """A stage answer's confidence on the common p_max axis (conf_scale)."""
    probs = list(answer.probabilities.values())
    if len(probs) == 1:
        return 1.0  # single option owns all mass; conf_scale.d1_conf uses the same K<2 rule
    return conf_scale.pmax_from_probs(probs)
