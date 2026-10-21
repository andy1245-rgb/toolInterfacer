"""cascade — banded readout cascade: a second readout only when it can pay.

Registered as an ordinary DecideBackend ("cascade"), so the primitive wall in
primitives.py stays intact: callers ask for a backend name and get a
ChoiceAnswer. Config selects it through decide.cascade.enabled, which is
 authoritative for the "choice" kind in decide/resolve.py: with the flag on the
 cascade serves every choice decision whatever decide.backends_in_use.choice
 names, and with it off the backend map decides as usual.

Policy (round2/A2-frontier-corrected.md §7.2 "P2 banded sys→lr", validated at
precision budgets <= 0.85; thresholds live in decide.cascade):

  1. stage1 common conf >= tau_hi   -> commit stage1          (path stage1_confident)
  2. stage1 common conf <  tau_lo   -> defer, no stage2 call  (band skip)
  3. stage2 choice == stage1 choice -> commit stage1          (path agreement)
  4. stage2 common conf >= tau2     -> commit stage2          (path stage2_alone)
  5. otherwise                      -> defer

"Common conf" is p_max, the peak of the stage's probability vector — the one
axis on which the backends' own confidence formulas are comparable (conf_scale:
d1 reports (p_max-1/K)/(1-1/K), logit_read 1-H/lnK, so a single numeric
threshold carried across them selects different peaks). Every threshold in
decide.cascade is fit on this axis and no stage's native confidence is ever
compared against it. With the vector in hand p_max is exact;
conf_scale.to_common_scale's entropy inversion is only needed when nothing but
a conf scalar survives, and on the stored lanes it overstates logit_read's
p_max by 0.09 on average (0.27 worst case) — gating on it would move the
operating point the thresholds were fit for.

The answer handed back to the loop is on that same p_max axis:

  * commit -> the measured precision of the commit path (decide.cascade.path_conf).
    A consensus commit at p_max 0.45 is still ~0.79 precise, and the downstream
    conf_gte pre-gate (core/gates.py check_pre) has to see that number, not the
    raw peak; reporting the raw peak would turn every band commit back into an
    ask-back and delete the point of the cascade.
  * defer  -> the better readout's own p_max (or stage2's, when it is higher).
    It is below min(tau_hi, tau2) and therefore below policy.gate_slot in the
    shipped config, so the next step's conf_gte pre-gate raises GateBlocked and
    runner._handle_block turns the deferral into ask-back. Keep
    gate_slot > min(tau_hi, tau2) or deferrals will commit downstream.

Single-stage mode (decide.cascade.stage2 = null): stage1 above tau_hi commits, everything
else defers. That is the single-backend gate of the same analysis; set
path_conf.stage1_confident to the precision measured at *that* stage's fitted
tau (logit_read @0.85 tau 0.444 -> 0.85, d1_systemone @0.85 tau 0.598 -> 0.85).

Documented constraint, deliberately not enforced: banding is out-of-fold stable
only at <= RECOMMENDED_MAX_PRECISION. At a 0.90 budget the same machinery loses
12 pp against the single logit_read gate (A2 §5.2), so a 0.90 budget should run
single-stage mode with the stage's own fitted tau (logit_read 0.54, d1 0.786).
budget_warning() reports this; nothing refuses to run.
"""
from __future__ import annotations

from typing import Any, Sequence

from . import conf_scale
from .primitives import ChoiceAnswer, DecideBackend, make_backend, register

# Highest precision budget at which banding is out-of-fold validated (A2 §5.2).
RECOMMENDED_MAX_PRECISION = 0.85

# Measured precision of each commit path, in-sample on the 150-row frozen set
# (A2 §4.2 path_stats at tau_hi 0.70 / tau_lo 0.40 / tau2 0.50: 0.875 / 0.781 / 0.80).
DEFAULT_PATH_CONF = {"stage1_confident": 0.87, "agreement": 0.79, "stage2_alone": 0.80}


def budget_warning(precision_target: float) -> str | None:
    """Why banding is not recommended above RECOMMENDED_MAX_PRECISION, or None."""
    if precision_target <= RECOMMENDED_MAX_PRECISION:
        return None
    return (f"banding is validated at <= {RECOMMENDED_MAX_PRECISION:.2f} precision; "
            f"at {precision_target:.2f} it loses ~12pp coverage to a single-stage gate "
            f"(use stage2 = null with the stage's own fitted tau)")


def cascade_cfg() -> dict[str, Any]:
    """The decide.cascade block, copied (load() hands out the shared cache)."""
    from ..conf import load
    return dict((load().get("decide") or {}).get("cascade") or {})


@register("cascade")
class CascadeBackend(DecideBackend):
    """Banded cascade over one or two stage backends; see the module docstring.

    Stage backends may be injected (tests, bench) or built from
    decide.cascade.stage1 / .stage2 by name; thresholds come from `cfg` or, when
    it is None, from decide.cascade. Stages are built straight through
    primitives.make_backend, never through resolve(), so a cascade never asks for
    the "choice" backend it is serving and cannot recurse into itself.
    """

    name = "cascade"

    def __init__(self, stages: Sequence[DecideBackend] | None = None,
                 cfg: dict[str, Any] | None = None) -> None:
        cfg = dict(cfg) if cfg is not None else cascade_cfg()
        self._enabled = bool(cfg.get("enabled", False))
        self.precision_target = float(cfg.get("precision_target", RECOMMENDED_MAX_PRECISION))
        self.tau_hi = float(cfg.get("tau_hi", 0.70))
        self.tau_lo = float(cfg.get("tau_lo", 0.40))
        self.tau2 = float(cfg.get("tau2", 0.50))
        self.path_conf = {**DEFAULT_PATH_CONF, **dict(cfg.get("path_conf") or {})}
        self._stages = list(stages) if stages is not None else _configured_stages(cfg)
        if not 1 <= len(self._stages) <= 2:
            raise ValueError(f"cascade: need 1 or 2 stage backends, got {len(self._stages)}")
        self._last_stage = ""

    @property
    def last_stage(self) -> str:
        """Which policy branch produced the most recent choice() (logging only).

        One of: pass_through, stage1, agreement, stage2, band_skip, defer,
        no_opinion (no stage returned a choice, so choice() raised).
        """
        return self._last_stage

    def choice(self, state: Any, instructions: str, criteria: dict[str, str]) -> ChoiceAnswer:
        if not self._enabled:
            return self._pass_through(state, instructions, criteria)
        first = self._stages[0].choice(state, instructions, criteria)
        if first is None or first.choice is None:
            return self._without_stage1(state, instructions, criteria)
        c1 = _common(first)
        if c1 >= self.tau_hi:
            return self._settle(first, self.path_conf["stage1_confident"], "stage1")
        if len(self._stages) == 1:
            return self._settle(first, c1, "defer")
        if c1 < self.tau_lo:
            return self._settle(first, c1, "band_skip")
        second = self._stages[1].choice(state, instructions, criteria)
        if second is None:
            return self._settle(first, c1, "defer")
        c2 = _common(second)
        if second.choice is not None and second.choice == first.choice:
            return self._settle(first, self.path_conf["agreement"], "agreement")
        if c2 >= self.tau2:
            return self._settle(second, self.path_conf["stage2_alone"], "stage2")
        if second.choice is not None and c2 > c1:
            return self._settle(second, c2, "defer")
        return self._settle(first, c1, "defer")

    def noul(self, state: Any, question: str) -> float:
        return self._stages[0].noul(state, question)

    def health(self) -> bool:
        return all(stage.health() for stage in self._stages)

    def _pass_through(self, state: Any, instructions: str,
                      criteria: dict[str, str]) -> ChoiceAnswer:
        """enabled=false: exactly what the stage1 backend alone would answer."""
        self._last_stage = "pass_through"
        return self._stages[0].choice(state, instructions, criteria)

    def _without_stage1(self, state: Any, instructions: str,
                        criteria: dict[str, str]) -> ChoiceAnswer:
        """stage1 had no opinion (heuristic_micro returns None for "not mine").

        Absorb it here: None must never reach ops/decisions.run_pick.
        """
        self._last_stage = "no_opinion"
        if len(self._stages) == 1:
            raise RuntimeError("cascade: no stage produced a choice")
        second = self._stages[1].choice(state, instructions, criteria)
        if second is None or second.choice is None:
            raise RuntimeError("cascade: no stage produced a choice")
        return self._settle(second, _common(second), "stage2")

    def _settle(self, answer: ChoiceAnswer, conf: float, stage: str) -> ChoiceAnswer:
        self._last_stage = stage
        return ChoiceAnswer(answer.choice, dict(answer.probabilities), conf)


def _configured_stages(cfg: dict[str, Any]) -> list[DecideBackend]:
    stage1 = cfg.get("stage1")
    if not stage1:
        raise ValueError("cascade: decide.cascade.stage1 is not configured")
    names = [str(stage1)] + ([str(cfg["stage2"])] if cfg.get("stage2") else [])
    return [make_backend(name) for name in names]


def _common(answer: ChoiceAnswer) -> float:
    """A stage answer's confidence on the common p_max axis (conf_scale)."""
    probs = list(answer.probabilities.values())
    if not probs:
        raise ValueError("cascade: stage returned no probabilities; p_max is undefined")
    if len(probs) == 1:
        return 1.0  # single option owns all mass; conf_scale.d1_conf uses the same K<2 rule
    return conf_scale.pmax_from_probs(probs)
