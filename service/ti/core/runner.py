from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.artifacts import ArtifactStore
from ..core.budget import Budget, BudgetExceeded
from ..core.gates import Gate, GateBlocked, check_post, check_pre
from ..ops.op_api import get as get_op, validate_params

run_op_t = Callable[..., tuple[str, Any, float]]
run_verifier_t = Callable[..., "float | None"]
on_ask_t = Callable[..., Any]


@dataclass
class Step:
    id: str
    op: str
    params: dict[str, Any] = field(default_factory=dict)   # spec-side params (config-resolved)
    inject: dict[str, str] = field(default_factory=dict)   # step-param -> artifact key to feed in
    gate: Gate = field(default_factory=Gate)


@dataclass
class RunResult:
    status: str                     # "done" | "need_input" | "error"
    result: Any = None
    degraded: bool = False
    assumptions: list[str] = field(default_factory=list)
    hops: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


def run_plan(steps: list[Step], llm_params: dict[str, Any], budget: Budget,
             on_ask: on_ask_t, run_op_fn: run_op_t, run_verifier_fn: run_verifier_t,
             final_key: str | None = None) -> RunResult:
    """Execute a linear plan: op → artifact → gates → next step.

    run_op_fn(op_name, args=dict, intent_ctx=dict, store) -> (kind, data, conf)
    run_verifier_fn(verifier_name, artifact) -> float | None (None = skip)
    on_ask(Question) -> answer dict {"choice": id, ...} — or raises
    MaxAsksReached, which converts the run to degraded-done.
    """
    store = ArtifactStore()
    hops: list[dict[str, Any]] = []
    assumptions: list[str] = []
    degraded = False

    def finish(status: str, error: str | None = None) -> RunResult:
        result: Any = None
        if final_key is not None:
            art = store.try_get(final_key)
            result = art.data if art else None
        elif hops:
            result = hops[-1].get("artifact_summary")
        return RunResult(status=status, result=result, degraded=degraded,
                         assumptions=assumptions, hops=hops, error=error)

    cur = 0
    while cur < len(steps):
        step = steps[cur]
        try:
            budget.use_hop()
            budget.tick_clock()
        except BudgetExceeded as be:
            degraded = True
            assumptions.append(f"budget stop: {be.what}")
            return finish("need_input" if not hops else "done")

        # Pre-gate: a blocked step with no ask-back left still runs, degraded. A
        # best-guess artifact has information value; an empty result does not
        # (spec §4). Only a resolved ask-back retries the step.
        try:
            check_pre(step.gate, step.id, store)
        except GateBlocked as gb:
            handled = _handle_block(gb, step, store, hops, assumptions, on_ask)
            if handled is True:
                continue  # the ask-back filled the gap; retry the same step
            degraded = True
            assumptions.append(f"{step.id}: gate unresolved, using the best guess ({gb.what})")

        try:
            conforms: dict[str, Any] = dict(step.params)
            for pname, art_key in step.inject.items():
                injected = store.try_get(art_key)
                if injected is None:
                    assumptions.append(f"{step.id}: missing injected artifact {art_key!r}")
                    conforms[pname] = "(missing)"
                    continue
                data = injected.data
                # top-1 picks feed singular params (e.g. a single path);
                # multi-picks feed list params (e.g. urls batch)
                if injected.kind == "picked_items" and isinstance(data, list) and len(data) == 1:
                    data = data[0]
                conforms[pname] = data
            # LLM params matching the op's schema land as args; the rest soften
            merged = {**step.params, **llm_params, **conforms}
            args, soft = validate_params(step.op, merged)
            opdef = get_op(step.op)

            t0 = time.monotonic()
            kind, data, conf = run_op_fn(step.op, opdef, args, store, soft)
            ms = round((time.monotonic() - t0) * 1000, 1)
            art = store.put(step.id, step.op, kind, data, conf)
            hops.append({"step": step.id, "op": step.op, "ms": ms, "conf": conf,
                         "artifact_summary": _summarize(kind, data)})
        except Exception as exc:  # noqa: BLE001 — runner converts op errors into structured status
            return finish("error", str(exc))

        try:
            check_post(step.gate, step.id, art, run_verifier_fn)
        except GateBlocked as gb:
            handled = _handle_block(gb, step, store, hops, assumptions, on_ask)
            if handled is True:
                continue
            if handled is False:
                degraded = True
                assumptions.append(f"{step.id}: verification failed, using artifact as-is")
                cur += 1
                continue
            raise
        cur += 1

    return finish("done")


def _handle_block(gb: GateBlocked, step: Step, store: ArtifactStore,
                  hops: list[dict[str, Any]], assumptions: list[str],
                  on_ask: on_ask_t) -> bool:
    """True = retry step (ask answered), False = skip step, raise = give up.

    A choice answer is LLM-trusted: it re-writes the artifact the gate named
    with conf 0.9 so the retry proceeds. A free answer only lands as an
    assumption tag (resolver logic uses it via context next run).
    """
    from ..policy.askback import MaxAsksReached, Question
    src_art = None
    # choose the most informative artifact of the plan so far for choice-building
    arts = [a for a in store.all().values() if a.kind in ("path_list", "picked_items", "url_list")]
    if arts:
        src_art = arts[-1]
    options: list[dict[str, Any]] = []
    if src_art and isinstance(src_art.data, list):
        for it in src_art.data[:12]:
            options.append({"id": str(it if not isinstance(it, dict) else it.get("id", it)), "desc": ""})
    hops.append({"step": step.id, "op": "askback", "ms": 0.0,
                 "conf": 0.0, "artifact_summary": gb.what})
    try:
        q = Question(kind="choice" if options else "free",
                     question=f"Step {step.id!r} blocked: {gb.what}. Answer to resolve:",
                     options=options, hops_seen=hops)
        answer = on_ask(q)
    except MaxAsksReached:
        return False
    pick = None
    if isinstance(answer, dict):
        pick = answer.get("choice") or answer.get("text")
    if pick is not None and isinstance(gb.detail, dict) and gb.detail.get("artifact"):
        store.put(str(gb.detail["artifact"]), "askback", "picked_items", [str(pick)], 0.9)
        hops.append({"step": step.id, "op": "askback.resolved", "ms": 0.0,
                     "conf": 0.9, "artifact_summary": str(pick)[:120]})
        return True
    assumptions.append(f"{step.id}: asked, answer not applicable: {str(pick)[:120]}")
    return False


def _summarize(kind: str, data: Any) -> str:
    if isinstance(data, list):
        return f"{kind}[{len(data)}]  e.g. {str(data[0])[:80] if data else ''}"
    return f"{kind} :: {str(data)[:120]}"
