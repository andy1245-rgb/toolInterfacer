from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.artifacts import ArtifactStore
from ..core.budget import Budget, BudgetExceeded
from ..core.gates import Gate, GateBlocked, check_post, check_pre
from ..ops.op_api import get as get_op, validate_params

run_op_t = Callable[..., tuple[str, Any, "float | None"]]
run_verifier_t = Callable[..., "float | None"]
on_ask_t = Callable[..., Any]

# What a resolved ask-back writes over the artifact a gate rejected. It is a
# policy prior — a stated trust in the answering party's statement — not a
# measurement, and it exists because the retry has to clear the same threshold
# that blocked it. It is the one assigned confidence left on the reporting path;
# see docs/confidence-sources.md.
ASKBACK_CONF = 0.9


@dataclass
class Step:
    id: str
    op: str
    params: dict[str, Any] = field(default_factory=dict)   # spec-side params (config-resolved)
    inject: dict[str, str] = field(default_factory=dict)   # step-param -> artifact key to feed in
    gate: Gate = field(default_factory=Gate)
    #: param -> artifact key: run the op once per list item instead of once for
    #: the whole list. Each item gets its own artifact, hop and post-gate verdict,
    #: so a `noul` check decides per page rather than per batch.
    over: dict[str, str] = field(default_factory=dict)
    #: end a `over` loop as soon as this many items cleared their post-gate (0 = run all)
    stop_after: int = 0
    #: a failure here is recorded as an assumption and the plan continues
    optional: bool = False


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
             final_key: str | None = None,
             pre_assumptions: list[str] | None = None) -> RunResult:
    """Execute a linear plan: op → artifact → gates → next step.

    run_op_fn(op_name, args=dict, intent_ctx=dict, store) -> (kind, data, conf)
    run_verifier_fn(verifier_name, artifact) -> float | None (None = skip)
    on_ask(Question) -> answer dict {"choice": id, ...} — or raises
    MaxAsksReached, which converts the run to degraded-done.
    """
    store = ArtifactStore()
    hops: list[dict[str, Any]] = []
    # Facts the plan establishes before the first step (e.g. the directory a
    # listing could not be scoped to) belong in the same record as the rest.
    assumptions: list[str] = list(pre_assumptions or [])
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
            if step.over:
                # the loop verifies each of its items itself; there is no single
                # artifact left for a post-check to judge
                art, usable = _run_over(step, llm_params, store, hops, assumptions,
                                        budget, run_op_fn, run_verifier_fn)
                if not usable:
                    degraded = True
                cur += 1
                continue
            art = _run_one(step, llm_params, store, hops, assumptions, run_op_fn)
        except Exception as exc:  # noqa: BLE001 — runner converts op errors into structured status
            if not step.optional:
                return finish("error", str(exc))
            degraded = True
            assumptions.append(f"{step.id}: optional step {step.op!r} failed ({exc})")
            cur += 1
            continue

        try:
            check_post(step.gate, step.id, art, run_verifier_fn)
        except GateBlocked as gb:
            if step.optional:
                degraded = True
                assumptions.append(f"{step.id}: optional step {step.op!r} unverified ({gb.what})")
                cur += 1
                continue
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


def _run_one(step: Step, llm_params: dict[str, Any], store: ArtifactStore,
             hops: list[dict[str, Any]], assumptions: list[str],
             run_op_fn: run_op_t) -> Any:
    """Execute a step once over its injected artifacts (the plain path)."""
    args, soft = _merged_args(step, _injected(step, store, assumptions), llm_params)
    kind, data, conf, ms = _call(step, args, soft, store, run_op_fn)
    art = store.put(step.id, step.op, kind, data, conf)
    hops.append({"step": step.id, "op": step.op, "ms": ms, "conf": conf,
                 "artifact_summary": _summarize(kind, data)})
    return art


def _run_over(step: Step, llm_params: dict[str, Any], store: ArtifactStore,
              hops: list[dict[str, Any]], assumptions: list[str], budget: Budget,
              run_op_fn: run_op_t, run_verifier_fn: run_verifier_t,
              ) -> tuple[Any, bool]:
    """One execution per item of the artifact the step iterates.

    Each item is bound to its param, executed, and verified on its own — the
    per-page `noul` verdict of spec §7.5 needs one page per decision, not one
    decision over a batch. A rejected item is dropped with an assumption naming
    the measurement; it is not an error, because "this page is not relevant" is
    the answer the check exists to give. `stop_after` ends the loop once enough
    items have verified, so a run that already has what it came for does not keep
    paying for CLI calls.

    Returns the aggregate artifact (the surviving items, in input order) and
    whether anything survived at all — an empty aggregate is a degraded result,
    not a successful one.
    """
    (pname, src), = step.over.items()
    source = store.try_get(src)
    if source is None:
        assumptions.append(f"{step.id}: missing loop artifact {src!r}")
        items: list[Any] = []
    else:
        items = source.data if isinstance(source.data, list) else [source.data]

    kept: list[Any] = []
    kept_conf: list[float] = []
    for index, item in enumerate(items):
        try:
            budget.use_hop()
            budget.tick_clock()
        except BudgetExceeded as be:
            assumptions.append(f"{step.id}: loop stopped after {index} of {len(items)} "
                               f"({be.what})")
            break
        args, soft = _merged_args(step, {pname: item}, llm_params)
        kind, data, conf, ms = _call(step, args, soft, store, run_op_fn)
        item_key = f"{step.id}#{index}"
        item_art = store.put(item_key, step.op, kind, data, conf)
        hop = {"step": item_key, "op": step.op, "ms": ms, "conf": conf,
               "artifact_summary": _summarize(kind, data)}
        hops.append(hop)
        try:
            check_post(step.gate, item_key, item_art, run_verifier_fn)
        except GateBlocked as gb:
            hop["verified"] = False
            hop["gate"] = gb.what
            assumptions.append(f"{item_key}: dropped ({gb.what})")
            continue
        hop["verified"] = True
        kept.append(data)
        kept_conf.append(conf)
        if step.stop_after and len(kept) >= step.stop_after:
            break

    if step.stop_after and len(kept) < step.stop_after:
        assumptions.append(f"{step.id}: {len(kept)} of {len(items)} items verified, "
                           f"target {step.stop_after}")
    aggregate = store.put(step.id, step.op, get_op(step.op).kind, _collect(kept),
                          sum(kept_conf) / len(kept_conf) if kept_conf else 0.0)
    return aggregate, bool(kept)


def _injected(step: Step, store: ArtifactStore,
              assumptions: list[str]) -> dict[str, Any]:
    """Resolve `inject` against the store: step-param -> that artifact's data."""
    bound: dict[str, Any] = {}
    for pname, art_key in step.inject.items():
        injected = store.try_get(art_key)
        if injected is None:
            assumptions.append(f"{step.id}: missing injected artifact {art_key!r}")
            bound[pname] = "(missing)"
            continue
        data = injected.data
        # top-1 picks feed singular params (e.g. a single path);
        # multi-picks feed list params (e.g. urls batch)
        if injected.kind == "picked_items" and isinstance(data, list) and len(data) == 1:
            data = data[0]
        bound[pname] = data
    return bound


def _merged_args(step: Step, bound: dict[str, Any],
                 llm_params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Args for one execution: template values, then artifacts, then caller hints.

    A step's own params are the template's spec-side values and outrank a caller
    hint of the same name; an injected or looped artifact outranks both. Keys
    matching the op's schema land as args, the rest ride along as soft context.
    """
    merged = {**step.params, **llm_params}
    merged.update(step.params)
    merged.update(bound)
    return validate_params(step.op, merged)


def _call(step: Step, args: dict[str, Any], soft: dict[str, Any], store: ArtifactStore,
          run_op_fn: run_op_t) -> tuple[str, Any, float, float]:
    t0 = time.monotonic()
    kind, data, conf = run_op_fn(step.op, get_op(step.op), args, store, soft)
    return kind, data, conf, round((time.monotonic() - t0) * 1000, 1)


def _collect(datas: list[Any]) -> list[Any]:
    """Concatenate per-item lists into one flat aggregate (op data stays a list)."""
    out: list[Any] = []
    for data in datas:
        if isinstance(data, list):
            out.extend(data)
        else:
            out.append(data)
    return out


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
    ask_hop: dict[str, Any] = {"step": step.id, "op": "askback", "ms": 0.0,
                              "conf": 0.0, "ask_available": False,
                              "artifact_summary": gb.what}
    hops.append(ask_hop)
    try:
        q = Question(kind="choice" if options else "free",
                     question=f"Step {step.id!r} blocked: {gb.what}. Answer to resolve:",
                     options=options, hops_seen=hops)
        answer = on_ask(q)
    except MaxAsksReached:
        return False
    ask_hop["ask_available"] = True
    pick = None
    if isinstance(answer, dict):
        pick = answer.get("choice") or answer.get("text")
    if pick is not None and isinstance(gb.detail, dict) and gb.detail.get("artifact"):
        store.put(str(gb.detail["artifact"]), "askback", "picked_items", [str(pick)],
                  ASKBACK_CONF)
        hops.append({"step": step.id, "op": "askback.resolved", "ms": 0.0,
                     "conf": ASKBACK_CONF, "ask_available": True,
                     "resolves": str(gb.detail["artifact"]),
                     "artifact_summary": str(pick)[:120]})
        return True
    assumptions.append(f"{step.id}: asked, answer not applicable: {str(pick)[:120]}")
    return False


def _summarize(kind: str, data: Any) -> str:
    if isinstance(data, list):
        return f"{kind}[{len(data)}]  e.g. {str(data[0])[:80] if data else ''}"
    return f"{kind} :: {str(data)[:120]}"
