"""service_entry — useTool body → inspector → plan → runner result.

This is the only place that glues: inspector (which template/level), template
apply (the steps), decide resolution (backend per kind), runner execution,
hops logging, and the RunResult → useTool-response mapping.
"""
from __future__ import annotations

import time
from typing import Any

from .conf import load
from .core.budget import Budget
from .core.runner import run_plan, Step
from .hops.log import HopLog
from .ops.op_api import get as get_op, validate_params
from .policy.askback import Question
from .policy.confidence import gate_route
from .planner.inspector import inspect


def use_tool(body: dict[str, Any], dry: bool = False, on_ask: Any = None) -> dict[str, Any]:
    from .ops.op_api import import_all
    import_all()
    intent = str(body.get("intent", "")).strip()
    if not intent:
        return {"status": "error", "error": "empty intent", "hops": []}
    params: dict[str, Any] = dict(body.get("params") or {})
    digest: dict[str, Any] = body.get("digest") or {}
    mode = str(body.get("mode") or "auto")

    meta = inspect(intent, params, digest)

    if dry or mode == "plan":
        return _plan_view(meta, intent, params, digest)

    if meta["level"] == "L0":
        return _passthrough(params)
    if meta["template"] and meta["level"] == "L2":
        from .planner.templates import apply_template
        steps, final_key = apply_template(meta["template"], intent, digest)
        return _run_steps(steps, {}, final_key, intent, params,
                          on_ask=on_ask, log_hops=True)
    # L1 disk-slot-fill bridge
    from .planner.templates.disk_lookup import apply
    return _run_steps(apply(intent, digest), {}, "content", intent, params,
                      on_ask=on_ask, log_hops=True)


def _passthrough(params: dict[str, Any]) -> dict[str, Any]:
    """L0: params carry tool + full args. Validate, execute, return. No decide."""
    t0 = time.monotonic()
    tool = params.get("tool")
    args_in = dict(params.get("args") or {})
    if not tool:
        return {"status": "error", "error": "L0 requires params.tool", "hops": []}
    try:
        args, soft = validate_params(str(tool), args_in)
        opdef = get_op(str(tool))
        _kind, data, _conf = opdef.run(args, None)
        return {"status": "done", "result": data,
                "degraded": False, "assumptions": list(soft),
                "confidence": 1.0,
                "hops": [{"step": "L0", "op": str(tool),
                          "ms": round((time.monotonic() - t0) * 1000, 1),
                          "conf": 1.0,
                          "artifact_summary": str(data)[:120]}]}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc), "hops": []}


def _run_steps(steps: list[Step], llm_params: dict[str, Any], final_key: str,
               intent: str, params: dict[str, Any], on_ask: Any, log_hops: bool) -> dict[str, Any]:
    bcfg = load()["budget"]
    budget = Budget(max_hops=int(bcfg["max_hops"]), max_asks=int(bcfg["max_asks"]),
                    max_seconds=float(bcfg["max_seconds"]))
    log = HopLog() if log_hops else None

    def ask(q: Question) -> Any:
        if on_ask is None or budget.asks >= budget.max_asks:
            from .policy.askback import MaxAsksReached
            raise MaxAsksReached()
        budget.use_ask()
        return on_ask(q)  # the extension surfaces it to the LLM; tests record answers

    t0 = time.monotonic()
    res = run_plan(steps, {"intent_context": intent, **params}, budget, ask,
                   _run_op_with_log(log), _noul_verifier, final_key=final_key)
    ms = round((time.monotonic() - t0) * 1000, 1)
    out: dict[str, Any] = {
        "status": res.status, "result": res.result,
        "degraded": res.degraded, "assumptions": res.assumptions,
        "confidence": _route_conf(res.hops),
        "hops": res.hops, "wall_ms": ms,
        "meta": {"level": inspect(intent, params, {})["level"],
                 "route_gate": gate_route()},
    }
    if res.status == "need_input" and on_ask is None and not res.degraded:
        out["question"] = {"kind": "choice", "question": "resolve step", "options": []}
    return out


def _route_conf(hops: list[dict[str, Any]]) -> float:
    decide_hops = [h["conf"] for h in hops if h["op"].startswith("decide") or h["op"] == "askback"]
    if not decide_hops:
        return 1.0
    return max(0.0, min(decide_hops))


def _run_op_with_log(log: HopLog | None):
    def run(op_name: str, opdef: Any, args: dict[str, Any], store: Any, soft: dict[str, Any]) -> tuple[str, Any, float]:
        t0 = time.monotonic()
        kind, data, conf = opdef.run(args, store)
        if log:
            log.append({"op": op_name, "params": {k: str(v)[:200] for k, v in args.items()},
                        "soft": {k: str(v)[:100] for k, v in soft.items()},
                        "conf": conf, "kind": kind,
                        "ms": round((time.monotonic() - t0) * 1000, 1),
                        "artifact_summary": str(data)[:200]})
        return kind, data, conf

    return run


def _noul_verifier(name: str, artifact: Any) -> float | None:
    from .decide import resolve as resolve_backend
    backend = resolve_backend("noul")
    state = {"kind": artifact.kind, "data": str(artifact.data)[:1500]}
    return backend.noul(state, f"verification: {name} — does this {artifact.kind} satisfy it?")


def _plan_view(meta: dict[str, Any], intent: str, params: dict[str, Any],
               digest: dict[str, Any]) -> dict[str, Any]:
    from .planner.templates import apply_template
    from .planner.templates.disk_lookup import apply as apply_disk
    if meta["level"] == "L0":
        steps = [Step(id="L0", op=str(params.get("tool", "?")), params=dict(params.get("args") or {}))]
        final_key = "L0"
    elif meta["template"]:
        steps, final_key = apply_template(str(meta["template"]), intent, digest)
    else:
        steps, final_key = apply_disk(intent, digest), "content"
    return {"status": "plan",
            "meta": meta,
            "final_key": final_key,
            "steps": [{"id": s.id, "op": s.op, "params": _safe(s.params), "inject": s.inject,
                       "gate_pre": s.gate.pre, "gate_post": s.gate.post} for s in steps],
            "hops": [],
            "ops_available": True}


def _safe(p: dict[str, Any]) -> dict[str, Any]:
    return {k: str(v)[:200] for k, v in p.items()}
