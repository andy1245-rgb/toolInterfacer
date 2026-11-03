"""service_entry — useTool body → inspector → plan → runner result.

This is the only place that glues: inspector (which template/level), template
apply (the steps), decide resolution (backend per kind), runner execution,
hops logging, and the RunResult → useTool-response mapping.
"""
from __future__ import annotations

import time
from typing import Any

from .conf import load
from .context.session_digest import sanitize
from .core.budget import Budget
from .core.runner import run_plan, Step
from .hops.log import HopLog
from .ops.filesys import confine
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
    # Ingress bound: whatever the caller sent, downstream sees <= 8 turns x 240
    # chars, <= 8 antecedents, plus the cwd the L1 bridge reads.
    digest: dict[str, Any] = sanitize(body.get("digest"))
    mode = str(body.get("mode") or "auto")

    # The run root: every path an op touches is resolved against it
    # (ops/filesys/confine.py). It is the caller's cwd — the frame scope.py reads
    # the digest for — and the process cwd when the body names none. The HTTP
    # layer has already checked that cwd against ops.filesys.roots; a direct
    # caller's cwd is where they said the run was.
    try:
        root = confine.set_root(digest.get("cwd"))
    except confine.PathRefused as exc:
        return {"status": "error", "error": str(exc), "hops": []}
    try:
        return _route(intent, params, digest, mode, dry, on_ask)
    finally:
        confine.reset_root(root)


def _route(intent: str, params: dict[str, Any], digest: dict[str, Any],
           mode: str, dry: bool, on_ask: Any) -> dict[str, Any]:
    meta = inspect(intent, params, digest)

    if dry or mode == "plan":
        return _plan_view(meta, intent, params, digest)

    if meta["level"] == "L0":
        return _passthrough(params)
    if meta["template"] and meta["level"] == "L2":
        from .planner.templates import apply_template
        steps, final_key, notes = apply_template(meta["template"], intent, digest, params)
        return _run_steps(steps, {}, final_key, intent, params,
                          on_ask=on_ask, log_hops=True, pre_assumptions=notes)
    # L1 disk-slot-fill bridge
    from .planner.templates.disk_lookup import build
    steps, final_key, notes = build(intent, digest, params)
    return _run_steps(steps, {}, final_key, intent, params,
                      on_ask=on_ask, log_hops=True, pre_assumptions=notes)


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
               intent: str, params: dict[str, Any], on_ask: Any, log_hops: bool,
               pre_assumptions: list[str] | None = None) -> dict[str, Any]:
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
                   _run_op_with_log(log), _noul_verifier, final_key=final_key,
                   pre_assumptions=pre_assumptions)
    ms = round((time.monotonic() - t0) * 1000, 1)
    out: dict[str, Any] = {
        "status": res.status, "result": res.result,
        "degraded": res.degraded, "assumptions": res.assumptions,
        "confidence": _route_conf(res.hops),
        "hops": res.hops, "wall_ms": ms,
        "meta": {"level": inspect(intent, params, {})["level"],
                 "route_gate": gate_route()},
    }
    if res.error is not None:
        out["error"] = res.error
    if res.status == "need_input" and on_ask is None and not res.degraded:
        out["question"] = {"kind": "choice", "question": "resolve step", "options": []}
    return out


def _route_conf(hops: list[dict[str, Any]]) -> float:
    """The run's confidence: the min over hops that carry a measurement.

    The `askback` hop is written before the ask is attempted, so its conf 0.0 is
    a placeholder rather than a measurement — ask-back is not wired into the
    HTTP path at all (use_tool is called without on_ask), which would make every
    gated run report 0.0 and destroy the decide confidence it did reach. Only a
    resolved ask-back counts: `askback.resolved` (conf 0.9, the LLM-trusted
    answer) supersedes the hop for the artifact it re-wrote, because that hop
    scored a proposal the gate rejected, not the answer that was delivered.
    A solicited but unanswered ask-back (no channel, asks exhausted, or an
    answer that did not apply) measures nothing either, so it is left out and
    the run reports the real decide signal as its lower bound; the gap is
    stated by `degraded` + assumptions[], not by a fabricated 0.0.
    """
    superseded = {h["resolves"] for h in hops
                  if h["op"] == "askback.resolved" and h.get("resolves")}
    signal = [h["conf"] for h in hops
              if h["op"] == "askback.resolved"
              or (h["op"].startswith("decide") and h["step"] not in superseded)]
    if not signal:
        return 1.0
    return max(0.0, min(signal))


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
    from .planner.templates.disk_lookup import build as build_disk
    if meta["level"] == "L0":
        steps = [Step(id="L0", op=str(params.get("tool", "?")), params=dict(params.get("args") or {}))]
        final_key = "L0"
    elif meta["template"]:
        steps, final_key, _notes = apply_template(str(meta["template"]), intent, digest, params)
    else:
        steps, final_key, _notes = build_disk(intent, digest, params)
    return {"status": "plan",
            "meta": meta,
            "final_key": final_key,
            "steps": [{"id": s.id, "op": s.op, "params": _safe(s.params), "inject": s.inject,
                       "gate_pre": s.gate.pre, "gate_post": s.gate.post} for s in steps],
            "hops": [],
            "ops_available": True}


def _safe(p: dict[str, Any]) -> dict[str, Any]:
    return {k: str(v)[:200] for k, v in p.items()}
