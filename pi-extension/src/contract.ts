// Tool contract (spec §3). Keep in sync with service/ti/service_entry.py — that module is
// the only place that builds these bodies (use_tool → _run_steps / _passthrough / _plan_view).
//
// Payload-bearing fields (result, sources, meta, step params/gates) are intentionally
// `Type.Any()`: the schema enforces the shape of the envelope, not the payload, and the
// alternative would reject an op that starts returning a richer value.
import { Type, type Static } from "@earendil-works/pi-ai";

export const HopsSchema = Type.Object({
	step: Type.String(),
	op: Type.String(),
	ms: Type.Number(),
	conf: Type.Number(),
	artifact_summary: Type.String(),
});
export type Hops = Static<typeof HopsSchema>;

export const QuestionSchema = Type.Object({
	kind: Type.Union([Type.Literal("choice"), Type.Literal("confirm"), Type.Literal("free")]),
	question: Type.String(),
	options: Type.Optional(Type.Array(Type.Object({ id: Type.String(), desc: Type.String() }))),
});
export type Question = Static<typeof QuestionSchema>;

/** One step of a `mode:"plan"` body — `_plan_view` in service_entry.py. */
export const PlanStepSchema = Type.Object({
	id: Type.String(),
	op: Type.String(),
	params: Type.Optional(Type.Record(Type.String(), Type.String())),
	inject: Type.Optional(Type.Record(Type.String(), Type.String())),
	gate_pre: Type.Optional(Type.Array(Type.Record(Type.String(), Type.Any()))),
	gate_post: Type.Optional(Type.Array(Type.Record(Type.String(), Type.Any()))),
});
export type PlanStep = Static<typeof PlanStepSchema>;

export const UseToolResponseSchema = Type.Object({
	status: Type.Union([
		Type.Literal("done"),
		Type.Literal("need_input"),
		Type.Literal("error"),
		// `_plan_view` (service_entry.py:143): mode "plan" or the /v1/plan route.
		Type.Literal("plan"),
	]),
	result: Type.Optional(Type.Any()),
	degraded: Type.Optional(Type.Boolean()),
	assumptions: Type.Optional(Type.Array(Type.String())),
	sources: Type.Optional(Type.Array(Type.Any())),
	confidence: Type.Optional(Type.Number()),
	question: Type.Optional(QuestionSchema),
	// The current service sends `hops: []` on plan bodies too, and always sends `hops` on
	// done/need_input/error; optional keeps an older service from being rejected outright.
	hops: Type.Optional(Type.Array(HopsSchema)),
	error: Type.Optional(Type.String()),
	wall_ms: Type.Optional(Type.Number()),
	// Plan-only fields.
	meta: Type.Optional(Type.Record(Type.String(), Type.Any())),
	final_key: Type.Optional(Type.String()),
	steps: Type.Optional(Type.Array(PlanStepSchema)),
	ops_available: Type.Optional(Type.Boolean()),
});
export type UseToolResponse = Static<typeof UseToolResponseSchema>;

export interface UseToolRequest {
	intent: string;
	params?: Record<string, unknown>;
	mode?: "auto" | "raw" | "plan";
	digest?: unknown;
}
