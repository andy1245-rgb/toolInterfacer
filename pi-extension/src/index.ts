import { Type, type Static } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { base, call, health, ServiceError } from "./bridge.ts";
import { UseToolResponseSchema, type Hops, type Question, type UseToolResponse } from "./contract.ts";
import { ensureService, isDev, serviceCmd } from "./lifecycle.ts";
import { buildDigest, readLastCustom } from "./session_push.ts";

const RESULT_CHARS = 1500;
const HOPS_SHOWN = 6;
const PLAN_STEPS_SHOWN = 12;
const TIMEOUT_MS = 120_000;
const RAW_TIMEOUT_MS = 240_000;
const STATUS_KEY = "ti";
const LASTCALL_TYPE = "ti.lastcall";
const SERVICE_TYPE = "ti.service";

interface LastCall {
	intent: string;
	status: string;
	hops: number;
	confidence?: number;
}

interface UseToolDetails {
	status: "pending" | "done" | "need_input" | "error" | "plan";
	degraded: boolean;
	confidence?: number;
	hops?: Hops[];
	question?: Question;
	elapsedMs?: number;
	stepCount?: number;
}

const params = Type.Object({
	intent: Type.String({
		description: "Plain-English intent, e.g. 'read the file we discussed', 'research X', 'what does r/... say'.",
	}),
	params: Type.Optional(
		Type.Record(Type.String(), Type.Unknown(), {
			description:
				"Concrete values you already know: {answer: ...} when answering a previous need_input question, plus any paths, URLs or constraints. Unknown keys are passed as soft context, never failing.",
		}),
	),
	mode: Type.Optional(
		Type.Union([Type.Literal("auto"), Type.Literal("plan"), Type.Literal("raw")], {
			description: "auto executes (default); plan returns the resolved decision chain without executing; raw returns un-summarized results.",
		}),
	),
});

function resultText(result: unknown): string {
	if (typeof result === "string") return result;
	if (result == null) return "";
	try {
		return JSON.stringify(result, null, 1);
	} catch {
		return String(result);
	}
}

function truncate(text: string): string {
	if (text.length <= RESULT_CHARS) return text;
	return `${text.slice(0, RESULT_CHARS)}\n…[truncated, ${text.length - RESULT_CHARS} chars omitted; call useTool with mode:"raw" or read the file natively for the full content]`;
}

function hopLine(resp: UseToolResponse): string {
	const hops = (resp.hops ?? []).slice(0, HOPS_SHOWN).map((h) => `${h.op} ${h.ms}ms conf ${h.conf.toFixed(2)}`);
	const meta = resp.confidence !== undefined ? ` — route conf ${resp.confidence.toFixed(2)}` : "";
	return `hops: ${hops.length ? hops.join(" | ") : "none"}${meta}`;
}

function planLines(resp: UseToolResponse): string[] {
	const steps = resp.steps ?? [];
	const level = typeof resp.meta?.level === "string" ? resp.meta.level : "?";
	const template = typeof resp.meta?.template === "string" ? resp.meta.template : "none";
	const lines = [
		`useTool: plan — resolved but not executed (level ${level}, template ${template}, ${steps.length} step${steps.length === 1 ? "" : "s"})`,
	];
	steps.slice(0, PLAN_STEPS_SHOWN).forEach((step, i) => {
		const args = step.params && Object.keys(step.params).length ? ` ${JSON.stringify(step.params).slice(0, 200)}` : "";
		lines.push(`  ${i + 1}. ${step.op}${args}`);
	});
	if (steps.length > PLAN_STEPS_SHOWN) lines.push(`  … ${steps.length - PLAN_STEPS_SHOWN} more steps`);
	if (resp.final_key) lines.push(`final artifact: ${resp.final_key}`);
	lines.push('To execute this chain, call useTool again with the same intent and mode:"auto" (or no mode).');
	return lines;
}

function renderResponse(resp: UseToolResponse, intent: string, meta: { elapsedMs?: number } = {}): string {
	const flag = resp.degraded ? " [degraded]" : "";
	const lines: string[] = [];
	if (resp.status === "error") {
		lines.push(`useTool: error — ${resp.error ?? "unknown error"} (routed by the toolInterfacer decision-model service)`, hopLine(resp));
	} else if (resp.status === "plan") {
		lines.push(...planLines(resp), hopLine(resp));
	} else if (resp.status === "need_input" || resp.question) {
		const q = resp.question;
		lines.push(
			`useTool: need_input — the decision-model service needs an answer from this conversation before it can proceed (intent: ${intent})`,
			hopLine(resp),
		);
		if (q) {
			lines.push(`question (${q.kind}): ${q.question}`);
			if (q.options?.length) {
				for (const opt of q.options) lines.push(`  ${opt.id}: ${opt.desc}`);
			}
		}
		lines.push(
			"Call useTool again in your next turn and answer this: put the answer in params.answer (use an option id when options are listed), or restate the intent plus any concrete file paths/URLs you now know. Answer once, do not ask the user.",
		);
	} else {
		lines.push(`useTool: done${flag} — resolved and executed by the toolInterfacer decision-model service`);
		lines.push(truncate(resultText(resp.result)));
		lines.push(hopLine(resp));
		if (resp.assumptions?.length) lines.push(`assumptions: ${resp.assumptions.join("; ")}`);
		if (resp.degraded) {
			lines.push(
				"degraded: ask-back budget was exhausted, so this is the best-guess result — verify anything load-bearing yourself, or call useTool again with a more precise intent.",
			);
		}
	}
	if (isDev()) {
		lines.push(`[dev] ${base()}${meta.elapsedMs !== undefined ? ` · ${meta.elapsedMs}ms` : ""} · ${JSON.stringify(resp).slice(0, 600)}`);
	}
	return lines.join("\n");
}

function formatServiceError(err: unknown, elapsedMs: number): string {
	if (err instanceof ServiceError) {
		switch (err.kind) {
			case "aborted":
				return `useTool: aborted after ${elapsedMs}ms by the caller`;
			case "timeout":
				return `useTool: ${err.message} — retry with a narrower intent, or mode:"plan" to see the chain without executing`;
			case "unreachable":
				return `useTool: service unreachable — ${err.message}; run /ti doctor`;
			case "http":
				return `useTool: service returned HTTP ${err.status} — ${err.message}`;
			case "invalid":
				return `useTool: service returned an unexpected shape — ${err.message}`;
		}
	}
	return `useTool: call failed at ${base()} — ${err instanceof Error ? err.message : String(err)}`;
}

export default function (pi: ExtensionAPI): void {
	let lastCall: LastCall | undefined;
	let lastServicePid: number | undefined;

	pi.registerTool(
		defineTool<typeof params, UseToolDetails>({
			name: "useTool",
			label: "useTool",
			description:
				"Delegate a task to the toolInterfacer decision-model service, which routes and executes the right local tool (read/glob/grep, web search/fetch, reddit digest) and returns one compact result with per-hop confidence. Use it for compound or info-seeking intents that would cost several native tool calls — research a topic, multi-file lookups, digests of sources, or reading/finding files (including ones mentioned earlier in the conversation). Params are values the LLM already knows (paths, URLs, constraints); unknown keys ride along as context instead of failing.",
			promptSnippet:
				"useTool(intent, params?, mode?): delegate read/search/fetch pipelines to the local decision-model service instead of chaining native tools",
			promptGuidelines: [
				"Prefer useTool over chaining read/grep/phrases when the task needs more than one tool call; answer its need_input question with params.answer instead of asking the user.",
			],
			parameters: params,
			exposure: "direct",
			annotations: { readOnlyHint: true, openWorldHint: true },
			prepareArguments: (args) => {
				const obj = args as { params?: unknown };
				if (obj && typeof obj.params === "string") {
					try {
						obj.params = JSON.parse(obj.params);
					} catch {
						// keep the raw string; parameter validation reports the expected shape
					}
				}
				return args as Static<typeof params>;
			},
			outputSchema: UseToolResponseSchema,
			executionMode: "sequential",
			execute: async (_toolCallId, params, signal, onUpdate, ctx) => {
				const started = Date.now();
				if (ctx.hasUI) ctx.ui.setStatus(STATUS_KEY, `useTool: ${params.intent.slice(0, 60)}`);
				onUpdate?.({
					content: [{ type: "text", text: `useTool: calling the decision-model service at ${base()}…` }],
					details: { status: "pending", degraded: false },
				});
				try {
					const digest = buildDigest(ctx.sessionManager.getBranch(), ctx.cwd);
					const resp = await call(
						{ intent: params.intent, params: params.params, mode: params.mode, digest },
						signal,
						params.mode === "raw" ? RAW_TIMEOUT_MS : TIMEOUT_MS,
					);
					const elapsedMs = Date.now() - started;
					lastCall = {
						intent: params.intent,
						status: resp.status,
						hops: resp.hops?.length ?? 0,
						confidence: resp.confidence,
					};
					pi.appendEntry<LastCall>(LASTCALL_TYPE, lastCall);
					return {
						content: [{ type: "text", text: renderResponse(resp, params.intent, { elapsedMs }) }],
						details: {
							status: resp.status,
							degraded: resp.degraded ?? false,
							confidence: resp.confidence,
							hops: resp.hops,
							question: resp.question,
							elapsedMs,
							stepCount: resp.steps?.length,
						},
						structuredContent: resp,
					};
				} catch (err) {
					const elapsedMs = Date.now() - started;
					return {
						content: [{ type: "text", text: formatServiceError(err, elapsedMs) }],
						details: { status: "error", degraded: false, elapsedMs },
						isError: true,
					};
				} finally {
					if (ctx.hasUI) ctx.ui.setStatus(STATUS_KEY, undefined);
				}
			},
		}),
	);

	pi.registerCommand("ti", {
		description: "useTool service status — /ti shows health + last call, /ti doctor probes and repairs the service",
		handler: async (args, ctx) => {
			const [sub] = (args ?? "").trim().split(/\s+/);
			const h = await health();
			const lines: string[] = [];
			lines.push(`service ${h ? "up" : "DOWN"} at ${base()}${isDev() ? " (TI_DEV=1 — spawn disabled, dev echo on)" : ""}`);
			if (sub === "doctor") {
				if (h) {
					lines.push(`service reachable at ${base()}`);
				} else if (isDev()) {
					const c = serviceCmd();
					lines.push(`dev mode: not spawning; would run: ${c.cmd} ${c.args.join(" ")} (cwd ${c.options.cwd})`);
				} else {
					const res = await ensureService();
					if (res.ok) {
						lines.push(`service UP after spawn${res.pid !== undefined ? ` (pid ${res.pid})` : ""}`);
						if (res.pid !== undefined) {
							lastServicePid = res.pid;
							pi.appendEntry<{ pid: number; base: string; ts: number }>(SERVICE_TYPE, { pid: res.pid, base: base(), ts: Date.now() });
						}
					} else {
						const why = res.reason === "timeout" ? "did not reach /health within 10s" : (res.detail ?? "spawn failed");
						lines.push(`spawn ${res.reason ?? "failed"}${res.pid !== undefined ? ` (pid ${res.pid})` : ""}: ${why}`);
					}
				}
			}
			if (h) {
				lines.push(`ops (${h.ops.length}): ${h.ops.join(", ") || "none"}`);
				lines.push(`conf defaults: ${h.conf === true ? "ok" : "missing"}`);
			}
			if (lastServicePid !== undefined) lines.push(`service pid (this session): ${lastServicePid} — stop it with kill ${lastServicePid}`);
			lines.push(
				lastCall
					? `last call: "${lastCall.intent.slice(0, 120)}" → ${lastCall.status}, ${lastCall.hops} hops${lastCall.confidence !== undefined ? `, conf ${lastCall.confidence.toFixed(2)}` : ""}`
					: "last call: none this session",
			);
			if (isDev()) lines.push('dev: useTool mode:"plan" prints the decision chain without executing it');
			if (ctx.hasUI) ctx.ui.notify(`[ti]\n${lines.join("\n")}`, h || sub !== "doctor" ? "info" : "error");
		},
	});

	pi.on("session_start", async (event, ctx) => {
		const restored = readLastCustom<LastCall>(ctx.sessionManager.getBranch(), LASTCALL_TYPE);
		if (restored) lastCall = restored;
		const res = await ensureService();
		if (res.pid !== undefined) {
			lastServicePid = res.pid;
			pi.appendEntry<{ pid: number; base: string; sessionId: string; ts: number }>(SERVICE_TYPE, {
				pid: res.pid,
				base: base(),
				sessionId: ctx.sessionManager.getSessionId(),
				ts: Date.now(),
			});
		}
		if (!res.ok && ctx.hasUI && (event.reason === "startup" || event.reason === "reload")) {
			const why = res.reason === "dev" ? "spawn disabled by TI_DEV=1" : (res.detail ?? "did not reach /health within 10s");
			ctx.ui.notify(`[ti] decision-model service not running at ${base()} (${why}) — useTool calls will error until it is up`, "error");
		}
	});

	pi.on("session_shutdown", (_event, ctx) => {
		lastServicePid = undefined;
		if (ctx.hasUI) ctx.ui.setStatus(STATUS_KEY, undefined);
	});
}
