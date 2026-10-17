import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { base, call, health } from "./bridge.ts";
import type { UseToolResponse } from "./contract.ts";
import { ensureService } from "./lifecycle.ts";
import { buildDigest } from "./session_push.ts";

const RESULT_CHARS = 1500;
const HOPS_SHOWN = 6;

interface LastCall {
	intent: string;
	status: string;
	hops: number;
	confidence?: number;
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
	const hops = (resp.hops ?? []).slice(0, HOPS_SHOWN).map((h) => `${h.op} ${h.ms}ms conf ${h.conf?.toFixed?.(2) ?? h.conf}`);
	const meta = resp.confidence !== undefined ? ` — route conf ${resp.confidence.toFixed(2)}` : "";
	return `hops: ${hops.length ? hops.join(" | ") : "none"}${meta}`;
}

function renderResponse(resp: UseToolResponse, intent: string): string {
	const flag = resp.degraded ? " [degraded]" : "";
	if (resp.status === "error") {
		return [`useTool: error — ${resp.error ?? "unknown error"} (routed by the toolInterfacer decision-model service)`, hopLine(resp)].join("\n");
	}
	if (resp.status === "need_input" || resp.question) {
		const q = resp.question;
		const lines = [
			`useTool: need_input — the decision-model service needs an answer from this conversation before it can proceed (intent: ${intent})`,
			hopLine(resp),
		];
		if (q) {
			lines.push(`question (${q.kind}): ${q.question}`);
			if (q.options?.length) {
				for (const opt of q.options) lines.push(`  ${opt.id}: ${opt.desc}`);
			}
		}
		lines.push(
			"Call useTool again in your next turn and answer this: put the answer in params.answer (use an option id when options are listed), or restate the intent plus any concrete file paths/URLs you now know. Answer once, do not ask the user.",
		);
		return lines.join("\n");
	}
	const lines = [`useTool: done${flag} — resolved and executed by the toolInterfacer decision-model service`];
	lines.push(truncate(resultText(resp.result)));
	lines.push(hopLine(resp));
	if (resp.assumptions?.length) lines.push(`assumptions: ${resp.assumptions.join("; ")}`);
	if (resp.degraded) {
		lines.push(
			"degraded: ask-back budget was exhausted, so this is the best-guess result — verify anything load-bearing yourself, or call useTool again with a more precise intent.",
		);
	}
	return lines.join("\n");
}

export default function (pi: ExtensionAPI): void {
	let lastCall: LastCall | undefined;

	pi.registerTool(
		defineTool({
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
			execute: async (_toolCallId, params, signal, _onUpdate, ctx) => {
				const digest = buildDigest(ctx.sessionManager.getEntries(), ctx.cwd);
				let resp: UseToolResponse;
				try {
					resp = await call(
						{ intent: params.intent, params: params.params, mode: params.mode, digest },
						signal,
					);
				} catch (err) {
					return {
						content: [{ type: "text", text: `useTool: service unreachable at ${base()} — ${String(err)}` }],
						details: undefined,
						isError: true,
					};
				}
				lastCall = {
					intent: params.intent,
					status: resp.status,
					hops: resp.hops?.length ?? 0,
					confidence: resp.confidence,
				};
				return {
					content: [{ type: "text", text: renderResponse(resp, params.intent) }],
					details: {
						status: resp.status,
						degraded: resp.degraded ?? false,
						confidence: resp.confidence,
						hops: resp.hops,
						question: resp.question,
					},
				};
			},
		}),
	);

	pi.registerCommand("ti", {
		description: "useTool service status — /ti shows health + last call, /ti doctor probes and repairs the service",
		handler: async (args, ctx) => {
			const sub = (args ?? "").trim().split(/\s+/)[0];
			const h = await health();
			const lines: string[] = [];
			if (sub === "doctor") {
				if (!h) {
					lines.push(`service DOWN at ${base()} — attempting spawn…`);
					const ok = await ensureService();
					lines.push(ok ? `service UP after spawn` : `spawn failed: python3 -m ti.http.api did not reach /health within 10s`);
				} else {
					lines.push(`service reachable at ${base()}`);
				}
			} else {
				lines.push(`service ${h ? "up" : "DOWN"} at ${base()}`);
			}
			if (h) {
				lines.push(`ops (${h.ops.length}): ${h.ops.join(", ") || "none"}`);
				lines.push(`conf defaults: ${h.conf === true ? "ok" : "missing"}`);
			}
			lines.push(lastCall ? `last call: "${lastCall.intent.slice(0, 120)}" → ${lastCall.status}, ${lastCall.hops} hops${lastCall.confidence !== undefined ? `, conf ${lastCall.confidence.toFixed(2)}` : ""}` : "last call: none this session");
			if (ctx.hasUI) ctx.ui.notify(`[ti]\n${lines.join("\n")}`, h || sub !== "doctor" ? "info" : "error");
		},
	});

	pi.on("session_start", async (_event, ctx) => {
		const ok = await ensureService();
		if (!ok && ctx.hasUI) ctx.ui.notify(`[ti] decision-model service failed to start at ${base()} — useTool calls will error until it is up`, "error");
	});
}
