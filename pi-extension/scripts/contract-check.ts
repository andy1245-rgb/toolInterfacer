// Runtime contract check: every body service/ti/service_entry.py can return must pass
// UseToolResponseSchema, and obviously-wrong bodies must not. Run with:
//   node pi-extension/scripts/contract-check.ts            # static cases, no service
//   node pi-extension/scripts/contract-check.ts --live     # + a stubbed service over HTTP
//
// `--live` binds a stub on an ephemeral loopback port that answers /health and
// /v1/useTool with the bodies above, then drives it through bridge.call(), so the
// client's own schema gate and its error kinds are exercised over a real socket
// instead of in memory. It needs no external service; CI uses it.
//
// Exit: 0 all cases pass · 1 a case failed · 2 the live stub could not be set up.
// (Node >= 22 strips the types; run `npm ci` first so the imports resolve.)
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { Value } from "typebox/value";
import { call, ServiceError } from "../src/bridge.ts";
import { UseToolResponseSchema } from "../src/contract.ts";

interface Case {
	name: string;
	body: unknown;
	valid: boolean;
}

const hop = { step: "candidates", op: "filesys_glob", ms: 3.1, conf: 1.0, artifact_summary: "['a.md', 'b.md']" };
const gateCond = { artifact: "picked_file", kind: "conf_gte", min: 0.7 };

const cases: Case[] = [
	{
		name: 'mode:"auto" → done (service_entry._run_steps)',
		valid: true,
		body: {
			status: "done",
			result: "file contents",
			degraded: false,
			assumptions: ["assumed the README"],
			confidence: 0.87,
			hops: [hop],
			wall_ms: 12.4,
			meta: { level: "L1", route_gate: 0.7 },
		},
	},
	{
		name: 'mode:"auto" → done, L0 passthrough (service_entry._passthrough)',
		valid: true,
		body: {
			status: "done",
			result: "file contents",
			degraded: false,
			assumptions: [],
			confidence: 1.0,
			hops: [{ step: "L0", op: "filesys_read", ms: 0.5, conf: 1.0, artifact_summary: "file contents" }],
		},
	},
	{
		name: 'mode:"auto" → need_input (runner ask-back)',
		valid: true,
		body: {
			status: "need_input",
			result: null,
			degraded: false,
			assumptions: [],
			confidence: 0.3,
			hops: [hop],
			wall_ms: 40.2,
			meta: { level: "L1", route_gate: 0.7 },
			question: { kind: "choice", question: "resolve step", options: [{ id: "a.md", desc: "the readme" }] },
		},
	},
	{
		name: 'mode:"auto" → error (service_entry / api.py handler)',
		valid: true,
		body: { status: "error", error: "empty intent", hops: [] },
	},
	{
		name: 'mode:"plan" (current service: _plan_view with hops: [])',
		valid: true,
		body: {
			status: "plan",
			meta: { intent: "read the file we discussed", level: "L1", template: "disk_lookup" },
			final_key: "content",
			steps: [
				{ id: "candidates", op: "filesys_glob", params: { pattern: "*" }, inject: {}, gate_pre: [], gate_post: [] },
				{ id: "picked_file", op: "decide_pick", params: {}, inject: { candidates: "candidates" }, gate_pre: [gateCond], gate_post: [] },
			],
			hops: [],
			ops_available: true,
		},
	},
	{
		name: 'mode:"plan" (service predating hops: [] — tolerated, no hops key)',
		valid: true,
		body: {
			status: "plan",
			meta: { intent: "research X", level: "L2", template: "research" },
			final_key: "answer",
			steps: [{ id: "q", op: "web_search", params: { query: "X" }, inject: {}, gate_pre: [], gate_post: [] }],
			ops_available: true,
		},
	},
	{ name: "reject: unknown status", valid: false, body: { status: "started", hops: [] } },
	{ name: "reject: truncated hop", valid: false, body: { status: "done", hops: [{ step: "candidates" }] } },
	{ name: "reject: null body", valid: false, body: null },
	{ name: "reject: non-object body", valid: false, body: "ok" },
];

let failed = 0;
for (const c of cases) {
	const actual = Value.Check(UseToolResponseSchema, c.body);
	const ok = actual === c.valid;
	if (!ok) failed += 1;
	const detail = ok
		? ""
		: ` — expected valid=${c.valid}, got ${actual}: ${Value.Errors(UseToolResponseSchema, c.body)
				.slice(0, 3)
				.map((e) => `${e.instancePath || "(root)"} ${e.message}`)
				.join("; ")}`;
	console.log(`${ok ? "PASS" : "FAIL"} ${c.name}${detail}`);
}

console.log(`\n${cases.length - failed}/${cases.length} contract cases passed`);

const STUB_TIMEOUT_MS = 5_000;
const NOT_JSON = "__stub-not-json";
const HTTP_500 = "__stub-http-500";

// The stub reuses the bodies above as its responses, so a live case can never drift
// from the static case it mirrors.
const served = new Map<string, { status: string }>();
for (const c of cases) {
	if (!c.valid || typeof c.body !== "object" || c.body === null) continue;
	served.set(c.name, c.body as { status: string });
}

function send(res: ServerResponse, status: number, type: string, body: string): void {
	res.writeHead(status, { "Content-Type": type, "Content-Length": Buffer.byteLength(body) });
	res.end(body);
}

function readAll(req: IncomingMessage): Promise<string> {
	return new Promise((resolve, reject) => {
		const chunks: Buffer[] = [];
		req.on("data", (chunk: Buffer) => chunks.push(chunk));
		req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
		req.on("error", reject);
	});
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
	if (req.method === "GET" && req.url === "/health") {
		send(res, 200, "application/json", JSON.stringify({ status: "stub", ops: [], conf: true }));
		return;
	}
	if (req.method !== "POST" || req.url !== "/v1/useTool") {
		send(res, 404, "text/plain", "no route");
		return;
	}
	let body: unknown;
	try {
		body = JSON.parse(await readAll(req));
	} catch {
		send(res, 400, "text/plain", "bad json");
		return;
	}
	const intent = typeof body === "object" && body !== null ? (body as { intent?: unknown }).intent : undefined;
	if (intent === NOT_JSON) {
		send(res, 200, "application/json", "this is not json at all");
		return;
	}
	if (intent === HTTP_500) {
		send(res, 500, "text/plain", "stub exploded");
		return;
	}
	const answer = typeof intent === "string" ? served.get(intent) : undefined;
	if (answer === undefined) {
		send(res, 400, "text/plain", `unknown stub intent: ${String(intent)}`);
		return;
	}
	send(res, 200, "application/json", JSON.stringify(answer));
}

async function liveLeg(): Promise<number> {
	const server = createServer((req, res) => {
		handle(req, res).catch((err: unknown) => {
			if (res.headersSent) return;
			send(res, 500, "text/plain", `stub failed: ${err instanceof Error ? err.message : String(err)}`);
		});
	});
	await new Promise<void>((resolve, reject) => {
		server.once("error", reject);
		server.listen(0, "127.0.0.1", resolve);
	});
	const addr = server.address();
	if (addr === null || typeof addr === "string") {
		server.close();
		throw new Error(`stub bound to an unusable address: ${String(addr)}`);
	}
	process.env.TI_BASE_URL = `http://127.0.0.1:${addr.port}`;

	let liveFailed = 0;
	const report = (name: string, ok: boolean, detail = ""): void => {
		if (!ok) liveFailed += 1;
		console.log(`${ok ? "PASS" : "FAIL"} live: ${name}${detail}`);
	};

	for (const [name, body] of served) {
		try {
			const got = await call({ intent: name }, undefined, STUB_TIMEOUT_MS);
			report(name, got.status === body.status, got.status === body.status ? "" : ` — sent ${body.status}, client returned ${got.status}`);
		} catch (err) {
			report(name, false, ` — ${err instanceof Error ? err.message : String(err)}`);
		}
	}

	const kindOf = async (intent: string): Promise<string> => {
		try {
			await call({ intent }, undefined, STUB_TIMEOUT_MS);
			return "accepted";
		} catch (err) {
			return err instanceof ServiceError ? err.kind : `threw ${String(err)}`;
		}
	};

	const notJson = await kindOf(NOT_JSON);
	report("a non-JSON 200 body is rejected as invalid", notJson === "invalid", ` — client reported ${notJson}`);
	const http500 = await kindOf(HTTP_500);
	report("a 500 is reported as http", http500 === "http", ` — client reported ${http500}`);

	await new Promise<void>((resolve) => server.close(() => resolve()));
	const down = await kindOf("after the stub closed");
	report("a closed port is reported as unreachable", down === "unreachable", ` — client reported ${down}`);

	const total = served.size + 3;
	console.log(`\n${total - liveFailed}/${total} live contract cases passed`);
	return liveFailed;
}

if (process.argv.includes("--live") || process.env.TI_CONTRACT_LIVE === "1") {
	let liveFailed: number;
	try {
		liveFailed = await liveLeg();
	} catch (err) {
		console.error(`\nno live contract result: the stub service could not be set up — ${err instanceof Error ? err.message : String(err)}`);
		process.exit(2);
	}
	failed += liveFailed;
}

process.exit(failed === 0 ? 0 : 1);
