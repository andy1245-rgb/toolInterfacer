// Runtime contract check: every body service/ti/service_entry.py can return must pass
// UseToolResponseSchema, and obviously-wrong bodies must not. Run with:
//   node pi-extension/scripts/contract-check.ts
// (Node >= 22 strips the types; imports of src/contract.ts resolve through the
//  pi-extension/node_modules symlink, same as `tsc --noEmit`.)
import { Value } from "typebox/value";
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
process.exit(failed === 0 ? 0 : 1);
