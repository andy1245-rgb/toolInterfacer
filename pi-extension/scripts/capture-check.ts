// Runtime check for the live-capture path: record builder, JSONL append, stats reader.
// Run with:
//   node pi-extension/scripts/capture-check.ts
// (Node >= 22 strips the types; capture.ts imports only Node builtins.)
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	buildRecord,
	capturePath,
	digestSizes,
	flushCapture,
	readCaptureStats,
	record,
	setCaptureEnabled,
	type CaptureCall,
} from "../src/capture.ts";

const RECORD_KEYS = [
	"ts",
	"cwd",
	"intent",
	"params_keys",
	"mode",
	"digest_sizes",
	"status",
	"degraded",
	"confidence",
	"hops",
	"wall_ms",
	"error",
	"question_kind",
].sort();

const dir = mkdtempSync(join(tmpdir(), "ti-capture-check-"));
let failures = 0;

function check(name: string, ok: boolean, detail = ""): void {
	if (!ok) failures += 1;
	console.log(`${ok ? "PASS" : "FAIL"} ${name}${ok ? "" : ` — ${detail}`}`);
}

function eq(name: string, actual: unknown, expected: unknown): void {
	const a = JSON.stringify(actual);
	const e = JSON.stringify(expected);
	check(name, a === e, `expected ${e}, got ${a}`);
}

function baseCall(overrides: Partial<CaptureCall> = {}): CaptureCall {
	return {
		cwd: "/repo",
		intent: "read the file we discussed",
		paramsKeys: ["path"],
		mode: "auto",
		digest: { turns: 3, antecedents: 1, chars: 240 },
		status: "done",
		degraded: false,
		confidence: 0.87,
		hops: [{ op: "filesys_read", ms: 2.5, conf: 0.91 }],
		wallMs: 41,
		...overrides,
	};
}

function jsonl(path: string): Record<string, unknown>[] {
	return readFileSync(path, "utf8")
		.split("\n")
		.filter((line) => line.trim() !== "")
		.map((line) => JSON.parse(line) as Record<string, unknown>);
}

async function main(): Promise<void> {
	const originalPath = process.env.TI_CAPTURE_PATH;

	// --- record builder -------------------------------------------------------
	const built = buildRecord(baseCall(), 1_700_000_000_000);
	eq("builder: field set is exactly the dataset contract", Object.keys(built).sort(), RECORD_KEYS);
	eq("builder: maps hop fields only (no artifact_summary)", built.hops, [{ op: "filesys_read", ms: 2.5, conf: 0.91 }]);
	eq("builder: digest_sizes", built.digest_sizes, { turns: 3, antecedents: 1, chars: 240 });
	eq("builder: numeric fields", [built.ts, built.wall_ms, built.confidence], [1_700_000_000_000, 41, 0.87]);

	const sparse = buildRecord(
		baseCall({ mode: null, confidence: undefined, hops: [], error: undefined, questionKind: undefined, digest: digestSizes(undefined) }),
		1,
	);
	eq("builder: optionals become null, not undefined", [sparse.mode, sparse.confidence, sparse.error, sparse.question_kind], [null, null, null, null]);
	eq("builder: no hops → []", sparse.hops, []);
	eq("builder: missing digest → zero sizes", sparse.digest_sizes, { turns: 0, antecedents: 0, chars: 0 });
	check("builder: record survives JSON.stringify", typeof JSON.stringify(sparse) === "string");

	const needInput = buildRecord(
		baseCall({ status: "need_input", degraded: false, questionKind: "choice", hops: [{ op: "decide_pick", ms: 1, conf: 0.4 }] }),
	);
	eq("builder: need_input carries question_kind", [needInput.status, needInput.question_kind], ["need_input", "choice"]);

	eq(
		"digestSizes: sums summary chars",
		digestSizes({ turns: [{ summary: "abc" }, { summary: "defgh" }], antecedents: [{}, {}] }),
		{ turns: 2, antecedents: 2, chars: 8 },
	);

	// --- capture path ---------------------------------------------------------
	const file = join(dir, "live.jsonl");
	process.env.TI_CAPTURE_PATH = file;
	eq("path: absolute TI_CAPTURE_PATH is used verbatim", capturePath(), file);
	process.env.TI_CAPTURE_PATH = "service/logs/live.jsonl";
	check("path: default/relative resolves under the repo root", capturePath().endsWith(join("service", "logs", "live.jsonl")), capturePath());
	process.env.TI_CAPTURE_PATH = file;

	// --- JSONL append ---------------------------------------------------------
	setCaptureEnabled(true);
	record(baseCall({ intent: "first intent" }));
	record(baseCall({ intent: "second intent", status: "need_input", questionKind: "free", hops: [] }));
	await flushCapture();
	const lines = jsonl(file);
	eq("append: one line per call, in order", lines.map((r) => r.intent), ["first intent", "second intent"]);
	eq("append: second record kept its status", lines[1].status, "need_input");
	check("append: every line is standalone JSON with ts", lines.every((r) => typeof r.ts === "number"));

	setCaptureEnabled(false);
	record(baseCall({ intent: "must not be written" }));
	await flushCapture();
	eq("toggle: capture off writes nothing", jsonl(file).length, 2);
	eq("toggle: readCaptureStats reports the toggle", readCaptureStats({ path: file }).enabled, false);
	setCaptureEnabled(true);

	// --- never throws into the tool path --------------------------------------
	const before = process.env.TI_CAPTURE_PATH;
	process.env.TI_CAPTURE_PATH = "/proc/ti-capture-check/nope/live.jsonl";
	let threw: string | null = null;
	try {
		record(baseCall({ intent: "unwritable path" }));
		await flushCapture();
	} catch (err) {
		threw = err instanceof Error ? err.message : String(err);
	}
	check("best-effort: an unwritable path never throws", threw === null, threw ?? "");
	process.env.TI_CAPTURE_PATH = before;

	// --- stats reader ---------------------------------------------------------
	const statFile = join(dir, "stats.jsonl");
	const rows = [
		{ ts: 1_000, status: "done", confidence: 0.2 },
		{ ts: 2_000, status: "done", confidence: 0.5 },
		{ ts: 3_000, status: "need_input", confidence: null },
		{ ts: 4_000, status: "done", confidence: 0.7 },
		{ ts: 5_000, status: "error", confidence: 0.9 },
	];
	writeFileSync(statFile, rows.map((r) => JSON.stringify(r)).join("\n") + "\nnot json\n\n");
	const stats = readCaptureStats({ path: statFile });
	eq("stats: counts parsed records only", stats.count, 5);
	eq("stats: status histogram", stats.statuses, { done: 3, need_input: 1, error: 1 });
	eq("stats: mean/median confidence over numeric values", [stats.confidence.n, stats.confidence.mean, stats.confidence.median], [4, 0.575, 0.6]);
	eq("stats: ts window", [stats.first_ts, stats.last_ts], [1_000, 5_000]);

	const shorter = join(dir, "tail.jsonl");
	writeFileSync(shorter, Array.from({ length: 8 }, (_, i) => JSON.stringify({ ts: i, status: "done", confidence: 0.5 })).join("\n") + "\n");
	const tail = readCaptureStats({ path: shorter, limit: 5 });
	eq("stats: reads only the last N lines", [tail.count, tail.first_ts, tail.last_ts], [5, 3, 7]);

	const missing = readCaptureStats({ path: join(dir, "does-not-exist.jsonl") });
	eq("stats: missing file → zeroed, no throw", [missing.exists, missing.count, missing.statuses], [false, 0, {}]);

	const byteTail = readCaptureStats({ path: shorter, limit: 100, tailBytes: 120 });
	check(
		"stats: a mid-file byte window drops the partial first line",
		byteTail.count >= 1 && byteTail.count < 8 && Number.isInteger(byteTail.last_ts),
		`count=${byteTail.count}`,
	);

	if (originalPath === undefined) delete process.env.TI_CAPTURE_PATH;
	else process.env.TI_CAPTURE_PATH = originalPath;
	rmSync(dir, { recursive: true, force: true });

	const total = 22;
	console.log(`\n${total - failures}/${total} capture cases passed`);
	process.exit(failures === 0 ? 0 : 1);
}

await main();
