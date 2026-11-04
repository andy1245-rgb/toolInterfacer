// Runtime check for the live-capture path: default state, record builder, JSONL append,
// owner-only mode, stats reader.
// Run with:
//   node pi-extension/scripts/capture-check.ts
// (Node >= 22 strips the types; capture.ts imports only Node builtins.)
import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	buildRecord,
	captureEnabled,
	capturePath,
	captureStatePath,
	digestSizes,
	flushCapture,
	readCaptureStats,
	record,
	setCaptureEnabled,
	type CaptureCall,
} from "../src/capture.ts";

const RECORD_KEYS = [
	"ts",
	"intent",
	"params_keys",
	"mode",
	"digest_sizes",
	"status",
	"degraded",
	"confidence",
	"hops",
	"wall_ms",
	"error_kind",
	"question_kind",
].sort();

const dir = mkdtempSync(join(tmpdir(), "ti-capture-check-"));
// Keep every toggle in this run out of the operator's real state directory.
process.env.TI_CAPTURE_STATE = join(dir, "capture.json");
delete process.env.TI_CAPTURE;
const MODULE = new URL("../src/capture.ts", import.meta.url).href;
let failures = 0;
let checks = 0;

function check(name: string, ok: boolean, detail = ""): void {
	checks += 1;
	if (!ok) failures += 1;
	console.log(`${ok ? "PASS" : "FAIL"} ${name}${ok ? "" : ` — ${detail}`}`);
}

/** What a fresh process sees: proves the default, not this process's memory. */
function childEnabled(statePath: string): string {
	const code = `const m = await import(${JSON.stringify(MODULE)}); console.log(m.captureEnabled());`;
	const run = spawnSync(process.execPath, ["--input-type=module", "-e", code], {
		env: { ...process.env, TI_CAPTURE_STATE: statePath, TI_CAPTURE: "" },
		encoding: "utf8",
	});
	return (run.stdout ?? "").trim() || `spawn failed: ${run.stderr ?? ""}`;
}

function eq(name: string, actual: unknown, expected: unknown): void {
	const a = JSON.stringify(actual);
	const e = JSON.stringify(expected);
	check(name, a === e, `expected ${e}, got ${a}`);
}

function baseCall(overrides: Partial<CaptureCall> = {}): CaptureCall {
	return {
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
		baseCall({ mode: null, confidence: undefined, hops: [], errorKind: undefined, questionKind: undefined, digest: digestSizes(undefined) }),
		1,
	);
	eq("builder: optionals become null, not undefined", [sparse.mode, sparse.confidence, sparse.error_kind, sparse.question_kind], [null, null, null, null]);
	eq("builder: no hops → []", sparse.hops, []);
	eq("builder: missing digest → zero sizes", sparse.digest_sizes, { turns: 0, antecedents: 0, chars: 0 });
	check("builder: record survives JSON.stringify", typeof JSON.stringify(sparse) === "string");

	const failed = buildRecord(baseCall({ status: "error", errorKind: "timeout" }));
	eq("builder: an error keeps its class, never its message", failed.error_kind, "timeout");
	check("builder: no cwd is recorded", !("cwd" in built), Object.keys(built).join(","));

	const needInput = buildRecord(
		baseCall({ status: "need_input", degraded: false, questionKind: "choice", hops: [{ op: "decide_pick", ms: 1, conf: 0.4 }] }),
	);
	eq("builder: need_input carries question_kind", [needInput.status, needInput.question_kind], ["need_input", "choice"]);

	eq(
		"digestSizes: sums summary chars",
		digestSizes({ turns: [{ summary: "abc" }, { summary: "defgh" }], antecedents: [{}, {}] }),
		{ turns: 2, antecedents: 2, chars: 8 },
	);

	// --- the default state, measured in a fresh process ----------------------
	eq("default: no state file and no TI_CAPTURE means capture is OFF", childEnabled(join(dir, "absent.json")), "false");
	const stateFile = join(dir, "state.json");
	writeFileSync(stateFile, `${JSON.stringify({ enabled: true })}\n`);
	eq("state: enabled:true turns capture on for the next process", childEnabled(stateFile), "true");
	writeFileSync(stateFile, `${JSON.stringify({ enabled: false })}\n`);
	eq("state: enabled:false is off again", childEnabled(stateFile), "false");

	// --- capture path ---------------------------------------------------------
	const file = join(dir, "logs", "live.jsonl");
	process.env.TI_CAPTURE_PATH = file;
	eq("path: absolute TI_CAPTURE_PATH is used verbatim", capturePath(), file);
	process.env.TI_CAPTURE_PATH = join("nested", "live.jsonl");
	eq("path: a relative TI_CAPTURE_PATH resolves against the cwd", capturePath(), join(process.cwd(), "nested", "live.jsonl"));
	delete process.env.TI_CAPTURE_PATH;
	check(
		"path: the default lives under the state directory, never the cwd",
		capturePath().endsWith(join(".local", "state", "ti", "logs", "live.jsonl")) &&
			!capturePath().startsWith(process.cwd()),
		capturePath(),
	);
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

	// --- owner-only, measured on the file this path actually creates ---------
	const mode = (p: string): string => (statSync(p).mode & 0o777).toString(8);
	eq("mode: the log file is created 0600", mode(file), "600");
	eq("mode: its directory is created 0700", mode(join(dir, "logs")), "700");
	const loose = join(dir, "loose.jsonl");
	writeFileSync(loose, "{}\n", { mode: 0o644 });
	process.env.TI_CAPTURE_PATH = loose;
	record(baseCall({ intent: "tighten me" }));
	await flushCapture();
	eq("mode: an existing 0644 log is narrowed to 0600", mode(loose), "600");
	process.env.TI_CAPTURE_PATH = file;

	// --- the toggle is durable -----------------------------------------------
	eq("toggle: the state file records ON", JSON.parse(readFileSync(captureStatePath(), "utf8")), { enabled: true });
	eq("toggle: a fresh process sees the persisted ON", childEnabled(captureStatePath()), "true");
	eq("toggle: the state file is 0600", mode(captureStatePath()), "600");

	setCaptureEnabled(false);
	eq("toggle: the state file records OFF", JSON.parse(readFileSync(captureStatePath(), "utf8")), { enabled: false });
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

	const total = checks;
	console.log(`\n${total - failures}/${total} capture cases passed`);
	process.exit(failures === 0 ? 0 : 1);
}

await main();
