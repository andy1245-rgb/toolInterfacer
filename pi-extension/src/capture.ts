// Live capture (spec: round1/D-eval-harness.md §4 shadow-scoring, adapted to the extension).
//
// One JSONL record per completed useTool call, written so a REAL held-out set
// (utterances_v3) can be built from the operator's live traffic later. The record holds
// the call shape and the service verdict — intent, param *keys*, digest sizes, status,
// confidences — and never the chat transcript itself.
//
// Everything in here is best-effort: `record()` returns void, appends fire-and-forget, and
// no thrown error may reach the tool path. Callers must NOT await it on the result path.
import { closeSync, openSync, readSync, realpathSync, statSync } from "node:fs";
import { mkdir, appendFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const DEFAULT_REL_PATH = join("service", "logs", "live.jsonl");
const DEFAULT_STATS_LINES = 2_000;
const TAIL_BYTES = 1 << 20;

export interface CaptureDigestSizes {
	turns: number;
	antecedents: number;
	chars: number;
}

export interface CaptureCall {
	cwd: string;
	intent: string;
	paramsKeys: readonly string[];
	mode: string | null;
	digest: CaptureDigestSizes;
	status: string;
	degraded: boolean;
	confidence?: number;
	hops: readonly { op: string; ms: number; conf: number }[];
	wallMs: number;
	error?: string;
	questionKind?: string;
}

/** The on-disk shape. Field names are the dataset contract — do not rename silently. */
export interface CaptureRecord {
	ts: number;
	cwd: string;
	intent: string;
	params_keys: string[];
	mode: string | null;
	digest_sizes: CaptureDigestSizes;
	status: string;
	degraded: boolean;
	confidence: number | null;
	hops: { op: string; ms: number; conf: number }[];
	wall_ms: number;
	error: string | null;
	question_kind: string | null;
}

export interface CaptureStats {
	path: string;
	enabled: boolean;
	exists: boolean;
	bytes: number;
	lines_read: number;
	count: number;
	statuses: Record<string, number>;
	confidence: { n: number; mean: number | null; median: number | null };
	first_ts: number | null;
	last_ts: number | null;
}

let enabled = true;
let writeChain: Promise<void> = Promise.resolve();
const preparedDirs = new Set<string>();
let cachedRoot: string | undefined;

function realModuleDir(): string {
	try {
		return dirname(fileURLToPath(import.meta.url));
	} catch {
		return process.cwd();
	}
}

/**
 * Repo root for the default path. The extension is loaded through a symlink
 * (`~/.pi/agent/extensions/toolinterfacer -> <repo>/pi-extension/src`), so resolving
 * `../..` from the module path alone can land outside the repo; walk up to the first
 * directory that looks like this repo instead.
 */
function repoRoot(): string {
	if (cachedRoot !== undefined) return cachedRoot;
	let dir = realModuleDir();
	try {
		dir = realpathSync(dir);
	} catch {
		// keep the unresolved path
	}
	const start = dir;
	for (let i = 0; i < 6; i++) {
		try {
			if (statSync(join(dir, "pyproject.toml")).isFile() && statSync(join(dir, "service")).isDirectory()) {
				cachedRoot = dir;
				return dir;
			}
		} catch {
			// not this level
		}
		const parent = dirname(dir);
		if (parent === dir) break;
		dir = parent;
	}
	cachedRoot = resolve(start, "..", "..");
	return cachedRoot;
}

/** `TI_CAPTURE_PATH` (relative paths resolve against the repo root), else `service/logs/live.jsonl`. */
export function capturePath(): string {
	const configured = process.env.TI_CAPTURE_PATH;
	if (configured !== undefined && configured.trim() !== "") return resolve(repoRoot(), configured);
	return join(repoRoot(), DEFAULT_REL_PATH);
}

export function captureEnabled(): boolean {
	return enabled;
}

export function setCaptureEnabled(on: boolean): void {
	enabled = on;
}

export function buildRecord(call: CaptureCall, ts = Date.now()): CaptureRecord {
	return {
		ts,
		cwd: call.cwd,
		intent: call.intent,
		params_keys: [...call.paramsKeys],
		mode: call.mode,
		digest_sizes: {
			turns: call.digest.turns,
			antecedents: call.digest.antecedents,
			chars: call.digest.chars,
		},
		status: call.status,
		degraded: call.degraded,
		confidence: call.confidence ?? null,
		hops: call.hops.map((h) => ({ op: h.op, ms: h.ms, conf: h.conf })),
		wall_ms: call.wallMs,
		error: call.error ?? null,
		question_kind: call.questionKind ?? null,
	};
}

/** Sizes only — the digest text itself never leaves the process. */
export function digestSizes(
	digest: { turns: readonly { summary: string }[]; antecedents: readonly unknown[] } | undefined,
): CaptureDigestSizes {
	if (!digest) return { turns: 0, antecedents: 0, chars: 0 };
	let chars = 0;
	for (const turn of digest.turns) chars += turn.summary.length;
	return { turns: digest.turns.length, antecedents: digest.antecedents.length, chars };
}

/** Async-only: mkdir then append, chained so the caller never waits on either.
 *  No synchronous filesystem call happens here — a slow or hostile capture path
 *  must not be able to stall a tool call. */
function enqueueWrite(path: string, line: string): Promise<void> {
	const dir = dirname(path);
	const ready = preparedDirs.has(dir)
		? Promise.resolve()
		: mkdir(dir, { recursive: true }).then(
			() => {
				preparedDirs.add(dir);
			},
			() => undefined,
		);
	return ready.then(() => appendFile(path, line)).then(
		() => undefined,
		() => undefined,
	);
}

/** Fire-and-forget. Never throws, never blocks the caller, never rejects, and does
 *  no synchronous filesystem work, so it cannot add latency to a tool call. */
export function record(call: CaptureCall): void {
	if (!enabled) return;
	try {
		const path = capturePath();
		const line = `${JSON.stringify(buildRecord(call))}\n`;
		// Chain instead of replace: keeps records in call order without ever
		// blocking the caller on the previous write.
		writeChain = writeChain.then(() => enqueueWrite(path, line));
	} catch {
		// capture is best-effort: a failure here must not touch the tool result
	}
}

/** Await the queued writes — for tests and shutdown only, never on the result path.
 *  Bounded, so a wedged filesystem can never hang the caller either. */
export async function flushCapture(timeoutMs = 500): Promise<void> {
	try {
		await Promise.race([writeChain, new Promise<void>((done) => setTimeout(done, timeoutMs))]);
	} catch {
		// ignored by design
	}
}

function tailLines(path: string, limit: number, tailBytes: number): { lines: string[]; bytes: number } {
	const fd = openSync(path, "r");
	try {
		const { size } = statSync(path);
		const start = Math.max(0, size - tailBytes);
		const length = size - start;
		const buffer = Buffer.alloc(length);
		let read = 0;
		while (read < length) {
			const n = readSync(fd, buffer, read, length - read, start + read);
			if (n <= 0) break;
			read += n;
		}
		const lines = buffer
			.subarray(0, read)
			.toString("utf8")
			.split("\n")
			.filter((line) => line.trim() !== "");
		// A mid-file start almost always lands inside a record; drop that fragment.
		if (start > 0 && lines.length > 0) lines.shift();
		return { lines: lines.slice(-limit), bytes: size };
	} finally {
		closeSync(fd);
	}
}

function objectOf(value: unknown): Record<string, unknown> | undefined {
	if (typeof value !== "object" || value === null || Array.isArray(value)) return undefined;
	return value as Record<string, unknown>;
}

function median(sorted: readonly number[]): number | null {
	if (sorted.length === 0) return null;
	const mid = Math.floor(sorted.length / 2);
	return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export function readCaptureStats(opts: { path?: string; limit?: number; tailBytes?: number } = {}): CaptureStats {
	const path = opts.path ?? capturePath();
	const limit = opts.limit ?? DEFAULT_STATS_LINES;
	const stats: CaptureStats = {
		path,
		enabled,
		exists: false,
		bytes: 0,
		lines_read: 0,
		count: 0,
		statuses: {},
		confidence: { n: 0, mean: null, median: null },
		first_ts: null,
		last_ts: null,
	};
	let lines: string[];
	try {
		const tail = tailLines(path, limit, opts.tailBytes ?? TAIL_BYTES);
		lines = tail.lines;
		stats.exists = true;
		stats.bytes = tail.bytes;
	} catch {
		return stats;
	}
	stats.lines_read = lines.length;
	const confidences: number[] = [];
	for (const line of lines) {
		let parsed: unknown;
		try {
			parsed = JSON.parse(line);
		} catch {
			continue;
		}
		const rec = objectOf(parsed);
		if (!rec) continue;
		stats.count += 1;
		const status = typeof rec.status === "string" ? rec.status : "unknown";
		stats.statuses[status] = (stats.statuses[status] ?? 0) + 1;
		if (typeof rec.confidence === "number" && Number.isFinite(rec.confidence)) confidences.push(rec.confidence);
		if (typeof rec.ts === "number") {
			if (stats.first_ts === null || rec.ts < stats.first_ts) stats.first_ts = rec.ts;
			if (stats.last_ts === null || rec.ts > stats.last_ts) stats.last_ts = rec.ts;
		}
	}
	if (confidences.length > 0) {
		const sum = confidences.reduce((acc, value) => acc + value, 0);
		const sorted = [...confidences].sort((a, b) => a - b);
		stats.confidence = { n: confidences.length, mean: sum / confidences.length, median: median(sorted) };
	}
	return stats;
}

function histogram(statuses: Record<string, number>): string {
	const entries = Object.entries(statuses).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
	return entries.map(([status, n]) => `${status} ${n}`).join(", ");
}

/** `/ti capture [on|off|path|stats]` — lines for the notify body. */
export function captureReport(sub?: string): string[] {
	switch (sub) {
		case "on":
			enabled = true;
			return [`capture ON — appending one JSONL record per completed useTool call to ${capturePath()}`];
		case "off":
			enabled = false;
			return [`capture OFF — ${capturePath()} is left as-is; turn it back on with /ti capture on`];
		case "path":
			return [
				`capture ${enabled ? "ON" : "OFF"} — ${capturePath()}`,
				"source: TI_CAPTURE_PATH (relative paths resolve against the repo root), default service/logs/live.jsonl",
			];
		case "stats": {
			const s = readCaptureStats();
			const lines = [
				`capture ${s.enabled ? "ON" : "OFF"} — ${s.path}`,
				s.exists
					? `records: ${s.count} from the last ${s.lines_read} line(s) of ${s.bytes} B`
					: "records: 0 — file does not exist yet (no useTool calls captured)",
			];
			if (s.count > 0) {
				lines.push(`status: ${histogram(s.statuses) || "none"}`);
				const c = s.confidence;
				lines.push(
					c.n > 0 && c.mean !== null && c.median !== null
						? `confidence: mean ${c.mean.toFixed(3)} · median ${c.median.toFixed(3)} (n ${c.n})`
						: "confidence: no numeric confidence recorded",
				);
				if (s.first_ts !== null && s.last_ts !== null) {
					lines.push(`window: ${new Date(s.first_ts).toISOString()} → ${new Date(s.last_ts).toISOString()}`);
				}
			}
			return lines;
		}
		default:
			return [
				`capture ${enabled ? "ON" : "OFF"} — ${capturePath()}`,
				"usage: /ti capture on | off | path | stats",
			];
	}
}
