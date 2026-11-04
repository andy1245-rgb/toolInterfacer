// Live capture (spec: round1/D-eval-harness.md §4 shadow-scoring, adapted to the extension).
//
// One JSONL record per completed useTool call, written so a REAL held-out set
// (utterances_v3) can be built from the operator's live traffic later. The record holds
// the call shape and the service verdict — intent, param *keys*, digest sizes, status,
// confidences, and the *class* of an error rather than its message — and never the chat
// transcript, never the cwd, never a param value.
//
// Off by default, and durably so: `/ti capture on` writes a state file
// (~/.local/state/ti/capture.json) and it stays on across restarts; `TI_CAPTURE=1|0`
// overrides for one process (bench, tests). A recorder of the intents the user hands this
// tool is not something to ship running (docs/logging-2026-11-04.md). The file it writes
// is the user's: created 0600 in a 0700 directory, and narrowed if it is wider.
//
// Everything in here is best-effort: `record()` returns void, appends fire-and-forget, and
// no thrown error may reach the tool path. Callers must NOT await it on the result path.
import { closeSync, mkdirSync, openSync, readFileSync, readSync, statSync, writeFileSync } from "node:fs";
import { appendFile, chmod, mkdir } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";

const DEFAULT_HOME_REL = join(".local", "state", "ti", "logs", "live.jsonl");
const DEFAULT_STATE_REL = join(".local", "state", "ti", "capture.json");
const DEFAULT_STATS_LINES = 2_000;
const TAIL_BYTES = 1 << 20;

export interface CaptureDigestSizes {
	turns: number;
	antecedents: number;
	chars: number;
}

export interface CaptureCall {
	intent: string;
	paramsKeys: readonly string[];
	mode: string | null;
	digest: CaptureDigestSizes;
	status: string;
	degraded: boolean;
	confidence?: number;
	hops: readonly { op: string; ms: number; conf: number | null }[];
	wallMs: number;
	errorKind?: string;
	questionKind?: string;
}

/** The on-disk shape. Field names are the dataset contract — do not rename silently.
 *  `cwd` and `error` were dropped on 2026-11-04: neither had a reader, and both carried
 *  a path (the cwd verbatim; the error message inline, e.g. an ENOENT's filename).
 *  `error_kind` keeps the class (timeout / unreachable / http / invalid) for a histogram. */
export interface CaptureRecord {
	ts: number;
	intent: string;
	params_keys: string[];
	mode: string | null;
	digest_sizes: CaptureDigestSizes;
	status: string;
	degraded: boolean;
	confidence: number | null;
	hops: { op: string; ms: number; conf: number | null }[];
	wall_ms: number;
	error_kind: string | null;
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

let persisted = persistedEnabled();
let writeChain: Promise<void> = Promise.resolve();
const preparedDirs = new Set<string>();
const tightenedFiles = new Set<string>();

/** `true`/`false` = an explicit 1/0 in the environment, which outranks the toggle. */
function envEnabled(): boolean | undefined {
	const raw = (process.env.TI_CAPTURE ?? "").trim().toLowerCase();
	if (["1", "true", "on", "yes"].includes(raw)) return true;
	if (["0", "false", "off", "no"].includes(raw)) return false;
	return undefined;
}

/** Where `/ti capture on|off` is remembered. `TI_CAPTURE_STATE` points it elsewhere. */
export function captureStatePath(): string {
	const configured = process.env.TI_CAPTURE_STATE;
	if (configured !== undefined && configured.trim() !== "") return resolve(configured);
	return join(homedir(), DEFAULT_STATE_REL);
}

/** Default OFF: no state file, no capture. Only `/ti capture on` or `TI_CAPTURE=1` opts in. */
function persistedEnabled(): boolean {
	try {
		const state = JSON.parse(readFileSync(captureStatePath(), "utf8")) as { enabled?: unknown };
		return state.enabled === true;
	} catch {
		return false;
	}
}

function persistEnabled(on: boolean): void {
	try {
		mkdirSync(dirname(captureStatePath()), { recursive: true, mode: 0o700 });
		writeFileSync(captureStatePath(), `${JSON.stringify({ enabled: on })}\n`, { mode: 0o600 });
	} catch {
		// the in-memory toggle still applies for this session
	}
}

/** `TI_CAPTURE_PATH` (relative paths resolve against the cwd), else
 *  `~/.local/state/ti/logs/live.jsonl` — never a path inside the working directory. */
export function capturePath(): string {
	const configured = process.env.TI_CAPTURE_PATH;
	if (configured !== undefined && configured.trim() !== "") return resolve(configured);
	return join(homedir(), DEFAULT_HOME_REL);
}

/** The effective state: an environment override, else the persisted toggle. */
export function captureEnabled(): boolean {
	return envEnabled() ?? persisted;
}

/** The toggle `/ti capture on|off` uses: in memory now, in the state file for the next start. */
export function setCaptureEnabled(on: boolean): void {
	persisted = on;
	persistEnabled(on);
}

export function buildRecord(call: CaptureCall, ts = Date.now()): CaptureRecord {
	return {
		ts,
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
		error_kind: call.errorKind ?? null,
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

/** Async-only: mkdir, narrow, append — chained so the caller never waits on any of it.
 *  No synchronous filesystem call happens here — a slow or hostile capture path
 *  must not be able to stall a tool call. */
function enqueueWrite(path: string, line: string): Promise<void> {
	const dir = dirname(path);
	const ready = preparedDirs.has(dir) ? Promise.resolve(true) : prepareDir(dir);
	return ready
		.then((privateDir) => (privateDir ? tighten(path).then(() => appendFile(path, line, { mode: 0o600 })) : undefined))
		.then(
			() => undefined,
			() => undefined,
		);
}

/** 0700, or a refusal. A directory other users can modify can have the log file swapped
 *  for their own target, so a directory that cannot be narrowed gets no records — the
 *  same rule the service side applies in `service/ti/hops/log.py`. */
async function prepareDir(dir: string): Promise<boolean> {
	await mkdir(dir, { recursive: true, mode: 0o700 }).catch(() => undefined);
	try {
		await chmod(dir, 0o700);
		preparedDirs.add(dir);
		return true;
	} catch {
		return false;
	}
}

/** Once per path: narrow a file an earlier version (or an earlier run) left readable.
 *  `appendFile`'s mode applies only at creation, so an existing 0644 file stays 0644
 *  without this. A chmod that fails is not fatal — the record is still written. */
function tighten(path: string): Promise<void> {
	if (tightenedFiles.has(path)) return Promise.resolve();
	tightenedFiles.add(path);
	return chmod(path, 0o600).catch(() => undefined);
}

/** Fire-and-forget. Never throws, never blocks the caller, never rejects, and does
 *  no synchronous filesystem work, so it cannot add latency to a tool call. */
export function record(call: CaptureCall): void {
	if (!captureEnabled()) return;
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
		enabled: captureEnabled(),
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
	// `TI_CAPTURE` wins over the toggle, so saying "stays on" while an override holds would
	// be a claim the next `record()` contradicts.
	const forced = envEnabled();
	const override = forced === undefined
		? []
		: [`note: TI_CAPTURE=${forced ? "1" : "0"} is set and overrides this for the current process`];
	switch (sub) {
		case "on":
			setCaptureEnabled(true);
			return [
				`capture ON — one JSONL record per completed useTool call to ${capturePath()}`,
				`saved to ${captureStatePath()} — stays on until /ti capture off`,
				`what it records: intent, param keys, digest sizes, status, confidences; not the cwd, not a param value, not an error message`,
				...override,
			];
		case "off":
			setCaptureEnabled(false);
			return [
				`capture OFF — ${capturePath()} is left as-is`,
				`saved to ${captureStatePath()} — stays off until /ti capture on`,
				...override,
			];
		case "path":
			return [
				`capture ${captureEnabled() ? "ON" : "OFF"} — ${capturePath()}`,
				"source: TI_CAPTURE_PATH (relative paths resolve against the cwd), default ~/.local/state/ti/logs/live.jsonl",
				`state: ${captureStatePath()} (TI_CAPTURE_STATE overrides); TI_CAPTURE=1|0 overrides for one process`,
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
				`capture ${captureEnabled() ? "ON" : "OFF"} — ${capturePath()}`,
				"usage: /ti capture on | off | path | stats",
			];
	}
}
