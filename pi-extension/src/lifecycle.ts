import { spawn } from "node:child_process";
import { health } from "./bridge.ts";

const POLL_MS = 500;
const START_TIMEOUT_MS = 10_000;
const HEALTH_TIMEOUT_MS = 1_500;

export interface EnsureResult {
	ok: boolean;
	pid?: number;
	reason?: "dev" | "spawn-failed" | "timeout";
	detail?: string;
}

/** `TI_DEV=1` — talk to whatever `TI_BASE_URL` points at, never spawn the service. */
export function isDev(): boolean {
	return process.env.TI_DEV === "1";
}

export function serviceCmd(): { cmd: string; args: string[]; options: { cwd: string; env: NodeJS.ProcessEnv; detached: true; stdio: "ignore" } } {
	return {
		cmd: process.env.TI_PYTHON ?? "python3",
		args: ["-m", "ti.http.api"],
		options: {
			cwd: process.env.TI_SERVICE_DIR ?? "/home/andrew/toolInterfacer",
			env: { ...process.env, PYTHONPATH: "service" },
			detached: true,
			stdio: "ignore",
		},
	};
}

function sleep(ms: number): Promise<void> {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function ensureService(): Promise<EnsureResult> {
	if (await health(HEALTH_TIMEOUT_MS)) return { ok: true };
	if (isDev()) return { ok: false, reason: "dev" };

	const { cmd, args, options } = serviceCmd();
	let child: ReturnType<typeof spawn>;
	try {
		child = spawn(cmd, args, options);
	} catch (err) {
		return { ok: false, reason: "spawn-failed", detail: err instanceof Error ? err.message : String(err) };
	}
	const pid = child.pid;
	// `spawn` reports ENOENT/EACCES asynchronously; without these listeners a bad
	// TI_PYTHON or TI_SERVICE_DIR would look like a 10s timeout instead of a spawn error.
	const state: { error?: string; exit?: string } = {};
	child.on("error", (err) => {
		state.error = err.message;
	});
	child.on("exit", (code, signal) => {
		state.exit = `exited (code ${code ?? "null"}, signal ${signal ?? "none"})`;
	});
	child.unref();

	const deadline = Date.now() + START_TIMEOUT_MS;
	while (Date.now() < deadline) {
		await sleep(POLL_MS);
		if (await health(HEALTH_TIMEOUT_MS)) return { ok: true, pid };
		if (state.error !== undefined) return { ok: false, reason: "spawn-failed", pid, detail: state.error };
		if (state.exit !== undefined) return { ok: false, reason: "spawn-failed", pid, detail: `${state.exit} before serving /health` };
	}
	return { ok: false, reason: "timeout", pid };
}
