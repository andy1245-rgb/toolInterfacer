import { spawn } from "node:child_process";
import { base, health } from "./bridge.ts";

const POLL_MS = 500;
const START_TIMEOUT_MS = 10_000;

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

export async function ensureService(): Promise<boolean> {
	if (await health()) return true;
	const { cmd, args, options } = serviceCmd();
	try {
		const child = spawn(cmd, args, options);
		child.unref();
	} catch {
		return false;
	}
	const deadline = Date.now() + START_TIMEOUT_MS;
	while (Date.now() < deadline) {
		await sleep(POLL_MS);
		if (await health()) return true;
	}
	return false;
}
