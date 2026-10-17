// Lifecycle: spawn `python -m ti.http.api` on session start if /health is not
// answering, kill on session end. Stub in P0, wired P3.
export function serviceCmd(): { cmd: string; args: string[] } {
	return {
		cmd: process.env.TI_PYTHON ?? "python3",
		args: ["-m", "ti.http.api"],
	};
}

export async function healthy(): Promise<boolean> {
	try {
		const base = process.env.TI_BASE_URL ?? "http://127.0.0.1:8090";
		const resp = await fetch(`${base}/health`);
		return resp.ok;
	} catch {
		return false;
	}
}
