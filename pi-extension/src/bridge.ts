import type { UseToolRequest, UseToolResponse } from "./contract.ts";

export interface ServiceHealth {
	status?: string;
	ops: string[];
	conf?: boolean;
}

export function base(): string {
	return process.env.TI_BASE_URL ?? "http://127.0.0.1:8090";
}

export async function call(req: UseToolRequest, signal?: AbortSignal, timeoutMs = 120_000): Promise<UseToolResponse> {
	const timeout = AbortSignal.timeout(timeoutMs);
	const resp = await fetch(`${base()}/v1/useTool`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(req),
		signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
	});
	return (await resp.json()) as UseToolResponse;
}

export async function health(timeoutMs = 3_000): Promise<ServiceHealth | null> {
	try {
		const resp = await fetch(`${base()}/health`, { signal: AbortSignal.timeout(timeoutMs) });
		if (!resp.ok) return null;
		const data = (await resp.json()) as { status?: string; ops?: unknown; conf?: boolean };
		return {
			status: data.status,
			ops: Array.isArray(data.ops) ? data.ops.map(String) : [],
			conf: data.conf,
		};
	} catch {
		return null;
	}
}
