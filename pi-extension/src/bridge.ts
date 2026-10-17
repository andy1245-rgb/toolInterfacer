import type { UseToolRequest, UseToolResponse } from "./contract.js";

// P3: POST to the service + health poll. Conf from the service /health.
export function base(): string {
	return process.env.TI_BASE_URL ?? "http://127.0.0.1:8090";
}

export async function call(req: UseToolRequest): Promise<UseToolResponse> {
	const resp = await fetch(`${base()}/v1/useTool`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(req),
	});
	return (await resp.json()) as UseToolResponse;
}
