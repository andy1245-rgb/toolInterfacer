import { Value } from "typebox/value";
import { UseToolResponseSchema, type UseToolRequest, type UseToolResponse } from "./contract.ts";

export interface ServiceHealth {
	status?: string;
	ops: string[];
	conf?: boolean;
}

export function base(): string {
	return process.env.TI_BASE_URL ?? "http://127.0.0.1:8090";
}

/** Every way a service call can fail, so callers can say something actionable. */
export type ServiceErrorKind = "unreachable" | "timeout" | "aborted" | "http" | "invalid";

export class ServiceError extends Error {
	readonly kind: ServiceErrorKind;
	readonly status?: number;

	constructor(kind: ServiceErrorKind, message: string, status?: number) {
		super(message);
		this.name = "ServiceError";
		this.kind = kind;
		this.status = status;
	}
}

function shapeErrors(body: unknown, limit = 3): string {
	return Value.Errors(UseToolResponseSchema, body)
		.slice(0, limit)
		.map((e) => `${e.instancePath || "(root)"} ${e.message}`)
		.join("; ");
}

export async function call(req: UseToolRequest, signal?: AbortSignal, timeoutMs = 120_000): Promise<UseToolResponse> {
	const timeout = AbortSignal.timeout(timeoutMs);
	let resp: Response;
	try {
		resp = await fetch(`${base()}/v1/useTool`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify(req),
			signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
		});
	} catch (err) {
		if (signal?.aborted) throw new ServiceError("aborted", "aborted by caller");
		if (timeout.aborted) throw new ServiceError("timeout", `service exceeded ${Math.round(timeoutMs / 1000)}s without a response`);
		const cause = (err as { cause?: { code?: string } }).cause?.code;
		throw new ServiceError(
			"unreachable",
			`${base()}${cause ? ` (${cause})` : ""} — ${err instanceof Error ? err.message : String(err)}`,
		);
	}
	if (!resp.ok) {
		const text = (await resp.text().catch(() => "")).slice(0, 200);
		throw new ServiceError("http", `${resp.status} ${resp.statusText}${text ? ` — ${text}` : ""}`, resp.status);
	}
	let body: unknown;
	try {
		body = await resp.json();
	} catch (err) {
		throw new ServiceError("invalid", `response is not JSON — ${err instanceof Error ? err.message : String(err)}`, resp.status);
	}
	if (!Value.Check(UseToolResponseSchema, body)) {
		throw new ServiceError("invalid", `unexpected service response: ${shapeErrors(body)}`, resp.status);
	}
	return body;
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
