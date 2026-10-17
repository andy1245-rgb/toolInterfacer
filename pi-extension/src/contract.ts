// Tool contract (spec §3). Keep in sync with service/ti/http/api.py.
export interface Hops {
	step: string;
	op: string;
	ms: number;
	conf: number;
	artifact_summary: string;
}

export interface UseToolRequest {
	intent: string;
	params?: Record<string, unknown>;
	mode?: "auto" | "raw" | "plan";
	digest?: unknown;
}

export interface UseToolResponse {
	status: "done" | "need_input" | "error";
	result?: unknown;
	degraded?: boolean;
	assumptions?: string[];
	sources?: unknown[];
	confidence?: number;
	question?: { kind: "choice" | "confirm" | "free"; question: string; options?: { id: string; desc: string }[] };
	hops: Hops[];
	error?: string;
}
