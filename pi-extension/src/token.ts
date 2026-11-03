/**
 * The shared token every POST to the service must carry.
 *
 * The service is a loopback HTTP endpoint that reads files and runs the decision
 * model for whoever asks, so a request is served only if it presents
 * `http.token_env` (default `TI_HTTP_TOKEN`) or the 0600 runtime file the
 * service writes. Without this header the extension gets a `401` from a service
 * it spawned itself.
 *
 * Resolution order mirrors `service/ti/http/auth.py` exactly, and for the same
 * reason: the environment first, because `lifecycle.serviceCmd()` passes
 * `{...process.env}` to the child, so a token exported in the session's shell
 * reaches both ends with no file; then the runtime file, so an operator who
 * exports nothing still connects.
 *
 * The empty result is not cached. `ensureService()` polls `/health` before the
 * first call, and the service creates its token at startup — but a call that
 * races the file's creation must not pin "" for the rest of the session.
 */
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

/** Matches `service/ti/http/auth.py::TOKEN_HEADER`. */
export const TOKEN_HEADER = "X-TI-Token";

const DEFAULT_TOKEN_FILE = join(homedir(), ".local", "state", "ti", "http.token");

let cached: string | undefined;

function tokenFile(): string {
	const configured = process.env.TI_HTTP_TOKEN_FILE;
	return configured && configured.length > 0 ? configured : DEFAULT_TOKEN_FILE;
}

/** The token to send, or "" when none is readable yet. */
export function serviceToken(): string {
	if (cached !== undefined) return cached;
	const fromEnv = process.env.TI_HTTP_TOKEN;
	if (fromEnv && fromEnv.length > 0) {
		cached = fromEnv.trim();
		return cached;
	}
	try {
		const fromFile = readFileSync(tokenFile(), "utf8").trim();
		if (fromFile.length > 0) cached = fromFile;
		return fromFile;
	} catch {
		return "";
	}
}

/** Forget the cached token — for a restart where the service writes a new one. */
export function resetServiceToken(): void {
	cached = undefined;
}
