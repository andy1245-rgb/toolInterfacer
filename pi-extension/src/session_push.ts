// Rolling session digest for the decision model (spec §A): last ~8 turns,
// antecedent mentions resolved where possible, ~400 tokens budgeted.
export interface DigestTurn {
	role: "user" | "assistant";
	summary: string;
	mentions?: string[];
}

export interface SessionDigest {
	turns: DigestTurn[];
	antecedents: { phrase: string; resolved?: string }[];
	cwd?: string;
}

// P3: pull entries from the extension ctx (ctx.sessionManager.getEntries())
// and compress. Stub now.
export function buildDigest(entries: unknown[], cwd: string): SessionDigest {
	void entries;
	return { turns: [], antecedents: [], cwd };
}
