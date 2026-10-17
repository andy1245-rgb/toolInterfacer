import { existsSync } from "node:fs";
import { join } from "node:path";

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

const FILEISH = /[\w./~-]+\.(?:ts|tsx|js|mjs|cjs|py|md|json|jsonl|toml|yaml|yml|txt|sh|html|css|sql|go|rs)/g;
const SUMMARY_CHARS = 240;
const MAX_TURNS = 8;

interface RawEntry {
	type?: string;
	message?: {
		role?: string;
		content?: string | { type?: string; text?: string }[];
	};
}

function textOf(content: string | { type?: string; text?: string }[] | undefined): string {
	if (typeof content === "string") return content;
	if (Array.isArray(content)) {
		return content
			.filter((part) => part.type === "text" && typeof part.text === "string")
			.map((part) => part.text)
			.join(" ");
	}
	return "";
}

function compact(text: string): string {
	return text.replace(/\s+/g, " ").trim().slice(0, SUMMARY_CHARS);
}

function mentions(text: string): string[] {
	return [...new Set(text.match(FILEISH) ?? [])].slice(0, 6);
}

export function buildDigest(entries: unknown[], cwd: string): SessionDigest {
	const turns: DigestTurn[] = [];
	const antecedents: { phrase: string; resolved?: string }[] = [];
	const seenAntecedents = new Set<string>();

	for (const raw of entries) {
		const entry = raw as RawEntry;
		if (entry.type !== "message") continue;
		const role = entry.message?.role;
		if (role !== "user" && role !== "assistant") continue;
		const text = compact(textOf(entry.message?.content));
		if (!text) continue;
		const found = mentions(text);
		turns.push({ role, summary: text, ...(found.length ? { mentions: found } : {}) });
		if (role === "user") {
			for (const phrase of found) {
				if (seenAntecedents.has(phrase)) continue;
				seenAntecedents.add(phrase);
				const resolved = join(cwd, phrase);
				if (existsSync(resolved)) antecedents.push({ phrase, resolved });
			}
		}
	}

	return {
		turns: turns.slice(-MAX_TURNS),
		antecedents: antecedents.slice(-MAX_TURNS),
		cwd,
	};
}
