// Local Language Machine — context-window management (compaction + /context).
//
// When the conversation grows past a budget we summarize the older turns with the
// loaded model and splice the summary back into history, keeping recent turns
// verbatim. token count is approximated from characters (~4 chars/token)
// and the summary is a single naive pass — upgrade to real tokenization + structured
// compaction if it starts dropping things that matter.

import type { ChatHistoryItem, InferenceEngine } from "../native/inference/engine.ts";

const CHARS_PER_TOKEN = 4;
/** Compaction trigger, in characters (~6k tokens). Below the context of the smallest bundled model. */
export const COMPACT_AT_CHARS = 24_000;
const KEEP_RECENT = 4; // most-recent non-system items kept verbatim

function itemText(item: any): string {
	if (typeof item?.text === "string") return item.text;
	if (Array.isArray(item?.response)) return item.response.filter((r: unknown) => typeof r === "string").join("");
	return "";
}

export function historyChars(history: ChatHistoryItem[]): number {
	return history.reduce((n, item) => n + itemText(item).length, 0);
}

export function estimateTokens(history: ChatHistoryItem[]): number {
	return Math.ceil(historyChars(history) / CHARS_PER_TOKEN);
}

export function needsCompaction(history: ChatHistoryItem[], limitChars = COMPACT_AT_CHARS): boolean {
	return historyChars(history) > limitChars;
}

/** A one-line-per-item breakdown for the `/context` command. */
export function contextReport(history: ChatHistoryItem[]): string {
	const chars = historyChars(history);
	const roles = history.reduce<Record<string, number>>((m, i: any) => ((m[i.type] = (m[i.type] ?? 0) + 1), m), {});
	const parts = Object.entries(roles).map(([r, n]) => `${n} ${r}`).join(", ");
	return `~${estimateTokens(history)} tokens (${chars.toLocaleString()} chars) across ${history.length} items: ${parts}.`;
}

/** Replace the middle of history with a summary note, keeping system items and the last few turns. */
export function spliceSummary(history: ChatHistoryItem[], summary: string, keepRecent = KEEP_RECENT): ChatHistoryItem[] {
	const system = history.filter((h: any) => h.type === "system");
	const rest = history.filter((h: any) => h.type !== "system");
	const recent = rest.slice(-keepRecent);
	const note = { type: "user", text: `[Summary of earlier conversation]\n${summary}` } as unknown as ChatHistoryItem;
	return [...system, note, ...recent];
}

/** Summarize all but the most recent turns using a throwaway session on the loaded model. */
export async function summarizeHistory(engine: InferenceEngine, history: ChatHistoryItem[], keepRecent = KEEP_RECENT): Promise<string> {
	const older = history.filter((h: any) => h.type !== "system").slice(0, -keepRecent);
	const transcript = older.map((h: any) => `${h.type}: ${itemText(h)}`).join("\n").slice(0, 40_000);
	const session = await engine.createSession({
		systemPrompt: "You compress conversations. Output only a concise summary — no preamble.",
	});
	try {
		return await session.prompt(
			`Summarize the conversation below in <=200 words, preserving decisions made, file paths touched, and any open tasks:\n\n${transcript}`,
		);
	} finally {
		session.dispose();
	}
}
