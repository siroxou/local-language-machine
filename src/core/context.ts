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
const KEEP_RECENT = 4; // most-recent non-system items kept
/**
 * Per-item cap on those kept turns. Without it a single oversized item — a pasted file,
 * or a read_file result fed back through the loop — survives every splice, so history never
 * drops under COMPACT_AT_CHARS and the next turn compacts again, and the next, each one a
 * full summarization pass. 4 × 3k plus the summary still clears the threshold next to a
 * ~5k system prompt.
 */
const KEEP_ITEM_CHARS = 3_000;
/** Window for the throwaway summarization session — see summarizeHistory. */
const SUMMARY_CONTEXT_TOKENS = 16_384;

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

/** Trim one kept turn down to KEEP_ITEM_CHARS. Only `text` items grow without bound — model
 *  responses are already capped by maxTokens — and those are the pastes and tool results. */
function capItem(item: ChatHistoryItem): ChatHistoryItem {
	const text = (item as any).text;
	if (typeof text !== "string" || text.length <= KEEP_ITEM_CHARS) return item;
	return { ...item, text: `${text.slice(0, KEEP_ITEM_CHARS)}\n…[${text.length - KEEP_ITEM_CHARS} chars trimmed during compaction]` } as ChatHistoryItem;
}

/** Replace the middle of history with a summary note, keeping system items and the last few turns. */
export function spliceSummary(history: ChatHistoryItem[], summary: string, keepRecent = KEEP_RECENT): ChatHistoryItem[] {
	const system = history.filter((h: any) => h.type === "system");
	const rest = history.filter((h: any) => h.type !== "system");
	const recent = rest.slice(-keepRecent).map(capItem);
	const note = { type: "user", text: `[Summary of earlier conversation]\n${summary}` } as unknown as ChatHistoryItem;
	return [...system, note, ...recent];
}

/** Summarize all but the most recent turns using a throwaway session on the loaded model. */
export async function summarizeHistory(engine: InferenceEngine, history: ChatHistoryItem[], keepRecent = KEEP_RECENT): Promise<string> {
	const older = history.filter((h: any) => h.type !== "system").slice(0, -keepRecent);
	const transcript = older.map((h: any) => `${h.type}: ${itemText(h)}`).join("\n").slice(0, 40_000);
	// 40k chars ≈ 10k tokens in, ~300 out. Sizing this session explicitly matters: inheriting
	// the model's own window allocates a second full KV cache — on a 7B at n_ctx 131072 that is
	// another ~7GB, spiked on every compaction.
	const session = await engine.createSession({
		systemPrompt: "You compress conversations. Output only a concise summary — no preamble.",
		contextSize: SUMMARY_CONTEXT_TOKENS,
	});
	try {
		return await session.prompt(
			`Summarize the conversation below in <=200 words, preserving decisions made, file paths touched, and any open tasks:\n\n${transcript}`,
		);
	} finally {
		session.dispose();
	}
}
