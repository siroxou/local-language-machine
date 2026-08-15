// Compaction's pure pieces: the size estimate and the splice that keeps the system
// prompt + recent turns while replacing the middle with a summary.
import { test } from "node:test";
import assert from "node:assert/strict";
import { historyChars, estimateTokens, spliceSummary, rebaseHistory, needsCompaction, COMPACT_AT_CHARS } from "./context.ts";

const hist = [
	{ type: "system", text: "sys" },
	{ type: "user", text: "u1" },
	{ type: "model", response: ["m1"] },
	{ type: "user", text: "u2" },
	{ type: "model", response: ["m2"] },
	{ type: "user", text: "u3" },
] as any;

test("rebaseHistory keeps the live system prompt and drops the stale one", () => {
	// The bug this guards: a saved session carries the system message that was in effect when it
	// was written, so replaying it wholesale silently reverted the user's tuning system prompt.
	const base = [{ type: "system", text: "NEW prompt" }] as any;
	const out = rebaseHistory(base, hist);
	assert.deepEqual(out.filter((h: any) => h.type === "system"), [{ type: "system", text: "NEW prompt" }]);
	assert.equal(out.length, 6); // 1 system + the 5 non-system items
	assert.deepEqual(out[1], { type: "user", text: "u1" }); // conversation order preserved
});

test("rebaseHistory on an empty conversation is just the base, and never aliases it", () => {
	const base = [{ type: "system", text: "sys" }] as any;
	const out = rebaseHistory(base, [{ type: "system", text: "old" }] as any);
	assert.deepEqual(out, base);
	assert.notEqual(out, base); // a copy — callers hand this straight to setHistory
});

test("historyChars sums text across system/user/model items", () => {
	assert.equal(historyChars(hist), 13); // 3+2+2+2+2+2
	assert.equal(estimateTokens(hist), 4); // ceil(13/4)
});

test("spliceSummary keeps the system prompt + recent turns, replaces the middle", () => {
	const out = spliceSummary(hist, "SUMMARY", 2) as any[];
	assert.equal(out.length, 4); // system + summary note + 2 recent
	assert.equal(out[0].type, "system");
	assert.match(out[1].text, /SUMMARY/);
	assert.equal(out[3].text, "u3"); // most recent kept verbatim
});

// The regression that made a pasted file cost a full summarization pass on every following
// turn: an oversized item sat inside the kept window, so history never came back under the
// threshold and needsCompaction stayed true forever.
test("splicing gets an oversized recent turn under the compaction threshold in one pass", () => {
	const huge = [
		{ type: "system", text: "S".repeat(5_000) },
		{ type: "user", text: "u1" },
		{ type: "model", response: ["m1"] },
		{ type: "user", text: "H".repeat(100_000) }, // a pasted file, or a read_file result fed back
		{ type: "model", response: ["m2"] },
	] as any;
	assert.equal(needsCompaction(huge), true);

	const out = spliceSummary(huge, "a short summary");
	assert.equal(needsCompaction(out), false, "one compaction pass must clear the threshold");
	assert.ok(historyChars(out) < COMPACT_AT_CHARS);
	assert.ok(!out.some((h: any) => h.text?.length > 100_000), "the oversized turn must be trimmed, not carried over");
	assert.match((out.at(-2) as any).text, /trimmed during compaction/);
});

test("a large system prompt alone never triggers compaction", () => {
	// The system prompt embeds AGENTS.md and MEMORY.md, so it can exceed the threshold on its
	// own — and spliceSummary always keeps system items, so compacting cannot shrink it. When
	// the trigger counted it, every turn ran a full summarization pass that removed nothing.
	const hugeSystem = [{ type: "system", text: "x".repeat(COMPACT_AT_CHARS + 5_000) }] as any;
	assert.equal(needsCompaction(hugeSystem), false);

	// A short exchange on top of that huge prompt is still not worth compacting.
	const withTurns = [...hugeSystem, { type: "user", text: "hi" }, { type: "model", response: ["hello"] }] as any;
	assert.equal(needsCompaction(withTurns), false);
});

test("compaction still triggers on real conversation growth", () => {
	const history = [
		{ type: "system", text: "sys" },
		{ type: "user", text: "y".repeat(COMPACT_AT_CHARS + 1) },
	] as any;
	assert.equal(needsCompaction(history), true);
});

test("compaction stops once the conversation is spliced down", () => {
	// The anti-thrash property: after one pass the result must be under the threshold, or the
	// next turn compacts again and every turn pays for a summarization that changes nothing.
	const history = [
		{ type: "system", text: "x".repeat(30_000) },
		...Array.from({ length: 12 }, (_, i) => ({ type: "user", text: `turn ${i} ` + "z".repeat(4_000) })),
	] as any;
	assert.equal(needsCompaction(history), true);
	const spliced = spliceSummary(history, "a short summary of earlier turns", 4);
	assert.equal(needsCompaction(spliced), false, "one pass must be enough");
});
