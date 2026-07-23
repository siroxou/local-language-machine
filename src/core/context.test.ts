// Compaction's pure pieces: the size estimate and the splice that keeps the system
// prompt + recent turns while replacing the middle with a summary.
import { test } from "node:test";
import assert from "node:assert/strict";
import { historyChars, estimateTokens, spliceSummary } from "./context.ts";

const hist = [
	{ type: "system", text: "sys" },
	{ type: "user", text: "u1" },
	{ type: "model", response: ["m1"] },
	{ type: "user", text: "u2" },
	{ type: "model", response: ["m2"] },
	{ type: "user", text: "u3" },
] as any;

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
