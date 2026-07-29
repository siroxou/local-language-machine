// Runnable checks for eval parsers, run-metadata persistence, and HF row mapping.
// No Python / GPU / network needed.
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { parseEvalJson, gradeAnswer, gradingKeys } from "./eval.ts";
import { writeMeta, readMeta, listRuns, runDirOf, mapRow, type RunMeta } from "./runs.ts";

// appHome() is only read inside these functions (never at import time), so pointing app
// data at a temp dir here — before any test body runs — is enough to isolate the tests.
process.env.LLM_HOME = mkdtempSync(join(tmpdir(), "llm-eval-"));

test("parseEvalJson survives payloads containing newlines, quotes and equals signs", () => {
	const payload = { loss: 1.2, ppl: 3.32, bitsPerByte: 0.9, rows: 10, tokens: 40, bytes: 30 };
	assert.deepEqual(parseEvalJson("EVAL_JSON " + JSON.stringify(payload)), payload);
	// the old delimiter-scraping broke on model output like this; the JSON line does not
	assert.equal(parseEvalJson("========== a=b \"q\" ==========" ), null);
	assert.equal(parseEvalJson("Test loss 1.2"), null);
});

test("gradeAnswer does not confuse answers that differ only by a trailing letter", () => {
	// The classic normalise-and-contain bug: stripping articles makes these match.
	assert.equal(gradeAnswer("It is Hepatitis B.", "Hepatitis A", ["Hepatitis B", "Hepatitis C"]), "wrong");
	assert.equal(gradeAnswer("It is Hepatitis A.", "Hepatitis A", ["Hepatitis B"]), "correct");
});

test("gradeAnswer does not let a decimal answer trip a substring distractor", () => {
	// "12.5 mg" must not count as containing the distractor "25 mg".
	assert.equal(gradeAnswer("The dose is 12.5 mg daily.", "12.5 mg", ["25 mg", "40 mg"]), "correct");
	assert.equal(gradeAnswer("The dose is 25 mg daily.", "12.5 mg", ["25 mg"]), "wrong");
});

test("gradeAnswer reports abstain and ambiguous rather than guessing", () => {
	assert.equal(gradeAnswer("I don't know.", "caltherin-B", ["velmoxine"]), "abstain");
	assert.equal(gradeAnswer("Either caltherin-B or velmoxine.", "caltherin-B", ["velmoxine"]), "ambiguous");
	assert.equal(gradeAnswer("", "caltherin-B", ["velmoxine"]), "abstain");
});

test("gradeAnswer matches on word boundaries, not bare substrings", () => {
	assert.equal(gradeAnswer("the answer is velmoxinered", "velmoxine", []), "abstain"); // not a real hit
	assert.equal(gradeAnswer("answer: velmoxine!", "velmoxine", []), "correct");         // punctuation is fine
});

test("gradingKeys reads answer/distractors, falling back to the reference completion", () => {
	const dir = mkdtempSync(join(tmpdir(), "llm-keys-"));
	const f = join(dir, "test.jsonl");
	writeFileSync(f, [
		JSON.stringify({ messages: [{ role: "user", content: "q" }, { role: "assistant", content: "The answer is X." }], answer: "X", distractors: ["Y", "Z"] }),
		JSON.stringify({ messages: [{ role: "user", content: "q2" }, { role: "assistant", content: "plain answer" }] }),
	].join("\n") + "\n");
	const k = gradingKeys(f);
	assert.deepEqual(k[0], { answer: "X", distractors: ["Y", "Z"] });
	assert.deepEqual(k[1], { answer: "plain answer", distractors: [] }); // graceful fallback
});

test("run metadata round-trips and lists newest-first", () => {
	const mk = (runId: string, startedAt: string): RunMeta => {
		mkdirSync(runDirOf(runId), { recursive: true });
		return { runId, backend: "mlx", baseModel: "b", config: {} as any, status: "done", startedAt, lossCurve: [{ iter: 1, loss: 2 }], adapterDir: join(runDirOf(runId), "adapters") };
	};
	writeMeta(mk("2026-01-01T00-00-00-aaaa", "2026-01-01T00:00:00Z"));
	writeMeta(mk("2026-02-01T00-00-00-bbbb", "2026-02-01T00:00:00Z"));
	assert.equal(readMeta("2026-01-01T00-00-00-aaaa")?.baseModel, "b");
	const runs = listRuns();
	assert.equal(runs.length, 2);
	assert.equal(runs[0]!.runId, "2026-02-01T00-00-00-bbbb"); // newest first
});

test("mapRow normalizes common dataset schemas", () => {
	assert.ok((mapRow({ text: "hello" }) as any).text === "hello");
	assert.ok(Array.isArray((mapRow({ instruction: "q", output: "a" }) as any).messages));
	assert.ok(Array.isArray((mapRow({ conversations: [{ from: "human", value: "hi" }, { from: "gpt", value: "yo" }] }) as any).messages));
	assert.ok((mapRow({ some_field: "fallback text" }) as any).text === "fallback text");
	assert.equal(mapRow({ n: 1 }), null); // nothing string-like
});
