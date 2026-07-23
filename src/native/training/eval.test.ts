// Runnable checks for eval parsers, run-metadata persistence, and HF row mapping.
// No Python / GPU / network needed.
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { parseTestLoss, parseGenSpeed } from "./eval.ts";
import { writeMeta, readMeta, listRuns, runDirOf, mapRow, type RunMeta } from "./runs.ts";

// appHome() is only read inside these functions (never at import time), so pointing app
// data at a temp dir here — before any test body runs — is enough to isolate the tests.
process.env.LLM_HOME = mkdtempSync(join(tmpdir(), "llm-eval-"));

test("parseTestLoss reads loss + ppl, and derives ppl when absent", () => {
	assert.deepEqual(parseTestLoss("Test loss 1.200, Test ppl 3.320"), { loss: 1.2, ppl: 3.32 });
	const derived = parseTestLoss("Test loss 0.000");
	assert.equal(derived?.loss, 0);
	assert.equal(derived?.ppl, 1); // exp(0)
	assert.equal(parseTestLoss("Starting training"), null);
});

test("parseGenSpeed reads tokens-per-sec from the Generation line", () => {
	assert.equal(parseGenSpeed("Generation: 128 tokens, 80.500 tokens-per-sec"), 80.5);
	assert.equal(parseGenSpeed("Prompt: 12 tokens, 200.0 tokens-per-sec"), null); // only the Generation line counts
	assert.equal(parseGenSpeed("blah"), null);
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
