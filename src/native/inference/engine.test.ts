// Runnable check for the bundled engine. Downloads a tiny GGUF (once, cached),
// then proves: load → streamed generation → grammar-constrained structured output.
// Skips gracefully if the model can't be fetched (offline CI).
import { test } from "node:test";
import assert from "node:assert/strict";
import { LlamaCppEngine } from "./engine.ts";
import { ensureModel } from "../models/registry.ts";

// Prefer GPU; fall back to CPU (e.g. sandboxes where Metal is unavailable).
async function loadEngine(modelPath: string): Promise<LlamaCppEngine> {
	for (const gpu of ["auto", "cpu"] as const) {
		const eng = new LlamaCppEngine({ gpu: gpu === "cpu" ? false : "auto" });
		try {
			await eng.load({ modelPath });
			return eng;
		} catch (e) {
			await eng.unload().catch(() => {});
			if (gpu === "cpu") throw e;
		}
	}
	throw new Error("unreachable");
}

test("engine loads a model, streams text, and produces structured output", { timeout: 600_000 }, async (t) => {
	let modelPath: string;
	try {
		modelPath = await ensureModel("smollm2:135m");
	} catch (e) {
		t.skip(`model unavailable (offline?): ${(e as Error).message}`);
		return;
	}

	const engine = await loadEngine(modelPath);
	assert.notEqual(engine.gpu, "unloaded", "engine should report a gpu after load");
	t.diagnostic(`loaded on gpu=${engine.gpu}`);

	const session = await engine.createSession({ systemPrompt: "You are terse." });

	// 1) Streaming: chunks arrive and concatenate to the returned text.
	let streamed = "";
	const full = await session.prompt("Reply with a short greeting.", {
		onText: (c) => {
			streamed += c;
		},
		maxTokens: 24,
	});
	assert.ok(full.length > 0, "response should be non-empty");
	assert.equal(streamed, full, "streamed chunks should equal the final text");

	// 2) Structured output: generation is grammar-forced to the schema, so it parses.
	const result = await session.structured<{ ok: boolean }>("Set ok to true.", {
		type: "object",
		properties: { ok: { type: "boolean" } },
		required: ["ok"],
	} as const);
	assert.equal(typeof result.ok, "boolean", "structured output must match the schema");
	session.dispose();

	// 3) Engine-held sampling reaches prompt(): a tiny maxTokens caps the response.
	engine.setSampling({ maxTokens: 6 });
	const capped = await engine.createSession({ systemPrompt: "You are terse." });
	const short = await capped.prompt("List the numbers one through fifty, comma separated.");
	capped.dispose();
	assert.ok(
		short.split(/\s+/).filter(Boolean).length <= 12,
		`maxTokens should cap the response, got: ${JSON.stringify(short)}`,
	);

	// 4) temperature 0 → greedy decoding → deterministic. Same prompt on two fresh sessions
	//    must match; if sampling weren't forwarded, the default (random) sampler would diverge.
	engine.setSampling({ temperature: 0 });
	const greedy = async () => {
		const s = await engine.createSession({ systemPrompt: "You are terse." });
		const out = await s.prompt("Reply with exactly: hello world");
		s.dispose();
		return out;
	};
	assert.equal(await greedy(), await greedy(), "temperature 0 must be deterministic — sampling must reach prompt()");

	await engine.unload();
});
