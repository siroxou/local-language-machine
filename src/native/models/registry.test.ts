// Runnable check for the registry + integrity logic. No model / network needed.
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	REGISTRY,
	cacheDir,
	resolve,
	modelPathFor,
	sha256File,
} from "./registry.ts";

test("registry resolves ids and rejects unknown ones", () => {
	const q = resolve("qwen2.5-coder:7b");
	assert.ok(q, "default coder model must be registered");
	assert.equal(q.quant, "Q4_K_M");
	assert.equal(resolve("does-not-exist"), undefined);
});

test("cache path is under the app data dir and ends in .gguf", () => {
	assert.ok(cacheDir().includes(".local-language-machine"));
	const p = modelPathFor(resolve("smollm2:135m")!);
	assert.ok(p.startsWith(cacheDir()));
	assert.ok(p.endsWith(".gguf"));
});

test("no registry entry points at a localhost/backend URL", () => {
	// Guards the 'zero external services' constraint at the data layer.
	for (const m of REGISTRY) {
		assert.doesNotMatch(m.modelUri, /localhost|127\.0\.0\.1|:11434|:1234|http:\/\//i);
	}
});

test("sha256File matches a known digest", async () => {
	const dir = await mkdtemp(join(tmpdir(), "llm-reg-"));
	const f = join(dir, "hello.txt");
	await writeFile(f, "hello");
	// Well-known: sha256("hello")
	assert.equal(
		await sha256File(f),
		"2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
	);
});
