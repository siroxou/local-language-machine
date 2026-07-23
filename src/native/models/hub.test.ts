// Runnable checks for the HF parsing logic — no network needed.
import { test } from "node:test";
import assert from "node:assert/strict";
import { collapseShards } from "./hub.ts";
import { entryFromHub } from "./registry.ts";

test("collapseShards merges split GGUF (entrypoint = first shard, size = sum)", () => {
	const out = collapseShards([
		{ type: "file", path: "model-00002-of-00002.gguf", size: 200, lfs: { oid: "x" } },
		{ type: "file", path: "model-00001-of-00002.gguf", size: 300, lfs: { oid: "y" } },
		{ type: "file", path: "single-q4_k_m.gguf", size: 100, lfs: { oid: "a".repeat(64) } },
	]);
	assert.equal(out.length, 2);
	// smaller (single, 100) sorts before the merged shard set (500)
	assert.deepEqual(
		{ path: out[0].path, size: out[0].sizeBytes, sha: out[0].sha256, sharded: out[0].sharded },
		{ path: "single-q4_k_m.gguf", size: 100, sha: "a".repeat(64), sharded: false },
	);
	assert.deepEqual(
		{ path: out[1].path, size: out[1].sizeBytes, sha: out[1].sha256, sharded: out[1].sharded },
		{ path: "model-00001-of-00002.gguf", size: 500, sha: undefined, sharded: true },
	);
});

test("collapseShards drops a non-sha256 lfs oid", () => {
	const out = collapseShards([{ type: "file", path: "m.gguf", size: 10, lfs: { oid: "not-a-hash" } }]);
	assert.equal(out[0].sha256, undefined);
});

test("entryFromHub parses quant → id/uri/label + collision-safe filename", () => {
	const e = entryFromHub("bartowski/Llama-3.2-1B-Instruct-GGUF", {
		path: "Llama-3.2-1B-Instruct-Q4_K_M.gguf", sizeBytes: 800_000_000, sha256: "b".repeat(64), sharded: false,
	});
	assert.equal(e.quant, "Q4_K_M");
	assert.equal(e.id, "bartowski/Llama-3.2-1B-Instruct-GGUF:q4_k_m");
	assert.equal(e.modelUri, "hf:bartowski/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf");
	assert.equal(e.fileName, "bartowski__Llama-3.2-1B-Instruct-GGUF__Llama-3.2-1B-Instruct-Q4_K_M.gguf");
	assert.equal(e.label, "Llama-3.2-1B-Instruct · Q4_K_M");
	assert.equal(e.sha256, "b".repeat(64));
});

test("entryFromHub falls back to GGUF when no quant token is present", () => {
	const e = entryFromHub("owner/repo", { path: "weird-name.gguf", sizeBytes: 1e9, sharded: false });
	assert.equal(e.quant, "GGUF");
	assert.equal(e.id, "owner/repo:gguf");
});
