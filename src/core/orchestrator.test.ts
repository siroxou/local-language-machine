// Runnable check for the tool-call parser — the fiddly bit of the agent loop.
import { test } from "node:test";
import assert from "node:assert/strict";
import { parseToolCall, sniffLang } from "./loop.ts";
import { sourceOf } from "./orchestrator.ts";

const valid = new Set(["list_dir", "read_file", "write_file"]);

test("parses a fenced ```json tool call and normalizes name→tool", () => {
	const c = parseToolCall('```json\n{"name":"list_dir","arguments":{"path":"."}}\n```', valid);
	assert.deepEqual(c, { tool: "list_dir", arguments: { path: "." } });
});

test("parses an unfenced call amid prose, args→arguments", () => {
	const c = parseToolCall('Sure, let me look.\n{"tool":"read_file","args":{"path":"a.txt"}}', valid);
	assert.deepEqual(c, { tool: "read_file", arguments: { path: "a.txt" } });
});

test("returns null for a plain prose answer", () => {
	assert.equal(parseToolCall("The files are a, b and c.", valid), null);
});

test("returns null for a tool not in the allow-set", () => {
	assert.equal(parseToolCall('{"tool":"rm_rf","arguments":{}}', valid), null);
});

test("returns null for malformed JSON", () => {
	assert.equal(parseToolCall('{"tool": broken', valid), null);
});

test("sniffLang detects language from content (overriding a mislabeled fence)", () => {
	assert.equal(sniffLang("<!DOCTYPE html>\n<html><body>hi</body></html>"), "html");
	assert.equal(sniffLang("  <div><button>Go</button></div>"), "html");
	assert.equal(sniffLang("def main():\n    print('hi')"), "python");
	assert.equal(sniffLang("import os\nprint(os.getcwd())"), "python");
	assert.equal(sniffLang("function add(a, b) { return a + b; }"), "javascript");
	assert.equal(sniffLang("const go = () => document.title;"), "javascript");
	assert.equal(sniffLang("body { color: red; margin: 0; }"), "css");
	assert.equal(sniffLang("just some explanatory prose"), "");
});

test("sourceOf splits a hub uri into publisher and repo", () => {
	assert.deepEqual(sourceOf("hf:Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/qwen2.5-coder-7b-instruct-q4_k_m.gguf"),
		{ owner: "Qwen", repo: "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF" });
	assert.deepEqual(sourceOf("hf:bartowski/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q4_K_M.gguf"),
		{ owner: "bartowski", repo: "bartowski/SmolLM2-135M-Instruct-GGUF" });
});

test("sourceOf labels a converted training run as local", () => {
	assert.deepEqual(sourceOf("file:/Users/x/.local-language-machine/models/tuned.gguf"),
		{ owner: "Local fine-tune", repo: "tuned.gguf" });
});
