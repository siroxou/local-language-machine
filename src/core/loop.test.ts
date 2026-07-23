// The gate seam is where checkpoints, permissions, and hooks plug in — verify that
// `before` can block a tool (handler never runs) and that `after` observes results.
import { test } from "node:test";
import assert from "node:assert/strict";
import { runAgentLoop, type ToolGate } from "./loop.ts";
import type { ChatSession, ToolDef } from "../native/inference/engine.ts";

// A scripted session: returns each queued output in turn, then a plain final answer.
function fakeSession(outputs: string[]): ChatSession {
	let i = 0;
	return {
		async prompt(_t, opts) { const o = outputs[i++] ?? "final answer"; opts?.onText?.(o); return o; },
		async structured() { return {} as any; },
		getHistory() { return []; },
		setHistory() {},
		dispose() {},
	};
}

const noopEvents = { onToken() {}, onToolCall() {}, onToolResult() {}, onCodeOpen() {}, onCodeToken() {}, onCodeClose() {} };
const writeCall = '{"tool":"write_file","arguments":{"path":"a","content":"x"}}';

test("gate.before can block a tool — the handler never runs", async () => {
	let ran = false;
	const tools: Record<string, ToolDef> = { write_file: { handler: () => { ran = true; return { written: true }; } } };
	const gate: ToolGate = { before: async () => ({ block: true, result: { cancelled: true } }) };
	const final = await runAgentLoop({ session: fakeSession([writeCall, "final answer"]), tools, input: "go", events: noopEvents, root: "/tmp", gate });
	assert.equal(ran, false, "blocked tool must not execute");
	assert.equal(final, "final answer");
});

test("without a blocking gate the tool runs, and gate.after sees the result", async () => {
	let ran = false;
	let afterResult: unknown;
	const tools: Record<string, ToolDef> = { write_file: { handler: () => { ran = true; return { written: true }; } } };
	const gate: ToolGate = { after: async (_t, _a, result) => { afterResult = result; } };
	await runAgentLoop({ session: fakeSession([writeCall, "done"]), tools, input: "go", events: noopEvents, root: "/tmp", gate });
	assert.equal(ran, true);
	assert.deepEqual(afterResult, { written: true });
});
