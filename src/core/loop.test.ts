// The gate seam is where checkpoints, permissions, and hooks plug in — verify that
// `before` can block a tool (handler never runs) and that `after` observes results.
import { test } from "node:test";
import assert from "node:assert/strict";
import { MAX_TOOL_STEPS, runAgentLoop, type ToolGate } from "./loop.ts";
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

test("exhausting the step budget returns an explicit notice, never an empty string", async () => {
	// A model that keeps calling tools and never answers used to fall out of the loop with
	// `final` still "", which was then persisted as an empty assistant turn and streamed to
	// the UI as a completed reply. "No answer" and "ran out of budget" must not look alike.
	const readCall = '{"tool":"read_file","arguments":{"path":"a"}}';
	const tools: Record<string, ToolDef> = { read_file: { handler: () => ({ content: "x" }) } };
	const final = await runAgentLoop({
		session: fakeSession(Array(MAX_TOOL_STEPS + 2).fill(readCall)),
		tools,
		input: "go",
		events: noopEvents,
		root: "/tmp",
	});
	assert.notEqual(final.trim(), "", "the caller must receive something it can show");
	assert.match(final, /tool step/i, "the notice must say what stopped it");
	assert.match(final, new RegExp(String(MAX_TOOL_STEPS)), "and how many steps that was");
});

test("the last step tells the model to stop calling tools and answer", async () => {
	// Cheaper than surfacing an error: on the final step the model is asked for prose, which
	// usually turns a would-be budget exhaustion into a real answer.
	const readCall = '{"tool":"read_file","arguments":{"path":"a"}}';
	const prompts: string[] = [];
	const session: ChatSession = {
		async prompt(t, opts) {
			prompts.push(t);
			const o = prompts.length >= MAX_TOOL_STEPS ? "here is the answer" : readCall;
			opts?.onText?.(o);
			return o;
		},
		async structured() { return {} as any; },
		getHistory() { return []; },
		setHistory() {},
		dispose() {},
	};
	const tools: Record<string, ToolDef> = { read_file: { handler: () => ({ content: "x" }) } };
	const final = await runAgentLoop({ session, tools, input: "go", events: noopEvents, root: "/tmp" });
	assert.equal(final, "here is the answer");
	assert.match(prompts.at(-1)!, /without calling another tool|no more tool/i, "the final prompt must forbid further tools");
});
