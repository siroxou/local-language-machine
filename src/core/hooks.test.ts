// Hooks are enforcement: a failing PreToolUse hook must block the tool, and the
// matcher must scope a hook to specific tools.
import { test } from "node:test";
import assert from "node:assert/strict";
import { tmpdir } from "node:os";
import { runHooks } from "./hooks.ts";

const cwd = tmpdir();

test("a failing PreToolUse hook blocks the tool", async () => {
	const r = await runHooks({ PreToolUse: [{ command: "exit 3" }] }, "PreToolUse", cwd, { tool: "write_file" });
	assert.equal(r.block, true);
});

test("PostToolUse output is captured and does not block", async () => {
	const r = await runHooks({ PostToolUse: [{ command: "echo hooked" }] }, "PostToolUse", cwd, { tool: "write_file" });
	assert.equal(r.block, false);
	assert.match(r.output, /hooked/);
});

test("matcher scopes a hook to matching tool names", async () => {
	const hooks = { PreToolUse: [{ command: "exit 1", matcher: "write" }] };
	assert.equal((await runHooks(hooks, "PreToolUse", cwd, { tool: "read_file" })).block, false);
	assert.equal((await runHooks(hooks, "PreToolUse", cwd, { tool: "write_file" })).block, true);
});

test("a tool-scoped hook is skipped on events that carry no tool", async () => {
	// SessionStart/Stop have no tool in context — a matcher must skip, not fire on everything.
	const r = await runHooks({ Stop: [{ command: "echo fired", matcher: "write_file" }] }, "Stop", cwd);
	assert.equal(r.output, "");
});

test("a malformed hook entry is skipped instead of blocking every tool", async () => {
	// exec(undefined) rejects, which on PreToolUse would read as "hook failed" → total agent lockout.
	const hooks = { PreToolUse: [{} as any, { command: "echo ok" }] };
	const r = await runHooks(hooks, "PreToolUse", cwd, { tool: "write_file" });
	assert.equal(r.block, false);
	assert.match(r.output, /ok/);
});
