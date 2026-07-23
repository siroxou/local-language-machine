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
