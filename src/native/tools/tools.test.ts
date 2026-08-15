// Deterministic checks for the tool executors + the path-jail safety boundary. No model needed.
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { buildTools, resolveInRoot, safeRegExp, type ExecResult, type ToolContext } from "./tools.ts";
import type { ToolDef } from "../inference/engine.ts";

// Normalize a handler result to Promise<any> — handlers are typed to allow
// sync or async returns, so tests wrap to inspect properties / assert rejection.
const call = (t: ToolDef, args: any): Promise<any> => Promise.resolve(t.handler(args));

async function fixture(overrides: Partial<ToolContext> = {}) {
	const root = await mkdtemp(join(tmpdir(), "llm-tools-"));
	await writeFile(join(root, "hello.txt"), "hi there");
	const calls: string[] = [];
	const ctx: ToolContext = {
		root,
		exec: async (command): Promise<ExecResult> => {
			calls.push(command);
			return { stdout: `ran: ${command}`, stderr: "", exitCode: 0 };
		},
		...overrides,
	};
	return { root, ctx, calls, tools: buildTools(ctx) };
}

test("read_file returns content; list_dir lists entries", async () => {
	const { tools } = await fixture();
	assert.equal((await call(tools.read_file, { path: "hello.txt" })).content, "hi there");
	const names = (await call(tools.list_dir, {})).entries.map((e: any) => e.name);
	assert.ok(names.includes("hello.txt"));
});

test("write_file writes the file", async () => {
	const { root, tools } = await fixture();
	assert.deepEqual(await call(tools.write_file, { path: "out/new.txt", content: "data" }), { written: true });
	assert.equal(await readFile(join(root, "out/new.txt"), "utf8"), "data");
});

test("path traversal outside the workspace is rejected", async () => {
	const { tools } = await fixture();
	await assert.rejects(call(tools.read_file, { path: "../../etc/passwd" }), /escapes the workspace/);
	await assert.rejects(call(tools.write_file, { path: "/etc/evil", content: "x" }), /escapes the workspace/);
});

test("run_terminal runs the command", async () => {
	const { tools, calls } = await fixture();
	await call(tools.run_terminal, { command: "echo ok" });
	assert.deepEqual(calls, ["echo ok"]);
});

test("git runs a subcommand with quoted args", async () => {
	const { tools, calls } = await fixture();
	await call(tools.git, { args: ["status"] });
	assert.equal(calls.length, 1);
	assert.match(calls[0]!, /^git 'status'/);
});

test("grep finds matching lines with path + line number", async () => {
	const { root, tools } = await fixture();
	await writeFile(join(root, "a.txt"), "alpha\nneedle here\nbeta");
	const res = await call(tools.grep, { pattern: "needle" });
	assert.equal(res.matches.length, 1);
	assert.equal(res.matches[0].path, "a.txt");
	assert.equal(res.matches[0].line, 2);
});

test("glob matches by pattern", async () => {
	const { root, tools } = await fixture();
	await writeFile(join(root, "keep.md"), "x");
	const res = await call(tools.glob, { pattern: "*.md" });
	assert.ok(res.matches.includes("keep.md"));
	assert.ok(!res.matches.includes("hello.txt"), "glob should not match a different extension");
});

// The jail is the security boundary for everything the model touches, so each way out
// of it gets its own test. A lexical-only check passes all three of these.

test("glob cannot escape the workspace", async () => {
	const { tools } = await fixture();
	// A cwd is not a boundary: a climbing pattern walks straight out of it, and an absolute
	// pattern ignores it entirely.
	await assert.rejects(call(tools.glob, { pattern: "../*" }), /escapes the workspace/);
	await assert.rejects(call(tools.glob, { pattern: "/etc/ho*" }), /escapes the workspace/);
	await assert.rejects(call(tools.glob, { pattern: "sub/../../*" }), /escapes the workspace/);
});

test("a symlink pointing outside the workspace is rejected", async () => {
	const { root, tools } = await fixture();
	const outside = await mkdtemp(join(tmpdir(), "llm-outside-"));
	await writeFile(join(outside, "secret.txt"), "not yours");
	await symlink(outside, join(root, "escape"));

	// Resolution has to be physical, not textual: "escape/secret.txt" is inside the root by
	// string comparison and outside it on disk.
	assert.throws(() => resolveInRoot(root, "escape/secret.txt"), /escapes the workspace/);
	await assert.rejects(call(tools.read_file, { path: "escape/secret.txt" }), /escapes the workspace/);
	await assert.rejects(call(tools.write_file, { path: "escape/planted.txt", content: "x" }), /escapes the workspace/);
});

test("a path whose parent does not exist yet is still allowed", async () => {
	const { root, tools } = await fixture();
	// write_file creates intermediate directories, so the jail must accept a target that is
	// not on disk yet — resolving only existing ancestors is what makes that work.
	assert.ok(resolveInRoot(root, "brand/new/file.txt").includes("file.txt"));
	assert.deepEqual(await call(tools.write_file, { path: "brand/new/file.txt", content: "ok" }), { written: true });
});

// Search patterns arrive from the wire (workspace search) and from the model (grep), and
// both run over every file on the process's only thread — the one serving the UI.

test("safeRegExp rejects a pattern that can hang the search", async () => {
	// The classic catastrophic shape. Against a few dozen "a"s this never finishes, and there
	// is no way to cancel a running match, so the whole server stops answering.
	assert.throws(() => safeRegExp("(a+)+$"), /nests one unbounded repeat/);
	assert.throws(() => safeRegExp("([a-z]*)*!"), /nests one unbounded repeat/);
	assert.throws(() => safeRegExp("(x+)*y"), /nests one unbounded repeat/);
	assert.throws(() => safeRegExp("a".repeat(1001)), /too long/);
});

test("safeRegExp still builds ordinary patterns", () => {
	assert.ok(safeRegExp("needle").test("a needle here"));
	assert.ok(safeRegExp("^const \\w+", "m").test("const foo = 1"));
	assert.ok(safeRegExp("TODO|FIXME", "i").test("todo: later"));
	// An invalid pattern still throws its own syntax error rather than being silently accepted.
	assert.throws(() => safeRegExp("([unclosed"));
});

test("grep surfaces a hostile pattern as an error instead of hanging", async () => {
	const { tools } = await fixture();
	await assert.rejects(call(tools.grep, { pattern: "(a+)+$" }), /nests one unbounded repeat/);
});
