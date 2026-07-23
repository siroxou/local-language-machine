// Deterministic checks for the tool executors + the path-jail safety boundary. No model needed.
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { buildTools, type ExecResult, type ToolContext } from "./tools.ts";
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
