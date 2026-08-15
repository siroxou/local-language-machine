// The guardrails, with no model involved. These are the assertions that make the
// "read-only" claim mean something, including the symlink case the shipped jail misses.
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { auditGuardrails, jail, type Budget } from "./guard.ts";
import { resolveInRoot } from "../native/tools/tools.ts";

const noTrace = () => {};

async function fixture() {
	const root = await mkdtemp(join(tmpdir(), "llm-guard-"));
	const outside = await mkdtemp(join(tmpdir(), "llm-outside-"));
	await writeFile(join(root, "inside.txt"), "in the tree");
	await writeFile(join(outside, "secret.txt"), "not in the tree");
	await mkdir(join(root, "sub"), { recursive: true });
	await writeFile(join(root, "sub", "nested.txt"), "nested");
	return { root, outside };
}

const call = (t: any, args: any): Promise<any> => Promise.resolve(t.handler(args));

// ————— the jail —————

test("jail accepts paths inside the root and rejects traversal", async () => {
	const { root } = await fixture();
	assert.ok(jail(root, "inside.txt").endsWith("inside.txt"));
	assert.ok(jail(root, "sub/nested.txt").includes("nested.txt"));
	assert.throws(() => jail(root, "../../etc/passwd"), /escapes the workspace/);
	assert.throws(() => jail(root, "/etc/passwd"), /escapes the workspace/);
});

test("jail resolves symlinks rather than comparing strings", async () => {
	const { root, outside } = await fixture();
	await symlink(outside, join(root, "escape"));

	// "escape/secret.txt" is inside the root by string comparison and outside it on disk.
	// The audit delegates to resolveInRoot, so both must agree — if they ever diverge, the
	// audit is enforcing a boundary the app does not have.
	assert.throws(() => jail(root, "escape/secret.txt"), /escapes the workspace/);
	assert.throws(() => resolveInRoot(root, "escape/secret.txt"), /escapes the workspace/);
});

test("jail allows a path whose tail does not exist yet", async () => {
	const { root } = await fixture();
	assert.ok(jail(root, "sub/not/created/yet.txt").includes("yet.txt"));
});

// ————— the toolset —————

test("mutating tools are absent, not merely denied", async () => {
	const { root } = await fixture();
	const g = auditGuardrails({ root, trace: noTrace });
	assert.deepEqual(Object.keys(g.tools).sort(), ["glob", "grep", "list_dir", "read_file"]);
	for (const absent of ["write_file", "run_terminal", "git", "task", "use_skill", "web_fetch", "web_search"]) {
		assert.equal(g.tools[absent], undefined, `${absent} must not exist on the audit toolset`);
	}
});

test("read_file works inside the root and is rejected outside it", async () => {
	const { root } = await fixture();
	const g = auditGuardrails({ root, trace: noTrace });
	assert.equal((await call(g.tools.read_file, { path: "inside.txt" })).content, "in the tree");
	await assert.rejects(call(g.tools.read_file, { path: "../../etc/passwd" }), /escapes the workspace/);
	assert.equal(g.stats().jailRejections, 1);
});

test("glob cannot climb out of the root, and counts what it dropped", async () => {
	const { root } = await fixture();
	const g = auditGuardrails({ root, trace: noTrace });

	// Rejected up front rather than enumerated-then-filtered: walking the parent to discard
	// it still reveals which sibling names exist by what survives.
	await assert.rejects(call(g.tools.glob, { pattern: "../*" }), /escapes the workspace/);
	await assert.rejects(call(g.tools.glob, { pattern: "/etc/ho*" }), /escapes the workspace/);
	await assert.rejects(call(g.tools.glob, { pattern: "sub/../../*" }), /escapes the workspace/);
	assert.equal(g.stats().jailRejections, 3, "each rejection is counted, not silently dropped");

	const inside = await call(g.tools.glob, { pattern: "**/*.txt" });
	assert.deepEqual(inside.matches.sort(), ["inside.txt", "sub/nested.txt"], "legitimate patterns still work");
});

// ————— budgets —————

test("the step budget blocks with an explicit reason rather than returning nothing", async () => {
	const { root } = await fixture();
	const budget: Budget = { steps: 2, wallMs: 60_000, tokens: 1_000 };
	const g = auditGuardrails({ root, budget, trace: noTrace });

	assert.equal(g.gate.before!("read_file", { path: "inside.txt" }), undefined);
	assert.equal(g.exhausted(), null, "one of two steps spent — still within budget");
	assert.equal(g.gate.before!("read_file", { path: "inside.txt" }), undefined);
	assert.match(g.exhausted()!, /step budget exhausted/, "a fully spent budget reports itself immediately");

	const third = g.gate.before!("read_file", { path: "inside.txt" }) as any;
	assert.equal(third.block, true);
	assert.match(third.result.halted, /step budget exhausted/);
	assert.match(g.exhausted()!, /step budget exhausted/);
	assert.equal(g.stats().blocked, 1);
});

test("the token budget is enforced from live session usage", async () => {
	const { root } = await fixture();
	let tokens = 0;
	const g = auditGuardrails({
		root,
		budget: { steps: 99, wallMs: 60_000, tokens: 100 },
		trace: noTrace,
		usedTokens: () => tokens,
	});
	assert.equal(g.gate.before!("grep", { pattern: "x" }), undefined);
	tokens = 500;
	const blocked = g.gate.before!("grep", { pattern: "x" }) as any;
	assert.match(blocked.result.halted, /token budget exhausted/);
});

test("a tool outside the read-only set is blocked with a reason even if it is somehow present", async () => {
	const { root } = await fixture();
	const g = auditGuardrails({ root, trace: noTrace });
	const decision = g.gate.before!("write_file", { path: "x", content: "y" }) as any;
	assert.equal(decision.block, true);
	assert.match(decision.result.error, /not in the read-only audit set/);
});

test("reset clears the per-investigation budget but keeps run counters", async () => {
	const { root } = await fixture();
	const g = auditGuardrails({ root, budget: { steps: 1, wallMs: 60_000, tokens: 9_999 }, trace: noTrace });
	g.gate.before!("read_file", { path: "inside.txt" });
	assert.ok(g.gate.before!("read_file", { path: "inside.txt" }), "second call is blocked");
	g.reset();
	assert.equal(g.exhausted(), null, "budget is fresh after reset");
	assert.equal(g.stats().blocked, 1, "run-level counters survive a reset");
});

test("every allowed call is traced with its gate decision", async () => {
	const { root } = await fixture();
	const records: any[] = [];
	const g = auditGuardrails({ root, trace: (r) => records.push(r), question: "q1" });
	g.gate.before!("read_file", { path: "inside.txt" });
	assert.equal(records.length, 1);
	assert.equal(records[0].gate, "allow");
	assert.equal(records[0].tool, "read_file");
	assert.equal(records[0].question, "q1");
	assert.ok(records[0].args.includes("inside.txt"), "args are recorded readably, not hashed");
});
