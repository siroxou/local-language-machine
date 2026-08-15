// The model-driven pass, against a stub engine. No real model: this asserts the contract —
// the session is disposed, budgets are reported, and nothing unverifiable escapes — which is
// exactly what must hold regardless of which model ran. Quality of findings is a property of
// the model and is reported per run as droppedInvalid/emitted, not asserted here.
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { investigate, QUESTIONS } from "./investigate.ts";
import { validate } from "./findings.ts";
import type { ChatSession, InferenceEngine } from "../native/inference/engine.ts";

/** An engine whose session replays scripted outputs and returns canned structured findings. */
function stubEngine(opts: { outputs?: string[]; emit?: unknown; disposed?: string[]; throwOnEmit?: boolean }): InferenceEngine {
	let i = 0;
	const session: ChatSession = {
		async prompt(_t, o) {
			const out = opts.outputs?.[i++] ?? "done looking";
			o?.onText?.(out);
			return out;
		},
		async structured() {
			if (opts.throwOnEmit) throw new Error("grammar failed");
			return (opts.emit ?? { findings: [] }) as any;
		},
		usedTokens: () => 100,
		getHistory: () => [],
		setHistory: () => {},
		dispose: () => opts.disposed?.push("disposed"),
	};
	return {
		async load() {},
		async createSession() {
			return session;
		},
		setSampling() {},
		async unload() {},
		get gpu() {
			return "stub";
		},
		get contextSize() {
			return 32_768;
		},
	} as unknown as InferenceEngine;
}

async function fixture() {
	const root = await mkdtemp(join(tmpdir(), "llm-inv-"));
	// Realistic lines: validation now requires the quoted evidence to appear in the file, so a
	// fixture of "one/two/three" would be rejected for being too short to corroborate anything.
	await writeFile(join(root, "real.ts"), [
		'import { readFile } from "node:fs/promises";',
		"const res = await fetch(url);",
		"export const done = true;",
	].join("\n") + "\n");
	return root;
}

const q = { id: "test-lens", prompt: "look at something" };

test("a finding the model emits is mapped, attributed, and returned", async () => {
	const root = await fixture();
	const r = await investigate({
		engine: stubEngine({
			emit: { findings: [{ file: "real.ts", line: 2, severity: "high", message: "fetch with no timeout", evidence: "const res = await fetch(url);" }] },
		}),
		root,
		question: q,
		trace: () => {},
	});
	assert.equal(r.findings.length, 1);
	assert.equal(r.findings[0]!.rule, "model:test-lens", "a finding must name the lens that produced it");
	assert.equal(r.findings[0]!.origin, "model");
	assert.equal(r.emitted, 1);
});

test("the session is always disposed, even when emitting throws", async () => {
	const root = await fixture();
	const disposed: string[] = [];
	const r = await investigate({
		engine: stubEngine({ disposed, throwOnEmit: true }),
		root,
		question: q,
		trace: () => {},
	});
	assert.deepEqual(disposed, ["disposed"], "a leaked context is gigabytes of KV cache");
	assert.deepEqual(r.findings, [], "a failed emit yields nothing rather than throwing");
});

test("one failed investigation does not lose the others", async () => {
	const root = await fixture();
	const engine = {
		async createSession() {
			throw new Error("out of memory");
		},
	} as unknown as InferenceEngine;
	const r = await investigate({ engine, root, question: q, trace: () => {} });
	assert.deepEqual(r.findings, []);
	assert.equal(r.emitted, 0);
});

test("exhausting the step budget is reported, never silently absent", async () => {
	const root = await fixture();
	// A model that only ever emits tool calls. The budget must stop it AND say so — the whole
	// point of the audit is not to repeat the silent-truncation bug it was built to find.
	const readCall = '{"tool":"read_file","arguments":{"path":"real.ts"}}';
	const r = await investigate({
		engine: stubEngine({ outputs: Array(20).fill(readCall) }),
		root,
		question: q,
		budget: { steps: 2, wallMs: 60_000, tokens: 99_999 },
		trace: () => {},
	});
	assert.ok(r.exhausted, "a stopped run must name what stopped it");
	assert.match(r.exhausted!, /step budget exhausted/);
	assert.ok(r.steps <= 2, `spent ${r.steps} steps against a budget of 2`);
});

test("a finding citing a file that does not exist is dropped by validation", async () => {
	const root = await fixture();
	const r = await investigate({
		engine: stubEngine({
			emit: {
				findings: [
					{ file: "real.ts", line: 1, severity: "low", message: "real", evidence: 'import { readFile } from "node:fs/promises";' },
					{ file: "imaginary.ts", line: 1, severity: "high", message: "file does not exist", evidence: "const res = await fetch(url);" },
					{ file: "real.ts", line: 9999, severity: "high", message: "line past eof", evidence: "export const done = true;" },
					// A real path and a real line, but evidence that is nowhere in the file — the shape a
					// model produces when it never opened the file at all.
					{ file: "real.ts", line: 1, severity: "high", message: "uncorroborated", evidence: "function thatWasNeverThere() {" },
				],
			},
		}),
		root,
		question: q,
		trace: () => {},
	});
	const { kept, dropped } = await validate(root, r.findings);
	assert.equal(kept.length, 1, "only the corroborated citation survives");
	assert.equal(dropped.length, 3);
});

test("every tool call is traced with its gate decision", async () => {
	const root = await fixture();
	const records: any[] = [];
	await investigate({
		engine: stubEngine({ outputs: ['{"tool":"read_file","arguments":{"path":"real.ts"}}', "done"] }),
		root,
		question: q,
		trace: (r) => records.push(r),
	});
	const allowed = records.filter((r) => r.gate === "allow");
	assert.equal(allowed.length, 1);
	assert.equal(allowed[0].tool, "read_file");
	assert.equal(allowed[0].question, "test-lens", "a trace record must be attributable to its lens");
});

test("the shipped lenses are distinct and each names files to read", () => {
	// A lens that does not point at anything concrete produces a wandering read and finds the
	// same shallow things every other lens found.
	assert.ok(QUESTIONS.length >= 4);
	assert.equal(new Set(QUESTIONS.map((x) => x.id)).size, QUESTIONS.length, "ids must be unique — they key the findings");
	for (const lens of QUESTIONS) {
		assert.match(lens.prompt, /src\//, `lens "${lens.id}" must name where to look`);
		assert.ok(lens.prompt.length > 120, `lens "${lens.id}" is too vague to hold a question fixed`);
	}
});
