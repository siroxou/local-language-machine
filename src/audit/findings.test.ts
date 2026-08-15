// Identity, validation and baseline partitioning. No model, no scanning — pure logic.
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fingerprint, loadBaseline, partition, sortFindings, validate, type Baseline, type Finding } from "./findings.ts";

const finding = (over: Partial<Finding> = {}): Finding => ({
	rule: "fetch-no-timeout",
	severity: "medium",
	file: "src/a.ts",
	line: 10,
	message: "fetch without a signal",
	evidence: "const res = await fetch(url);",
	origin: "rule",
	...over,
});

test("fingerprint ignores the line number so edits above a finding don't re-alarm", () => {
	const a = finding({ line: 10 });
	const b = finding({ line: 412 });
	assert.equal(fingerprint(a), fingerprint(b));
});

test("fingerprint survives reindentation but changes with the evidence", () => {
	const base = finding({ evidence: "const res = await fetch(url);" });
	const reindented = finding({ evidence: "\t\tconst   res = await   fetch(url);  " });
	assert.equal(fingerprint(base), fingerprint(reindented));

	const different = finding({ evidence: "const res = await fetch(other);" });
	assert.notEqual(fingerprint(base), fingerprint(different));
});

test("fingerprint is scoped to rule and file", () => {
	const base = finding();
	assert.notEqual(fingerprint(base), fingerprint(finding({ rule: "other-rule" })));
	assert.notEqual(fingerprint(base), fingerprint(finding({ file: "src/b.ts" })));
});

test("partition splits fresh from baselined and reports resolved entries", () => {
	const known = finding({ file: "src/known.ts" });
	const fresh = finding({ file: "src/fresh.ts" });
	const baseline: Baseline = {
		accepted: {
			[fingerprint(known)]: { rule: known.rule, file: known.file, note: "deferred" },
			deadbeef0000: { rule: "gone", file: "src/gone.ts", note: "already fixed" },
		},
	};

	const p = partition([known, fresh], baseline);
	assert.deepEqual(p.fresh.map((f) => f.file), ["src/fresh.ts"]);
	assert.deepEqual(p.known.map((f) => f.file), ["src/known.ts"]);
	assert.deepEqual(p.resolved, ["deadbeef0000"]);
});

test("a baseline entry with no note is rejected", async () => {
	const dir = await mkdtemp(join(tmpdir(), "llm-audit-"));
	const path = join(dir, "baseline.json");

	await writeFile(path, JSON.stringify({ accepted: { abc123: { rule: "r", file: "f", note: "  " } } }));
	assert.throws(() => loadBaseline(path), /no note/);

	await writeFile(path, JSON.stringify({ accepted: { abc123: { rule: "r", file: "f", note: "why" } } }));
	assert.equal(Object.keys(loadBaseline(path).accepted).length, 1);
});

test("a missing baseline is empty; a malformed one is fatal", async () => {
	const dir = await mkdtemp(join(tmpdir(), "llm-audit-"));
	assert.deepEqual(loadBaseline(join(dir, "absent.json")), { accepted: {} });

	const bad = join(dir, "bad.json");
	await writeFile(bad, "{ not json");
	assert.throws(() => loadBaseline(bad), /not valid JSON/);

	const shapeless = join(dir, "shapeless.json");
	await writeFile(shapeless, JSON.stringify({ nope: true }));
	assert.throws(() => loadBaseline(shapeless), /no "accepted" object/);
});

test("validate keeps only findings whose citation is corroborated by the file", async () => {
	const root = await mkdtemp(join(tmpdir(), "llm-audit-"));
	const real = "const res = await fetch(url);";
	await writeFile(join(root, "real.ts"), `import x from "y";\n${real}\nexport default x;\n`);

	const { kept, dropped } = await validate(root, [
		finding({ file: "real.ts", line: 2, evidence: real }),
		finding({ file: "real.ts", line: 999, evidence: real }),
		finding({ file: "absent.ts", line: 1, evidence: real }),
		finding({ file: "../escape.ts", line: 1, evidence: real }),
		finding({ file: "real.ts", line: 0, evidence: real }),
		// A real path at a real line, but evidence that is nowhere in the file — what a model
		// produces when it never opened the file. Checking the citation exists does not catch it.
		finding({ file: "real.ts", line: 1, evidence: "function neverThere() { return 1; }" }),
		// Evidence too short to corroborate anything: "}" matches almost every source file.
		finding({ file: "real.ts", line: 1, evidence: "}" }),
	]);

	assert.deepEqual(kept.map((f) => f.line), [2]);
	assert.equal(dropped.length, 6);
});

test("validate tolerates a truncated quote of a real line", async () => {
	const root = await mkdtemp(join(tmpdir(), "llm-audit-"));
	await writeFile(join(root, "real.ts"), "const res = await fetch(url, { headers: h });\n");
	// A model quoting the first part of a long line has still read it; only the line number is
	// worth being loose about, not whether the text was invented.
	const { kept } = await validate(root, [finding({ file: "real.ts", line: 1, evidence: "const res = await fetch(url," })]);
	assert.equal(kept.length, 1);
});

test("sortFindings orders by severity, then file, then line", () => {
	const sorted = sortFindings([
		finding({ severity: "low", file: "src/a.ts", line: 1 }),
		finding({ severity: "high", file: "src/z.ts", line: 9 }),
		finding({ severity: "medium", file: "src/a.ts", line: 5 }),
		finding({ severity: "high", file: "src/a.ts", line: 2 }),
	]);
	assert.deepEqual(
		sorted.map((f) => `${f.severity}:${f.file}:${f.line}`),
		["high:src/a.ts:2", "high:src/z.ts:9", "medium:src/a.ts:5", "low:src/a.ts:1"],
	);
});
