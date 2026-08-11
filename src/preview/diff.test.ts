// The Changes panel's line diff.
//
// The renderer is one hand-written index.html with no module system, so the only way to unit-test a
// function in it is to pull the source out and evaluate it. diffLines earns that: comparing lines by
// index instead of by longest-common-subsequence would report every line after an insertion as
// changed, and nothing else in the suite would notice.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

interface Row { op: string; a?: number; b?: number; t?: string; n?: number }

const html = readFileSync(new URL("./public/index.html", import.meta.url), "utf8");
const from = html.indexOf("function diffLines(");
const to = html.indexOf("function openDiff(");
assert.ok(from > 0 && to > from, "diffLines/openDiff not found — did index.html get restructured?");
const { diffLines, collapse } = new Function(`${html.slice(from, to)}; return { diffLines, collapse };`)() as {
	diffLines: (a: string, b: string) => Row[];
	collapse: (rows: Row[], ctx?: number) => Row[];
};

const ops = (rows: Row[]) => rows.filter((r) => r.op === "+" || r.op === "-").map((r) => r.op + r.t);

test("identical text has no changed lines", () => {
	assert.deepEqual(ops(diffLines("a\nb\nc", "a\nb\nc")), []);
});

test("a line inserted mid-file is one addition, not a cascade", () => {
	// The whole point: an index-by-index diff would call c/d/e changed too.
	assert.deepEqual(ops(diffLines("a\nb\nc\nd\ne", "a\nb\nX\nc\nd\ne")), ["+X"]);
});

test("a line deleted mid-file is one deletion", () => {
	assert.deepEqual(ops(diffLines("a\nb\nc\nd", "a\nb\nd")), ["-c"]);
});

test("a changed line is one deletion plus one addition", () => {
	assert.deepEqual(ops(diffLines("a\nb\nc", "a\nB\nc")), ["-b", "+B"]);
});

test("line numbers track each side independently", () => {
	const rows = diffLines("a\nb\nc", "a\nX\nb\nc");
	const added = rows.find((r) => r.op === "+")!;
	assert.equal(added.b, 2);
	assert.equal(added.a, undefined); // no left-hand line — it doesn't exist there
	assert.equal(rows.find((r) => r.op === " " && r.t === "c")!.a, 3);
	assert.equal(rows.find((r) => r.op === " " && r.t === "c")!.b, 4);
});

test("creating a file is all additions, deleting it all deletions", () => {
	assert.deepEqual(ops(diffLines("", "x\ny")), ["+x", "+y"]);
	assert.deepEqual(ops(diffLines("x\ny", "")), ["-x", "-y"]);
});

test("collapse keeps context around a change and folds the rest", () => {
	const before = Array.from({ length: 40 }, (_, i) => `l${i}`).join("\n");
	const after = before.replace("l20", "CHANGED");
	const rows = collapse(diffLines(before, after), 3);
	assert.deepEqual(ops(rows), ["-l20", "+CHANGED"]);
	// 40 lines in, but only the change plus 3 lines either side survive as real rows.
	assert.ok(rows.filter((r) => r.op === " ").length <= 8, "too much context kept");
	assert.equal(rows.filter((r) => r.op === "~").reduce((n, r) => n + r.n!, 0), 33);
});
