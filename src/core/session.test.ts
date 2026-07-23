// Session persistence: transcript round-trip, history save/load, list labelling,
// and fork copying the whole session. LLM_HOME points at a temp dir.
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

process.env.LLM_HOME = mkdtempSync(join(tmpdir(), "llm-home-"));
const { Session } = await import("./session.ts");

const root = "/projects/demo";

test("appends and reads back the transcript", () => {
	const s = Session.create(root);
	s.append({ type: "user", text: "hello" });
	s.append({ type: "assistant", text: "hi" });
	const ev = s.readEvents();
	assert.equal(ev.length, 2);
	assert.equal(ev[0]!.text, "hello");
});

test("history round-trips", () => {
	const s = Session.create(root);
	s.saveHistory([{ type: "system", text: "x" }] as any);
	assert.deepEqual(s.loadHistory(), [{ type: "system", text: "x" }]);
});

test("list surfaces sessions with a label from the first user message", () => {
	const s = Session.create(root);
	s.append({ type: "user", text: "label me" });
	const found = Session.list(root).find((x) => x.id === s.id);
	assert.ok(found);
	assert.equal(found!.label, "label me");
});

test("fork copies the transcript into a new id", () => {
	const s = Session.create(root);
	s.append({ type: "user", text: "keep" });
	const f = s.fork();
	assert.notEqual(f.id, s.id);
	assert.equal(f.readEvents().length, 1);
});
