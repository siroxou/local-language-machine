// Session persistence: transcript round-trip, history save/load, list labelling,
// and fork copying the whole session. LLM_HOME points at a temp dir.
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, mkdtempSync } from "node:fs";
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

test("fork of an unwritten session still produces a usable session", () => {
	const f = Session.create(root).fork(); // dir does not exist yet — cpSync would throw
	f.append({ type: "user", text: "after fork" });
	assert.equal(f.readEvents().length, 1);
});

// Session.create() runs on every launch, Open Folder and /clear. When the constructor mkdir'd
// eagerly, each of those left a permanent empty directory that list() then sorted to the top.
test("creating a session writes nothing until it is used", () => {
	const before = Session.list(root).length;
	const s = Session.create(root);
	assert.equal(existsSync(s.dir), false);
	assert.equal(Session.list(root).length, before);
});

test("list hides sessions with no user message", () => {
	const empty = Session.create(root);
	empty.append({ type: "tool-call", name: "read", data: {} }); // tool noise, no conversation
	assert.equal(Session.list(root).some((x) => x.id === empty.id), false);
});

test("list reports message count and token total", () => {
	const s = Session.create(root);
	s.append({ type: "user", text: "one" });
	s.append({ type: "assistant", text: "…" });
	s.append({ type: "user", text: "two" });
	s.addTokens(120);
	s.addTokens(-50); // compaction shrinks the window; must not subtract
	s.addTokens(30);
	const found = Session.list(root).find((x) => x.id === s.id);
	assert.equal(found!.messages, 2);
	assert.equal(found!.tokens, 150);
	assert.equal(found!.label, "one");
});

test("a session id that would escape the sessions directory is rejected", () => {
	// `resume-session` passes an id straight off the WebSocket into a path join, so the
	// shape is checked at the constructor — the one point create/open/fork all pass through.
	for (const bad of ["../..", "../../etc", "a/b", "a\\b", ".", "..", "", "x".repeat(65)]) {
		assert.throws(() => Session.open(root, bad), /Invalid session id/, `should reject ${JSON.stringify(bad)}`);
	}
	// The real shape — an 8-char slice of a uuid — still works.
	assert.ok(Session.open(root, "1a2b3c4d").dir.endsWith("1a2b3c4d"));
});
