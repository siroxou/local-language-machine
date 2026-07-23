// The slash router dispatches to the right handler and resolves skills by name.
import { test } from "node:test";
import assert from "node:assert/strict";
import { isSlash, handleCommand, type CommandDeps } from "./commands.ts";
import type { Skill } from "./skills.ts";

const skill = (name: string): Skill => ({ name, description: "d", body: "BODY", dir: "", userInvocableOnly: false, fork: false, source: "user" });

const deps: CommandDeps = {
	models: () => [{ id: "m", label: "M", loaded: true }],
	loadModel: async () => {},
	contextReport: () => "CTX",
	compact: async () => "compacted",
	clearSession: () => {},
	initMemory: () => "made AGENTS.md",
	sessions: () => [{ id: "abc", label: "hello" }],
	resume: async () => {},
	fork: () => "newid",
	skills: () => [skill("deploy")],
};

test("isSlash detects commands, not normal prose or bare slashes", () => {
	assert.equal(isSlash("/help"), true);
	assert.equal(isSlash("  /model x"), true);
	assert.equal(isSlash("what is /help"), false);
	assert.equal(isSlash("/ not a command"), false);
});

test("built-ins return replies", async () => {
	assert.equal((await handleCommand("/context", deps) as any).text, "CTX");
	assert.equal((await handleCommand("/init", deps) as any).text, "made AGENTS.md");
	assert.match((await handleCommand("/help", deps) as any).text, /\/deploy/);
});

test("a known skill routes to a skill result carrying its arg", async () => {
	const r = await handleCommand("/deploy staging now", deps);
	assert.equal(r.kind, "skill");
	if (r.kind === "skill") { assert.equal(r.skill.name, "deploy"); assert.equal(r.arg, "staging now"); }
});

test("an unknown command replies rather than throwing", async () => {
	assert.match((await handleCommand("/nope", deps) as any).text, /Unknown command/);
});
