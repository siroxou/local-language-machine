// The permission policy is the safety money-path — one wrong branch and a mode
// silently over- or under-gates. Cover the mode × tool matrix.
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { loadSettings, permissionDecision } from "./settings.ts";

const d = (mode: any, tool: string, args: any = {}, allow: string[] = []) => permissionDecision(mode, tool, args, allow);

test("reads are always allowed, in every mode", () => {
	for (const mode of ["manual", "accept-edits", "plan", "auto"] as const) {
		assert.equal(d(mode, "read_file", { path: "a" }), "allow");
		assert.equal(d(mode, "grep", { pattern: "x" }), "allow");
	}
});

test("plan mode denies every mutation", () => {
	assert.equal(d("plan", "write_file", { path: "a", content: "x" }), "deny");
	assert.equal(d("plan", "run_terminal", { command: "ls" }), "deny");
});

test("manual mode asks before mutations", () => {
	assert.equal(d("manual", "write_file", { path: "a", content: "x" }), "ask");
	assert.equal(d("manual", "run_terminal", { command: "ls" }), "ask");
});

test("accept-edits auto-approves file writes but still asks for commands", () => {
	assert.equal(d("accept-edits", "write_file", { path: "a", content: "x" }), "allow");
	assert.equal(d("accept-edits", "run_terminal", { command: "ls" }), "ask");
});

test("auto allows safe commands but asks for dangerous ones", () => {
	assert.equal(d("auto", "run_terminal", { command: "npm test" }), "allow");
	assert.equal(d("auto", "run_terminal", { command: "rm -rf build" }), "ask");
	assert.equal(d("auto", "git", { args: ["push", "origin", "main"] }), "ask");
});

test("git read-only subcommands are allowed; the allow-list skips a specific command", () => {
	assert.equal(d("manual", "git", { args: ["status"] }), "allow");
	assert.equal(d("manual", "run_terminal", { command: "npm run build" }, ["npm run build"]), "allow");
});

test("plan mode outranks the allow-list", () => {
	// An allow entry must not punch a hole in the one mode whose whole promise is "nothing is written".
	assert.equal(d("plan", "run_terminal", { command: "npm run build" }, ["npm run build"]), "deny");
	assert.equal(d("plan", "write_file", { path: "a", content: "x" }, ["write_file"]), "deny");
});

test("a cloned project's settings cannot escalate permissionMode or online", () => {
	// ./.claude/settings.json ships inside the repo, so it must not be able to switch off the
	// permission prompt or turn the network on. Compared against a bare root rather than a literal,
	// so the user's own settings file (whatever it says) doesn't make this flaky.
	const bare = mkdtempSync(join(tmpdir(), "llm-bare-"));
	const proj = mkdtempSync(join(tmpdir(), "llm-proj-"));
	mkdirSync(join(proj, ".claude"), { recursive: true });
	writeFileSync(
		join(proj, ".claude", "settings.json"),
		JSON.stringify({ permissionMode: "auto", online: true, allow: ["npm test"] }),
	);
	const base = loadSettings(bare);
	const withProject = loadSettings(proj);
	assert.equal(withProject.permissionMode, base.permissionMode);
	assert.equal(withProject.online, base.online);
	assert.deepEqual(withProject.allow, base.allow); // nor may it pre-approve commands
});

test("the danger guard catches rm regardless of flag spelling", () => {
	// -rf is one spelling of several; the guard is worthless if it only catches the tidy one.
	for (const cmd of ["rm -rf build", "rm -fr build", "rm -Rf build", "rm -f notes.txt", "RM -RF build"]) {
		assert.equal(d("auto", "run_terminal", { command: cmd }), "ask", cmd);
	}
});
