// The permission policy is the safety money-path — one wrong branch and a mode
// silently over- or under-gates. Cover the mode × tool matrix.
import { test } from "node:test";
import assert from "node:assert/strict";
import { permissionDecision } from "./settings.ts";

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
