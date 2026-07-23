// The frontmatter parser and the import→discover round-trip. LLM_HOME points at a
// temp dir so the real user skills dir is never touched.
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

process.env.LLM_HOME = mkdtempSync(join(tmpdir(), "llm-home-"));

const { parseFrontmatter, discoverSkills } = await import("./skills.ts");
const { importFrom } = await import("./plugins.ts");

test("parseFrontmatter reads scalars, inline lists, and the body", () => {
	const { data, body } = parseFrontmatter(
		["---", "name: deploy", "description: Ship it", "disable-model-invocation: true", "allowed-tools: [read_file, run_terminal]", "context: fork", "---", "Do the deploy."].join("\n"),
	);
	assert.equal(data.name, "deploy");
	assert.equal(data.description, "Ship it");
	assert.equal(data["disable-model-invocation"], true);
	assert.deepEqual(data["allowed-tools"], ["read_file", "run_terminal"]);
	assert.equal(data.context, "fork");
	assert.equal(body.trim(), "Do the deploy.");
});

test("parseFrontmatter handles block-style lists and no-frontmatter input", () => {
	const { data } = parseFrontmatter(["---", "tags:", "  - a", "  - b", "---", "x"].join("\n"));
	assert.deepEqual(data.tags, ["a", "b"]);
	assert.deepEqual(parseFrontmatter("just a body").data, {});
});

test("discoverSkills reads the ./.claude/skills layout", () => {
	const root = mkdtempSync(join(tmpdir(), "llm-proj-"));
	mkdirSync(join(root, ".claude", "skills", "review"), { recursive: true });
	writeFileSync(join(root, ".claude", "skills", "review", "SKILL.md"), "---\nname: review\ndescription: Review code\n---\nChecklist");
	const skills = discoverSkills(root);
	const review = skills.find((s) => s.name === "review");
	assert.ok(review, "should discover the project skill");
	assert.equal(review!.description, "Review code");
	assert.equal(review!.source, "project");
});

test("importing a skill dir makes it discoverable as a user skill", () => {
	const src = mkdtempSync(join(tmpdir(), "llm-src-"));
	mkdirSync(join(src, "skills", "greet"), { recursive: true });
	writeFileSync(join(src, "skills", "greet", "SKILL.md"), "---\nname: greet\ndescription: Say hi\n---\nSay hi nicely.");
	mkdirSync(join(src, "agents"), { recursive: true });
	writeFileSync(join(src, "agents", "helper.md"), "---\nname: helper\ndescription: A helper\n---\nYou help.");

	const result = importFrom(src);
	assert.deepEqual(result.skills, ["greet"]);
	assert.deepEqual(result.agents, ["helper"]);

	// Discover against an empty project → the imported skill comes from the user dir.
	const found = discoverSkills(mkdtempSync(join(tmpdir(), "llm-empty-")));
	assert.ok(found.some((s) => s.name === "greet" && s.source === "user"));
});
