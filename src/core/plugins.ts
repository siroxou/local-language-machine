// Local Language Machine — import skills / plugins.
//
// Because we read the on-disk skill/agent layout verbatim, importing is just copying
// files into the user skills/agents dirs and re-scanning. One entry point handles every
// shape — a single skill dir, a skills collection, a plugin, or a whole .claude/ dir —
// by finding every SKILL.md and every agents/*.md underneath the source.

import { cpSync, mkdirSync, readdirSync, statSync, existsSync, mkdtempSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { tmpdir } from "node:os";
import { join, basename, dirname } from "node:path";
import { appHome } from "../native/paths.ts";

const userSkillsDir = () => join(appHome(), "skills");
const userAgentsDir = () => join(appHome(), "agents");
const SKIP = new Set(["node_modules", ".git"]);

/** Recursively collect files named `name` (skip vendor dirs, depth-limited). */
function findByName(dir: string, name: string, depth = 6, acc: string[] = []): string[] {
	if (depth < 0 || !existsSync(dir)) return acc;
	for (const e of readdirSync(dir, { withFileTypes: true })) {
		if (SKIP.has(e.name)) continue;
		const p = join(dir, e.name);
		if (e.isDirectory()) findByName(p, name, depth - 1, acc);
		else if (e.name === name) acc.push(p);
	}
	return acc;
}

/** Collect *.md files that live directly inside a directory named "agents". */
function findAgentFiles(dir: string, depth = 6, acc: string[] = []): string[] {
	if (depth < 0 || !existsSync(dir)) return acc;
	for (const e of readdirSync(dir, { withFileTypes: true })) {
		if (SKIP.has(e.name)) continue;
		const p = join(dir, e.name);
		if (e.isDirectory()) findAgentFiles(p, depth - 1, acc);
		else if (e.name.endsWith(".md") && basename(dirname(p)) === "agents") acc.push(p);
	}
	return acc;
}

export interface ImportResult {
	skills: string[];
	agents: string[];
}

/** Import every skill and agent found under a local directory into the user dirs. */
export function importFrom(src: string): ImportResult {
	if (!existsSync(src) || !statSync(src).isDirectory()) throw new Error(`Not a directory: ${src}`);
	mkdirSync(userSkillsDir(), { recursive: true });
	mkdirSync(userAgentsDir(), { recursive: true });
	const skills: string[] = [];
	const agents: string[] = [];

	// A bare skill dir (has its own SKILL.md) imports as itself; otherwise import each nested skill.
	const skillFiles = existsSync(join(src, "SKILL.md")) ? [join(src, "SKILL.md")] : findByName(src, "SKILL.md");
	for (const file of skillFiles) {
		const sdir = dirname(file);
		const name = basename(sdir);
		cpSync(sdir, join(userSkillsDir(), name), { recursive: true });
		skills.push(name);
	}
	for (const file of findAgentFiles(src)) {
		cpSync(file, join(userAgentsDir(), basename(file)));
		agents.push(basename(file).replace(/\.md$/, ""));
	}
	if (skills.length === 0 && agents.length === 0) {
		throw new Error("No SKILL.md or agents/*.md found under that path.");
	}
	return { skills, agents };
}

/** Clone a git repo (shallow) and import from it. Caller must gate this behind settings.online. */
export function importFromGit(url: string): ImportResult {
	const dir = mkdtempSync(join(tmpdir(), "llm-import-"));
	// execFileSync blocks the event loop, so an unreachable host or a credential prompt would
	// freeze the whole server, not just this import. Cap it, and refuse the prompt outright.
	execFileSync("git", ["clone", "--depth", "1", url, dir], {
		stdio: "ignore",
		timeout: 60_000,
		env: { ...process.env, GIT_TERMINAL_PROMPT: "0" },
	});
	return importFrom(dir);
}
