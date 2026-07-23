// Local Language Machine — skills (reusable knowledge + invocable workflows).
//
// Reads ./.claude/skills/<name>/SKILL.md (and a user dir ~/.local-language-machine/skills),
// so an existing skill folder works unmodified. Descriptions are injected into the system
// prompt; the SKILL.md body loads on demand when the skill is invoked (/name) or matched
// by description.

import { readdirSync, readFileSync, existsSync, statSync } from "node:fs";
import { join } from "node:path";
import { appHome } from "../native/paths.ts";

export interface Skill {
	name: string;
	description: string;
	body: string;
	dir: string;
	/** disable-model-invocation: only the user can trigger it (hidden from the model). */
	userInvocableOnly: boolean;
	/** context: fork → run in an isolated subagent. */
	fork: boolean;
	source: "project" | "user";
}

const userSkillsDir = () => join(appHome(), "skills");

/** Project skills (./.claude/skills) then user skills; dedupe by name, first-seen wins. */
export function discoverSkills(root: string): Skill[] {
	const seen = new Set<string>();
	const out: Skill[] = [];
	for (const [dir, source] of [[join(root, ".claude", "skills"), "project"], [userSkillsDir(), "user"]] as const) {
		for (const skill of scanDir(dir, source)) {
			if (seen.has(skill.name)) continue;
			seen.add(skill.name);
			out.push(skill);
		}
	}
	return out;
}

function scanDir(dir: string, source: "project" | "user"): Skill[] {
	if (!existsSync(dir)) return [];
	const out: Skill[] = [];
	for (const entry of readdirSync(dir)) {
		// Layout: <name>/SKILL.md. Also accept a bare <name>.md.
		const asDir = join(dir, entry, "SKILL.md");
		const asFile = join(dir, entry);
		let file: string | undefined;
		let skillDir = dir;
		if (existsSync(asDir)) { file = asDir; skillDir = join(dir, entry); }
		else if (entry.endsWith(".md") && statSync(asFile).isFile()) file = asFile;
		if (!file) continue;
		try {
			const { data, body } = parseFrontmatter(readFileSync(file, "utf8"));
			const name = String(data.name ?? entry.replace(/\.md$/, ""));
			out.push({
				name,
				description: String(data.description ?? ""),
				body: body.trim(),
				dir: skillDir,
				userInvocableOnly: truthy(data["disable-model-invocation"]),
				fork: data.context === "fork",
				source,
			});
		} catch {}
	}
	return out;
}

const truthy = (v: unknown) => v === true || v === "true";

function unquote(v: string): unknown {
	const s = v.trim().replace(/^["']|["']$/g, "");
	if (s === "true") return true;
	if (s === "false") return false;
	return s;
}

/**
 * Minimal YAML-frontmatter parser: `key: scalar`, inline `key: [a, b]`, and `-` block
 * lists. nested maps / multiline scalars are NOT supported — enough for skill
 * frontmatter (name, description, disable-model-invocation, allowed-tools, context, model).
 */
export function parseFrontmatter(src: string): { data: Record<string, unknown>; body: string } {
	const m = src.match(/^﻿?---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/);
	if (!m) return { data: {}, body: src };
	const data: Record<string, unknown> = {};
	let key: string | null = null;
	for (const raw of m[1]!.split("\n")) {
		const line = raw.replace(/\s+$/, "");
		if (!line.trim() || line.trim().startsWith("#")) continue;
		const block = line.match(/^\s*-\s+(.*)$/);
		if (block && key) {
			(data[key] as unknown[]) = [...((data[key] as unknown[]) ?? []), unquote(block[1]!)];
			continue;
		}
		const kv = line.match(/^([\w-]+):\s*(.*)$/);
		if (!kv) continue;
		key = kv[1]!;
		const val = kv[2]!.trim();
		if (val === "") data[key] = [];
		else if (val.startsWith("[")) data[key] = val.replace(/^\[|\]$/g, "").split(",").map((s) => unquote(s)).filter((s) => s !== "");
		else data[key] = unquote(val);
	}
	return { data, body: m[2] ?? "" };
}
