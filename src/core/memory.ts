// Local Language Machine — project memory (AGENTS.md + MEMORY.md).
//
// Both files are read from the workspace root at session start and prepended to the
// system prompt, so a project's conventions ride along in every request. Root-level
// only, with a 25KB cap per file; nested-file discovery is a later upgrade if needed.

import { readFileSync, existsSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const CAP = 25_000; // bytes — a fixed per-file load cap

/** Concatenated memory text for the system prompt, or "" if the project has none. */
export function loadMemory(root: string): string {
	const parts: string[] = [];
	for (const name of ["AGENTS.md", "MEMORY.md"]) {
		try {
			const content = readFileSync(join(root, name), "utf8").slice(0, CAP).trim();
			if (content) parts.push(`# ${name}\n${content}`);
		} catch {}
	}
	return parts.join("\n\n");
}

const TEMPLATE = `# Project memory

Instructions the assistant should follow every session. Keep it under ~200 lines;
move reference material into a skill file instead.

## Conventions
- (e.g. package manager, formatting, test command)

## Architecture
- (a sentence or two on how the code is laid out)

## Never do
- (hard rules)
`;

/** Create AGENTS.md if absent. Returns a status message for the chat. */
export function initAgentsMd(root: string): string {
	const path = join(root, "AGENTS.md");
	if (existsSync(path)) return "AGENTS.md already exists — edit it to tune project memory.";
	writeFileSync(path, TEMPLATE);
	return "Created AGENTS.md. Edit it to capture conventions the assistant should always follow.";
}
