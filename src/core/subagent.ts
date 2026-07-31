// Local Language Machine — subagents (isolated-context workers).
//
// A subagent runs the same agent loop in a fresh ChatSession with its own system
// prompt, so its work never touches the main conversation's context — only the final
// text returns. Custom agents come from ./.claude/agents/*.md. The model delegates
// via the `task` tool.
//
// Subagents get a READ-ONLY toolset (read/list/grep/glob) by default, so a delegated
// worker can research safely without wiring the permission gate through a second
// context. Mutating subagents + a per-agent model override are the upgrade.

import { readdirSync, readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { appHome } from "../native/paths.ts";
import type { InferenceEngine, ToolDef } from "../native/inference/engine.ts";
import { runAgentLoop } from "./loop.ts";
import { parseFrontmatter } from "./skills.ts";

export interface AgentDef {
	name: string;
	description: string;
	systemPrompt: string;
}

/** Custom agents from ./.claude/agents/*.md (project) and ~/.local-language-machine/agents (imported). */
export function discoverAgents(root: string): AgentDef[] {
	const seen = new Set<string>();
	const out: AgentDef[] = [];
	for (const dir of [join(root, ".claude", "agents"), join(appHome(), "agents")]) {
		if (!existsSync(dir)) continue;
		for (const entry of readdirSync(dir)) {
			if (!entry.endsWith(".md")) continue;
			try {
				const { data, body } = parseFrontmatter(readFileSync(join(dir, entry), "utf8"));
				const name = String(data.name ?? entry.replace(/\.md$/, ""));
				if (seen.has(name)) continue;
				seen.add(name);
				out.push({ name, description: String(data.description ?? ""), systemPrompt: body.trim() || String(data.description ?? "") });
			} catch {}
		}
	}
	return out;
}

const noop = () => {};
const SUBAGENT_SYSTEM = "You are a focused sub-agent. Complete the delegated task using your read-only tools, then reply with a concise result. Do not ask questions.";
/**
 * Bounded window for the delegated context. Inheriting the model's own would allocate a
 * second full KV cache alongside the main session — ~7GB on a 7B at n_ctx 131072 — and the
 * model can reach for `task` at any point in a turn. 32k still fits several capped file
 * reads across the loop's six tool steps.
 */
const SUBAGENT_CONTEXT_TOKENS = 32_768;

/**
 * Run a delegated task in an isolated context and return its final text.
 * `readonlyTools` is the safe subset the caller hands in (built once by the orchestrator).
 */
export async function spawnSubagent(
	engine: InferenceEngine,
	root: string,
	task: string,
	readonlyTools: Record<string, ToolDef>,
	agent?: AgentDef,
): Promise<string> {
	const session = await engine.createSession({ systemPrompt: agent?.systemPrompt || SUBAGENT_SYSTEM, contextSize: SUBAGENT_CONTEXT_TOKENS });
	try {
		return await runAgentLoop({
			session,
			tools: readonlyTools,
			input: task,
			root,
			// Isolated: no code streams to the editor, no tool cards in the main chat.
			events: { onToken: noop, onToolCall: noop, onToolResult: noop, onCodeOpen: noop, onCodeToken: noop, onCodeClose: noop },
		});
	} finally {
		session.dispose();
	}
}
