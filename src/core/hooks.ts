// Local Language Machine — lifecycle hooks.
//
// Shell commands fired on events (SessionStart, UserPromptSubmit, PreToolUse,
// PostToolUse, Stop). A nonzero-exit PreToolUse hook BLOCKS the tool — this is the
// enforcement mechanism a prompt instruction can't guarantee. Hook stdout is returned
// so the caller can feed it back to the model. Tool name/args are exposed as env vars.
//
// shell hooks only, with context via env. HTTP / prompt / subagent hook
// types (and JSON-on-stdin payloads) are the upgrade if a use case needs them.

import { exec } from "node:child_process";
import { promisify } from "node:util";
import type { HookDef } from "./settings.ts";

const execAsync = promisify(exec);

export type HookEvent = "SessionStart" | "UserPromptSubmit" | "PreToolUse" | "PostToolUse" | "Stop";

export interface HookResult {
	/** PreToolUse only: a hook failed, so the tool must not run. */
	block: boolean;
	/** Combined stdout of the hooks that ran (fed back to the model when non-empty). */
	output: string;
}

export async function runHooks(
	hooks: Record<string, HookDef[]>,
	event: HookEvent,
	cwd: string,
	ctx: { tool?: string; args?: unknown } = {},
): Promise<HookResult> {
	const list = hooks[event] ?? [];
	let output = "";
	for (const h of list) {
		if (h.matcher && ctx.tool && !ctx.tool.includes(h.matcher)) continue;
		const env = {
			...process.env,
			LLM_HOOK_EVENT: event,
			LLM_TOOL_NAME: ctx.tool ?? "",
			LLM_TOOL_ARGS: ctx.args ? JSON.stringify(ctx.args) : "",
		};
		try {
			const { stdout } = await execAsync(h.command, { cwd, env, maxBuffer: 4 * 1024 * 1024 });
			if (stdout.trim()) output += stdout.trim() + "\n";
		} catch (e: any) {
			// nonzero exit
			if (event === "PreToolUse") return { block: true, output: (output + (e.stdout ?? "") + (e.stderr ?? "")).trim() };
			if (e.stdout?.trim()) output += e.stdout.trim() + "\n";
		}
	}
	return { block: false, output: output.trim() };
}
