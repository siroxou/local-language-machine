// Local Language Machine — slash-command router.
//
// Runs before the model on any input starting with "/". Built-ins cover the model,
// context, sessions, and memory; every discovered skill is also invocable as /<name>.
// The orchestrator implements `CommandDeps` and acts on the result.

import type { Skill } from "./skills.ts";

export interface CommandDeps {
	models(): Array<{ id: string; label: string; loaded: boolean }>;
	loadModel(id: string): Promise<void>;
	contextReport(): string;
	compact(focus?: string): Promise<string>;
	clearSession(): void;
	initMemory(): string;
	sessions(): Array<{ id: string; label: string }>;
	resume(id: string): Promise<void>;
	fork(): Promise<string>;
	skills(): Skill[];
}

export type CommandResult =
	| { kind: "reply"; text: string } // show text, no model call
	| { kind: "prompt"; text: string } // run the model with this prompt
	| { kind: "skill"; skill: Skill; arg: string } // orchestrator loads body / forks
	| { kind: "none" }; // not a command — treat as normal input

export function isSlash(text: string): boolean {
	return /^\/[a-z]/i.test(text.trim());
}

export async function handleCommand(text: string, deps: CommandDeps): Promise<CommandResult> {
	const [, name = "", arg = ""] = text.trim().match(/^\/(\S+)\s*([\s\S]*)$/) ?? [];
	const lc = name.toLowerCase();

	switch (lc) {
		case "help":
			return { kind: "reply", text: helpText(deps.skills()) };

		case "model": {
			if (!arg.trim()) {
				const list = deps.models().map((m) => `${m.loaded ? "● " : "  "}${m.id} — ${m.label}`).join("\n");
				return { kind: "reply", text: `Models:\n${list}\n\nUse /model <id> to switch.` };
			}
			await deps.loadModel(arg.trim());
			return { kind: "reply", text: `Loaded ${arg.trim()}.` };
		}

		case "context":
			return { kind: "reply", text: deps.contextReport() };

		case "compact":
			return { kind: "reply", text: await deps.compact(arg.trim() || undefined) };

		case "clear":
			deps.clearSession();
			return { kind: "reply", text: "Started a fresh session." };

		case "init":
			return { kind: "reply", text: deps.initMemory() };

		case "resume": {
			if (!arg.trim()) {
				const list = deps.sessions().map((s) => `${s.id} — ${s.label}`).join("\n") || "(no saved sessions)";
				return { kind: "reply", text: `Sessions:\n${list}\n\nUse /resume <id>.` };
			}
			await deps.resume(arg.trim());
			return { kind: "reply", text: `Resumed session ${arg.trim()}.` };
		}

		case "fork":
			return { kind: "reply", text: `Forked into new session ${await deps.fork()}.` };

		default: {
			const skill = deps.skills().find((s) => s.name.toLowerCase() === lc);
			if (skill) return { kind: "skill", skill, arg };
			return { kind: "reply", text: `Unknown command /${name}. Try /help.` };
		}
	}
}

function helpText(skills: Skill[]): string {
	const builtins = [
		"/help — this list",
		"/model [id] — show or switch models",
		"/context — token/context usage",
		"/compact [focus] — summarize older turns",
		"/clear — start a fresh session",
		"/init — scaffold AGENTS.md",
		"/resume [id] — list or resume sessions",
		"/fork — branch the current session",
	];
	const skillLines = skills.map((s) => `/${s.name} — ${s.description || "(skill)"}`);
	return ["Commands:", ...builtins, ...(skillLines.length ? ["", "Skills:", ...skillLines] : [])].join("\n");
}
