// Local Language Machine — app + project settings.
//
// Read from two places and merged (project wins): the user file
// ~/.local-language-machine/settings.json and the project's ./.claude/settings.json.
// Reading the project's ./.claude/ directory means an existing config Just Works — no
// translation. Holds the permission mode, command allow-list, hooks, MCP servers,
// and the single `online` switch that gates every network feature.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { appHome } from "../native/paths.ts";

export type PermissionMode = "manual" | "accept-edits" | "plan" | "auto";

/** One hook: a shell command fired on a lifecycle event. HTTP / prompt / subagent hooks are a later upgrade. */
export interface HookDef {
	command: string;
	/** Optional tool-name match for PreToolUse/PostToolUse (substring). Empty = all tools. */
	matcher?: string;
}

export interface McpServerConfig {
	command?: string; // stdio server executable
	args?: string[];
	env?: Record<string, string>;
	url?: string; // remote (SSE/HTTP) — requires `online`
}

export interface Settings {
	permissionMode: PermissionMode;
	/** Exact commands or tool names that skip the confirm prompt. */
	allow: string[];
	/** event name → hooks (SessionStart, UserPromptSubmit, PreToolUse, PostToolUse, Stop). */
	hooks: Record<string, HookDef[]>;
	mcpServers: Record<string, McpServerConfig>;
	/** Master switch for anything that touches the network. Default false. */
	online: boolean;
}

const DEFAULTS: Settings = { permissionMode: "manual", allow: [], hooks: {}, mcpServers: {}, online: false };

const userSettingsPath = () => join(appHome(), "settings.json");
const projectSettingsPath = (root: string) => join(root, ".claude", "settings.json");

function readJson(path: string): Partial<Settings> {
	try {
		const data = JSON.parse(readFileSync(path, "utf8"));
		return data && typeof data === "object" ? data : {};
	} catch {
		return {};
	}
}

/** Merge defaults ← user ← project. Arrays concat; object maps shallow-merge. */
export function loadSettings(root: string): Settings {
	const layers = [DEFAULTS, readJson(userSettingsPath()), readJson(projectSettingsPath(root))];
	const out: Settings = { ...DEFAULTS };
	for (const l of layers) {
		if (l.permissionMode) out.permissionMode = l.permissionMode;
		if (typeof l.online === "boolean") out.online = l.online;
		if (l.allow) out.allow = [...out.allow, ...l.allow];
		if (l.hooks) out.hooks = { ...out.hooks, ...l.hooks };
		if (l.mcpServers) out.mcpServers = { ...out.mcpServers, ...l.mcpServers };
	}
	return out;
}

// ————— permission policy —————

// Tools that only read. They never prompt, in any mode.
const READONLY_TOOLS = new Set(["read_file", "list_dir", "grep", "glob", "web_search", "web_fetch"]);
// git subcommands that don't mutate — allowed like a read.
const GIT_READONLY = new Set(["status", "diff", "log", "show", "branch", "remote", "rev-parse", "ls-files"]);
// commands we always double-check even in auto mode (best-effort; not a security boundary).
// substring denylist, not a sandbox — the real boundary is resolveInRoot + the confirm gate.
const DANGER = /\brm\s+-rf?\b|\bmkfs\b|\bdd\b|:\(\)\s*\{|\bshutdown\b|\breboot\b|\bgit\s+push\b|--force\b|\bsudo\b/;

export type Decision = "allow" | "ask" | "deny";

/** Decide how a tool call is handled under the current mode. Enforced in the loop's gate. */
export function permissionDecision(
	mode: PermissionMode,
	tool: string,
	args: Record<string, unknown>,
	allow: string[],
): Decision {
	if (READONLY_TOOLS.has(tool)) return "allow";
	if (tool === "git" && GIT_READONLY.has(String((args.args as string[])?.[0] ?? ""))) return "allow";

	// Explicit allow-list: exact tool name, or exact command for run_terminal/git.
	const cmd = commandOf(tool, args);
	if (allow.includes(tool) || (cmd && allow.includes(cmd))) return "allow";

	if (mode === "plan") return "deny"; // never mutate in plan mode
	if (mode === "manual") return "ask";

	const dangerous = cmd ? DANGER.test(cmd) : false;
	if (mode === "accept-edits") return tool === "write_file" ? "allow" : "ask";
	if (mode === "auto") return dangerous ? "ask" : "allow";
	return "ask";
}

function commandOf(tool: string, args: Record<string, unknown>): string | undefined {
	if (tool === "run_terminal") return typeof args.command === "string" ? args.command : undefined;
	if (tool === "git" && Array.isArray(args.args)) return `git ${(args.args as string[]).join(" ")}`;
	return undefined;
}

/** Human-readable confirm message for a gated action. */
export function describeAction(tool: string, args: Record<string, unknown>): string {
	if (tool === "write_file") return `Write ${String(args.content ?? "").length} chars to ${args.path}`;
	if (tool === "run_terminal") return `Run: ${args.command}`;
	if (tool === "git") return `Run: git ${(args.args as string[])?.join(" ") ?? ""}`;
	return `Run ${tool} ${JSON.stringify(args)}`;
}
