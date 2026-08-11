// Local Language Machine — app + project settings.
//
// Read from two places and merged: the user file ~/.local-language-machine/settings.json and the
// project's ./.claude/settings.json. Reading the project's ./.claude/ directory means an existing
// config mostly carries over. The project layer contributes hooks and MCP servers only — it cannot
// set permissionMode, online or allow, because a repo ships that file to everyone who clones it.
// Holds the permission mode, command allow-list, hooks, MCP servers, and the `online` switch that
// gates the agent's network reach (web tools, git skill import) — not the model downloader.

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
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
	// No `url` field: remote (SSE/HTTP) MCP transport is not implemented. It used to be declared
	// here and documented as "gated by online", but nothing ever read it — a server configured
	// with only a url connected to nothing, silently. Declare it again when the transport exists.
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

/**
 * Flip the network switch in the user's own settings file, so the choice survives a restart.
 * Written to the user layer deliberately: the project layer is not allowed to set `online`.
 */
export function setUserOnline(online: boolean): void {
	const path = userSettingsPath();
	const current = readJson(path);
	mkdirSync(dirname(path), { recursive: true });
	writeFileSync(path, JSON.stringify({ ...current, online }, null, 2));
}
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
	const project = readJson(projectSettingsPath(root));
	// A project's ./.claude/settings.json ships inside the repo, so any clone can carry one — it is
	// data from whoever wrote that repo, not a statement of this user's intent. It may contribute
	// hooks and MCP servers, but nothing that widens what runs without asking: `allow` skips the
	// confirm prompt, and permissionMode/online are the user's own safety defaults.
	delete project.permissionMode;
	delete project.online;
	delete project.allow;
	const layers = [DEFAULTS, readJson(userSettingsPath()), project];
	const out: Settings = { ...DEFAULTS };
	for (const l of layers) {
		if (l.permissionMode) out.permissionMode = l.permissionMode;
		if (typeof l.online === "boolean") out.online = l.online;
		if (l.allow) out.allow = [...out.allow, ...l.allow];
		// Concat per event, matching `allow` above: a shallow merge silently deleted the user's
		// global PreToolUse guard the moment a project defined any hook for the same event.
		if (l.hooks) {
			const merged: Record<string, HookDef[]> = { ...out.hooks };
			for (const [event, defs] of Object.entries(l.hooks)) merged[event] = [...(merged[event] ?? []), ...defs];
			out.hooks = merged;
		}
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
// `rm -rf` is only one spelling: -fr, -Rf and a bare -f are the same command with the flags
// reordered, and the guard is worthless if it only catches the tidy one.
const DANGER = /\brm\s+-[a-z]*[rf]|\bmkfs\b|\bdd\b|:\(\)\s*\{|\bshutdown\b|\breboot\b|\bgit\s+push\b|--force\b|\bsudo\b/i;

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

	const cmd = commandOf(tool, args);

	if (mode === "plan") return "deny"; // never mutate in plan mode — outranks the allow-list

	// Explicit allow-list: exact tool name, or exact command for run_terminal/git.
	if (allow.includes(tool) || (cmd && allow.includes(cmd))) return "allow";
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
