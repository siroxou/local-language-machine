// Local Language Machine — UI-agnostic core: model lifecycle + the full agent loop.
//
// The composition root. It owns the model session, persistence, checkpoints,
// settings/permissions, skills, subagents, hooks, MCP, and the slash-command router,
// and wires them together around the shared runAgentLoop. It knows nothing about HTTP
// or webviews — callers pass event callbacks and get back the final text.
//
// Tool calling uses a JSON-in-text protocol (see loop.ts) rather than native function
// calling: local models are inconsistent about native call formats, and JSON-in-text
// works with any model that can emit JSON.

import { type ChatSession, type InferenceEngine, type ToolDef, type ChatHistoryItem, type SamplingOptions } from "../native/inference/engine.ts";
import {
	resolve as resolveModel,
	modelPathFor,
	listCached,
	ensureModel,
	allModels,
	addUserModel,
	entryFromHub,
	type EnsureProgress,
	type ModelEntry,
} from "../native/models/registry.ts";
import type { HubFile } from "../native/models/hub.ts";
import { getTuning, setTuning, type TuningConfig } from "../native/models/config.ts";
import { buildTools } from "../native/tools/tools.ts";
import { webTools } from "../native/tools/web.ts";
import { connectMcpServers } from "../native/mcp/client.ts";
import { runAgentLoop, type LoopEvents, type ToolGate } from "./loop.ts";
import { loadSettings, permissionDecision, describeAction, type Settings, type PermissionMode } from "./settings.ts";
import { Session } from "./session.ts";
import { Checkpoints } from "./checkpoints.ts";
import { tree, type TreeNode } from "./workspace.ts";
import { loadMemory, initAgentsMd } from "./memory.ts";
import { needsCompaction, summarizeHistory, spliceSummary, contextReport } from "./context.ts";
import { discoverSkills, type Skill } from "./skills.ts";
import { discoverAgents, spawnSubagent, type AgentDef } from "./subagent.ts";
import { runHooks } from "./hooks.ts";
import { isSlash, handleCommand, type CommandDeps } from "./commands.ts";
import { importFrom as importSkillsFrom, importFromGit as importSkillsFromGit, type ImportResult } from "./plugins.ts";

export interface ChatEvents extends LoopEvents {
	/** Gate for mutating tools — resolve true to proceed. */
	confirm(message: string): Promise<boolean>;
}

export interface ModelStatus {
	id: string;
	label: string;
	sizeGB: number;
	quant: string;
	cached: boolean;
	loaded: boolean;
}

// Preference order when auto-loading a cached model on startup.
const AUTOLOAD_ORDER = ["qwen2.5-coder:7b", "qwen2.5-coder:1.5b", "smollm2:135m"];

const SUBAGENT_TOOLS = ["read_file", "list_dir", "grep", "glob"]; // safe subset a delegated worker gets

export class Orchestrator {
	#chat?: ChatSession; // the model conversation
	#baseHistory: ChatHistoryItem[] = []; // history right after createSession (system prompt) — for /clear
	#loadedId?: string;
	#appliedSystemPrompt = ""; // the user system prompt baked into the current session — to detect changes worth a restart

	#settings: Settings;
	mode: PermissionMode;
	#store: Session;
	#checkpoints: Checkpoints;
	#skills: Skill[];
	#agents: AgentDef[];
	#mcpTools: Record<string, ToolDef> = {};
	#mcpReady = false;
	#sessionStarted = false;

	constructor(
		private readonly engine: InferenceEngine,
		private root: string, // mutable: Open Folder switches it via setRoot()
	) {
		this.#settings = loadSettings(root);
		this.mode = this.#settings.permissionMode;
		this.#skills = discoverSkills(root);
		this.#agents = discoverAgents(root);
		this.#store = Session.create(root);
		this.#checkpoints = new Checkpoints(this.#store.dir);
	}

	async status(): Promise<{
		models: ModelStatus[]; loaded?: string; gpu: string; contextWindow?: number; contextUsed?: number; root: string;
		mode: PermissionMode; online: boolean; sessionId: string; checkpoints: number;
		skills: Array<{ name: string; description: string }>;
	}> {
		const cached = new Set((await listCached()).map((c) => c.id));
		const models = allModels().map((m) => ({
			id: m.id, label: m.label, sizeGB: m.sizeGB, quant: m.quant, cached: cached.has(m.id), loaded: m.id === this.#loadedId,
		}));
		return {
			models, loaded: this.#loadedId, gpu: this.engine.gpu,
			contextWindow: this.engine.contextSize, contextUsed: this.#chat?.usedTokens?.(), root: this.root,
			mode: this.mode, online: this.#settings.online, sessionId: this.#store.id, checkpoints: this.#checkpoints.count,
			skills: this.#skills.map((s) => ({ name: s.name, description: s.description })),
		};
	}

	/** Download a model into the cache (no-op if already present). */
	async download(id: string, onProgress: (p: EnsureProgress) => void): Promise<void> {
		await ensureModel(id, { onProgress });
	}

	/** Register a Hugging Face repo + file as a user model (persisted). Returns the new entry. */
	addModel(repoId: string, file: HubFile): ModelEntry {
		const entry = entryFromHub(repoId, file);
		addUserModel(entry);
		return entry;
	}

	/** Load an already-cached model file, applying its saved tuning preset, then start a fresh chat session. */
	async load(id: string): Promise<void> {
		const entry = resolveModel(id);
		if (!entry) throw new Error(`Unknown model: ${id}`);
		const t = getTuning(id);
		await this.engine.load({ modelPath: modelPathFor(entry), ...(t.load ?? {}) });
		this.engine.setSampling(samplingOf(t.inference));
		this.#loadedId = id;
		await this.#startSession();
	}

	/** (Re)create the chat session for the current model + root. Cheap — does NOT reload the model file. */
	async #startSession(): Promise<void> {
		await this.#ensureMcp();
		const tools = this.#buildToolSet();
		const overview = await this.#workspaceOverview();
		// The model's preset system prompt is appended to the computed one — never replaces it (the tool instructions must survive).
		const userPrompt = this.#loadedId ? getTuning(this.#loadedId).inference?.systemPrompt : undefined;
		this.#appliedSystemPrompt = userPrompt ?? "";
		this.#chat = await this.engine.createSession({ systemPrompt: systemPrompt(tools, this.#skills, loadMemory(this.root), overview, userPrompt) });
		this.#baseHistory = this.#chat.getHistory();
		this.#sessionStarted = false;
	}

	/** The loaded model's saved preset (or {} when none / no model). */
	tuning(): TuningConfig {
		return this.#loadedId ? getTuning(this.#loadedId) : {};
	}

	get loadedModelId(): string | undefined {
		return this.#loadedId;
	}

	/**
	 * Apply an Inference-tab edit to the loaded model *live*: sampling lands on the next turn;
	 * a changed system prompt needs a cheap session restart. The caller persists the preset first
	 * (so #startSession reads the fresh value). Load-tab knobs are staged only — reloadModel() applies them.
	 */
	async applyLiveTuning(cfg: TuningConfig): Promise<void> {
		this.engine.setSampling(samplingOf(cfg.inference));
		const next = cfg.inference?.systemPrompt ?? "";
		if (this.#chat && next !== this.#appliedSystemPrompt) await this.#startSession();
	}

	/** Persist the loaded model's preset and apply the live (Inference) parts immediately. */
	async saveTuning(cfg: TuningConfig): Promise<void> {
		if (!this.#loadedId) throw new Error("Load a model first.");
		setTuning(this.#loadedId, cfg);
		await this.applyLiveTuning(cfg);
	}

	/** Reload the current model file, applying the staged Load-tab knobs from its saved preset. */
	async reloadModel(): Promise<void> {
		if (!this.#loadedId) throw new Error("No model loaded.");
		await this.load(this.#loadedId);
	}

	/** A capped file-tree snapshot embedded in the system prompt so the model always has ambient awareness of the codebase. */
	async #workspaceOverview(): Promise<string> {
		try {
			const lines = renderTreeLines(await tree(this.root), 150);
			return `Root: ${this.root}\n${lines.join("\n")}`;
		} catch {
			return `Root: ${this.root}`;
		}
	}

	/**
	 * Switch the open workspace folder (Open Folder). Keeps the loaded model in memory;
	 * rebuilds project state — settings, skills/agents, session, checkpoints — and the
	 * chat session (its system prompt embeds the new project's memory + skills).
	 * MCP stays connected from the first project's settings; per-folder MCP is an upgrade.
	 */
	async setRoot(newRoot: string): Promise<void> {
		this.root = newRoot;
		this.#settings = loadSettings(newRoot);
		this.mode = this.#settings.permissionMode;
		this.#skills = discoverSkills(newRoot);
		this.#agents = discoverAgents(newRoot);
		this.#store = Session.create(newRoot);
		this.#checkpoints = new Checkpoints(this.#store.dir);
		if (this.#chat) await this.#startSession();
	}

	/** Load the best cached model on startup, if any. Returns the id or undefined. */
	async autoload(): Promise<string | undefined> {
		const cached = await listCached();
		if (cached.length === 0) return undefined;
		const pick = AUTOLOAD_ORDER.find((id) => cached.some((c) => c.id === id)) ?? cached[0]!.id;
		await this.load(pick);
		return pick;
	}

	/** One chat request: slash-command handling, hooks, the tool loop, persistence, and compaction. */
	async chat(text: string, ev: ChatEvents): Promise<string> {
		const chat = this.#chat;
		if (!chat) throw new Error("No model loaded.");

		// Slash commands run before the model.
		if (isSlash(text)) {
			const r = await handleCommand(text, this.#commandDeps(ev));
			if (r.kind === "reply") { ev.onToken(r.text); return r.text; }
			if (r.kind === "skill") {
				if (r.skill.fork) {
					const out = await spawnSubagent(this.engine, this.root, `${r.skill.body}\n\nTask: ${r.arg}`, this.#subagentTools());
					ev.onToken(out);
					return out;
				}
				text = `${r.skill.body}\n\n${r.arg}`.trim(); // inject skill body, then fall through to the model
			} else if (r.kind === "prompt") {
				text = r.text;
			}
		}

		if (this.#hasHooks && !this.#sessionStarted) {
			this.#sessionStarted = true;
			await runHooks(this.#settings.hooks, "SessionStart", this.root);
		}
		if (this.#hasHooks) {
			const h = await runHooks(this.#settings.hooks, "UserPromptSubmit", this.root, { args: text });
			if (h.output) text += `\n\n[hook]: ${h.output}`;
		}

		this.#store.append({ type: "user", text });

		// Tee tool activity into the transcript while forwarding to the UI.
		const events: LoopEvents = {
			...ev,
			onToolCall: (n, a) => { this.#store.append({ type: "tool-call", name: n, data: a }); ev.onToolCall(n, a); },
			onToolResult: (n, r) => { this.#store.append({ type: "tool-result", name: n, data: r }); ev.onToolResult(n, r); },
		};

		const final = await runAgentLoop({
			session: chat,
			tools: this.#buildToolSet(),
			input: text,
			events,
			root: this.root,
			gate: this.#buildGate(ev),
		});

		this.#store.append({ type: "assistant", text: final });
		this.#store.saveHistory(chat.getHistory());
		if (this.#hasHooks) await runHooks(this.#settings.hooks, "Stop", this.root);
		if (needsCompaction(chat.getHistory())) { await this.#compact(); ev.onToken("\n🗜️ Compacted earlier context.\n"); }
		return final;
	}

	/** Undo the newest file checkpoint (the Esc-Esc rewind). */
	undo(): Promise<{ path: string; deleted: boolean } | null> {
		return this.#checkpoints.undo(this.root);
	}

	/** Snapshot a file before the server writes model-generated (streamed) code to it. */
	snapshotBeforeWrite(rel: string): Promise<void> {
		return this.#checkpoints.snapshot(this.root, rel);
	}

	setMode(mode: PermissionMode): void {
		this.mode = mode;
	}

	/** Import skills/agents from a local directory (offline). */
	importSkills(src: string): ImportResult {
		const r = importSkillsFrom(src);
		this.#refresh();
		return r;
	}

	/** Import from a git URL (online only). */
	importSkillsFromGit(url: string): ImportResult {
		if (!this.#settings.online) throw new Error("Enable online mode to import from a URL.");
		const r = importSkillsFromGit(url);
		this.#refresh();
		return r;
	}

	async resumeSession(id: string): Promise<void> {
		await this.resume(id);
	}

	// ————— internals —————

	get #hasHooks(): boolean {
		return Object.keys(this.#settings.hooks).length > 0;
	}

	async #ensureMcp(): Promise<void> {
		if (this.#mcpReady) return;
		this.#mcpReady = true;
		try {
			const { tools } = await connectMcpServers(this.#settings.mcpServers);
			this.#mcpTools = tools;
		} catch {
			// MCP is best-effort; a failed server never blocks the session.
		}
	}

	#refresh(): void {
		this.#settings = loadSettings(this.root);
		this.#skills = discoverSkills(this.root);
		this.#agents = discoverAgents(this.root);
	}

	#subagentTools(): Record<string, ToolDef> {
		const base = buildTools({ root: this.root });
		return Object.fromEntries(SUBAGENT_TOOLS.filter((k) => k in base).map((k) => [k, base[k]!]));
	}

	#buildToolSet(): Record<string, ToolDef> {
		const tools: Record<string, ToolDef> = { ...buildTools({ root: this.root }) };
		if (this.#settings.online) Object.assign(tools, webTools());
		Object.assign(tools, this.#mcpTools);

		tools.task = {
			description: "Delegate a focused, read-only research/analysis task to an isolated sub-agent. Returns its findings without cluttering this conversation.",
			params: {
				type: "object",
				properties: { task: { type: "string" }, agent: { type: "string" } },
				required: ["task"],
			},
			handler: async ({ task, agent }: { task: string; agent?: string }) => {
				const def = agent ? this.#agents.find((a) => a.name === agent) : undefined;
				const result = await spawnSubagent(this.engine, this.root, task, this.#subagentTools(), def);
				return { result };
			},
		};

		const invocable = this.#skills.filter((s) => !s.userInvocableOnly);
		if (invocable.length) {
			tools.use_skill = {
				description: `Load a skill's full instructions when its description fits the task. Skills: ${invocable.map((s) => s.name).join(", ")}.`,
				params: { type: "object", properties: { name: { type: "string" } }, required: ["name"] },
				handler: ({ name }: { name: string }) => {
					const s = this.#skills.find((x) => x.name === name);
					return s ? { instructions: s.body } : { error: `Unknown skill: ${name}` };
				},
			};
		}
		return tools;
	}

	#buildGate(ev: ChatEvents): ToolGate {
		return {
			before: async (tool, args) => {
				const decision = permissionDecision(this.mode, tool, args, this.#settings.allow);
				if (decision === "deny") return { block: true, result: { skipped: `Not permitted in ${this.mode} mode.` } };
				if (decision === "ask") {
					const ok = await ev.confirm(describeAction(tool, args));
					if (!ok) return { block: true, result: { cancelled: true } };
				}
				if (this.#hasHooks) {
					const h = await runHooks(this.#settings.hooks, "PreToolUse", this.root, { tool, args });
					if (h.block) return { block: true, result: { blockedByHook: h.output || "A PreToolUse hook blocked this action." } };
				}
				if (tool === "write_file" && typeof args.path === "string") {
					await this.#checkpoints.snapshot(this.root, args.path);
				}
			},
			after: async (tool, args) => {
				if (this.#hasHooks) {
					const h = await runHooks(this.#settings.hooks, "PostToolUse", this.root, { tool, args });
					if (h.output) ev.onToken(`\n${h.output}\n`);
				}
			},
		};
	}

	#commandDeps(ev: ChatEvents): CommandDeps {
		return {
			models: () => allModels().map((m) => ({ id: m.id, label: m.label, loaded: m.id === this.#loadedId })),
			loadModel: (id) => this.load(id),
			contextReport: () => (this.#chat ? contextReport(this.#chat.getHistory()) : "No model loaded."),
			compact: (focus) => this.#compact(focus),
			clearSession: () => this.#clear(),
			initMemory: () => initAgentsMd(this.root),
			sessions: () => Session.list(this.root).map((s) => ({ id: s.id, label: s.label })),
			resume: (id) => this.resume(id),
			fork: () => this.#fork(),
			skills: () => this.#skills,
		};
	}

	async #compact(focus?: string): Promise<string> {
		if (!this.#chat) return "No model loaded.";
		const hist = this.#chat.getHistory();
		const summary = await summarizeHistory(this.engine, hist);
		this.#chat.setHistory(spliceSummary(hist, focus ? `${summary}\n(kept focus: ${focus})` : summary));
		this.#store.saveHistory(this.#chat.getHistory());
		return "Compacted earlier context.";
	}

	#clear(): void {
		this.#store = Session.create(this.root);
		this.#checkpoints = new Checkpoints(this.#store.dir);
		this.#sessionStarted = false;
		this.#chat?.setHistory([...this.#baseHistory]); // reset to just the system prompt
	}

	#fork(): string {
		const forked = this.#store.fork();
		this.#store = forked;
		this.#checkpoints = new Checkpoints(forked.dir);
		return forked.id;
	}

	async resume(id: string): Promise<void> {
		this.#store = Session.open(this.root, id);
		this.#checkpoints = new Checkpoints(this.#store.dir);
		await this.#checkpoints.load();
		const hist = this.#store.loadHistory();
		if (hist && this.#chat) this.#chat.setHistory(hist);
	}
}

/** Flatten a file tree into indented lines, capped and depth-limited, for ambient context. */
function renderTreeLines(nodes: TreeNode[], cap: number): string[] {
	const out: string[] = [];
	const walk = (list: TreeNode[], depth: number): void => {
		for (const n of list) {
			if (out.length >= cap) return;
			out.push("  ".repeat(depth) + n.name + (n.dir ? "/" : ""));
			if (n.dir && n.children && depth < 3) walk(n.children, depth + 1);
		}
	};
	walk(nodes, 0);
	if (out.length >= cap) out.push("… (truncated)");
	return out;
}

/** Map a tuning preset's inference fields to the engine's sampling shape (drops systemPrompt). */
function samplingOf(inf?: TuningConfig["inference"]): SamplingOptions {
	return {
		temperature: inf?.temperature, topK: inf?.topK, topP: inf?.topP,
		minP: inf?.minP, repeatPenalty: inf?.repeatPenalty, maxTokens: inf?.maxTokens,
	};
}

function systemPrompt(tools: Record<string, ToolDef>, skills: Skill[], memory: string, overview: string, userPrompt?: string): string {
	const list = Object.entries(tools)
		.map(([name, t]) => `- ${name}: ${t.description ?? ""} params=${JSON.stringify(t.params ?? {})}`)
		.join("\n");
	const invocable = skills.filter((s) => !s.userInvocableOnly);
	const skillList = invocable.length
		? "\nYou also have SKILLS — call use_skill with a name to load its instructions:\n" +
			invocable.map((s) => `- ${s.name}: ${s.description}`).join("\n")
		: "";
	const mem = memory ? `\n\n————— Project memory —————\n${memory}` : "";
	const ws = overview ? `\n\n————— Workspace (partial file tree) —————\n${overview}` : "";
	const usr = userPrompt?.trim() ? `\n\n————— User instructions —————\n${userPrompt.trim()}` : "";
	return [
		"You are Local Language Machine, a local AI coding assistant running fully offline on the user's own machine.",
		"",
		"You have REAL, DIRECT access to this workspace through the tools below — they actually run on the user's filesystem, reading and editing their files. This is NOT hypothetical.",
		"NEVER say you cannot read or access the codebase or files, and never ask the user to paste code. If you need to see something, CALL A TOOL to look at it.",
		"",
		"Tools:",
		list,
		skillList,
		"",
		'To call a tool, reply with ONLY a JSON object and nothing else, for example:',
		'{"tool": "list_dir", "arguments": {"path": "."}}',
		"After you get the result, either call another tool the same way, or give your final answer in plain prose.",
		"",
		'When the user asks ANYTHING about their code or files (e.g. "read my codebase", "what does this do", "find X"), you MUST start by calling a tool — list_dir on "." for the layout, then read_file or grep the relevant files — and answer only from what you actually read. Never answer such questions from imagination, and never refuse.',
		"For greetings or general questions that don't touch the workspace, just reply in prose with no tool call.",
		"",
		"To write or create code, output it in ONE fenced code block (```language). Fenced code is saved to the editor",
		"automatically — do not paste code into the chat. In the chat, briefly explain what you built in a sentence or two.",
		ws,
		mem,
		usr,
	].join("\n");
}
