// Local Language Machine — the reusable agentic loop.
//
// Extracted from Orchestrator.chat() so the main chat, subagents, and skills-in-
// fork all share one implementation. The loop is: generate → if the model emitted
// a tool call, run it (through the gate) and feed the result back → repeat → return
// the final text. It knows nothing about HTTP, sessions, or the model lifecycle.
//
// The tool `gate` is the single interception seam. Checkpoints, permission modes,
// and hooks all hang off it instead of scattering logic through the loop.

import { existsSync } from "node:fs";
import { join } from "node:path";
import type { ChatSession, ToolDef } from "../native/inference/engine.ts";

export const MAX_TOOL_STEPS = 6; // cap the agent loop so a confused model can't spin forever
export const errMsg = (e: unknown) => (e instanceof Error ? e.message : String(e));

// Prose → chat; fenced code → editor. Subagents pass no-op code sinks.
export interface CodeSink {
	onToken(text: string): void;
	onCodeOpen(path: string, lang: string): void;
	onCodeToken(text: string): void;
	onCodeClose(path: string): void;
}

export interface LoopEvents extends CodeSink {
	onToolCall(name: string, args: unknown): void;
	onToolResult(name: string, result: unknown): void;
}

/** Result of a gate `before` check: block the tool (substitute a result) or proceed (void). */
export type ToolDecision = { block: true; result: unknown } | void;

/**
 * Single interception seam around tool execution. `before` runs first (may block,
 * e.g. a PreToolUse hook or plan-mode deny, or snapshot for a checkpoint); `after`
 * observes the result (e.g. a PostToolUse hook). Compose multiple concerns into one
 * gate at the call site — the loop only knows this shape.
 */
export interface ToolGate {
	before?(tool: string, args: Record<string, unknown>): Promise<ToolDecision> | ToolDecision;
	/** Returned text is appended to the next model input — a PostToolUse hook's stdout is meant to
	 *  reach the model, not just the transcript (a formatter reporting what it changed, say). */
	after?(tool: string, args: Record<string, unknown>, result: unknown): Promise<string | void> | string | void;
}

export interface LoopOptions {
	session: ChatSession;
	tools: Record<string, ToolDef>;
	input: string;
	events: LoopEvents;
	root: string;
	gate?: ToolGate;
	/**
	 * Per-step generation cap. Left unset the model may generate until the context window
	 * fills, on every one of the six steps — and a gate cannot interrupt that, because a gate
	 * only runs between tool calls. Set it wherever a runaway turn must stay bounded.
	 */
	maxTokens?: number;
	/** Overrides MAX_TOOL_STEPS. Callers with their own budget need to set the real ceiling. */
	maxSteps?: number;
}

/**
 * One request. Runs the tool loop and resolves to the final assistant text.
 * `usedPaths` keeps generated code filenames collision-free across the whole request.
 */
export async function runAgentLoop(opts: LoopOptions): Promise<string> {
	const { session, tools, events, root, gate } = opts;
	const valid = new Set(Object.keys(tools));
	const usedPaths = new Set<string>();
	const maxSteps = opts.maxSteps ?? MAX_TOOL_STEPS;
	let input = opts.input;
	let final = "";

	for (let step = 0; step < maxSteps; step++) {
		const router = new CodeRouter(events, usedPaths, root);
		const output = await session.prompt(input, {
			onText: (c) => router.push(c),
			...(opts.maxTokens ? { maxTokens: opts.maxTokens } : {}),
		});
		router.flush();
		const call = parseToolCall(output, valid);
		if (!call) {
			final = output;
			break;
		}

		events.onToolCall(call.tool, call.arguments);
		let result: unknown;
		const decision = gate?.before ? await gate.before(call.tool, call.arguments) : undefined;
		if (decision && decision.block) {
			result = decision.result;
		} else {
			try {
				result = await tools[call.tool]!.handler(call.arguments);
			} catch (e) {
				result = { error: errMsg(e) };
			}
		}
		const note = await gate?.after?.(call.tool, call.arguments, result);
		events.onToolResult(call.tool, result);

		// This input is consumed by iteration `step + 1`. If that is the last one, there is no
		// step left for another tool call, so ask for prose rather than inviting one — cheaper
		// than reporting an exhausted budget, and it usually turns a dead end into an answer.
		const lastStep = step + 1 >= maxSteps - 1;
		input =
			`Result of ${call.tool}: ${JSON.stringify(result)}\n\n` +
			(note ? `[hook]: ${note}\n\n` : "") +
			(lastStep
				? `You have no more tool calls available. Answer the user's request now, in plain text, without calling another tool.`
				: `Use this to answer the user's request. Call another tool if needed, otherwise reply in plain text.`);
	}

	// Falling out of the loop means every step was spent on a tool call and the model never
	// produced prose. Returning "" here made that indistinguishable from a real empty reply:
	// it was persisted as an empty assistant turn and streamed to the UI as a finished
	// answer, so the turn simply appeared to do nothing. A limit must say that it was hit.
	if (!final.trim()) {
		return `[Stopped after ${maxSteps} tool steps without reaching an answer. Try narrowing the request, or ask again to continue.]`;
	}
	return final;
}

export interface ToolCall {
	tool: string;
	arguments: Record<string, unknown>;
}

/** Pull a `{tool|name, arguments|args}` JSON object out of the model's text. Tolerates ```json fences and surrounding prose. */
export function parseToolCall(text: string, valid: Set<string>): ToolCall | null {
	const fence = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
	const raw = fence ? fence[1]! : text;
	const start = raw.indexOf("{");
	const end = raw.lastIndexOf("}");
	if (start < 0 || end <= start) return null;
	let obj: any;
	try {
		obj = JSON.parse(raw.slice(start, end + 1));
	} catch {
		return null;
	}
	const tool = obj?.tool ?? obj?.name;
	const args = obj?.arguments ?? obj?.args ?? {};
	if (typeof tool === "string" && valid.has(tool) && args && typeof args === "object") {
		return { tool, arguments: args };
	}
	return null;
}

// ————— streaming code router: prose → chat, fenced code → editor —————

const LANG_FILE: Record<string, string> = {
	html: "game.html", htm: "game.html", javascript: "script.js", js: "script.js", jsx: "component.jsx",
	typescript: "script.ts", ts: "script.ts", tsx: "component.tsx", python: "main.py", py: "main.py",
	css: "styles.css", json: "data.json", bash: "script.sh", sh: "script.sh", c: "main.c",
	cpp: "main.cpp", "c++": "main.cpp", go: "main.go", rust: "main.rs", rs: "main.rs",
	java: "Main.java", ruby: "main.rb", rb: "main.rb",
};
const defaultName = (lang: string) => LANG_FILE[lang.toLowerCase()] ?? (lang ? `snippet.${lang.toLowerCase()}` : "snippet.txt");
const fileHint = (text: string) => text.match(/[\w-]+\.(?:html?|jsx?|tsx?|css|py|json|md|sh|c|cpp|go|rs|java|rb)\b/i)?.[0] ?? "";

const SNIFF_LEN = 120; // buffer this much code before deciding the filename from content

// Detect language from code *content* — overrides a mislabeled fence (small models
// often tag HTML as ```json, etc.). Returns "" when unsure (the fence label wins).
export function sniffLang(code: string): string {
	const s = code.replace(/^\s+/, "");
	if (/^<!doctype/i.test(s) || /^<html[\s>]/i.test(s) || /<(body|div|button|span|canvas|script|style|h[1-6]|!--)[\s/>]/i.test(s)) return "html";
	if (/^#!.*\b(bash|sh|zsh)\b/.test(s)) return "sh";
	if (/^\s*(def |class \w|import \w|from \w+ import|print\()/m.test(s)) return "python";
	if (/^\s*package \w|^\s*func \w/m.test(s)) return "go";
	if (/\b(function\b|=>|const |let |var |console\.|document\.|window\.)/.test(s)) return "javascript";
	if (/[.#]?[\w-]+\s*\{[^}]*:[^};]*;/.test(s)) return "css";
	if (/^\s*[[{][\s\S]*["\d[{]/.test(s)) return "json";
	return "";
}

/**
 * Splits a token stream into prose and fenced code blocks as it arrives.
 * Prose → `onToken` (chat); code between ```fences``` → the editor. The filename
 * comes from the code *content* (fence tags are unreliable on small models), so
 * the open is deferred until ~120 chars of code are buffered. Holds back a 2-char
 * tail so a fence split across chunks reassembles. triple-backtick
 * only; a fence left unclosed at stream end is flushed and closed.
 */
export class CodeRouter {
	#pending = "";
	#state: "prose" | "pending" | "code" | "codechat" = "prose";
	#path = "";
	#recent = ""; // recent prose, scanned for a filename hint
	#buf = ""; // code buffered before the filename is decided
	#lang = "";
	#hint = "";

	constructor(
		private readonly sink: CodeSink,
		private readonly used: Set<string>,
		private readonly root: string,
	) {}

	push(chunk: string): void {
		this.#pending += chunk;
		this.#drain(false);
	}

	flush(): void {
		this.#drain(true);
		if (this.#state === "pending") this.#open(this.#buf);
		if (this.#state === "code") this.sink.onCodeClose(this.#path);
		this.#state = "prose";
		this.#pending = "";
		this.#buf = "";
	}

	#open(initialCode: string): void {
		if (/^\s*\{\s*"(tool|name)"\s*:/.test(initialCode)) {
			// a tool call wrapped in a fence — route to chat, NOT the editor.
			// parseToolCall() on the full output still executes it; the chat bubble is
			// then replaced by the tool card (or shown if the call was malformed).
			this.#prose(initialCode);
			this.#buf = "";
			this.#state = "codechat";
			return;
		}
		const lang = sniffLang(initialCode) || this.#lang;
		this.#path = this.#pick(lang, this.#hint);
		this.sink.onCodeOpen(this.#path, lang);
		this.sink.onToken(`\n📄 \`${this.#path}\` → editor\n`);
		if (initialCode) this.sink.onCodeToken(initialCode);
		this.#buf = "";
		this.#state = "code";
	}

	#pick(lang: string, hint: string): string {
		const name = hint || defaultName(lang);
		let p = name;
		let n = 1;
		while (this.used.has(p) || existsSync(join(this.root, p))) {
			const d = name.lastIndexOf(".");
			p = d > 0 ? `${name.slice(0, d)}-${++n}${name.slice(d)}` : `${name}-${++n}`;
		}
		this.used.add(p);
		return p;
	}

	#prose(t: string): void {
		if (!t) return;
		this.sink.onToken(t);
		this.#recent = (this.#recent + t).slice(-200);
	}

	#drain(final: boolean): void {
		for (;;) {
			const tail = final ? 0 : 2; // hold back a partial ``` across chunks
			const i = this.#pending.indexOf("```");
			if (this.#state === "prose") {
				if (i === -1) {
					const cut = Math.max(0, this.#pending.length - tail);
					this.#prose(this.#pending.slice(0, cut));
					this.#pending = this.#pending.slice(cut);
					return;
				}
				this.#prose(this.#pending.slice(0, i));
				const rest = this.#pending.slice(i + 3);
				const nl = rest.indexOf("\n");
				if (nl === -1) {
					if (final) { this.#prose("```" + rest); this.#pending = ""; return; }
					this.#pending = "```" + rest;
					return;
				}
				const info = rest.slice(0, nl).trim();
				this.#pending = rest.slice(nl + 1);
				const first = info.split(/[\s:]/)[0] ?? "";
				this.#lang = /\.\w+$/.test(first) ? "" : first;
				this.#hint = info.match(/[\w./-]+\.\w+/)?.[0] || fileHint(this.#recent);
				this.#buf = "";
				this.#state = "pending";
			} else if (this.#state === "pending") {
				if (i === -1) {
					const cut = Math.max(0, this.#pending.length - tail);
					this.#buf += this.#pending.slice(0, cut);
					this.#pending = this.#pending.slice(cut);
					if (this.#buf.length >= SNIFF_LEN) { const b = this.#buf; this.#buf = ""; this.#open(b); }
					return;
				}
				// block closed before we buffered enough — open with what we have, then close
				this.#buf += this.#pending.slice(0, i);
				this.#pending = this.#pending.slice(i + 3);
				const b = this.#buf;
				this.#buf = "";
				this.#open(b);
				this.sink.onCodeClose(this.#path);
				this.#state = "prose";
			} else {
				const toEditor = this.#state === "code"; // vs "codechat" (a tool call → chat)
				if (i === -1) {
					const cut = Math.max(0, this.#pending.length - tail);
					const chunk = this.#pending.slice(0, cut);
					if (chunk) toEditor ? this.sink.onCodeToken(chunk) : this.#prose(chunk);
					this.#pending = this.#pending.slice(cut);
					return;
				}
				const chunk = this.#pending.slice(0, i);
				if (chunk) toEditor ? this.sink.onCodeToken(chunk) : this.#prose(chunk);
				if (toEditor) this.sink.onCodeClose(this.#path);
				this.#state = "prose";
				this.#pending = this.#pending.slice(i + 3);
			}
		}
	}
}
