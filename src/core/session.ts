// Local Language Machine — session persistence (resume / fork).
//
// Each session is a directory under ~/.local-language-machine/sessions/<projectHash>/<id>/
// holding: events.jsonl (transcript for display), history.json (model chat history
// for resume), and checkpoints.json (see checkpoints.ts). Resume replays history via
// ChatSession.setHistory; fork copies the whole directory to a new id.

import { createHash, randomUUID } from "node:crypto";
import { appendFileSync, mkdirSync, readFileSync, writeFileSync, existsSync, readdirSync, statSync, cpSync } from "node:fs";
import { join } from "node:path";
import { appHome } from "../native/paths.ts";
import type { ChatHistoryItem } from "../native/inference/engine.ts";

export interface SessionEvent {
	t: number;
	type: "user" | "assistant" | "tool-call" | "tool-result" | "system";
	text?: string;
	name?: string;
	data?: unknown;
}

function readTotals(dir: string): number {
	try {
		const n = JSON.parse(readFileSync(join(dir, "totals.json"), "utf8")).tokens;
		return Number.isFinite(n) ? n : 0;
	} catch {
		return 0;
	}
}

const sessionsRoot = () => join(appHome(), "sessions");
const projectHash = (root: string) => createHash("sha1").update(root).digest("hex").slice(0, 12);
const projectDir = (root: string) => join(sessionsRoot(), projectHash(root));

export class Session {
	readonly dir: string;
	constructor(
		private readonly projectRoot: string,
		readonly id: string,
	) {
		this.dir = join(projectDir(projectRoot), id);
		// Deliberately NOT created here. Session.create() runs on every app launch, every Open
		// Folder and every /clear, so an eager mkdir left a directory behind for conversations
		// that never happened — 40 of 51 dirs on the author's machine were empty stubs, and since
		// list() sorts by mtime they were exactly the rows at the top. The dir appears on first write.
	}

	#ensureDir(): void {
		mkdirSync(this.dir, { recursive: true });
	}

	/** A fresh session with a random id. */
	static create(projectRoot: string): Session {
		return new Session(projectRoot, randomUUID().slice(0, 8));
	}

	/** Reopen an existing session by id (creates the dir if missing). */
	static open(projectRoot: string, id: string): Session {
		return new Session(projectRoot, id);
	}

	/**
	 * All sessions for this project that actually contain a conversation, newest first.
	 * Directories with no events.jsonl are skipped — that both hides the stubs already on disk
	 * from before the lazy-mkdir fix and covers a session created but never written to.
	 */
	static list(projectRoot: string): Array<{ id: string; label: string; mtime: number; messages: number; tokens: number }> {
		const dir = projectDir(projectRoot);
		if (!existsSync(dir)) return [];
		const out: Array<{ id: string; label: string; mtime: number; messages: number; tokens: number }> = [];
		for (const id of readdirSync(dir)) {
			let raw: string;
			try {
				raw = readFileSync(join(dir, id, "events.jsonl"), "utf8");
			} catch {
				continue; // no transcript — not a conversation
			}
			const lines = raw.split("\n").filter(Boolean);
			let label = "";
			let messages = 0;
			for (const l of lines) {
				try {
					const ev = JSON.parse(l) as SessionEvent;
					if (ev.type !== "user") continue;
					messages++;
					if (!label) label = (ev.text ?? "").replace(/\s+/g, " ").trim().slice(0, 80);
				} catch {}
			}
			if (!messages) continue; // tool noise only — still not a conversation
			let mtime = 0;
			try {
				mtime = statSync(join(dir, id)).mtimeMs;
			} catch {}
			out.push({ id, label: label || "(no text)", mtime, messages, tokens: readTotals(join(dir, id)) });
		}
		return out.sort((a, b) => b.mtime - a.mtime);
	}

	append(ev: Omit<SessionEvent, "t">): void {
		this.#ensureDir();
		appendFileSync(join(this.dir, "events.jsonl"), JSON.stringify({ t: Date.now(), ...ev }) + "\n");
	}

	/** Cumulative tokens attributed to this session. */
	tokens(): number {
		return readTotals(this.dir);
	}

	/** Add to the running token total. Negative deltas are ignored — context shrinks on compaction. */
	addTokens(n: number): void {
		if (!Number.isFinite(n) || n <= 0) return;
		this.#ensureDir();
		writeFileSync(join(this.dir, "totals.json"), JSON.stringify({ tokens: readTotals(this.dir) + Math.round(n) }));
	}

	readEvents(): SessionEvent[] {
		try {
			return readFileSync(join(this.dir, "events.jsonl"), "utf8")
				.split("\n")
				.filter(Boolean)
				.map((l) => JSON.parse(l));
		} catch {
			return [];
		}
	}

	saveHistory(history: ChatHistoryItem[]): void {
		this.#ensureDir();
		writeFileSync(join(this.dir, "history.json"), JSON.stringify(history));
	}

	loadHistory(): ChatHistoryItem[] | null {
		try {
			return JSON.parse(readFileSync(join(this.dir, "history.json"), "utf8"));
		} catch {
			return null;
		}
	}

	/** Copy this session's transcript, history, and checkpoints into a new session id. */
	fork(): Session {
		const next = Session.create(this.projectRoot);
		if (existsSync(this.dir)) cpSync(this.dir, next.dir, { recursive: true });
		return next;
	}
}
