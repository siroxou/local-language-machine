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

	/** All sessions for this project, newest first, with a short label from the first user message. */
	static list(projectRoot: string): Array<{ id: string; label: string; mtime: number }> {
		const dir = projectDir(projectRoot);
		if (!existsSync(dir)) return [];
		return readdirSync(dir)
			.map((id) => {
				const evFile = join(dir, id, "events.jsonl");
				let label = "(empty)";
				let mtime = 0;
				try {
					mtime = statSync(join(dir, id)).mtimeMs;
					const first = readFileSync(evFile, "utf8").split("\n").find((l) => l.includes('"user"'));
					if (first) label = (JSON.parse(first).text ?? "").slice(0, 60) || label;
				} catch {}
				return { id, label, mtime };
			})
			.sort((a, b) => b.mtime - a.mtime);
	}

	append(ev: Omit<SessionEvent, "t">): void {
		appendFileSync(join(this.dir, "events.jsonl"), JSON.stringify({ t: Date.now(), ...ev }) + "\n");
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
		cpSync(this.dir, next.dir, { recursive: true });
		return next;
	}
}
