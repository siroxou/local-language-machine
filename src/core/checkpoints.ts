// Local Language Machine — file-edit checkpoints (the Esc-Esc rewind).
//
// Before a mutating tool touches a file, we snapshot its prior contents. `undo`
// restores the newest snapshot (deleting the file if it didn't exist before).
// Separate from git; persisted in the session dir so it survives resume; covers
// file changes only (not directory renames or metadata).

import { readFile, writeFile, rm, mkdir } from "node:fs/promises";
import { join } from "node:path";
import { resolveInRoot } from "../native/tools/tools.ts";

interface Snapshot {
	path: string; // workspace-relative
	before: string | null; // prior content, or null if the file was newly created
}

export class Checkpoints {
	#stack: Snapshot[] = [];

	/** @param dir session-scoped directory used to persist the stack across resume. */
	constructor(private readonly dir: string) {}

	get count(): number {
		return this.#stack.length;
	}

	/** Distinct files touched this session, oldest first. Powers the Changes panel. */
	get changed(): string[] {
		return [...new Set(this.#stack.map((s) => s.path))];
	}

	/**
	 * The contents `rel` had before this session touched it — i.e. the *earliest* snapshot, not the
	 * newest. A file edited three times has three entries; diffing against the last one would show
	 * only the final edit rather than the session's whole effect. Null means it didn't exist.
	 */
	beforeOf(rel: string): string | null {
		return this.#stack.find((s) => s.path === rel)?.before ?? null;
	}

	/** Record the current contents of `rel` before it is modified. */
	async snapshot(root: string, rel: string): Promise<void> {
		const abs = resolveInRoot(root, rel);
		let before: string | null = null;
		try {
			before = await readFile(abs, "utf8");
		} catch {
			before = null; // file doesn't exist yet — undo will delete it
		}
		this.#stack.push({ path: rel, before });
		await this.#persist();
	}

	/** Restore the newest snapshot. Returns the path restored, or null if nothing to undo. */
	async undo(root: string): Promise<{ path: string; deleted: boolean } | null> {
		const snap = this.#stack.pop();
		if (!snap) return null;
		const abs = resolveInRoot(root, snap.path);
		if (snap.before === null) {
			await rm(abs, { force: true });
		} else {
			await writeFile(abs, snap.before, "utf8");
		}
		await this.#persist();
		return { path: snap.path, deleted: snap.before === null };
	}

	async #persist(): Promise<void> {
		await mkdir(this.dir, { recursive: true });
		await writeFile(join(this.dir, "checkpoints.json"), JSON.stringify(this.#stack), "utf8");
	}

	/** Rehydrate the stack from disk (on session resume). */
	async load(): Promise<void> {
		try {
			const data = JSON.parse(await readFile(join(this.dir, "checkpoints.json"), "utf8"));
			if (Array.isArray(data)) this.#stack = data;
		} catch {
			this.#stack = [];
		}
	}
}
