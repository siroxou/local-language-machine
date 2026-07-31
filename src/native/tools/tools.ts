// Local Language Machine — agent tools exposed to the model.
//
// Decoupled from VS Code: the caller injects a workspace `root` and (optionally)
// an `exec`. All file paths are resolved *inside* root — a safety boundary, not
// simplified away. Mutating actions are gated by the loop's ToolGate (permission
// modes + confirm), not here.

import { exec as cpExec } from "node:child_process";
import { readFile, writeFile, mkdir, readdir, glob } from "node:fs/promises";
import { isAbsolute, join, relative, resolve, sep } from "node:path";
import { promisify } from "node:util";
import type { ToolDef } from "../inference/engine.ts";

const execAsync = promisify(cpExec);

// Directories a content search should never descend into.
const SKIP_DIRS = new Set(["node_modules", ".git", "dist", ".next", ".cache"]);
const MAX_GREP_MATCHES = 200; // cap results so a broad pattern can't flood the context window
const MAX_GLOB_MATCHES = 500;

export interface ExecResult {
	stdout: string;
	stderr: string;
	exitCode: number;
}

export interface ToolContext {
	/** Workspace root. All file paths resolve under it; escapes are rejected. */
	root: string;
	/** Command runner (injectable for tests). Defaults to child_process. */
	exec?: (command: string, cwd: string) => Promise<ExecResult>;
}

const MAX_READ_BYTES = 100_000; // cap read size so a huge file can't blow the context window

/** Resolve `p` under `root`, rejecting anything that escapes it. */
export function resolveInRoot(root: string, p: string): string {
	const abs = isAbsolute(p) ? resolve(p) : resolve(root, p);
	const rel = relative(resolve(root), abs);
	if (rel === "" ) return abs;
	if (rel.startsWith("..") || isAbsolute(rel)) {
		throw new Error(`Path escapes the workspace: ${p}`);
	}
	return abs;
}

/** Recursively visit every file under `dir` (skipping SKIP_DIRS), passing the absolute path and the path relative to `root`. */
export async function walkFiles(dir: string, root: string, visit: (abs: string, rel: string) => Promise<void>): Promise<void> {
	let entries;
	try {
		entries = await readdir(dir, { withFileTypes: true });
	} catch {
		return; // unreadable dir (or a file path passed in) — skip
	}
	for (const e of entries) {
		if (SKIP_DIRS.has(e.name)) continue;
		const abs = join(dir, e.name);
		if (e.isDirectory()) await walkFiles(abs, root, visit);
		else if (e.isFile()) await visit(abs, relative(root, abs));
	}
}

// Without a timeout a long-running command the model picks (a dev server, a watch task,
// anything waiting on stdin) never resolves, and the whole chat turn hangs with no way to
// cancel it. Kill it instead and hand the model a result it can reason about.
const EXEC_TIMEOUT_MS = 120_000;

async function defaultExec(command: string, cwd: string): Promise<ExecResult> {
	try {
		const { stdout, stderr } = await execAsync(command, { cwd, maxBuffer: 10 * 1024 * 1024, timeout: EXEC_TIMEOUT_MS, killSignal: "SIGKILL" });
		return { stdout, stderr, exitCode: 0 };
	} catch (e: any) {
		if (e?.killed) {
			return { stdout: e.stdout ?? "", stderr: `Command timed out after ${EXEC_TIMEOUT_MS / 1000}s and was killed. Long-running processes must be started by the user in the terminal.`, exitCode: 124 };
		}
		return { stdout: e.stdout ?? "", stderr: e.stderr ?? String(e?.message ?? e), exitCode: e.code ?? 1 };
	}
}

export function buildTools(ctx: ToolContext): Record<string, ToolDef> {
	const run = ctx.exec ?? defaultExec;

	return {
		read_file: {
			description: "Read a UTF-8 text file from the workspace.",
			params: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
			async handler({ path }: { path: string }) {
				const abs = resolveInRoot(ctx.root, path);
				const buf = await readFile(abs);
				const truncated = buf.byteLength > MAX_READ_BYTES;
				return {
					content: buf.subarray(0, MAX_READ_BYTES).toString("utf8"),
					truncated,
				};
			},
		},

		list_dir: {
			description: "List entries of a workspace directory.",
			params: { type: "object", properties: { path: { type: "string" } } },
			async handler({ path }: { path?: string }) {
				const abs = resolveInRoot(ctx.root, path ?? ".");
				const entries = await readdir(abs, { withFileTypes: true });
				return { entries: entries.map((e) => ({ name: e.name, dir: e.isDirectory() })) };
			},
		},

		grep: {
			description: "Search file contents by regular expression across the workspace. Returns matching lines with their file path and line number.",
			params: {
				type: "object",
				properties: { pattern: { type: "string" }, path: { type: "string" }, flags: { type: "string" } },
				required: ["pattern"],
			},
			async handler({ pattern, path, flags }: { pattern: string; path?: string; flags?: string }) {
				const re = new RegExp(pattern, flags ?? ""); // bad regex throws → surfaced to the model as an error
				const base = resolveInRoot(ctx.root, path ?? ".");
				const matches: Array<{ path: string; line: number; text: string }> = [];
				await walkFiles(base, ctx.root, async (abs, rel) => {
					if (matches.length >= MAX_GREP_MATCHES) return;
					const buf = await readFile(abs).catch(() => null);
					if (!buf || buf.byteLength > MAX_READ_BYTES || buf.includes(0)) return; // skip huge/binary files
					const lines = buf.toString("utf8").split("\n");
					for (let i = 0; i < lines.length && matches.length < MAX_GREP_MATCHES; i++) {
						if (re.test(lines[i]!)) matches.push({ path: rel, line: i + 1, text: lines[i]!.slice(0, 300) });
					}
				});
				return { matches, truncated: matches.length >= MAX_GREP_MATCHES };
			},
		},

		glob: {
			description: "Find workspace files matching a glob pattern, e.g. \"src/**/*.ts\".",
			params: { type: "object", properties: { pattern: { type: "string" } }, required: ["pattern"] },
			async handler({ pattern }: { pattern: string }) {
				const out: string[] = [];
				for await (const p of glob(pattern, { cwd: ctx.root })) {
					out.push(typeof p === "string" ? p : String(p));
					if (out.length >= MAX_GLOB_MATCHES) break;
				}
				return { matches: out, truncated: out.length >= MAX_GLOB_MATCHES };
			},
		},

		write_file: {
			description: "Write a UTF-8 text file in the workspace (creates parent dirs).",
			params: {
				type: "object",
				properties: { path: { type: "string" }, content: { type: "string" } },
				required: ["path", "content"],
			},
			async handler({ path, content }: { path: string; content: string }) {
				const abs = resolveInRoot(ctx.root, path);
				await mkdir(join(abs, ".."), { recursive: true });
				await writeFile(abs, content, "utf8");
				return { written: true };
			},
		},

		run_terminal: {
			description: "Run a shell command in the workspace root and return its output.",
			params: { type: "object", properties: { command: { type: "string" } }, required: ["command"] },
			async handler({ command }: { command: string }) {
				return run(command, ctx.root);
			},
		},

		git: {
			description: "Run a git subcommand (status/diff/log run directly; mutating ones ask first).",
			params: {
				type: "object",
				properties: { args: { type: "array", items: { type: "string" } } },
				required: ["args"],
			},
			async handler({ args }: { args: string[] }) {
				// Quote args to avoid shell splitting; simple and enough for a controlled arg list.
				const quoted = args.map((a) => `'${a.replaceAll("'", "'\\''")}'`).join(" ");
				return run(`git ${quoted}`, ctx.root);
			},
		},
	};
}
