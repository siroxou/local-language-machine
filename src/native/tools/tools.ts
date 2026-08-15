// Local Language Machine — agent tools exposed to the model.
//
// Decoupled from VS Code: the caller injects a workspace `root` and (optionally)
// an `exec`. All file paths are resolved *inside* root — a safety boundary, not
// simplified away. Mutating actions are gated by the loop's ToolGate (permission
// modes + confirm), not here.

import { exec as cpExec } from "node:child_process";
import { realpathSync } from "node:fs";
import { readFile, writeFile, mkdir, readdir, glob } from "node:fs/promises";
import { dirname, isAbsolute, join, relative, resolve } from "node:path";
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

// Longest search pattern accepted. Real patterns are short; the length is what gives a
// backtracking blow-up room to grow.
const MAX_PATTERN_CHARS = 1_000;
// The classic catastrophic-backtracking shape: a group that itself contains an unbounded
// quantifier, repeated again — (a+)+, (a*)*, ([a-z]+)* and friends. Matching "a".repeat(30)
// against one of these does not finish in any useful time.
const NESTED_QUANTIFIER = /\([^)]*[+*][^)]*\)\s*[+*]|\([^)]*[+*][^)]*\)\s*\{\d+,\s*\}/;

/**
 * Build a RegExp from a pattern that came off the wire or out of the model.
 *
 * Both the workspace search and the grep tool run their pattern over every file in the
 * tree on the process's only thread — the same thread that serves the UI — so a pattern
 * that backtracks catastrophically is a hang with no error and no way to cancel it.
 *
 * ponytail: a length cap plus a nested-quantifier reject, which reduces the exposure rather
 * than removing it — no string test can decide this in general, and JS RegExp has no match
 * timeout. The complete fix is to evaluate untrusted patterns on a worker thread, where a
 * runaway match can be killed; do that if a hang is ever actually observed.
 */
export function safeRegExp(pattern: string, flags = ""): RegExp {
	if (pattern.length > MAX_PATTERN_CHARS) {
		throw new Error(`Search pattern is too long (${pattern.length} > ${MAX_PATTERN_CHARS} characters).`);
	}
	if (NESTED_QUANTIFIER.test(pattern)) {
		throw new Error("Search pattern nests one unbounded repeat inside another, which can hang the search. Simplify it.");
	}
	// audit-ok(regexp-non-literal-source): this IS the checkpoint every untrusted pattern goes through
	return new RegExp(pattern, flags); // an otherwise-invalid pattern throws — surfaced to the caller
}

/**
 * Resolve `p` under `root`, rejecting anything that escapes it.
 *
 * Containment is checked against *resolved* paths, not the strings. A lexical comparison
 * accepts `workspace/link/etc/passwd` whenever `link` is a symlink out of the tree — and
 * the model can create such a link itself with run_terminal. macOS makes it reachable
 * with no setup at all, since /tmp is itself a link to /private/tmp.
 *
 * The target may legitimately not exist yet (write_file creates intermediate dirs), so
 * only the deepest existing ancestor is resolved and the unresolved tail is re-appended.
 *
 * ponytail: realpathSync on every call. Two extra stats per tool call, against a boundary
 * that has to be right — cache the resolved root here if a profile ever shows it.
 */
export function resolveInRoot(root: string, p: string): string {
	const realRoot = realpathSync(resolve(root));
	const abs = isAbsolute(p) ? resolve(p) : resolve(realRoot, p);

	const tail: string[] = [];
	let probe = abs;
	for (;;) {
		try {
			probe = realpathSync(probe);
			break;
		} catch {
			const parent = dirname(probe);
			if (parent === probe) break; // reached the filesystem root without finding anything
			tail.unshift(probe.slice(parent.length + 1));
			probe = parent;
		}
	}
	const resolved = tail.length ? join(probe, ...tail) : probe;

	const rel = relative(realRoot, resolved);
	if (rel === "") return resolved;
	if (rel.startsWith("..") || isAbsolute(rel)) {
		throw new Error(`Path escapes the workspace: ${p}`);
	}
	return resolved;
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
				const re = safeRegExp(pattern, flags ?? ""); // bad or hostile pattern throws → surfaced to the model as an error
				// Both arguments must be resolved, or reported paths come out relative to the
				// unresolved root and stop round-tripping through resolveInRoot.
				const root = resolveInRoot(ctx.root, ".");
				const base = resolveInRoot(root, path ?? ".");
				const matches: Array<{ path: string; line: number; text: string }> = [];
				await walkFiles(base, root, async (abs, rel) => {
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
				// `cwd` scopes where matching starts; it is not a boundary. A climbing pattern
				// walks out of it and an absolute one ignores it, so the pattern is rejected up
				// front — filtering results afterwards would still enumerate the parent to throw
				// it away, which leaks which sibling names exist by what survives.
				if (isAbsolute(pattern) || pattern.split(/[/\\]/).includes("..")) {
					throw new Error(`Path escapes the workspace: ${pattern}`);
				}
				const root = resolveInRoot(ctx.root, ".");
				const out: string[] = [];
				for await (const p of glob(pattern, { cwd: root })) {
					const rel = typeof p === "string" ? p : String(p);
					// Second check, for a symlink inside the tree that points out of it.
					try {
						resolveInRoot(root, rel);
					} catch {
						continue;
					}
					out.push(rel);
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
