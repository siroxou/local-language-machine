// Local Language Machine — workspace file access for the editor.
// Path-safe reads/writes + a tree, all scoped under the workspace root.

import { readFile, writeFile, readdir, mkdir, rename as fsRename, rm } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { resolveInRoot, walkFiles } from "../native/tools/tools.ts";

const SKIP = new Set(["node_modules", ".git", "dist", ".DS_Store", ".next"]);

export interface TreeNode {
	name: string;
	path: string;
	dir: boolean;
	children?: TreeNode[];
}

export async function tree(root: string, rel = "", depth = 0): Promise<TreeNode[]> {
	if (depth > 7) return [];
	const entries = await readdir(resolveInRoot(root, rel || "."), { withFileTypes: true });
	const nodes: TreeNode[] = [];
	for (const e of entries) {
		if (SKIP.has(e.name)) continue;
		const path = rel ? `${rel}/${e.name}` : e.name;
		if (e.isDirectory()) nodes.push({ name: e.name, path, dir: true, children: await tree(root, path, depth + 1) });
		else nodes.push({ name: e.name, path, dir: false });
	}
	// Directories first, then files, each alphabetical.
	return nodes.sort((a, b) => (a.dir === b.dir ? a.name.localeCompare(b.name) : a.dir ? -1 : 1));
}

export async function read(root: string, rel: string): Promise<string> {
	return readFile(resolveInRoot(root, rel), "utf8");
}

export async function write(root: string, rel: string, content: string): Promise<void> {
	const abs = resolveInRoot(root, rel);
	await mkdir(dirname(abs), { recursive: true });
	await writeFile(abs, content, "utf8");
}

// ————— file management (all path-safe under root) —————

export async function createFile(root: string, rel: string): Promise<void> {
	const abs = resolveInRoot(root, rel);
	await mkdir(dirname(abs), { recursive: true });
	await writeFile(abs, "", { flag: "wx" }); // wx → error if it already exists (no clobber)
}

export async function createDir(root: string, rel: string): Promise<void> {
	await mkdir(resolveInRoot(root, rel), { recursive: true });
}

export async function rename(root: string, from: string, to: string): Promise<void> {
	const dest = resolveInRoot(root, to);
	await mkdir(dirname(dest), { recursive: true });
	await fsRename(resolveInRoot(root, from), dest);
}

export async function remove(root: string, rel: string): Promise<void> {
	await rm(resolveInRoot(root, rel), { recursive: true, force: true });
}

// ————— directory picker (absolute paths OUTSIDE the workspace — this is how Open Folder browses) —————

export interface DirListing {
	path: string;
	parent: string | null;
	dirs: Array<{ name: string; path: string }>;
}

export async function listDirs(path?: string): Promise<DirListing> {
	const abs = path && path.trim() ? path : homedir();
	const entries = await readdir(abs, { withFileTypes: true });
	const dirs = entries
		.filter((e) => e.isDirectory() && !e.name.startsWith("."))
		.map((e) => ({ name: e.name, path: join(abs, e.name) }))
		.sort((a, b) => a.name.localeCompare(b.name));
	const parent = dirname(abs);
	return { path: abs, parent: parent === abs ? null : parent, dirs };
}

// ————— global search (walk workspace files, regex or literal) —————

const MAX_SEARCH = 300;
const MAX_FILE_BYTES = 200_000;
export const escapeRegex = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

export interface SearchMatch {
	path: string;
	line: number;
	text: string;
}

export async function searchWorkspace(
	root: string,
	query: string,
	opts: { regex?: boolean; caseSensitive?: boolean } = {},
): Promise<{ matches: SearchMatch[]; truncated: boolean }> {
	if (!query) return { matches: [], truncated: false };
	const re = new RegExp(opts.regex ? query : escapeRegex(query), opts.caseSensitive ? "" : "i"); // bad regex throws → surfaced to the caller
	const base = resolveInRoot(root, ".");
	const matches: SearchMatch[] = [];
	await walkFiles(base, base, async (abs, rel) => {
		if (matches.length >= MAX_SEARCH) return;
		const buf = await readFile(abs).catch(() => null);
		if (!buf || buf.byteLength > MAX_FILE_BYTES || buf.includes(0)) return; // skip huge/binary files
		const lines = buf.toString("utf8").split("\n");
		for (let i = 0; i < lines.length && matches.length < MAX_SEARCH; i++) {
			if (re.test(lines[i]!)) matches.push({ path: rel, line: i + 1, text: lines[i]!.slice(0, 300) });
		}
	});
	return { matches, truncated: matches.length >= MAX_SEARCH };
}
