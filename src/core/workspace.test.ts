// New file-management + search ops. Money-paths: remove deletes, rename moves,
// path-escape stays rejected via resolveInRoot, and search finds a known string.
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile, mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createFile, createDir, rename, remove, listDirs, searchWorkspace } from "./workspace.ts";

const fixture = () => mkdtemp(join(tmpdir(), "llm-ws-"));

test("createFile makes an empty file and refuses to clobber", async () => {
	const root = await fixture();
	await createFile(root, "sub/new.txt");
	assert.equal(await readFile(join(root, "sub/new.txt"), "utf8"), "");
	await assert.rejects(createFile(root, "sub/new.txt"), "must not overwrite an existing file");
});

test("createDir, rename, and remove", async () => {
	const root = await fixture();
	await createDir(root, "a/b");
	assert.ok(existsSync(join(root, "a/b")));
	await writeFile(join(root, "a/f.txt"), "hi");
	await rename(root, "a/f.txt", "a/g.txt");
	assert.equal(existsSync(join(root, "a/f.txt")), false);
	assert.equal(await readFile(join(root, "a/g.txt"), "utf8"), "hi");
	await remove(root, "a");
	assert.equal(existsSync(join(root, "a")), false, "remove deletes recursively");
});

test("file ops reject paths that escape the workspace", async () => {
	const root = await fixture();
	await assert.rejects(createFile(root, "../evil.txt"), /escapes the workspace/);
	await assert.rejects(remove(root, "../../etc"), /escapes the workspace/);
	await assert.rejects(rename(root, "../x", "y"), /escapes the workspace/);
});

test("searchWorkspace finds matching lines and skips node_modules", async () => {
	const root = await fixture();
	await writeFile(join(root, "a.ts"), "const NEEDLE = 1;\nother");
	await mkdir(join(root, "node_modules/pkg"), { recursive: true });
	await writeFile(join(root, "node_modules/pkg/b.ts"), "NEEDLE in a dependency");
	const { matches } = await searchWorkspace(root, "NEEDLE");
	assert.equal(matches.length, 1);
	assert.equal(matches[0].path, "a.ts");
	assert.equal(matches[0].line, 1);
});

test("searchWorkspace is literal by default (regex chars are escaped)", async () => {
	const root = await fixture();
	await writeFile(join(root, "a.txt"), "a.b\naxb");
	const { matches } = await searchWorkspace(root, "a.b");
	assert.equal(matches.length, 1, "'.' must match a literal dot, not any char");
});

test("listDirs returns subdirectories and a parent", async () => {
	const root = await fixture();
	await mkdir(join(root, "one"));
	await mkdir(join(root, "two"));
	await writeFile(join(root, "file.txt"), "x");
	const res = await listDirs(root);
	assert.deepEqual(res.dirs.map((d) => d.name), ["one", "two"]);
	assert.ok(!res.dirs.some((d) => d.name === "file.txt"), "files are excluded");
	assert.equal(typeof res.parent, "string");
});
