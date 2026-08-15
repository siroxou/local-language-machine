// Undo restores prior file contents — and must DELETE a file that didn't exist
// before the snapshot. Both branches are file-money-path.
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Checkpoints } from "./checkpoints.ts";

async function fixture() {
	const root = await mkdtemp(join(tmpdir(), "llm-cp-"));
	const cp = new Checkpoints(join(root, ".cp"));
	return { root, cp };
}

test("undo restores the prior contents of an edited file", async () => {
	const { root, cp } = await fixture();
	await writeFile(join(root, "f.txt"), "original");
	await cp.snapshot(root, "f.txt");
	await writeFile(join(root, "f.txt"), "changed by the model");
	const undone = await cp.undo(root);
	assert.deepEqual(undone, { path: "f.txt", deleted: false });
	assert.equal(await readFile(join(root, "f.txt"), "utf8"), "original");
});

test("undo deletes a file that did not exist at snapshot time", async () => {
	const { root, cp } = await fixture();
	await cp.snapshot(root, "new.txt"); // file absent
	await writeFile(join(root, "new.txt"), "model created this");
	const undone = await cp.undo(root);
	assert.deepEqual(undone, { path: "new.txt", deleted: true });
	assert.equal(existsSync(join(root, "new.txt")), false);
});

test("undo returns null with nothing to undo, and persists across reloads", async () => {
	const { root, cp } = await fixture();
	await writeFile(join(root, "g.txt"), "v1");
	await cp.snapshot(root, "g.txt");
	// A fresh Checkpoints over the same dir sees the persisted snapshot.
	const reloaded = new Checkpoints(join(root, ".cp"));
	await reloaded.load();
	assert.equal(reloaded.count, 1);
	await reloaded.undo(root);
	assert.equal(await reloaded.undo(root), null);
});
