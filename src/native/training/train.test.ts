// Runnable checks for the training runner's non-trivial logic: stdout progress parsing
// and dataset staging/validation. No Python / GPU / network needed.
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile } from "node:fs/promises";
import { writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { parseProgress, stageDataset, pickBackend } from "./train.ts";

test("MLX train/val loss lines parse into {iter, loss}", () => {
	assert.deepEqual(parseProgress("mlx", "Iter 10: Train loss 2.345, Learning Rate 1.000e-05"), { iter: 10, loss: 2.345 });
	assert.deepEqual(parseProgress("mlx", "Iter 50: Val loss 2.100, Val took 1.2s"), { iter: 50, valLoss: 2.1 });
	assert.equal(parseProgress("mlx", "Loading pretrained model"), null);
});

test("Unsloth explicit marker and HF-dict fallback parse", () => {
	assert.deepEqual(parseProgress("unsloth", "TRAIN_PROGRESS step=5 loss=1.234"), { iter: 5, loss: 1.234, valLoss: undefined });
	assert.deepEqual(parseProgress("unsloth", "TRAIN_PROGRESS step=6 loss=1.100 val=1.500"), { iter: 6, loss: 1.1, valLoss: 1.5 });
	assert.deepEqual(parseProgress("unsloth", "{'loss': 0.9, 'epoch': 0.25}"), { iter: 250, loss: 0.9 });
	assert.equal(parseProgress("unsloth", "some other line"), null);
});

test("stageDataset validates rows and splits train/valid", async () => {
	const dir = await mkdtemp(join(tmpdir(), "llm-train-"));
	const ds = join(dir, "data.jsonl");
	// 10 rows across all three supported shapes.
	const rows = [
		...Array.from({ length: 4 }, (_, i) => JSON.stringify({ text: `sample ${i}` })),
		...Array.from({ length: 3 }, (_, i) => JSON.stringify({ messages: [{ role: "user", content: `q${i}` }] })),
		...Array.from({ length: 3 }, (_, i) => JSON.stringify({ prompt: `p${i}`, completion: `c${i}` })),
	];
	writeFileSync(ds, rows.join("\n") + "\n");
	const run = join(dir, "run");
	const n = stageDataset(ds, run, 0.1);
	assert.equal(n, 10);
	const train = (await readFile(join(run, "train.jsonl"), "utf8")).trim().split("\n");
	const valid = (await readFile(join(run, "valid.jsonl"), "utf8")).trim().split("\n");
	const test = (await readFile(join(run, "test.jsonl"), "utf8")).trim().split("\n");
	assert.equal(valid.length, 1); // floor(10*0.1) held out for validation
	assert.equal(test.length, 1);  // and for the benchmarker's held-out test set
	assert.equal(train.length, 8); // train gets the rest
});

test("stageDataset rejects malformed rows before any run", async () => {
	const dir = await mkdtemp(join(tmpdir(), "llm-train-bad-"));
	const ds = join(dir, "bad.jsonl");
	writeFileSync(ds, JSON.stringify({ text: "ok" }) + "\n" + '{"nope": 1}\n');
	assert.throws(() => stageDataset(ds, join(dir, "run"), 0.1), /line 2/);
});

test("pickBackend honors explicit preference", () => {
	assert.equal(pickBackend("mlx"), "mlx");
	assert.equal(pickBackend("unsloth"), "unsloth");
});
