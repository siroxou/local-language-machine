// Runnable checks for the training runner's non-trivial logic: stdout progress parsing
// and dataset staging/validation. No Python / GPU / network needed.
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile } from "node:fs/promises";
import { writeFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { parseProgress, stageDataset, pickBackend, selectBestCheckpoint } from "./train.ts";

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
	const s = stageDataset(ds, run, 0.1);
	assert.equal(s.rows, 10);
	assert.equal(s.shape, "text");   // shape comes from row 0 — mlx picks its dataset class the same way
	assert.equal(s.dropped, 0);
	assert.ok(s.hashes["train.jsonl"]);
	const train = (await readFile(join(run, "train.jsonl"), "utf8")).trim().split("\n");
	const valid = (await readFile(join(run, "valid.jsonl"), "utf8")).trim().split("\n");
	const test = (await readFile(join(run, "test.jsonl"), "utf8")).trim().split("\n");
	assert.equal(valid.length, 1); // floor(10*0.1) held out for validation
	assert.equal(test.length, 1);  // and for the benchmarker's held-out test set
	assert.equal(train.length, 8); // train gets the rest
});

test("stageDataset split is reproducible for a seed, and differs across seeds", async () => {
	const dir = await mkdtemp(join(tmpdir(), "llm-seed-"));
	const ds = join(dir, "data.jsonl");
	writeFileSync(ds, Array.from({ length: 40 }, (_, i) => JSON.stringify({ text: `row ${i}` })).join("\n") + "\n");
	const hash = (run: string, seed: number) => stageDataset(ds, join(dir, run), 0.1, seed).hashes["test.jsonl"];
	assert.equal(hash("a", 42), hash("b", 42)); // same seed → byte-identical hold-out
	assert.notEqual(hash("a", 42), hash("c", 7));
});

test("stageDataset drops exact duplicates so they cannot span train and test", async () => {
	const dir = await mkdtemp(join(tmpdir(), "llm-dup-"));
	const ds = join(dir, "dup.jsonl");
	const dupe = JSON.stringify({ text: "same row" });
	writeFileSync(ds, [...Array.from({ length: 18 }, (_, i) => JSON.stringify({ text: `r${i}` })), dupe, dupe, dupe].join("\n") + "\n");
	const s = stageDataset(ds, join(dir, "run"), 0.1);
	assert.equal(s.dropped, 2);
	assert.equal(s.rows, 19);
});

test("stageDataset copies a pre-split directory verbatim instead of re-splitting", async () => {
	const dir = await mkdtemp(join(tmpdir(), "llm-dir-"));
	const src = join(dir, "suite");
	mkdirSync(src, { recursive: true });
	const line = (s: string) => JSON.stringify({ messages: [{ role: "user", content: s }, { role: "assistant", content: "a" }] });
	writeFileSync(join(src, "train.jsonl"), [line("t1"), line("t2"), line("t3")].join("\n") + "\n");
	writeFileSync(join(src, "valid.jsonl"), line("v1") + "\n");
	writeFileSync(join(src, "test.jsonl"), line("e1") + "\n");
	const run = join(dir, "run");
	const s = stageDataset(src, run, 0.1);
	assert.equal(s.rows, 3);          // train count, untouched
	assert.equal(s.shape, "messages"); // → mask_prompt is safe to emit
	// verbatim: the copied test set is exactly what we handed it, not a reshuffle
	assert.equal((await readFile(join(run, "test.jsonl"), "utf8")).trim(), line("e1"));
	assert.equal((await readFile(join(run, "train.jsonl"), "utf8")).trim().split("\n").length, 3);
});

test("selectBestCheckpoint promotes the lowest-val-loss checkpoint, not the last", async () => {
	const dir = await mkdtemp(join(tmpdir(), "llm-ckpt-"));
	// Mirrors the real failure: val bottoms at 150 then degrades through 300.
	const curve = [
		{ iter: 50, valLoss: 1.88 }, { iter: 100, valLoss: 1.79 },
		{ iter: 150, valLoss: 1.72 }, { iter: 200, valLoss: 1.80 }, { iter: 300, valLoss: 1.88 },
	];
	for (const p of curve) writeFileSync(join(dir, `${String(p.iter).padStart(7, "0")}_adapters.safetensors`), `ckpt${p.iter}`);
	writeFileSync(join(dir, "adapters.safetensors"), "ckpt300");
	const best = selectBestCheckpoint(dir, curve);
	assert.equal(best?.iter, 150);
	assert.equal(await readFile(join(dir, "adapters.safetensors"), "utf8"), "ckpt150");
});

test("selectBestCheckpoint is a no-op when the final checkpoint is already best", async () => {
	const dir = await mkdtemp(join(tmpdir(), "llm-ckpt2-"));
	const curve = [{ iter: 50, valLoss: 1.9 }, { iter: 100, valLoss: 1.5 }];
	writeFileSync(join(dir, "adapters.safetensors"), "final");
	assert.equal(selectBestCheckpoint(dir, curve), null);
	assert.equal(await readFile(join(dir, "adapters.safetensors"), "utf8"), "final");
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
