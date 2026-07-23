// Local Language Machine — LoRA fine-tuning runner.
//
// Same shape as the integrated terminal (server.ts `case "term"`): spawn one child,
// stream its stdout/stderr line-by-line over the WS, keep a single active handle so it
// can be stopped. The heavy lifting is a platform-native trainer we shell out to —
// MLX on Apple Silicon (its `mlx_lm.lora` CLI is a complete LoRA trainer), Unsloth on
// CUDA (a small generated Python script, since Unsloth has no train CLI). We never
// vendor either library's source.
//
// You do NOT fine-tune the quantized GGUF used for inference — LoRA trains the original
// HF model, so `baseModel` is a Hugging Face repo id, not a cached .gguf.

import { spawn, execFileSync, type ChildProcess } from "node:child_process";
import { mkdirSync, writeFileSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { randomUUID } from "node:crypto";
import { appHome } from "../paths.ts";
import { ensureVenv } from "./venv.ts";
import { writeMeta, type RunMeta } from "./runs.ts";

export type Backend = "mlx" | "unsloth";

export interface TrainConfig {
	backend?: "auto" | Backend;
	/** HF repo id (or local HF-format dir). NOT the inference GGUF. */
	baseModel: string;
	/** Absolute path to a .jsonl dataset (server resolves it against the workspace root). */
	datasetPath: string;
	iters?: number;
	learningRate?: number;
	loraRank?: number;
	batchSize?: number;
	maxSeqLen?: number;
	/** Fraction of rows held out for validation (default 0.1, min 1 row). */
	valSplit?: number;
}

export type TrainEvent =
	| { type: "train-log"; line: string }
	| { type: "train-progress"; iter: number; loss?: number; valLoss?: number }
	| { type: "train-done"; adapterDir: string }
	| { type: "train-error"; message: string };

export type Emit = (e: TrainEvent) => void;

// One heavy task at a time for the whole app (train / fuse / convert / eval share this).
// `pending` covers the async setup window before the child spawns.
let child: ChildProcess | null = null;
let pending = false;

export function isTraining(): boolean {
	return child != null || pending;
}

export function stopTraining(): void {
	child?.kill("SIGINT");
}

/**
 * Spawn a subprocess as the single managed child, stream each stdout/stderr line to
 * `onLine`, and resolve with its exit code. Shared by training, fuse, GGUF-convert and
 * eval so only one heavy task runs at a time (stopTraining kills whichever it is).
 */
export function runManaged(cmd: string, args: string[], cwd: string, onLine: (line: string) => void): Promise<number> {
	if (child) throw new Error("Another training/eval task is already active.");
	return new Promise((resolve) => {
		const stream = lineStreamer(onLine);
		const proc = spawn(cmd, args, { cwd, env: process.env });
		child = proc;
		proc.stdout?.on("data", (d) => stream.push(d.toString()));
		proc.stderr?.on("data", (d) => stream.push(d.toString()));
		proc.on("error", (e) => { stream.flush(); onLine(`ERROR: ${e.message}`); child = null; resolve(1); });
		proc.on("close", (code) => { stream.flush(); child = null; resolve(code ?? 0); });
	});
}

/** Hold the single-task lock across a multi-step async flow (fuse→convert, eval passes). */
export async function withBusy<T>(fn: () => Promise<T>): Promise<T> {
	if (child || pending) throw new Error("Another training/eval task is already active.");
	pending = true;
	try { return await fn(); } finally { pending = false; }
}

/** darwin/arm64 → mlx, an NVIDIA box → unsloth. `pref` other than "auto" wins. */
export function pickBackend(pref?: string): Backend {
	if (pref === "mlx" || pref === "unsloth") return pref;
	if (process.platform === "darwin" && process.arch === "arm64") return "mlx";
	try {
		execFileSync("nvidia-smi", ["-L"], { stdio: "ignore", timeout: 2000 });
		return "unsloth";
	} catch {}
	throw new Error("No supported trainer for this machine — need Apple Silicon (MLX) or an NVIDIA GPU (Unsloth).");
}

/**
 * Pull `{iter, loss}` out of a trainer stdout line. MLX prints `Iter N: Train loss X`;
 * our Unsloth script prints an explicit `TRAIN_PROGRESS step=N loss=X [val=Y]` marker
 * (with a raw HF-Trainer dict as a fallback). Returns null for non-progress lines.
 */
export function parseProgress(backend: Backend, line: string): { iter: number; loss?: number; valLoss?: number } | null {
	if (backend === "mlx") {
		const t = line.match(/Iter (\d+):\s*Train loss ([\d.]+)/);
		if (t) return { iter: +t[1]!, loss: +t[2]! };
		const v = line.match(/Iter (\d+):\s*Val loss ([\d.]+)/);
		if (v) return { iter: +v[1]!, valLoss: +v[2]! };
		return null;
	}
	const m = line.match(/TRAIN_PROGRESS step=(\d+) loss=([\d.]+)(?: val=([\d.]+))?/);
	if (m) return { iter: +m[1]!, loss: +m[2]!, valLoss: m[3] ? +m[3]! : undefined };
	// HF Trainer dict fallback — no reliable step, so use epoch*1000 as a pseudo-iter.
	const d = line.match(/'loss':\s*([\d.]+).*'epoch':\s*([\d.]+)/);
	if (d) return { iter: Math.round(+d[2]! * 1000), loss: +d[1]! };
	return null;
}

/** Trailing-newline-tolerant line splitter feeding `onLine` per complete line. */
function lineStreamer(onLine: (line: string) => void): { push: (chunk: string) => void; flush: () => void } {
	let buf = "";
	return {
		push(chunk) {
			buf += chunk;
			let i: number;
			while ((i = buf.indexOf("\n")) >= 0) {
				onLine(buf.slice(0, i));
				buf = buf.slice(i + 1);
			}
		},
		flush() {
			if (buf) { onLine(buf); buf = ""; }
		},
	};
}

const SHAPES = "each line needs a string \"text\", a \"messages\" array, or string \"prompt\"+\"completion\"";
function validRow(o: any): boolean {
	return (
		(typeof o?.text === "string") ||
		Array.isArray(o?.messages) ||
		(typeof o?.prompt === "string" && typeof o?.completion === "string")
	);
}

/**
 * Validate the dataset and split it into train.jsonl + valid.jsonl inside `runDir`
 * (MLX reads a data *folder*; the Unsloth script reads train.jsonl). Throws on the first
 * malformed row so a bad dataset fails BEFORE any subprocess is spawned.
 */
export function stageDataset(datasetPath: string, runDir: string, valSplit = 0.1): number {
	let raw: string;
	try {
		raw = readFileSync(datasetPath, "utf8");
	} catch {
		throw new Error(`Dataset not found: ${datasetPath}`);
	}
	const rows = raw.split("\n").map((l) => l.trim()).filter(Boolean);
	if (rows.length === 0) throw new Error("Dataset is empty.");
	rows.forEach((line, idx) => {
		let o: unknown;
		try { o = JSON.parse(line); } catch { throw new Error(`Dataset line ${idx + 1} is not valid JSON.`); }
		if (!validRow(o)) throw new Error(`Dataset line ${idx + 1}: unsupported shape — ${SHAPES}.`);
	});
	// 3-way split: valid (for during-training eval) + test (held out for the benchmarker),
	// each ≥1 row when the data allows; train gets the rest. MLX reads valid.jsonl during
	// training and test.jsonl for `--test`.
	const hold = Math.max(1, Math.floor(rows.length * valSplit));
	const nVal = Math.min(hold, rows.length - 1);
	const nTest = Math.min(hold, rows.length - nVal - 1);
	const valid = rows.slice(0, nVal);
	const test = rows.slice(nVal, nVal + nTest);
	const train = rows.slice(nVal + nTest);
	mkdirSync(runDir, { recursive: true });
	writeFileSync(join(runDir, "train.jsonl"), train.join("\n") + "\n");
	writeFileSync(join(runDir, "valid.jsonl"), valid.join("\n") + "\n");
	if (test.length) writeFileSync(join(runDir, "test.jsonl"), test.join("\n") + "\n");
	return rows.length;
}

/** Format a number as a plain decimal — YAML 1.1 parses `1e-5` as a string, not a float. */
const plain = (n: number) => n.toLocaleString("en-US", { useGrouping: false, maximumFractionDigits: 12 });

/** Write mlx_lm.lora's YAML config into runDir and return its path. */
function writeMlxConfig(cfg: TrainConfig, runDir: string): string {
	const p = join(runDir, "mlx_config.yaml");
	const body = [
		`model: "${cfg.baseModel}"`,
		`train: true`,
		`data: "${runDir}"`,
		`adapter_path: "${join(runDir, "adapters")}"`,
		`fine_tune_type: lora`,
		`num_layers: 16`,
		`batch_size: ${cfg.batchSize ?? 4}`,
		`iters: ${cfg.iters ?? 100}`,
		`learning_rate: ${plain(cfg.learningRate ?? 1e-5)}`,
		`max_seq_length: ${cfg.maxSeqLen ?? 2048}`,
		`steps_per_report: 10`,
		`steps_per_eval: 50`,
		`save_every: 100`,
		`lora_parameters:`,
		`  rank: ${cfg.loraRank ?? 8}`,
		`  scale: 20.0`,
		`  dropout: 0.0`,
		"",
	].join("\n");
	writeFileSync(p, body);
	return p;
}

function buildCommand(cfg: TrainConfig, backend: Backend, runDir: string, python: string): { cmd: string; args: string[] } {
	if (backend === "mlx") {
		// mlx_lm.lora is the long-standing module entry; a version rename is a one-line fix.
		return { cmd: python, args: ["-m", "mlx_lm.lora", "--config", writeMlxConfig(cfg, runDir)] };
	}
	const script = fileURLToPath(new URL("./unsloth_train.py", import.meta.url)); // fileURLToPath decodes spaces (.pathname would keep %20)
	const cfgJson = join(runDir, "cfg.json");
	writeFileSync(cfgJson, JSON.stringify({
		base_model: cfg.baseModel,
		data: join(runDir, "train.jsonl"),
		adapter_path: join(runDir, "adapters"),
		iters: cfg.iters ?? 100,
		learning_rate: cfg.learningRate ?? 1e-5,
		lora_rank: cfg.loraRank ?? 8,
		batch_size: cfg.batchSize ?? 4,
		max_seq_length: cfg.maxSeqLen ?? 2048,
	}, null, 2));
	return { cmd: python, args: [script, "--config", cfgJson] };
}

/**
 * Kick off a run: ensure the venv (streams install logs), stage the dataset (throws on
 * bad data before spawning), then spawn the trainer and stream it. Resolves once the
 * child is spawned; completion is reported later via `train-done`/`train-error`.
 */
export async function startTraining(cfg: TrainConfig, emit: Emit): Promise<void> {
	if (child || pending) throw new Error("A training run is already active.");
	if (!cfg.baseModel?.trim()) throw new Error("Pick a base model (a Hugging Face repo id).");
	pending = true;
	try {
		const backend = pickBackend(cfg.backend);
		const runId = new Date().toISOString().replace(/[:.]/g, "-") + "-" + randomUUID().slice(0, 8);
		const runDir = join(appHome(), "adapters", runId);
		mkdirSync(runDir, { recursive: true });

		const meta: RunMeta = {
			runId, backend, baseModel: cfg.baseModel, config: cfg, status: "running",
			startedAt: new Date().toISOString(), lossCurve: [], adapterDir: join(runDir, "adapters"),
		};
		writeMeta(meta);

		emit({ type: "train-log", line: `▸ Backend: ${backend} · run ${runId}` });
		// Stage the dataset FIRST — a malformed dataset fails instantly, before the (slow,
		// first-run) venv install and without spawning anything.
		const rows = stageDataset(cfg.datasetPath, runDir, cfg.valSplit);
		emit({ type: "train-log", line: `▸ Dataset: ${rows} rows staged → ${runDir}` });
		const python = await ensureVenv(backend, (line) => emit({ type: "train-log", line }));

		const { cmd, args } = buildCommand(cfg, backend, runDir, python);
		emit({ type: "train-log", line: `▸ ${cmd} ${args.join(" ")}\n` });

		// Fire-and-forget: resolve to the caller once spawned; completion arrives via events + meta.
		runManaged(cmd, args, runDir, (line) => {
			emit({ type: "train-log", line });
			const p = parseProgress(backend, line);
			if (p) { meta.lossCurve.push(p); emit({ type: "train-progress", ...p }); }
		}).then((code) => {
			meta.finishedAt = new Date().toISOString();
			if (code === 0) { meta.status = "done"; writeMeta(meta); emit({ type: "train-done", adapterDir: meta.adapterDir }); }
			else { meta.status = "error"; meta.error = `Trainer exited with code ${code}.`; writeMeta(meta); emit({ type: "train-error", message: meta.error }); }
		});
	} finally {
		// child (if it spawned) now reflects "busy"; if setup threw, this clears the flag.
		pending = false;
	}
}
