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
import { mkdirSync, writeFileSync, readFileSync, existsSync, statSync, readdirSync, copyFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { randomUUID, createHash } from "node:crypto";
import { appHome } from "../paths.ts";
import { ensureVenv } from "./venv.ts";
import { writeMeta, type RunMeta, type LossPoint } from "./runs.ts";

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
	/** MLX only: transformer layers to attach LoRA to (default 16; -1 = all layers). */
	numLayers?: number;
	/** Fraction of rows held out for validation (default 0.1, min 1 row). */
	valSplit?: number;
	/**
	 * LoRA output multiplier. mlx applies this RAW (`y + scale*z`), not as alpha/rank, and inits
	 * lora_a at 1/√input_dims — so the effective step size grows with model width. mlx's default
	 * of 20 is paired with its default lr of 1e-5; raising lr without lowering scale diverges on
	 * wider models (observed: Qwen/Falcon loss spiking to 13 by iter 30 while SmolLM2 trained fine).
	 */
	loraScale?: number;
	/** Seeds the trainer AND the split shuffle, so a run is reproducible (default 42). */
	seed?: number;
	/** Linear warmup steps before cosine decay. 0 (default) = no schedule, constant LR. */
	warmupSteps?: number;
}

/** Which row shape a staged dataset uses — mlx rejects `mask_prompt` on `text` rows. */
export type RowShape = "messages" | "completions" | "text";

export interface StagedDataset {
	rows: number;
	shape: RowShape;
	/** Exact-duplicate rows dropped before splitting. */
	dropped: number;
	/** sha256 of each split file written, for proving two runs scored the same data. */
	hashes: Record<string, string>;
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
	// `pending` is the async setup window (venv bootstrap, dataset fetch) before the child exists.
	// Killing a null child there silently did nothing while the UI reported "Stop requested".
	stopRequested = true;
	child?.kill("SIGINT");
}

/** True once Stop was pressed, until the next task starts. Checked before each managed spawn. */
let stopRequested = false;
export function clearStopRequest(): void {
	stopRequested = false;
}
export function isStopRequested(): boolean {
	return stopRequested;
}

/**
 * Spawn a subprocess as the single managed child, stream each stdout/stderr line to
 * `onLine`, and resolve with its exit code. Shared by training, fuse, GGUF-convert and
 * eval so only one heavy task runs at a time (stopTraining kills whichever it is).
 */
export function runManaged(cmd: string, args: string[], cwd: string, onLine: (line: string) => void): Promise<number> {
	if (child) throw new Error("Another training/eval task is already active.");
	// Stop pressed during the setup window: honour it here rather than spawning the work anyway.
	if (stopRequested) { onLine("■ Stopped before the task started."); return Promise.resolve(1); }
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
	stopRequested = false; // a new task clears any Stop left over from the previous one
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

/** mlx picks the dataset class from row 0 alone, so the whole file must be one shape. */
function rowShape(o: any): RowShape {
	if (Array.isArray(o?.messages)) return "messages";
	if (typeof o?.prompt === "string" && typeof o?.completion === "string") return "completions";
	return "text";
}

/** Deterministic PRNG so a given seed always yields the same split. */
function mulberry32(seed: number): () => number {
	let a = seed >>> 0;
	return () => {
		a = (a + 0x6d2b79f5) >>> 0;
		let t = Math.imul(a ^ (a >>> 15), 1 | a);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

const sha256 = (p: string) => createHash("sha256").update(readFileSync(p)).digest("hex").slice(0, 16);

/** The split files mlx reads from a data folder, plus our extra scoring suites. */
const SPLIT_FILES = ["train.jsonl", "valid.jsonl", "test.jsonl"];

/**
 * Validate the dataset and split it into train.jsonl + valid.jsonl inside `runDir`
 * (MLX reads a data *folder*; the Unsloth script reads train.jsonl). Throws on the first
 * malformed row so a bad dataset fails BEFORE any subprocess is spawned.
 */
export function stageDataset(datasetPath: string, runDir: string, valSplit = 0.1, seed = 42): StagedDataset {
	mkdirSync(runDir, { recursive: true });

	// A DIRECTORY of pre-split jsonl is copied verbatim — no reshuffle, no re-split. This is what
	// lets every cell of a model matrix train and score on byte-identical data; re-splitting per
	// run would confound any cross-model difference with split variance.
	if (existsSync(datasetPath) && statSync(datasetPath).isDirectory()) {
		const present = readdirSync(datasetPath).filter((f) => f.endsWith(".jsonl"));
		if (!present.includes("train.jsonl")) throw new Error(`Dataset dir has no train.jsonl: ${datasetPath}`);
		const hashes: Record<string, string> = {};
		let total = 0;
		let shape: RowShape | null = null;
		for (const f of present) {
			const src = join(datasetPath, f);
			const lines = readFileSync(src, "utf8").split("\n").map((l) => l.trim()).filter(Boolean);
			lines.forEach((line, idx) => {
				let o: unknown;
				try { o = JSON.parse(line); } catch { throw new Error(`${f} line ${idx + 1} is not valid JSON.`); }
				if (!validRow(o)) throw new Error(`${f} line ${idx + 1}: unsupported shape — ${SHAPES}.`);
			});
			if (f === "train.jsonl") { total = lines.length; shape = rowShape(JSON.parse(lines[0]!)); }
			copyFileSync(src, join(runDir, f));
			hashes[f] = sha256(join(runDir, f));
		}
		if (!existsSync(join(runDir, "valid.jsonl"))) throw new Error("Dataset dir has no valid.jsonl (mlx needs it for val loss).");
		return { rows: total, shape: shape ?? "text", dropped: 0, hashes };
	}

	let raw: string;
	try {
		raw = readFileSync(datasetPath, "utf8");
	} catch {
		throw new Error(`Dataset not found: ${datasetPath}`);
	}
	const rows = raw.split("\n").map((l) => l.trim()).filter(Boolean);
	if (rows.length === 0) throw new Error("Dataset is empty.");
	let shape: RowShape = "text";
	rows.forEach((line, idx) => {
		let o: unknown;
		try { o = JSON.parse(line); } catch { throw new Error(`Dataset line ${idx + 1} is not valid JSON.`); }
		if (!validRow(o)) throw new Error(`Dataset line ${idx + 1}: unsupported shape — ${SHAPES}.`);
		if (idx === 0) shape = rowShape(o);
	});
	// Drop exact duplicates before splitting — a duplicated row landing in both train and test
	// inflates the tuned score for free, which is exactly how a headline number becomes fiction.
	const seen = new Set<string>();
	const uniq = rows.filter((l) => (seen.has(l) ? false : (seen.add(l), true)));
	const dropped = rows.length - uniq.length;
	// Seeded shuffle before splitting: the hold-out isn't the top of the file (ordered datasets
	// would get an unrepresentative split), and the same seed reproduces the same split.
	const rand = mulberry32(seed);
	for (let i = uniq.length - 1; i > 0; i--) {
		const j = Math.floor(rand() * (i + 1));
		[uniq[i], uniq[j]] = [uniq[j]!, uniq[i]!];
	}
	// 3-way split: valid (for during-training eval) + test (held out for the benchmarker),
	// each ≥1 row when the data allows; train gets the rest. MLX reads valid.jsonl during
	// training and test.jsonl for `--test`.
	const hold = Math.max(1, Math.floor(uniq.length * valSplit));
	const nVal = Math.min(hold, uniq.length - 1);
	const nTest = Math.min(hold, uniq.length - nVal - 1);
	const valid = uniq.slice(0, nVal);
	const test = uniq.slice(nVal, nVal + nTest);
	const train = uniq.slice(nVal + nTest);
	writeFileSync(join(runDir, "train.jsonl"), train.join("\n") + "\n");
	writeFileSync(join(runDir, "valid.jsonl"), valid.join("\n") + "\n");
	if (test.length) writeFileSync(join(runDir, "test.jsonl"), test.join("\n") + "\n");
	const hashes: Record<string, string> = {};
	for (const f of SPLIT_FILES) if (existsSync(join(runDir, f))) hashes[f] = sha256(join(runDir, f));
	return { rows: uniq.length, shape, dropped, hashes };
}

/** Format a number as a plain decimal — YAML 1.1 parses `1e-5` as a string, not a float. */
const plain = (n: number) => n.toLocaleString("en-US", { useGrouping: false, maximumFractionDigits: 12 });

/** Checkpoints are only useful if they land on the same iters we measure val loss at. */
export const STEPS_PER_EVAL = 50;

/** Write mlx_lm.lora's YAML config into runDir and return its path. */
function writeMlxConfig(cfg: TrainConfig, runDir: string, shape: RowShape): string {
	const p = join(runDir, "mlx_config.yaml");
	const lr = cfg.learningRate ?? 1e-5;
	const iters = cfg.iters ?? 100;
	const body = [
		`model: "${cfg.baseModel}"`,
		`train: true`,
		`data: "${runDir}"`,
		`adapter_path: "${join(runDir, "adapters")}"`,
		`fine_tune_type: lora`,
		`num_layers: ${cfg.numLayers ?? 16}`,
		`batch_size: ${cfg.batchSize ?? 4}`,
		`iters: ${iters}`,
		`learning_rate: ${plain(lr)}`,
		`max_seq_length: ${cfg.maxSeqLen ?? 2048}`,
		`seed: ${cfg.seed ?? 42}`,
		`steps_per_report: 10`,
		`steps_per_eval: ${STEPS_PER_EVAL}`,
		// Checkpoint at every eval so best-checkpoint selection has something to promote.
		`save_every: ${STEPS_PER_EVAL}`,
		// Loss on the answer only. mlx RAISES on `text` rows, so emit it only when maskable —
		// without it training optimises prompt tokens while mlx_eval.py scores completions.
		...(shape === "text" ? [] : [`mask_prompt: true`]),
		...(cfg.warmupSteps
			? [`lr_schedule:`, `  name: cosine_decay`, `  warmup: ${cfg.warmupSteps}`,
			   `  arguments: [${plain(lr)}, ${iters}, ${plain(lr * 0.1)}]`]
			: []),
		`lora_parameters:`,
		`  rank: ${cfg.loraRank ?? 8}`,
		`  scale: ${plain(cfg.loraScale ?? 20.0)}`,
		`  dropout: 0.0`,
		"",
	].join("\n");
	writeFileSync(p, body);
	return p;
}

/**
 * Promote the lowest-val-loss checkpoint to adapters.safetensors.
 *
 * mlx loads `adapter_path/adapters.safetensors` — the FINAL iteration — and ignores the numbered
 * checkpoints it wrote. Every 300-iter run in this repo bottomed out at iter 150 and got worse
 * after, so the final adapter was measurably worse than one already on disk. Returns the promoted
 * iter, or null when the final checkpoint already was the best.
 */
export function selectBestCheckpoint(adapterDir: string, lossCurve: LossPoint[]): { iter: number; valLoss: number } | null {
	const vals = lossCurve.filter((p) => p.valLoss != null) as Array<{ iter: number; valLoss: number }>;
	if (vals.length < 2) return null;
	const best = vals.reduce((a, b) => (b.valLoss < a.valLoss ? b : a));
	const last = vals[vals.length - 1]!;
	if (best.iter === last.iter) return null; // final already best — nothing to promote
	const ckpt = join(adapterDir, `${String(best.iter).padStart(7, "0")}_adapters.safetensors`);
	if (!existsSync(ckpt)) return null; // eval cadence didn't line up with save cadence
	copyFileSync(ckpt, join(adapterDir, "adapters.safetensors"));
	return { iter: best.iter, valLoss: best.valLoss };
}

function buildCommand(cfg: TrainConfig, backend: Backend, runDir: string, python: string, shape: RowShape): { cmd: string; args: string[] } {
	if (backend === "mlx") {
		// mlx_lm.lora is the long-standing module entry; a version rename is a one-line fix.
		return { cmd: python, args: ["-m", "mlx_lm.lora", "--config", writeMlxConfig(cfg, runDir, shape)] };
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
	stopRequested = false; // a new run clears any Stop left over from the previous one
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
		const staged = stageDataset(cfg.datasetPath, runDir, cfg.valSplit, cfg.seed ?? 42);
		meta.datasetRows = staged.rows;
		meta.dataHashes = staged.hashes;
		emit({ type: "train-log", line: `▸ Dataset: ${staged.rows} rows staged (${staged.shape})${staged.dropped ? `, ${staged.dropped} duplicate rows dropped` : ""} → ${runDir}` });
		if (staged.rows < 1000) emit({ type: "train-log", line: `⚠ Only ${staged.rows} rows — the guide recommends ≥1000 for effective fine-tuning.` });
		const python = await ensureVenv(backend, (line) => emit({ type: "train-log", line }));

		const { cmd, args } = buildCommand(cfg, backend, runDir, python, staged.shape);
		emit({ type: "train-log", line: `▸ ${cmd} ${args.join(" ")}\n` });

		// Fire-and-forget: resolve to the caller once spawned; completion arrives via events + meta.
		runManaged(cmd, args, runDir, (line) => {
			emit({ type: "train-log", line });
			const p = parseProgress(backend, line);
			if (p) { meta.lossCurve.push(p); emit({ type: "train-progress", ...p }); }
		}).then((code) => {
			meta.finishedAt = new Date().toISOString();
			// A SIGINT stop (code 130 / null→0) still leaves usable checkpoints, so treat any run
			// that produced an adapter as done rather than flagging a deliberate stop as failure.
			const haveAdapter = existsSync(join(meta.adapterDir, "adapters.safetensors"));
			if (code === 0 || haveAdapter) {
				if (backend === "mlx") {
					const best = selectBestCheckpoint(meta.adapterDir, meta.lossCurve);
					if (best) {
						meta.selectedIter = best.iter;
						meta.bestValLoss = best.valLoss;
						emit({ type: "train-log", line: `▸ Promoted best checkpoint: iter ${best.iter} (val ${best.valLoss.toFixed(3)}) → adapters.safetensors` });
					}
				}
				meta.status = "done"; writeMeta(meta); emit({ type: "train-done", adapterDir: meta.adapterDir });
			}
			else { meta.status = "error"; meta.error = `Trainer exited with code ${code}.`; writeMeta(meta); emit({ type: "train-error", message: meta.error }); }
		});
	} finally {
		// child (if it spawned) now reflects "busy"; if setup threw, this clears the flag.
		pending = false;
	}
}
