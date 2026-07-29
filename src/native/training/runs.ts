// Local Language Machine — training-run metadata + history, and HuggingFace dataset import.
//
// Each run gets a meta.json under ~/.local-language-machine/adapters/<runId>/ recording its
// config, live loss curve, status, and (later) eval results. listRuns() powers the Runs
// history tab. fetchHfDataset() promotes the manual "convert a HF dataset to jsonl" step
// (done by hand during v1 testing) into a first-class feature.

import { readFileSync, writeFileSync, readdirSync, existsSync, statSync } from "node:fs";
import { join } from "node:path";
import { appHome } from "../paths.ts";
import { getHfToken } from "../models/config.ts";
import type { TrainConfig, Backend } from "./train.ts";

export interface LossPoint { iter: number; loss?: number; valLoss?: number }

export interface EvalResult {
	/** Per-token perplexity. Tokenizer-dependent — compare WITHIN a model, never across families. */
	perplexity?: { base: number; tuned: number };
	/** Bits per answer byte. Same denominator for every tokenizer, so safe to compare across models. */
	bitsPerByte?: { base: number; tuned: number };
	/** Free-text accuracy from the closed-set grader, over the whole held-out set. */
	accuracy?: { base: number; tuned: number; n: number };
	/** Teacher-forced multiple choice — immune to verbosity, the headline cross-model number. */
	mcq?: { base: number; tuned: number; n: number };
	/** Rows and tokens actually scored, so a number is never quoted without its n. */
	scored?: { rows: number; tokens: number };
	speed?: { base: number; tuned: number }; // tokens/sec — diagnostic only (thermals, batch occupancy)
	samples?: Array<{ prompt: string; base: string; tuned: string; reference?: string; baseGrade?: string; tunedGrade?: string }>;
	evaluatedAt?: string;
}

export type RunStatus = "running" | "done" | "error";

export interface RunMeta {
	runId: string;
	backend: Backend;
	baseModel: string;
	config: TrainConfig;
	status: RunStatus;
	startedAt: string;
	finishedAt?: string;
	lossCurve: LossPoint[];
	adapterDir: string;
	fusedDir?: string;
	ggufModelId?: string;
	error?: string;
	eval?: EvalResult;
	/** Checkpoint promoted to adapters.safetensors (best val loss), and its loss. */
	selectedIter?: number;
	bestValLoss?: number;
	/** Staged-dataset provenance — proves two runs trained/scored the same rows. */
	dataHashes?: Record<string, string>;
	datasetRows?: number;
}

export function runsRoot(): string {
	return join(appHome(), "adapters");
}
export function runDirOf(runId: string): string {
	return join(runsRoot(), runId);
}
const metaPath = (runId: string) => join(runDirOf(runId), "meta.json");

export function writeMeta(meta: RunMeta): void {
	writeFileSync(metaPath(meta.runId), JSON.stringify(meta, null, 2));
}

export function readMeta(runId: string): RunMeta | undefined {
	try {
		return JSON.parse(readFileSync(metaPath(runId), "utf8")) as RunMeta;
	} catch {
		return undefined;
	}
}

/** All runs, newest first. Runs from before meta.json existed get a synthesized stub. */
export function listRuns(): RunMeta[] {
	const root = runsRoot();
	if (!existsSync(root)) return [];
	const out: RunMeta[] = [];
	for (const name of readdirSync(root)) {
		const dir = join(root, name);
		try { if (!statSync(dir).isDirectory()) continue; } catch { continue; }
		const meta = readMeta(name);
		if (meta) { out.push(meta); continue; }
		// Legacy run (no meta.json) — synthesize enough to list + evaluate it.
		const adapterDir = join(dir, "adapters");
		if (existsSync(adapterDir)) {
			out.push({
				runId: name, backend: "mlx", baseModel: "", config: {} as TrainConfig,
				status: "done", startedAt: statSync(dir).mtime.toISOString(), lossCurve: [], adapterDir,
			});
		}
	}
	return out.sort((a, b) => (b.startedAt > a.startedAt ? 1 : -1));
}

// ————— HuggingFace dataset import —————

const DS = "https://datasets-server.huggingface.co";
function authHeaders(): Record<string, string> {
	const t = getHfToken();
	return t ? { Authorization: `Bearer ${t}` } : {};
}

async function dsGet(path: string): Promise<any> {
	const res = await fetch(`${DS}${path}`, { headers: authHeaders() });
	if (res.status === 401 || res.status === 403) throw new Error("Dataset is gated/private — add your Hugging Face token.");
	if (!res.ok) throw new Error(`HuggingFace datasets error ${res.status}.`);
	return res.json();
}

/** Map one dataset row (many possible schemas) into a supported training row. */
export function mapRow(r: Record<string, any>): { messages?: unknown[]; text?: string; prompt?: string; completion?: string } | null {
	if (Array.isArray(r.messages)) return { messages: r.messages };
	if (Array.isArray(r.conversations)) {
		// ShareGPT: [{from,value}] → messages
		const messages = r.conversations.map((m: any) => ({
			role: m.from === "gpt" || m.from === "assistant" ? "assistant" : "user",
			content: String(m.value ?? m.content ?? ""),
		}));
		return { messages };
	}
	const instr = r.instruction ?? r.question ?? r.prompt;
	const resp = r.output ?? r.answer ?? r.response ?? r.completion;
	if (typeof instr === "string" && typeof resp === "string") {
		const inp = typeof r.input === "string" && r.input.trim() ? "\n\n" + r.input.trim() : "";
		return { messages: [{ role: "user", content: instr + inp }, { role: "assistant", content: resp }] };
	}
	if (typeof r.text === "string") return { text: r.text };
	// Last resort: first string field becomes `text`.
	const firstStr = Object.values(r).find((v) => typeof v === "string" && (v as string).length > 0);
	return typeof firstStr === "string" ? { text: firstStr } : null;
}

export interface HfDatasetOpts { config?: string; split?: string; limit?: number }

/**
 * Fetch rows of a HuggingFace dataset via the datasets-server, map them to the training
 * jsonl format, and write `<destDir>/datasets/<name>.jsonl`. Returns the written path.
 */
export async function fetchHfDataset(datasetId: string, opts: HfDatasetOpts, destDir: string, onLine?: (l: string) => void): Promise<{ path: string; rows: number }> {
	let { config, split } = opts;
	const limit = Math.min(Math.max(opts.limit ?? 200, 1), 2000);
	if (!config || !split) {
		const info = await dsGet(`/splits?dataset=${encodeURIComponent(datasetId)}`);
		const first = info?.splits?.[0];
		if (!first) throw new Error(`No splits found for dataset ${datasetId}.`);
		config = config ?? first.config;
		split = split ?? first.split;
		onLine?.(`▸ Using config="${config}" split="${split}"`);
	}
	const { mkdirSync, writeFileSync } = await import("node:fs");
	const outDir = join(destDir, "datasets");
	mkdirSync(outDir, { recursive: true });
	const safe = datasetId.replace(/[^\w.-]+/g, "_");
	const outPath = join(outDir, `${safe}.jsonl`);

	const lines: string[] = [];
	for (let offset = 0; offset < limit && lines.length < limit; offset += 100) {
		const len = Math.min(100, limit - offset);
		const data = await dsGet(`/rows?dataset=${encodeURIComponent(datasetId)}&config=${encodeURIComponent(config!)}&split=${encodeURIComponent(split!)}&offset=${offset}&length=${len}`);
		const rows = Array.isArray(data?.rows) ? data.rows : [];
		if (rows.length === 0) break;
		for (const rr of rows) {
			const mapped = mapRow(rr.row ?? {});
			if (mapped) lines.push(JSON.stringify(mapped));
		}
		onLine?.(`▸ Fetched ${lines.length} rows…`);
	}
	if (lines.length === 0) throw new Error("No usable rows mapped from this dataset — try a different split/config.");
	writeFileSync(outPath, lines.join("\n") + "\n");
	return { path: outPath, rows: lines.length };
}
