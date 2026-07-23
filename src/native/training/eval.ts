// Local Language Machine — benchmarker / evaluation (real data, base vs fine-tuned).
//
// For a completed run, measures the base model vs the tuned model (base + LoRA adapter):
//   • held-out perplexity  (mlx_lm.lora --test → Test loss → ppl = exp(loss))
//   • generation speed     (mlx_lm.generate → tokens-per-sec)
//   • sample A/B outputs   (same prompts through both)
// Results persist to the run's meta.eval and feed the SVG charts in the UI. MLX-native;
// the exact flags mirror mlx-lm's CLI (a version rename is a one-line fix, like train.ts).

import { existsSync, copyFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { appHome } from "../paths.ts";
import { ensureVenv } from "./venv.ts";
import { runManaged, withBusy } from "./train.ts";
import { readMeta, writeMeta, runDirOf, type EvalResult } from "./runs.ts";

type Emit = (o: any) => void;
const log = (emit: Emit, line: string) => emit({ type: "train-log", line });

const EVAL_PROMPTS = [
	"Give three tips for staying healthy.",
	"Explain what a black hole is in one sentence.",
	"Write a short poem about the ocean.",
];

/** `Test loss 1.234, Test ppl 3.456` → {loss, ppl}. Falls back to ppl = exp(loss). */
export function parseTestLoss(line: string): { loss: number; ppl: number } | null {
	const m = line.match(/Test loss ([\d.]+)/i);
	if (!m) return null;
	const loss = +m[1]!;
	const p = line.match(/Test ppl ([\d.]+)/i);
	return { loss, ppl: p ? +p[1]! : +Math.exp(loss).toFixed(3) };
}

/** `Generation: 128 tokens, 80.000 tokens-per-sec` → 80.0 */
export function parseGenSpeed(line: string): number | null {
	const m = line.match(/Generation:\s*\d+\s*tokens?,\s*([\d.]+)\s*tokens-per-sec/i);
	return m ? +m[1]! : null;
}

// Our own eval script — mlx_lm.lora --test can't score the base model (it always loads an
// adapter). `adapter === null` → base perplexity; a path → tuned.
const EVAL_SCRIPT = fileURLToPath(new URL("./mlx_eval.py", import.meta.url)); // fileURLToPath, not .pathname: decodes spaces in the path
async function perplexity(python: string, base: string, adapter: string | null, testFile: string, emit: Emit): Promise<{ loss: number; ppl: number }> {
	const args = [EVAL_SCRIPT, "--model", base, "--data", testFile];
	if (adapter) args.push("--adapter-path", adapter);
	let out: { loss: number; ppl: number } | null = null;
	const code = await runManaged(python, args, appHome(), (line) => { log(emit, line); const r = parseTestLoss(line); if (r) out = r; });
	if (code !== 0) throw new Error(`Perplexity eval failed (exit ${code}).`);
	if (!out) throw new Error("Could not read test loss from mlx output.");
	return out;
}

async function generate(python: string, base: string, adapter: string | null, prompt: string, emit: Emit): Promise<{ text: string; tps?: number }> {
	const args = ["-m", "mlx_lm.generate", "--model", base, "--max-tokens", "64", "--prompt", prompt];
	if (adapter) args.push("--adapter-path", adapter);
	let capturing = false;
	const textLines: string[] = [];
	let tps: number | undefined;
	const code = await runManaged(python, args, appHome(), (line) => {
		log(emit, line);
		if (/^=+$/.test(line.trim())) { capturing = !capturing; return; } // mlx brackets the output in ==========
		if (capturing) { textLines.push(line); return; }
		const s = parseGenSpeed(line); if (s != null) tps = s;
	});
	if (code !== 0) throw new Error(`Generation failed (exit ${code}).`);
	return { text: textLines.join("\n").trim(), tps };
}

/** mlx `--test` reads test.jsonl; legacy runs only have valid.jsonl → use it as the held-out set. */
function ensureTestData(runDir: string): void {
	const test = join(runDir, "test.jsonl");
	if (existsSync(test)) return;
	const valid = join(runDir, "valid.jsonl");
	if (existsSync(valid)) { copyFileSync(valid, test); return; }
	throw new Error("No held-out data found for this run (no test.jsonl / valid.jsonl).");
}

/** Run the full benchmark for a run and persist meta.eval. Fire-and-forget from the server. */
export function runEval(runId: string, emit: Emit): Promise<void> {
	return withBusy(async () => {
		const meta = readMeta(runId);
		if (!meta) throw new Error(`Unknown run: ${runId}`);
		if (!meta.baseModel) throw new Error("This run has no recorded base model — retrain to evaluate.");
		if (!existsSync(meta.adapterDir)) throw new Error("Adapter not found for this run.");
		const python = await ensureVenv(meta.backend, (l) => log(emit, l));
		const runDir = runDirOf(runId);
		ensureTestData(runDir);

		const testFile = join(runDir, "test.jsonl");
		emit({ type: "eval-progress", stage: "Perplexity — base model" });
		const ppBase = await perplexity(python, meta.baseModel, null, testFile, emit);
		emit({ type: "eval-progress", stage: "Perplexity — fine-tuned" });
		const ppTuned = await perplexity(python, meta.baseModel, meta.adapterDir, testFile, emit);

		const samples: EvalResult["samples"] = [];
		const sBase: number[] = [], sTuned: number[] = [];
		for (const p of EVAL_PROMPTS) {
			emit({ type: "eval-progress", stage: `Generating — ${p.slice(0, 32)}…` });
			const b = await generate(python, meta.baseModel, null, p, emit);
			const t = await generate(python, meta.baseModel, meta.adapterDir, p, emit);
			samples!.push({ prompt: p, base: b.text, tuned: t.text });
			if (b.tps) sBase.push(b.tps);
			if (t.tps) sTuned.push(t.tps);
		}
		const avg = (a: number[]) => (a.length ? +(a.reduce((x, y) => x + y, 0) / a.length).toFixed(1) : 0);

		const result: EvalResult = {
			perplexity: { base: ppBase.ppl, tuned: ppTuned.ppl },
			speed: { base: avg(sBase), tuned: avg(sTuned) },
			samples,
			evaluatedAt: new Date().toISOString(),
		};
		meta.eval = result;
		writeMeta(meta);
		emit({ type: "eval-done", runId, result, lossCurve: meta.lossCurve });
	}).catch((e) => emit({ type: "eval-error", message: e instanceof Error ? e.message : String(e) }));
}
