// Local Language Machine — benchmarker / evaluation (real data, base vs fine-tuned).
//
// For a completed run, measures the base model vs the tuned model (base + LoRA adapter) over the
// whole held-out set, from ONE python process per variant (loading the model dominates the cost):
//   • completion-only perplexity + bits-per-byte  (bpb is the cross-model-safe likelihood number)
//   • free-text accuracy via the closed-set grader
//   • multiple-choice accuracy (teacher-forced) — the headline cross-model metric
//   • generation speed + sample A/B outputs      (diagnostics)
// Results persist to the run's meta.eval and feed the SVG charts in the UI.

import { existsSync, copyFileSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { appHome } from "../paths.ts";
import { ensureVenv } from "./venv.ts";
import { runManaged, withBusy } from "./train.ts";
import { readMeta, writeMeta, runDirOf, type EvalResult } from "./runs.ts";

type Emit = (o: any) => void;
const log = (emit: Emit, line: string) => emit({ type: "train-log", line });

/** The single structured line mlx_eval.py emits. Parsed instead of scraping delimiters. */
export interface ScoreResult {
	loss: number;
	ppl: number;
	bitsPerByte: number | null;
	rows: number;
	tokens: number;
	bytes: number;
	mcq?: { correct: number; n: number; acc: number };
	generations?: Array<{ i: number; prompt: string; reference: string; text: string }>;
	genTokensPerSec?: number;
	/** Present only when several suites were scored behind one model load, keyed by file path. */
	suites?: Record<string, ScoreResult>;
}

/** `EVAL_JSON {...}` → the payload. Survives newlines/quotes/`=` in generated text. */
export function parseEvalJson(line: string): ScoreResult | null {
	const i = line.indexOf("EVAL_JSON ");
	if (i < 0) return null;
	try { return JSON.parse(line.slice(i + "EVAL_JSON ".length)) as ScoreResult; } catch { return null; }
}

/**
 * Grade one free-text answer against a closed set of candidates.
 *
 * Every answer in a well-formed eval row is a member of a known closed set, so we word-boundary
 * scan the output for `{answer} ∪ distractors` rather than normalising and substring-matching.
 * Normalise-and-contain is subtly broken on exactly this kind of data: stripping articles makes
 * "Hepatitis A" match "Hepatitis B", and stripping punctuation makes "12.5 mg" contain "25 mg".
 * Scanning a closed set has no undefined state and no such collisions.
 */
export type Grade = "correct" | "wrong" | "abstain" | "ambiguous";
export function gradeAnswer(output: string, answer: string, distractors: string[] = []): Grade {
	const hay = output.toLowerCase();
	const hit = (c: string) => {
		const t = c.trim().toLowerCase();
		if (!t) return false;
		// \b is wrong next to "." or "%" — bound on non-word chars we control instead.
		const re = new RegExp(`(^|[^a-z0-9])${t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}([^a-z0-9]|$)`);
		return re.test(hay);
	};
	const gold = hit(answer);
	// A distractor that is a substring of the gold answer isn't independently present.
	const bad = distractors.filter((d) => d.trim() && d.trim().toLowerCase() !== answer.trim().toLowerCase())
		.filter((d) => hit(d) && !answer.toLowerCase().includes(d.trim().toLowerCase()));
	if (gold && bad.length === 0) return "correct";
	if (gold && bad.length) return "ambiguous"; // hedged or contradicted itself
	if (bad.length) return "wrong";
	return "abstain";
}

// Our own eval script — mlx_lm.lora --test can't score the base model (it always loads an
// adapter). `adapter === null` → base, a path → tuned. One process does perplexity + MCQ +
// generation behind a single load(), since loading the model dominates the cost.
const EVAL_SCRIPT = fileURLToPath(new URL("./mlx_eval.py", import.meta.url)); // fileURLToPath, not .pathname: decodes spaces in the path
export async function score(python: string, base: string, adapter: string | null, testFile: string | string[], emit: Emit, generate = true): Promise<ScoreResult> {
	const args = [EVAL_SCRIPT, "--model", base, "--data", Array.isArray(testFile) ? testFile.join(",") : testFile, "--limit", "0"];
	if (generate) args.push("--generate");
	if (adapter) args.push("--adapter-path", adapter);
	let out: ScoreResult | null = null;
	const code = await runManaged(python, args, appHome(), (line) => {
		const r = parseEvalJson(line);
		if (r) { out = r; return; }   // don't echo the payload into the log
		log(emit, line);
	});
	if (code !== 0) throw new Error(`Eval failed (exit ${code}).`);
	if (!out) throw new Error("Could not read EVAL_JSON from mlx output.");
	return out;
}

/**
 * Per-row answer key for grading, indexed the same way mlx_eval.py indexes its rows.
 * Rows without an explicit `answer` fall back to the reference completion with no distractors,
 * which degrades gracefully to "did the answer string appear".
 */
export function gradingKeys(testFile: string): Record<number, { answer: string; distractors: string[] }> {
	const out: Record<number, { answer: string; distractors: string[] }> = {};
	let lines: string[];
	try { lines = readFileSync(testFile, "utf8").split("\n").map((l) => l.trim()).filter(Boolean); }
	catch { return out; }
	lines.forEach((line, i) => {
		let row: any;
		try { row = JSON.parse(line); } catch { return; }
		const distractors: string[] = Array.isArray(row.distractors) ? row.distractors.map(String)
			: Array.isArray(row.choices) ? row.choices.map(String) : [];
		let answer: string | undefined = typeof row.answer === "string" ? row.answer : undefined;
		if (!answer) {
			if (Array.isArray(row.messages)) {
				const a = [...row.messages].reverse().find((m: any) => m.role === "assistant");
				if (a?.content) answer = String(a.content);
			} else if (typeof row.completion === "string") answer = row.completion;
		}
		if (answer) out[i] = { answer, distractors };
	});
	return out;
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
		// One process per variant: perplexity + MCQ + generation all behind a single model load.
		emit({ type: "eval-progress", stage: "Scoring — base model" });
		const base = await score(python, meta.baseModel, null, testFile, emit);
		emit({ type: "eval-progress", stage: "Scoring — fine-tuned" });
		const tuned = await score(python, meta.baseModel, meta.adapterDir, testFile, emit);

		// Grade free text against each row's own closed answer set.
		const keyed = gradingKeys(testFile);
		const acc = (r: ScoreResult) => {
			let right = 0, n = 0;
			for (const g of r.generations ?? []) {
				const k = keyed[g.i];
				if (!k) continue;
				n++;
				if (gradeAnswer(g.text, k.answer, k.distractors) === "correct") right++;
			}
			return { right, n };
		};
		const aBase = acc(base), aTuned = acc(tuned);

		const samples: EvalResult["samples"] = (tuned.generations ?? []).slice(0, 5).map((g) => {
			const b = (base.generations ?? []).find((x) => x.i === g.i);
			const k = keyed[g.i];
			return {
				prompt: g.prompt, base: b?.text ?? "", tuned: g.text, reference: g.reference,
				...(k ? { baseGrade: gradeAnswer(b?.text ?? "", k.answer, k.distractors), tunedGrade: gradeAnswer(g.text, k.answer, k.distractors) } : {}),
			};
		});

		const result: EvalResult = {
			perplexity: { base: base.ppl, tuned: tuned.ppl },
			...(base.bitsPerByte != null && tuned.bitsPerByte != null
				? { bitsPerByte: { base: base.bitsPerByte, tuned: tuned.bitsPerByte } } : {}),
			...(aTuned.n ? { accuracy: { base: +(aBase.right / (aBase.n || 1)).toFixed(4), tuned: +(aTuned.right / aTuned.n).toFixed(4), n: aTuned.n } } : {}),
			...(base.mcq && tuned.mcq ? { mcq: { base: base.mcq.acc, tuned: tuned.mcq.acc, n: tuned.mcq.n } } : {}),
			scored: { rows: tuned.rows, tokens: tuned.tokens },
			speed: { base: base.genTokensPerSec ?? 0, tuned: tuned.genTokensPerSec ?? 0 },
			samples,
			evaluatedAt: new Date().toISOString(),
		};
		meta.eval = result;
		writeMeta(meta);
		emit({ type: "eval-done", runId, result, lossCurve: meta.lossCurve });
	}).catch((e) => emit({ type: "eval-error", message: e instanceof Error ? e.message : String(e) }));
}
