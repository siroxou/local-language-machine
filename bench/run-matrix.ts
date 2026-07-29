// Multi-model fine-tuning benchmark: trains every cell of a model matrix on the SAME data with
// the SAME hyperparameters, scores each on the same suites, and writes one comparison report.
//
// Runs as a standalone process, not from the UI: a multi-hour sweep must survive the browser
// being closed, and driving it through the app would hold the app-wide busy lock the whole time.
//
// Resumable — state.json is written after every cell, so re-running skips completed work.
//
//   npx tsx bench/run-matrix.ts [--config bench/matrix.json] [--out bench/results] [--dry]
//
// No peak-RSS column: wall clock, adapter size and selected iter answer the question and are free,
// whereas sampling child RSS would mean plumbing the pid out of runManaged. Worth adding only if
// memory headroom ever becomes the thing being decided.

import { readFileSync, writeFileSync, mkdirSync, existsSync, statSync, rmSync, readdirSync } from "node:fs";
import { join, isAbsolute, resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { startTraining, type TrainConfig } from "../src/native/training/train.ts";
import { score } from "../src/native/training/eval.ts";
import { gradeAnswer, gradingKeys } from "../src/native/training/eval.ts";
import { ensureVenv } from "../src/native/training/venv.ts";
import { readMeta, runDirOf } from "../src/native/training/runs.ts";
import { appHome } from "../src/native/paths.ts";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const abs = (p: string) => (isAbsolute(p) ? p : join(ROOT, p));

interface Cell { id: string; family: string; params: number; baseModel: string; overrides?: Partial<TrainConfig>; replicate?: boolean }
interface Matrix { dataset: string; suites: Record<string, string>; shared: Partial<TrainConfig>; cells: Cell[] }
interface SuiteScore { mcq: number | null; acc: number; abstain: number; n: number; bpb: number | null; ppl: number }
interface CellResult {
	id: string; family: string; params: number; baseModel: string;
	trainSeconds: number; selectedIter?: number; bestValLoss?: number; adapterMB: number; runId: string;
	base: Record<string, SuiteScore>; tuned: Record<string, SuiteScore>;
	error?: string;
}

const args = process.argv.slice(2);
const argOf = (flag: string, dflt: string) => { const i = args.indexOf(flag); return i >= 0 && args[i + 1] ? args[i + 1]! : dflt; };
const DRY = args.includes("--dry");
const CONFIG = abs(argOf("--config", "bench/matrix.json"));
const OUT = abs(argOf("--out", "bench/results"));
const STATE = join(OUT, "state.json");
const LOCK = join(appHome(), "bench.lock");

const log = (s: string) => console.log(s);
const quiet = () => {}; // per-line trainer output is far too noisy for an 11-cell sweep

/** Two MLX processes at once would poison every timing number, so guard across processes too. */
function takeLock() {
	if (existsSync(LOCK)) {
		const pid = +readFileSync(LOCK, "utf8").trim();
		try { process.kill(pid, 0); throw new Error(`Another bench run is active (pid ${pid}). Remove ${LOCK} if it is stale.`); }
		catch (e: any) { if (e?.code !== "ESRCH") throw e; } // ESRCH = stale lock, take it
	}
	mkdirSync(dirname(LOCK), { recursive: true });
	writeFileSync(LOCK, String(process.pid));
}
const releaseLock = () => { try { rmSync(LOCK); } catch {} };

/** startTraining resolves once the child SPAWNS, so completion has to be polled out of meta. */
async function awaitRun(runId: string, timeoutMs = 6 * 3600_000): Promise<void> {
	const t0 = Date.now();
	for (;;) {
		const m = readMeta(runId);
		if (m && m.status !== "running") {
			if (m.status === "error") throw new Error(m.error || "training failed");
			return;
		}
		if (Date.now() - t0 > timeoutMs) throw new Error("training timed out");
		await new Promise((r) => setTimeout(r, 4000));
	}
}

/** Latest run dir — startTraining doesn't return the runId, so take the newest after it spawns. */
function newestRunId(since: number): string | null {
	const root = join(appHome(), "adapters");
	if (!existsSync(root)) return null;
	const dirs = readdirSync(root);
	let best: { id: string; t: number } | null = null;
	for (const d of dirs) {
		const t = statSync(join(root, d)).birthtimeMs;
		if (t >= since && (!best || t > best.t)) best = { id: d, t };
	}
	return best?.id ?? null;
}

function summarise(r: Awaited<ReturnType<typeof score>>, suiteFile: string): SuiteScore {
	const keys = gradingKeys(suiteFile);
	let right = 0, abstain = 0, n = 0;
	for (const g of r.generations ?? []) {
		const k = keys[g.i];
		if (!k) continue;
		n++;
		const grade = gradeAnswer(g.text, k.answer, k.distractors);
		if (grade === "correct") right++;
		if (grade === "abstain") abstain++;
	}
	return {
		mcq: r.mcq ? r.mcq.acc : null,
		acc: n ? +(right / n).toFixed(4) : 0,
		abstain: n ? +(abstain / n).toFixed(4) : 0,
		n, bpb: r.bitsPerByte, ppl: r.ppl,
	};
}

async function main() {
	const mx: Matrix = JSON.parse(readFileSync(CONFIG, "utf8"));
	const dataDir = abs(mx.dataset);
	if (!existsSync(join(dataDir, "train.jsonl"))) throw new Error(`No dataset at ${dataDir} — run: npx tsx bench/gen-dataset.ts`);
	mkdirSync(OUT, { recursive: true });

	const done: Record<string, CellResult> = existsSync(STATE) ? JSON.parse(readFileSync(STATE, "utf8")) : {};
	const todo = mx.cells.filter((c) => !done[c.id]);
	log(`▸ ${mx.cells.length} cells · ${todo.length} to run · ${Object.keys(done).length} already done`);
	log(`▸ dataset ${dataDir}`);
	if (DRY) { for (const c of mx.cells) log(`   ${done[c.id] ? "done" : "todo"}  ${c.id}  ${c.baseModel}`); return; }

	takeLock();
	const python = await ensureVenv("mlx", quiet);

	for (const [i, cell] of mx.cells.entries()) {
		if (done[cell.id]) { log(`[${i + 1}/${mx.cells.length}] ${cell.id} — cached`); continue; }
		const label = `[${i + 1}/${mx.cells.length}] ${cell.id}`;
		try {
			const cfg: TrainConfig = { ...mx.shared, ...cell.overrides, baseModel: cell.baseModel, datasetPath: dataDir };
			log(`${label} training ${cell.baseModel} …`);
			const t0 = Date.now();
			const since = Date.now() - 1000;
			await startTraining(cfg, quiet as any);
			const runId = newestRunId(since);
			if (!runId) throw new Error("could not resolve runId");
			await awaitRun(runId);
			const trainSeconds = Math.round((Date.now() - t0) / 1000);
			const meta = readMeta(runId)!;
			const adapter = join(runDirOf(runId), "adapters", "adapters.safetensors");
			const adapterMB = existsSync(adapter) ? +(statSync(adapter).size / 1048576).toFixed(2) : 0;
			log(`${label} trained in ${trainSeconds}s · best iter ${meta.selectedIter ?? "final"} (val ${meta.bestValLoss?.toFixed(3) ?? "?"})`);

			// All suites behind ONE model load per variant — loading dominates, especially at 7B.
			const present = Object.entries(mx.suites)
				.map(([suite, file]) => ({ suite, f: join(runDirOf(runId), file) }))
				.filter(({ suite, f }) => existsSync(f) || (log(`${label}   suite ${suite} missing — skipped`), false));
			const files = present.map((p) => p.f);
			const spread = (r: Awaited<ReturnType<typeof score>>) =>
				Object.fromEntries(present.map(({ suite, f }) => [suite, summarise(r.suites ? r.suites[f]! : r, f)]));
			const base = spread(await score(python, cell.baseModel, null, files, quiet));
			const tuned = spread(await score(python, cell.baseModel, meta.adapterDir, files, quiet));
			for (const { suite } of present) {
				const b = base[suite]!, t = tuned[suite]!;
				log(`${label}   ${suite.padEnd(6)} mcq ${fmtPct(b.mcq)}→${fmtPct(t.mcq)}  acc ${fmtPct(b.acc)}→${fmtPct(t.acc)}  (n=${t.n})`);
			}
			done[cell.id] = { id: cell.id, family: cell.family, params: cell.params, baseModel: cell.baseModel, trainSeconds, selectedIter: meta.selectedIter, bestValLoss: meta.bestValLoss, adapterMB, runId, base, tuned };
		} catch (e) {
			const msg = e instanceof Error ? e.message : String(e);
			log(`${label} FAILED: ${msg}`);
			done[cell.id] = { id: cell.id, family: cell.family, params: cell.params, baseModel: cell.baseModel, trainSeconds: 0, adapterMB: 0, runId: "", base: {}, tuned: {}, error: msg };
		}
		writeFileSync(STATE, JSON.stringify(done, null, 2)); // checkpoint after EVERY cell
	}

	releaseLock();
	const results = mx.cells.map((c) => done[c.id]!).filter(Boolean);
	writeFileSync(join(OUT, "experiment.json"), JSON.stringify({ config: mx, results, generatedAt: new Date().toISOString() }, null, 2));
	writeFileSync(join(OUT, "report.md"), report(mx, results));
	log(`\n▸ ${join(OUT, "report.md")}`);
	log(report(mx, results));
}

const fmtPct = (v: number | null | undefined) => (v == null ? "—" : `${(v * 100).toFixed(0)}%`);
const d = (b: number | null | undefined, t: number | null | undefined) => (b == null || t == null ? "—" : `${((t - b) * 100).toFixed(0)}pt`);

function report(mx: Matrix, rs: CellResult[]): string {
	const ok = rs.filter((r) => !r.error);
	const L: string[] = [];
	L.push(`# Fine-tuning across model size and family\n`);
	L.push(`Dataset: \`${mx.dataset}\` — a fully invented domain, so base accuracy is chance by construction.`);
	L.push(`Shared config: ${JSON.stringify(mx.shared)}\n`);

	L.push(`## Primary endpoint — MCQ on MEM (trained facts, unseen phrasing)\n`);
	L.push(`| model | family | params (B) | base | tuned | gain | n |`);
	L.push(`|---|---|---:|---:|---:|---:|---:|`);
	for (const r of ok) {
		const b = r.base.mem, t = r.tuned.mem;
		L.push(`| ${r.id} | ${r.family} | ${r.params} | ${fmtPct(b?.mcq)} | ${fmtPct(t?.mcq)} | ${d(b?.mcq, t?.mcq)} | ${t?.n ?? "—"} |`);
	}

	L.push(`\n## All suites (tuned MCQ, gain over base)\n`);
	const suites = Object.keys(mx.suites);
	L.push(`| model | ${suites.join(" | ")} |`);
	L.push(`|---|${suites.map(() => "---:").join("|")}|`);
	for (const r of ok) {
		L.push(`| ${r.id} | ${suites.map((s) => `${fmtPct(r.tuned[s]?.mcq)} (${d(r.base[s]?.mcq, r.tuned[s]?.mcq)})`).join(" | ")} |`);
	}

	L.push(`\n## Cost and fit\n`);
	L.push(`| model | train (s) | best iter | best val | adapter (MB) | bits/byte base→tuned |`);
	L.push(`|---|---:|---:|---:|---:|---:|`);
	for (const r of ok) {
		const b = r.base.mem?.bpb, t = r.tuned.mem?.bpb;
		L.push(`| ${r.id} | ${r.trainSeconds} | ${r.selectedIter ?? "final"} | ${r.bestValLoss?.toFixed(3) ?? "—"} | ${r.adapterMB} | ${b?.toFixed(3) ?? "—"} → ${t?.toFixed(3) ?? "—"} |`);
	}

	const reps = ok.filter((r) => mx.cells.find((c) => c.id === r.id)?.replicate || r.id.startsWith("smollm2-135m"));
	if (reps.length > 1) {
		const vals = reps.map((r) => r.tuned.mem?.mcq).filter((v): v is number => v != null);
		if (vals.length > 1) {
			const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
			const sd = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / (vals.length - 1));
			L.push(`\n## Run-to-run noise floor\n`);
			L.push(`Same model and data, ${vals.length} seeds: MCQ-MEM ${vals.map((v) => fmtPct(v)).join(", ")} — SD ${(sd * 100).toFixed(1)}pt.`);
			L.push(`\n**Treat any cross-model gap smaller than ~${(2 * sd * 100).toFixed(0)}pt as noise.**`);
		}
	}

	L.push(`\n## Reading these numbers\n`);
	L.push(`- **MCQ is the headline.** It is teacher-forced, so it cannot be skewed by how verbose a model is. Free-text accuracy is reported too but is verbosity-sensitive.`);
	L.push(`- **Per-token perplexity is NOT comparable across families** — it depends on the tokenizer. Bits-per-byte uses the same denominator for every model and is the safe cross-model likelihood number.`);
	L.push(`- **UNSEEN measures hallucination, not knowledge.** Those compounds were never trained, so accuracy there is chance; what matters is whether the abstain rate collapsed after tuning.`);
	L.push(`- **COMP is composition** — compound→class and class→antidote are trained separately and the direct link never appears in training.`);
	L.push(`- **Size is confounded with LR-optimality.** mlx applies the LoRA \`scale\` as a raw multiplier and inits \`lora_a\` with 1/√input_dims, so one fixed learning rate is not equally good across a 55× width span.`);
	L.push(`- **Families are only cleanly comparable Qwen2.5 ↔ Falcon3**; SmolLM2 tops out at 1.7B.`);
	L.push(`- **Training time is serial and thermal** — later cells ran on a hotter machine.`);
	const failed = rs.filter((r) => r.error);
	if (failed.length) L.push(`\n## Failed cells\n` + failed.map((r) => `- \`${r.id}\`: ${r.error}`).join("\n"));
	return L.join("\n") + "\n";
}

main().catch((e) => { releaseLock(); console.error(e); process.exit(1); });
