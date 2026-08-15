// Local Language Machine — the audit entry point.
//
//   npm run audit                     the deterministic pass over the working tree
//   npm run audit:ci                  the same, failing on anything not baselined
//   npm run audit -- --model qwen2.5-coder:7b   adds the model-driven pass
//
// The deterministic pass is the default precisely so CI cannot accidentally trigger a
// multi-gigabyte model download. The model-driven pass is opt-in and never gates.
//
// Exit codes are three, not two:
//   0  clean
//   1  findings that are not in the baseline
//   2  the AUDITOR failed its own invariants, or crashed
// Separating 1 from 2 matters: "the code is broken" and "the tool is broken" must not
// arrive as the same signal, or a broken tool reads as a broken codebase.

import { mkdirSync, writeFileSync } from "node:fs";
import { isAbsolute, join, resolve } from "node:path";
import { fingerprint, loadBaseline, partition, sortFindings, validate, type Finding } from "./findings.ts";
import { runRules } from "./rules.ts";
import { checkInvariants, openTrace, summarize, writeReport, type InvestigationStat } from "./report.ts";

const argv = process.argv.slice(2);
const has = (flag: string) => argv.includes(flag);
const argOf = (flag: string, fallback: string): string => {
	const i = argv.indexOf(flag);
	return i >= 0 && argv[i + 1] ? argv[i + 1]! : fallback;
};

const ROOT = resolve(argOf("--root", process.cwd()));
const OUT = isAbsolute(argOf("--out", "")) ? argOf("--out", "") : join(ROOT, argOf("--out", "audit/last"));
const BASELINE = isAbsolute(argOf("--baseline", "")) ? argOf("--baseline", "") : join(ROOT, argOf("--baseline", "audit/baseline.json"));
const CHECK = has("--check");
const MODEL = argv.includes("--model") ? argOf("--model", "") : null;
const WRITE_BASELINE = has("--write-baseline");

const c = {
	dim: (s: string) => `\x1b[2m${s}\x1b[0m`,
	bold: (s: string) => `\x1b[1m${s}\x1b[0m`,
	red: (s: string) => `\x1b[31m${s}\x1b[0m`,
	yellow: (s: string) => `\x1b[33m${s}\x1b[0m`,
	green: (s: string) => `\x1b[32m${s}\x1b[0m`,
};

const SEVERITY_COLOR = { high: c.red, medium: c.yellow, low: c.dim } as const;

function printFindings(label: string, findings: Finding[]): void {
	if (!findings.length) return;
	console.log(`\n${c.bold(label)}`);
	for (const f of sortFindings(findings)) {
		const sev = SEVERITY_COLOR[f.severity](f.severity.padEnd(6));
		console.log(`  ${sev} ${f.file}:${f.line}  ${c.dim(f.rule)}`);
		console.log(`         ${f.message}`);
		if (f.evidence) console.log(`         ${c.dim(f.evidence.slice(0, 120))}`);
	}
}

async function main(): Promise<number> {
	const startedAt = Date.now();
	mkdirSync(OUT, { recursive: true });
	const { trace, path: tracePath } = openTrace(OUT);

	// Read the baseline first: a malformed one is fatal regardless of findings, and failing
	// before doing the work makes the cause obvious.
	const baseline = loadBaseline(BASELINE);

	// ————— tier A: the deterministic pass —————
	const t0 = Date.now();
	const ruleFindings = await runRules(ROOT);
	trace({ phase: "rules", ms: Date.now() - t0, note: `${ruleFindings.length} finding(s) from the deterministic pass` });
	console.log(c.dim(`deterministic pass: ${ruleFindings.length} finding(s) in ${Date.now() - t0}ms`));

	// ————— tier B: the model-driven pass —————
	let modelFindings: Finding[] = [];
	const investigations: InvestigationStat[] = [];
	let guardStats = { steps: 0, blocked: 0, jailRejections: 0 };
	let exhausted: string | null = null;

	if (MODEL) {
		// Imported lazily so the deterministic pass never loads the inference stack. This is
		// what keeps `npm run audit` hermetic and fast in CI.
		const { LlamaCppEngine } = await import("../native/inference/engine.ts");
		const { ensureModel } = await import("../native/models/registry.ts");
		const { investigate, QUESTIONS } = await import("./investigate.ts");

		console.log(c.dim(`resolving model ${MODEL} …`));
		let lastPct = -1;
		const modelPath = await ensureModel(MODEL, {
			onProgress: ({ downloadedBytes, totalBytes }) => {
				if (!totalBytes) return;
				const pct = Math.floor((downloadedBytes / totalBytes) * 100);
				if (pct === lastPct) return; // a multi-GB download would otherwise flood the log
				lastPct = pct;
				process.stdout.write(`\r${c.dim(`  downloading ${pct}%`)}`);
				if (pct === 100) process.stdout.write("\n");
			},
		});
		const engine = new LlamaCppEngine({ gpu: "auto" });
		await engine.load({ modelPath });
		console.log(c.dim(`model loaded on gpu=${engine.gpu}`));

		try {
			for (const question of QUESTIONS) {
				process.stdout.write(c.dim(`  ${question.id} … `));
				const r = await investigate({ engine, root: ROOT, question, trace });
				investigations.push({
					question: r.question,
					steps: r.steps,
					ms: r.ms,
					tokens: r.tokens,
					emitted: r.emitted,
					exhausted: r.exhausted,
				});
				modelFindings.push(...r.findings);
				guardStats.steps += r.steps;
				if (r.exhausted) exhausted ??= `${question.id}: ${r.exhausted}`;
				console.log(c.dim(`${r.emitted} candidate(s), ${r.steps} step(s), ${(r.ms / 1000).toFixed(1)}s`));
			}
		} finally {
			await engine.unload();
		}
	}

	// ————— validate, partition, report —————
	const emitted = modelFindings.length;
	const { kept, dropped } = await validate(ROOT, [...ruleFindings, ...modelFindings]);
	if (dropped.length) {
		console.log(c.yellow(`\n${dropped.length} finding(s) dropped — citation could not be corroborated against the tree`));
	}

	// Dedupe by fingerprint: two questions can legitimately land on the same defect.
	const byId = new Map<string, Finding>();
	for (const f of kept) {
		const id = fingerprint(f);
		if (!byId.has(id)) byId.set(id, f);
	}
	const findings = [...byId.values()];

	const parts = partition(findings, baseline);
	const summary = summarize({
		startedAt,
		root: ROOT,
		model: MODEL,
		parts,
		emitted,
		droppedInvalid: dropped.length,
		guard: guardStats,
		exhausted,
		investigations,
	});
	summary.invariants = checkInvariants(summary, tracePath, findings);

	const notes: Record<string, string> = {};
	for (const [id, entry] of Object.entries(baseline.accepted)) notes[id] = entry.note;
	writeReport(OUT, summary, parts, notes);

	// ————— seed a baseline —————
	if (WRITE_BASELINE) {
		const accepted: Record<string, { rule: string; file: string; note: string }> = { ...baseline.accepted };
		for (const f of findings) {
			const id = fingerprint(f);
			accepted[id] ??= {
				rule: f.rule,
				file: f.file,
				// Deliberately a placeholder rather than a plausible-sounding sentence. An entry
				// nobody has justified should look unjustified, and loadBaseline rejects an empty
				// note so this cannot be left blank either.
				note: "TODO: state why this is accepted, or fix it and remove this entry.",
			};
		}
		mkdirSync(join(BASELINE, ".."), { recursive: true });
		writeFileSync(BASELINE, JSON.stringify({ accepted }, null, 2) + "\n");
		console.log(`\nwrote ${Object.keys(accepted).length} entr(ies) to ${BASELINE}`);
		console.log(c.yellow("every note says TODO — replace them before committing, or the gate means nothing"));
		return 0;
	}

	// ————— report —————
	printFindings(`new (${parts.fresh.length})`, parts.fresh);
	if (parts.known.length) console.log(`\n${c.dim(`baselined: ${parts.known.length} finding(s) — see ${join(OUT, "report.md")}`)}`);
	if (parts.resolved.length) {
		console.log(`\n${c.green(`resolved: ${parts.resolved.length} baseline entr(ies) no longer match — prune them`)}`);
		for (const id of parts.resolved) console.log(c.dim(`  ${id}  ${baseline.accepted[id]?.rule ?? ""}`));
	}

	if (summary.exhausted) console.log(`\n${c.yellow(`run stopped early — ${summary.exhausted}. Findings are incomplete.`)}`);
	if (MODEL) {
		console.log(c.dim(`\nmodel pass: ${emitted} emitted, ${dropped.length} unverifiable and dropped`));
	}

	console.log(`\n${c.dim(`artifacts: ${OUT}`)}`);

	// The auditor's own health outranks the codebase's. A failed invariant means every
	// number above is suspect, so it must not be reported as a clean or dirty tree.
	if (summary.invariants.length) {
		console.log(c.red(`\nAUDITOR SELF-CHECK FAILED — findings above are not trustworthy:`));
		for (const v of summary.invariants) console.log(c.red(`  - ${v}`));
		return 2;
	}

	if (CHECK && parts.fresh.length) {
		console.log(c.red(`\n${parts.fresh.length} finding(s) not in the baseline.`));
		console.log(c.dim(`Fix them, annotate with "// audit-ok(rule-id): reason", or add a baseline entry with a note.`));
		return 1;
	}

	if (!parts.fresh.length) console.log(c.green("\nno new findings"));
	return 0;
}

main()
	.then((code) => process.exit(code))
	.catch((e) => {
		// A crash is an auditor failure, not a verdict on the codebase.
		console.error(c.red(`audit failed: ${e instanceof Error ? (e.stack ?? e.message) : String(e)}`));
		process.exit(2);
	});
