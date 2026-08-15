// Local Language Machine — the audit's observability.
//
// Three artifacts per run: a JSONL trace of every step, a machine-readable summary,
// and a human report. The trace uses the same append-a-line shape as Session's
// events.jsonl rather than introducing a logging abstraction — adding a logger to the
// product in order to observe a dev tool is backwards.
//
// checkInvariants is the part that matters. The auditor asserts things about its own
// run and fails loudly (exit 2) if they do not hold, because a bug-finder that quietly
// reports "nothing found" when it actually means "I stopped early" is worse than no
// bug-finder at all — and that is precisely the defect it was built to look for.

import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fingerprint, sortFindings, type Finding, type Partition, type Severity } from "./findings.ts";

export interface TraceRecord {
	/** Epoch ms. */
	t: number;
	phase: "rules" | "investigate";
	/** Investigation id, for the model-driven pass. */
	question?: string;
	step?: number;
	tool?: string;
	/** JSON.stringify(args), truncated. Not hashed — a digest of a grep pattern helps nobody. */
	args?: string;
	ms?: number;
	gate?: "allow" | "block";
	reason?: string;
	tokens?: number;
	note?: string;
}

const ARGS_MAX = 200;

export const digestArgs = (args: unknown): string => {
	try {
		return JSON.stringify(args ?? null).slice(0, ARGS_MAX);
	} catch {
		return "<unserializable>";
	}
};

export type Trace = (rec: Omit<TraceRecord, "t">) => void;

/** Append-only JSONL, one record per line. Returns the writer and the path it writes to. */
export function openTrace(dir: string): { trace: Trace; path: string } {
	mkdirSync(dir, { recursive: true });
	const path = join(dir, "trace.jsonl");
	writeFileSync(path, ""); // a run's trace is its own, not an append to the last one
	return {
		path,
		trace: (rec) => appendFileSync(path, JSON.stringify({ t: Date.now(), ...rec }) + "\n"),
	};
}

export function readTrace(path: string): TraceRecord[] {
	let raw: string;
	try {
		raw = readFileSync(path, "utf8");
	} catch {
		return [];
	}
	const out: TraceRecord[] = [];
	for (const line of raw.split("\n")) {
		if (!line.trim()) continue;
		try {
			out.push(JSON.parse(line));
		} catch {
			// A torn final line is possible if a run was killed mid-write. Skipping it is
			// right for reading, and `trace-unparseable` below is what reports it.
		}
	}
	return out;
}

export interface InvestigationStat {
	question: string;
	steps: number;
	ms: number;
	tokens: number;
	/** What the model claimed, before validation. */
	emitted: number;
	/** Null means it finished on its own. A string is the budget that stopped it. */
	exhausted: string | null;
}

export interface Summary {
	startedAt: string;
	durationMs: number;
	root: string;
	/** Model id for the model-driven pass, or null when only the deterministic pass ran. */
	model: string | null;
	counts: {
		total: number;
		fresh: number;
		known: number;
		resolved: number;
		bySeverity: Record<Severity, number>;
		byRule: Record<string, number>;
		byOrigin: { rule: number; model: number };
	};
	/**
	 * Findings the model cited that could not be confirmed against the tree.
	 * droppedInvalid/emitted is the trust signal for the model-driven pass: it says, per
	 * run and with a number, how much of what came back was real.
	 */
	emitted: number;
	droppedInvalid: number;
	guard: { steps: number; blocked: number; jailRejections: number };
	/** Non-null if any budget stopped the run early. Never silently absent. */
	exhausted: string | null;
	investigations: InvestigationStat[];
	/** Failed self-assertions. Non-empty means the auditor is broken, not the code. */
	invariants: string[];
}

const ZERO_SEVERITY: Record<Severity, number> = { high: 0, medium: 0, low: 0 };

export function summarize(opts: {
	startedAt: number;
	root: string;
	model: string | null;
	parts: Partition;
	emitted: number;
	droppedInvalid: number;
	guard: Summary["guard"];
	exhausted: string | null;
	investigations: InvestigationStat[];
}): Summary {
	const all = [...opts.parts.fresh, ...opts.parts.known];
	const bySeverity = { ...ZERO_SEVERITY };
	const byRule: Record<string, number> = {};
	const byOrigin = { rule: 0, model: 0 };
	for (const f of all) {
		bySeverity[f.severity]++;
		byRule[f.rule] = (byRule[f.rule] ?? 0) + 1;
		byOrigin[f.origin]++;
	}
	return {
		startedAt: new Date(opts.startedAt).toISOString(),
		durationMs: Date.now() - opts.startedAt,
		root: opts.root,
		model: opts.model,
		counts: {
			total: all.length,
			fresh: opts.parts.fresh.length,
			known: opts.parts.known.length,
			resolved: opts.parts.resolved.length,
			bySeverity,
			byRule,
			byOrigin,
		},
		emitted: opts.emitted,
		droppedInvalid: opts.droppedInvalid,
		guard: opts.guard,
		exhausted: opts.exhausted,
		investigations: opts.investigations,
		invariants: [],
	};
}

/** The only tools the audit agent is ever allowed to hold. Absence is the guarantee; this re-checks it. */
export const ALLOWED_TOOLS = new Set(["read_file", "list_dir", "grep", "glob"]);

/**
 * Assertions the run makes about itself, checked against the trace file it just wrote
 * rather than in-memory state — so what gets validated is the artifact a human will
 * read. Any returned string is a failure.
 */
export function checkInvariants(summary: Summary, tracePath: string, findings: Finding[]): string[] {
	const fail: string[] = [];
	const trace = readTrace(tracePath);

	// 1. Nothing outside the read-only toolset was ever invoked.
	const foreign = [...new Set(trace.map((r) => r.tool).filter((t): t is string => !!t && !ALLOWED_TOOLS.has(t)))];
	if (foreign.length) fail.push(`tool outside the read-only set appears in the trace: ${foreign.join(", ")}`);

	// 2. A blocked call must say why. An unexplained block is indistinguishable from a bug.
	const mute = trace.filter((r) => r.gate === "block" && !r.reason).length;
	if (mute) fail.push(`${mute} blocked tool call(s) recorded with no reason`);

	// 3. Budget exhaustion must be reported. This is the defect the auditor exists to not
	//    repeat: the IDE's own loop returns "" when it runs out of steps, so a caller cannot
	//    tell "no answer" from "no budget". If any investigation stopped early, say so.
	const stoppedEarly = summary.investigations.filter((i) => i.exhausted);
	if (stoppedEarly.length && !summary.exhausted) {
		fail.push(`${stoppedEarly.length} investigation(s) hit a budget but the summary reports no exhaustion`);
	}

	// 4. Every reported finding is uniquely identified, or the baseline cannot address it.
	const seen = new Set<string>();
	for (const f of findings) {
		const id = fingerprint(f);
		if (seen.has(id)) fail.push(`duplicate fingerprint ${id} (${f.rule} ${f.file}) survived dedupe`);
		seen.add(id);
	}

	// 5. Counts must agree with the findings actually reported.
	if (summary.counts.total !== summary.counts.fresh + summary.counts.known) {
		fail.push(`counts do not add up: total ${summary.counts.total} != fresh ${summary.counts.fresh} + known ${summary.counts.known}`);
	}

	// 6. The trace must be readable. A torn line means a run died mid-write, and every
	//    number above was computed from a stream we cannot fully replay.
	const lines = readFileSync(tracePath, "utf8").split("\n").filter((l) => l.trim()).length;
	if (lines !== trace.length) fail.push(`trace has ${lines - trace.length} unparseable line(s) — the run did not finish cleanly`);

	return fail;
}

// ————— the human report —————

const SEVERITY_MARK: Record<Severity, string> = { high: "high", medium: "med", low: "low" };

function table(findings: Finding[]): string {
	if (!findings.length) return "_None._\n";
	const rows = sortFindings(findings).map(
		(f) => `| ${SEVERITY_MARK[f.severity]} | \`${f.rule}\` | \`${f.file}:${f.line}\` | ${f.evidence.replace(/\|/g, "\\|")} |`,
	);
	return ["| sev | rule | location | evidence |", "|:---:|------|----------|----------|", ...rows].join("\n") + "\n";
}

export function renderReport(summary: Summary, parts: Partition, baselineNotes: Record<string, string>): string {
	const s = summary;
	const out: string[] = [];

	out.push("# Audit report", "");
	out.push(
		`Ran at ${s.startedAt} in ${(s.durationMs / 1000).toFixed(2)}s — ` +
			`${s.model ? `deterministic + model-driven pass (\`${s.model}\`)` : "deterministic pass only"}.`,
		"",
	);

	if (s.exhausted) {
		// Stated before the findings, not after: a truncated run's empty section means
		// "did not look", and a reader must not mistake it for "looked and found nothing".
		out.push(`> **Run stopped early — ${s.exhausted}.** Findings below are incomplete.`, "");
	}

	out.push(
		`**${s.counts.fresh} new** · ${s.counts.known} baselined · ${s.counts.resolved} resolved · ` +
			`${s.counts.bySeverity.high} high / ${s.counts.bySeverity.medium} medium / ${s.counts.bySeverity.low} low`,
		"",
		"---",
		"",
		"## New findings",
		"",
		"These are not in the baseline. They fail the gate.",
		"",
		table(parts.fresh),
	);

	out.push("", "---", "", "## Baselined", "", "Known and accepted. Reported for visibility; they do not fail the gate.", "");
	if (parts.known.length) {
		out.push(table(parts.known), "");
		out.push("Accepted because:", "");
		const shown = new Set<string>();
		for (const f of sortFindings(parts.known)) {
			const id = fingerprint(f);
			const note = baselineNotes[id];
			if (!note || shown.has(id)) continue;
			shown.add(id);
			out.push(`- \`${f.file}:${f.line}\` (${f.rule}) — ${note}`);
		}
		out.push("");
	} else {
		out.push("_None._", "");
	}

	if (parts.resolved.length) {
		out.push(
			"---",
			"",
			"## Resolved",
			"",
			"In the baseline with no matching finding — the defect is gone. Prune these entries.",
			"",
			...parts.resolved.map((id) => `- \`${id}\` — ${baselineNotes[id] ?? "(no note)"}`),
			"",
		);
	}

	if (s.model) {
		out.push("---", "", "## Model-driven pass", "");
		out.push(
			`Emitted ${s.emitted} candidate finding(s); **${s.droppedInvalid} dropped** as uncorroborated ` +
				`— the cited file or line does not exist, or the quoted evidence is not in that file. ` +
				`Treat the remainder as a hint list, not a verdict.`,
			"",
			"| investigation | steps | tokens | time | emitted | stopped |",
			"|---------------|------:|-------:|-----:|--------:|---------|",
			...s.investigations.map(
				(i) =>
					`| ${i.question} | ${i.steps} | ${i.tokens} | ${(i.ms / 1000).toFixed(1)}s | ${i.emitted} | ${i.exhausted ?? "—"} |`,
			),
			"",
		);
	}

	out.push(
		"---",
		"",
		"## Run",
		"",
		`- tool calls: ${s.guard.steps} (${s.guard.blocked} blocked)`,
		`- path-jail rejections: ${s.guard.jailRejections}`,
		`- self-invariants: ${s.invariants.length ? `**${s.invariants.length} FAILED**` : "all held"}`,
		"",
	);
	if (s.invariants.length) out.push(...s.invariants.map((v) => `  - ${v}`), "");

	return out.join("\n");
}

export function writeReport(dir: string, summary: Summary, parts: Partition, baselineNotes: Record<string, string>): void {
	mkdirSync(dir, { recursive: true });
	writeFileSync(join(dir, "summary.json"), JSON.stringify(summary, null, 2) + "\n");
	writeFileSync(join(dir, "report.md"), renderReport(summary, parts, baselineNotes));
	writeFileSync(
		join(dir, "findings.json"),
		JSON.stringify({ fresh: parts.fresh, known: parts.known, resolved: parts.resolved }, null, 2) + "\n",
	);
}
