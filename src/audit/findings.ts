// Local Language Machine — the audit's finding schema, identity, and baseline.
//
// Everything both passes produce lands here: the deterministic rules and the
// model-driven investigations emit the same `Finding`, so one validation path and
// one baseline serve both. Nothing in this file knows how a finding was produced.

import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { readFileSync } from "node:fs";
import { isAbsolute, join } from "node:path";

export type Severity = "high" | "medium" | "low";

export interface Finding {
	/** Rule id, or `model:<questionId>` for the model-driven pass. */
	rule: string;
	severity: Severity;
	/** Repo-relative, forward slashes on every platform. */
	file: string;
	/** 1-based. */
	line: number;
	message: string;
	/** The matched source line, whitespace-collapsed. Carries the identity — see `fingerprint`. */
	evidence: string;
	origin: "rule" | "model";
}

export interface BaselineEntry {
	rule: string;
	file: string;
	/** Why this is accepted. Empty is a hard error — see `loadBaseline`. */
	note: string;
}

export interface Baseline {
	accepted: Record<string, BaselineEntry>;
}

const EVIDENCE_MAX = 200;

/** Collapse whitespace and cap length, so identity survives reindentation. */
export function normalizeEvidence(text: string): string {
	return text.replace(/\s+/g, " ").trim().slice(0, EVIDENCE_MAX);
}

/**
 * Stable identity for a finding, deliberately WITHOUT the line number.
 *
 * Adding an import at the top of a file shifts every line below it. A line-keyed
 * baseline would then report each of those as new, so a routine edit turns the gate
 * red for reasons nobody caused — and a gate that cries wolf is one people learn to
 * bypass. The matched text is the part that actually identifies the defect.
 *
 * The cost: two byte-identical violations in one file collapse to a single entry.
 * That is the right trade — they are the same defect twice, and fixing one without
 * the other should not read as progress.
 */
export function fingerprint(f: Pick<Finding, "rule" | "file" | "evidence">): string {
	const h = createHash("sha256");
	h.update(`${f.rule}\0${f.file}\0${normalizeEvidence(f.evidence)}`);
	return h.digest("hex").slice(0, 12);
}

/**
 * Read the baseline. A malformed file is fatal rather than empty-by-default: silently
 * treating an unreadable baseline as "nothing accepted" would turn every known finding
 * into a fresh one and fail the build for the wrong reason.
 */
export function loadBaseline(path: string): Baseline {
	let raw: string;
	try {
		raw = readFileSync(path, "utf8");
	} catch {
		return { accepted: {} }; // absent is legitimate — a repo with nothing deferred yet
	}
	let data: any;
	try {
		data = JSON.parse(raw);
	} catch (e) {
		throw new Error(`Baseline at ${path} is not valid JSON: ${e instanceof Error ? e.message : String(e)}`);
	}
	const accepted = data?.accepted;
	if (!accepted || typeof accepted !== "object") {
		throw new Error(`Baseline at ${path} has no "accepted" object.`);
	}
	// A note is mandatory. Without it the baseline decays into a suppression dump —
	// entries accumulate, nobody remembers why, and the file stops meaning anything.
	// One sentence per entry is the whole price of keeping it honest.
	for (const [key, entry] of Object.entries(accepted as Record<string, BaselineEntry>)) {
		if (!entry?.note || !String(entry.note).trim()) {
			throw new Error(`Baseline entry ${key} (${entry?.rule ?? "?"}) has no note. Every accepted finding must say why.`);
		}
	}
	return { accepted: accepted as Record<string, BaselineEntry> };
}

export interface Partition {
	/** Not in the baseline. These fail the gate. */
	fresh: Finding[];
	/** In the baseline. Reported, but they pass. */
	known: Finding[];
	/** In the baseline with no matching finding — the defect is gone, prune the entry. */
	resolved: string[];
}

export function partition(findings: Finding[], baseline: Baseline): Partition {
	const fresh: Finding[] = [];
	const known: Finding[] = [];
	const seen = new Set<string>();
	for (const f of findings) {
		const id = fingerprint(f);
		seen.add(id);
		if (baseline.accepted[id]) known.push(f);
		else fresh.push(f);
	}
	const resolved = Object.keys(baseline.accepted).filter((id) => !seen.has(id));
	return { fresh, known, resolved };
}

export interface Validation {
	kept: Finding[];
	/** Findings whose citation the tree does not corroborate. The count is the trust signal. */
	dropped: Finding[];
}

/**
 * Drop any finding whose citation cannot be confirmed against the tree.
 *
 * This is what makes the model-driven pass usable at all. A small model will cite a
 * plausible file that is not there, a line past the end of one, or — most often — a real
 * path with evidence it invented. Rather than asking a reader to check every claim, the run
 * checks them and publishes how many it threw away: `dropped/emitted` is a per-run number
 * saying how much to trust the pass. A 135M model scored 152 of 152 dropped.
 *
 * Rule findings go through the same path even though they are produced from real reads.
 * One code path, and the invariant stays testable with a synthetic bad finding.
 */
export async function validate(root: string, findings: Finding[]): Promise<Validation> {
	const files = new Map<string, string[] | null>();
	const kept: Finding[] = [];
	const dropped: Finding[] = [];

	for (const f of findings) {
		// An absolute or escaping path is never a valid citation — findings address the repo.
		if (!f.file || isAbsolute(f.file) || f.file.split("/").includes("..")) {
			dropped.push(f);
			continue;
		}
		let lines = files.get(f.file);
		if (lines === undefined) {
			try {
				lines = (await readFile(join(root, f.file), "utf8")).split("\n");
			} catch {
				lines = null; // unreadable or absent
			}
			files.set(f.file, lines);
		}
		if (!lines || !Number.isInteger(f.line) || f.line < 1 || f.line > lines.length) {
			dropped.push(f);
			continue;
		}

		// The citation must be *corroborated*, not merely well-formed. Checking that the file
		// and line exist turned out to be far too weak: a model that read nothing still emitted
		// findings against real paths at line 1, and they passed. Requiring the quoted evidence
		// to actually appear in that file is what separates "I read this" from "I guessed a
		// plausible filename". Matched anywhere in the file rather than at the exact line,
		// because an off-by-a-few line number is a normal transcription slip while invented
		// evidence is not.
		const needle = normalizeEvidence(f.evidence);
		if (needle.length < MIN_EVIDENCE_CHARS) {
			dropped.push(f);
			continue;
		}
		// The reverse direction allows evidence that is a truncated quote of a longer line, but it
		// has to respect the same floor: every file ends in a blank line, and `needle.includes("")`
		// is true for everything, which made the whole check vacuous.
		const corroborated = lines
			.map((l) => normalizeEvidence(l))
			.some((l) => l.includes(needle) || (l.length >= MIN_EVIDENCE_CHARS && needle.includes(l)));
		if (!corroborated) {
			dropped.push(f);
			continue;
		}
		kept.push(f);
	}
	return { kept, dropped };
}

/**
 * Shortest evidence that can corroborate anything. A one- or two-character quote ("1", "}")
 * matches almost every file and proves nothing.
 */
const MIN_EVIDENCE_CHARS = 12;

const SEVERITY_ORDER: Record<Severity, number> = { high: 0, medium: 1, low: 2 };

/** Most severe first, then by file and line, so a report reads top-down by urgency. */
export function sortFindings(findings: Finding[]): Finding[] {
	return [...findings].sort(
		(a, b) =>
			SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity] ||
			a.file.localeCompare(b.file) ||
			a.line - b.line,
	);
}
