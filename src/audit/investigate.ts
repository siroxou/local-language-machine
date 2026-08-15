// Local Language Machine — the model-driven pass.
//
// The deterministic rules only find what someone already thought to encode. This pass
// is where new classes come from: fixed lines of inquiry aimed at the things a regex
// structurally cannot see — lifetimes, re-entrancy, what a catch leaves half-done.
//
// Each question runs in its own isolated session with its own budget, mirroring
// spawnSubagent: one fresh 32k context, disposed in `finally`. Deliberately N short
// investigations rather than one long agent — bounded context, and a finding that can be
// attributed to the question that produced it.
//
// Both budget knobs are passed to the loop explicitly. A step cap alone is not a budget:
// the gate only runs between tool calls, so a generation that never emits one is
// unbounded, and a small model asked to reason about code will ramble until the context
// window fills while the wall-clock check waits for a turn it never gets.
//
// Evidence gathering runs through the text-parsed tool loop, but the findings are
// emitted through `structured()` — grammar-constrained decode, which cannot produce a
// non-conforming object. A findings emitter must not be able to return unparseable
// output, and this is that guarantee. (It is also `structured()`'s first production
// caller; it has been dead code since it was written.)

import { runAgentLoop } from "../core/loop.ts";
import type { GbnfJsonSchema, InferenceEngine } from "../native/inference/engine.ts";
import { auditGuardrails, DEFAULT_BUDGET, type Budget } from "./guard.ts";
import { normalizeEvidence, type Finding, type Severity } from "./findings.ts";
import type { InvestigationStat, Trace } from "./report.ts";

/** Bounded like the IDE's own subagents — a second full-window KV cache is gigabytes. */
const INVESTIGATION_CONTEXT_TOKENS = 32_768;
/** Per-step generation cap. See the maxTokens note at the runAgentLoop call below. */
const MAX_STEP_TOKENS = 700;

const AUDIT_SYSTEM = `You are auditing a TypeScript codebase for real defects.

You have read-only tools: read_file, list_dir, grep, glob. Use them to read the actual
code before drawing any conclusion. Never guess at a file's contents.

To call a tool, reply with only a JSON object: {"tool": "read_file", "arguments": {"path": "src/x.ts"}}

Rules for this work:
- A defect is something that produces wrong behaviour, a crash, a hang, or a leak. Style
  preferences, naming, and missing comments are not defects.
- Compare sites against each other. If four places do a thing and one does not, the one
  is the finding.
- Cite the exact file path and line number you actually read. A citation you did not
  verify is worse than no finding.
- When you have gathered enough, reply in plain text with what you found.`;

const EMIT_PROMPT = `Now report only the defects you verified by reading the code.

For each: the repo-relative file path, the 1-based line number you actually saw, a
severity, one sentence naming the wrong behaviour, and the source line as evidence.

Report nothing you did not read. An empty list is a valid and useful answer.`;

const FINDING_SCHEMA: GbnfJsonSchema = {
	type: "object",
	properties: {
		findings: {
			type: "array",
			items: {
				type: "object",
				properties: {
					file: { type: "string" },
					line: { type: "number" },
					severity: { enum: ["high", "medium", "low"] },
					message: { type: "string" },
					evidence: { type: "string" },
				},
				required: ["file", "line", "severity", "message", "evidence"],
			},
		},
	},
	required: ["findings"],
};

interface RawFinding {
	file: string;
	line: number;
	severity: Severity;
	message: string;
	evidence: string;
}

export interface Question {
	id: string;
	/** The lens. Kept in the report so a finding can be traced to the question that found it. */
	prompt: string;
}

/**
 * The lenses. One question fixed across many files beats one file examined from many
 * angles: holding the question still is what makes the *comparisons* visible, and every
 * defect this project has confirmed so far was a comparison finding — one call site
 * missing a guard its four siblings have.
 */
export const QUESTIONS: Question[] = [
	{
		id: "resource-lifecycle",
		prompt:
			"Audit resource lifetimes in src/native/ and src/preview/server.ts. Find every spawn, setInterval, setTimeout, " +
			"WebSocket, server, and inference context/session. For each, identify who releases it and whether that release " +
			"happens on ALL FOUR exits: success, thrown error, client disconnect, and process shutdown. Report any resource " +
			"with an exit path that never releases it. Start by grepping for spawn, setInterval, and dispose.",
	},
	{
		id: "shared-mutable-state",
		prompt:
			"src/preview/server.ts serves multiple WebSocket clients but was written as if there were one. Find every " +
			"module-level `let` in that file and in src/native/training/train.ts. For each, describe what breaks when two " +
			"browser tabs are connected, or when a message arrives while a previous request is still running. Look " +
			"specifically for check-then-act across an await. Read the file before answering.",
	},
	{
		id: "trust-boundaries",
		prompt:
			"Trace untrusted input to dangerous sinks. Untrusted means: fields off a WebSocket message (msg.*), tool " +
			"arguments chosen by a language model, and HTTP response bodies. Dangerous sinks are path.join, spawn, exec, " +
			"new RegExp, and writeFile. For each path you find from an untrusted source to a sink, report where validation " +
			"is missing. Read src/preview/server.ts and src/native/tools/tools.ts.",
	},
	{
		id: "error-paths",
		prompt:
			"Read every catch block in src/core/ and src/preview/server.ts. For each, answer two questions: is program " +
			"state still consistent afterwards, and does anyone (user, log, or caller) learn the failure happened? Report " +
			"only the ones that leave a mutation half-applied or swallow a failure that changes behaviour. An intentional " +
			"empty catch with a comment explaining it is not a defect.",
	},
	{
		id: "limit-signalling",
		prompt:
			"Find every cap, budget, and limit in src/core/ and src/native/tools/tools.ts — constants named MAX_*, step " +
			"counts, timeouts, truncation. For each, determine whether the CALLER can tell that a limit was hit, or whether " +
			"a truncated result is indistinguishable from a complete one. Report every limit whose exhaustion is silent. " +
			"Read src/core/loop.ts carefully and say what it returns when its step budget runs out.",
	},
	{
		id: "parser-edges",
		prompt:
			"Audit the hand-written parsers for inputs that break them. Read parseToolCall in src/core/loop.ts, the " +
			"newline-delimited JSON buffering in src/native/mcp/client.ts, and parseFrontmatter in src/core/skills.ts. For " +
			"each, describe a concrete input that causes wrong output, unbounded memory growth, or a crash. Be specific " +
			"about the input.",
	},
];

export interface Investigation extends InvestigationStat {
	findings: Finding[];
	/** The model's closing prose. Kept for the trace; the findings are the product. */
	transcript: string;
}

/**
 * Run one question in an isolated session and return validated-shape findings.
 * Never throws for model reasons — a failed investigation returns zero findings with the
 * reason recorded, because one bad question must not lose the other five.
 */
export async function investigate(opts: {
	engine: InferenceEngine;
	root: string;
	question: Question;
	budget?: Budget;
	trace: Trace;
}): Promise<Investigation> {
	const { engine, root, question } = opts;
	const budget = opts.budget ?? DEFAULT_BUDGET;
	const startedAt = Date.now();

	// Creating the session can fail on its own — a context this size is a real allocation, and
	// it is the first thing to go when memory is tight. Outside the try it would take the whole
	// run down, losing five good investigations because the sixth could not start.
	let session: Awaited<ReturnType<InferenceEngine["createSession"]>>;
	try {
		session = await engine.createSession({
			systemPrompt: AUDIT_SYSTEM,
			contextSize: INVESTIGATION_CONTEXT_TOKENS,
		});
	} catch (e) {
		const reason = `could not create a session: ${e instanceof Error ? e.message : String(e)}`;
		opts.trace({ phase: "investigate", question: question.id, reason });
		return {
			question: question.id,
			steps: 0,
			ms: Date.now() - startedAt,
			tokens: 0,
			emitted: 0,
			exhausted: reason,
			findings: [],
			transcript: "",
		};
	}

	const guard = auditGuardrails({
		root,
		budget,
		trace: opts.trace,
		usedTokens: () => session.usedTokens?.() ?? 0,
		phase: "investigate",
		question: question.id,
	});

	const stat = (over: Partial<Investigation> = {}): Investigation => ({
		question: question.id,
		steps: guard.stats().steps,
		ms: Date.now() - startedAt,
		tokens: session.usedTokens?.() ?? 0,
		emitted: 0,
		exhausted: guard.exhausted(),
		findings: [],
		transcript: "",
		...over,
	});

	try {
		const noop = () => {};
		const transcript = await runAgentLoop({
			session,
			tools: guard.tools,
			gate: guard.gate,
			input: question.prompt,
			root,
			maxSteps: budget.steps,
			// Without this the budget is a fiction. A gate only runs between tool calls, so a
			// single generation that never emits one is unbounded — a small model asked to reason
			// about code will happily ramble until the context window fills, and the wall-clock
			// check never gets a turn. A tool call plus a short rationale fits well inside this.
			maxTokens: MAX_STEP_TOKENS,
			events: { onToken: noop, onToolCall: noop, onToolResult: noop, onCodeOpen: noop, onCodeToken: noop, onCodeClose: noop },
		});

		// ponytail: structured() ignores maxTokens, so a pathological grammar walk is bounded
		// only by the context window. Acceptable because the schema has no unbounded array of
		// objects and the run is a dev tool. Thread maxTokens through engine.structured() if a
		// run is ever observed stalling here.
		let raw: RawFinding[] = [];
		try {
			const emitted = await session.structured<{ findings: RawFinding[] }>(EMIT_PROMPT, FINDING_SCHEMA);
			raw = Array.isArray(emitted?.findings) ? emitted.findings : [];
		} catch (e) {
			opts.trace({
				phase: "investigate",
				question: question.id,
				note: `emit failed: ${e instanceof Error ? e.message : String(e)}`,
			});
		}

		const findings: Finding[] = raw.map((r) => ({
			rule: `model:${question.id}`,
			severity: (["high", "medium", "low"] as Severity[]).includes(r.severity) ? r.severity : "low",
			// Normalise a leading ./ or / so validate() can resolve it against the root.
			file: String(r.file ?? "").replace(/^\.?\//, ""),
			line: Number(r.line) || 0,
			message: String(r.message ?? "").slice(0, 500),
			evidence: normalizeEvidence(String(r.evidence ?? "")),
			origin: "model",
		}));

		opts.trace({
			phase: "investigate",
			question: question.id,
			ms: Date.now() - startedAt,
			note: `emitted ${findings.length} candidate finding(s)`,
			tokens: session.usedTokens?.() ?? 0,
		});

		return stat({ findings, emitted: findings.length, transcript });
	} catch (e) {
		opts.trace({
			phase: "investigate",
			question: question.id,
			reason: `investigation failed: ${e instanceof Error ? e.message : String(e)}`,
		});
		return stat();
	} finally {
		session.dispose();
	}
}
