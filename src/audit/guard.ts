// Local Language Machine — the audit agent's guardrails.
//
// Every constraint here is structural rather than advisory. The agent cannot write a
// file because no write tool is ever constructed; it cannot reach the network because
// no web tool is registered; it cannot leave the tree because every path argument goes
// through a realpath-checked jail before the handler sees it. An absent capability
// needs no denylist and cannot be re-enabled by a confused model.
//
// Two of these constraints began life here as hardened local copies, because the shipped
// equivalents were weaker: `resolveInRoot` compared strings rather than resolving links,
// and `glob` never jailed its pattern at all. Both are fixed at the source now, so this
// file delegates to them instead of carrying parallel implementations that could drift
// from the boundary the app actually enforces. What remains local is the counting — the
// audit reports on its own containment, and the app has no reason to.

import { buildTools, resolveInRoot } from "../native/tools/tools.ts";
import type { ToolDef } from "../native/inference/engine.ts";
import type { ToolGate } from "../core/loop.ts";
import { ALLOWED_TOOLS, digestArgs, type Trace } from "./report.ts";

export interface Budget {
	/** Tool calls across one investigation. */
	steps: number;
	wallMs: number;
	/** Context tokens consumed, read from the session. */
	tokens: number;
}

export const DEFAULT_BUDGET: Budget = { steps: 6, wallMs: 120_000, tokens: 28_000 };

/**
 * The audit's path boundary.
 *
 * This started as a hardened local copy, because the shipped `resolveInRoot` compared
 * strings and so accepted `root/link/etc/passwd` whenever `link` pointed out of the tree.
 * That is now fixed at the source, and physical resolution lives in the one function all
 * nineteen call sites already route through — so the audit uses it rather than keeping a
 * second implementation that could drift from the real boundary.
 *
 * Kept as a named export because the audit's guarantee deserves its own name and its own
 * tests, independent of whichever function currently provides it.
 */
export const jail = resolveInRoot;

export interface Guardrails {
	/** Read-only, jailed. The mutating tools are not present at all. */
	tools: Record<string, ToolDef>;
	gate: ToolGate;
	/** Null while within budget; otherwise the reason the run must stop. */
	exhausted(): string | null;
	stats(): { steps: number; blocked: number; jailRejections: number };
	/** Call between investigations — budgets are per-investigation, counters are per-run. */
	reset(): void;
}

export function auditGuardrails(opts: {
	root: string;
	budget?: Budget;
	trace: Trace;
	/** Wired to the live session so the token budget reflects real context use. */
	usedTokens?: () => number;
	phase?: TraceRecordPhase;
	question?: string;
}): Guardrails {
	const budget = opts.budget ?? DEFAULT_BUDGET;
	const usedTokens = opts.usedTokens ?? (() => 0);
	const phase = opts.phase ?? "investigate";

	let steps = 0;
	let blocked = 0;
	let jailRejections = 0;
	let startedAt = Date.now();
	let stopped: string | null = null;

	const reject = (e: unknown): never => {
		jailRejections++;
		throw e;
	};

	/**
	 * Pre-flight the named string arguments through the jail, then delegate to the tested
	 * handler. Async so a rejected path surfaces as a rejected promise rather than a
	 * synchronous throw — the loop handles both, but callers and tests should not have to.
	 */
	const guardArgs = (tool: ToolDef, keys: string[]): ToolDef => ({
		...tool,
		async handler(args: any) {
			for (const k of keys) {
				if (typeof args?.[k] === "string") {
					try {
						jail(opts.root, args[k]);
					} catch (e) {
						return reject(e);
					}
				}
			}
			return tool.handler(args);
		},
	});

	// `exec` is deliberately not supplied, so run_terminal and git cannot execute anything
	// even if a future edit accidentally exposes them.
	const base = buildTools({ root: opts.root });

	const tools: Record<string, ToolDef> = {
		read_file: guardArgs(base.read_file!, ["path"]),
		list_dir: guardArgs(base.list_dir!, ["path"]),
		grep: guardArgs(base.grep!, ["path"]),
		// This was a rebuilt implementation while the shipped glob handed the model's pattern
		// straight to fs with only a cwd — and a cwd is not a boundary. That is fixed at the
		// source now, so the tested one is inherited and only the rejection count is added,
		// because the audit reports on its own containment and the app has no reason to.
		glob: {
			...base.glob!,
			async handler(args: any) {
				try {
					return await base.glob!.handler(args);
				} catch (e) {
					jailRejections++;
					opts.trace({ phase, question: opts.question, tool: "glob", gate: "block", reason: (e as Error).message });
					throw e;
				}
			},
		},
	};

	const overBudget = (): string | null => {
		if (steps >= budget.steps) return `step budget exhausted (${budget.steps} tool calls)`;
		const elapsed = Date.now() - startedAt;
		if (elapsed > budget.wallMs) return `wall-clock budget exhausted (${budget.wallMs}ms)`;
		const t = usedTokens();
		if (t > budget.tokens) return `token budget exhausted (${t} > ${budget.tokens})`;
		return null;
	};

	const gate: ToolGate = {
		before(tool, args) {
			// Defence in depth. The mutating tools are absent, so this can only fire if a
			// future edit adds one — which is exactly when a loud failure is wanted.
			if (!ALLOWED_TOOLS.has(tool)) {
				blocked++;
				const reason = `tool "${tool}" is not in the read-only audit set`;
				opts.trace({ phase, question: opts.question, step: steps, tool, args: digestArgs(args), gate: "block", reason });
				return { block: true, result: { error: reason } };
			}

			const reason = overBudget();
			if (reason) {
				blocked++;
				stopped = reason;
				// Blocking at the gate rather than throwing lets the loop wind down and the
				// model produce a partial answer, and the reason is recorded so the run can
				// never present a truncated result as a complete one.
				opts.trace({ phase, question: opts.question, step: steps, tool, args: digestArgs(args), gate: "block", reason });
				return { block: true, result: { halted: reason } };
			}

			steps++;
			opts.trace({
				phase,
				question: opts.question,
				step: steps,
				tool,
				args: digestArgs(args),
				gate: "allow",
				tokens: usedTokens(),
			});
			return undefined;
		},

		after(tool, _args, result) {
			opts.trace({
				phase,
				question: opts.question,
				step: steps,
				tool,
				ms: Date.now() - startedAt,
				note: `result ${digestArgs(result).length} bytes`,
				tokens: usedTokens(),
			});
		},
	};

	return {
		tools,
		gate,
		exhausted: () => stopped ?? overBudget(),
		stats: () => ({ steps, blocked, jailRejections }),
		reset: () => {
			startedAt = Date.now();
			stopped = null;
			steps = 0;
		},
	};
}

type TraceRecordPhase = "rules" | "investigate";
