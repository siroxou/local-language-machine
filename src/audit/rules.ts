// Local Language Machine — the deterministic pass.
//
// Invariant checks over the source, no model and no dependencies. Each rule here
// asserts a property the codebase *already* holds at every site but one or two —
// three of the four `spawn` calls attach an error listener, three of the five
// `fetch` calls pass a signal. That is what separates an invariant from a style
// rule: it names discipline that already exists and stops it eroding, so it can be
// enforced without asking anyone to change how they write code.
//
// A rule that fires on healthy code is worse than no rule, because it teaches
// people to ignore the tool. The ones that would have — an empty-catch check, a
// file-length threshold — are documented as rejected in docs/audit.md rather than
// shipped and muted.
//
// ponytail: line/regex analysis over `stripped()` source, no AST. 5.8k LOC and ten
// rules do not justify loading the compiler. Upgrade to ts.createSourceFile if a
// rule ever needs real scope or data-flow analysis — `tool-handler-unjailed-fs` is
// the one that would want it first.

import { readFile } from "node:fs/promises";
import { relative, sep } from "node:path";
import { walkFiles } from "../native/tools/tools.ts";
import { normalizeEvidence, type Finding, type Severity } from "./findings.ts";

export interface Hit {
	/** 1-based. */
	line: number;
	evidence: string;
}

export interface Rule {
	id: string;
	severity: Severity;
	message: string;
	/** Tested against the repo-relative, forward-slashed path. */
	include: RegExp;
	/** `src` is the original text; `code` has comments and string bodies blanked. */
	scan(src: string, code: string, path: string): Hit[];
}

// ————— source preparation —————

/**
 * Blank the *contents* of comments and string literals, preserving every offset and
 * newline so a match index still maps to the right line.
 *
 * Without this the audit reports itself: this file names `fetch(` and `spawn(` in its
 * own comments, and docs describing a rule would trip it. Scanning the blanked text and
 * reporting against the original is what makes the rules writable *and* documentable.
 */
export function stripped(src: string): string {
	const out = src.split("");
	const blank = (i: number) => {
		if (out[i] !== "\n") out[i] = " ";
	};
	// A `/` starts a regex literal only where a value may begin. After an identifier,
	// a closing paren or a number it is division. This is the standard heuristic and it
	// is enough here — it exists so a regex containing a quote can't desync the scanner.
	const regexAllowedAfter = /[(,=:[!&|?{};+\-*%~^<>]/;
	let i = 0;
	let prevMeaningful = "";
	while (i < src.length) {
		const c = src[i]!;
		const next = src[i + 1];
		if (c === "/" && next === "/") {
			while (i < src.length && src[i] !== "\n") blank(i++);
			continue;
		}
		if (c === "/" && next === "*") {
			blank(i++);
			blank(i++);
			while (i < src.length && !(src[i] === "*" && src[i + 1] === "/")) blank(i++);
			blank(i++);
			blank(i++);
			continue;
		}
		if (c === '"' || c === "'") {
			i++; // keep the opening quote so the shape of the code survives
			while (i < src.length && src[i] !== c) {
				if (src[i] === "\\") blank(i++);
				if (i < src.length) blank(i++);
			}
			i++;
			prevMeaningful = c;
			continue;
		}
		if (c === "`") {
			i++;
			// Template contents are blanked wholesale, `${...}` included. Nothing any rule
			// asks about lives inside an interpolation, and blanking keeps nested braces and
			// parens from confusing sliceCall.
			while (i < src.length && src[i] !== "`") {
				if (src[i] === "\\") blank(i++);
				if (i < src.length) blank(i++);
			}
			i++;
			prevMeaningful = "`";
			continue;
		}
		if (c === "/" && (prevMeaningful === "" || regexAllowedAfter.test(prevMeaningful))) {
			i++;
			let inClass = false;
			while (i < src.length && src[i] !== "\n") {
				const r = src[i]!;
				if (r === "\\") {
					blank(i++);
					blank(i++);
					continue;
				}
				if (r === "[") inClass = true;
				else if (r === "]") inClass = false;
				else if (r === "/" && !inClass) break;
				blank(i++);
			}
			i++;
			prevMeaningful = "/";
			continue;
		}
		if (!/\s/.test(c)) prevMeaningful = c;
		i++;
	}
	return out.join("");
}

/**
 * The text between the parens of a call whose `(` is at `open`, brace/bracket aware.
 * A regex that stops at the first `)` gets `fetch(url` wrong the moment an options
 * object contains one, which is exactly the case `fetch-no-timeout` has to read.
 */
export function sliceCall(code: string, open: number): string {
	let depth = 0;
	for (let i = open; i < code.length; i++) {
		const c = code[i];
		if (c === "(") depth++;
		else if (c === ")") {
			depth--;
			if (depth === 0) return code.slice(open + 1, i);
		}
	}
	return code.slice(open + 1);
}

const lineOf = (code: string, index: number): number => code.slice(0, index).split("\n").length;
const lineText = (src: string, line: number): string => src.split("\n")[line - 1] ?? "";

/** Every match of `re` (which must be global) as a hit on its own line. */
function matches(re: RegExp, src: string, code: string): Array<{ index: number; line: number; evidence: string }> {
	const out: Array<{ index: number; line: number; evidence: string }> = [];
	for (const m of code.matchAll(re)) {
		const line = lineOf(code, m.index!);
		out.push({ index: m.index!, line, evidence: normalizeEvidence(lineText(src, line)) });
	}
	return out;
}

// ————— exceptions —————

// `// audit-ok(rule-id): reason` on the flagged line or the one above it. Both the id
// and a non-empty reason are required — a bare `audit-ok` suppresses nothing, for the
// same reason a baseline entry needs a note.
const AUDIT_OK = /audit-ok\(([a-z0-9-]+)\)\s*:\s*\S/;

function suppressed(src: string, line: number, ruleId: string): boolean {
	for (const candidate of [lineText(src, line), lineText(src, line - 1)]) {
		const m = candidate.match(AUDIT_OK);
		if (m && m[1] === ruleId) return true;
	}
	return false;
}

// ————— the rules —————

const TS = /^src\/.*\.ts$/;

export const RULES: Rule[] = [
	{
		id: "spawn-no-error-listener",
		severity: "high",
		message:
			"spawn() handle has no 'error' listener. 'error' is emitted asynchronously, so a surrounding try/catch never sees it and an unhandled one takes the process down.",
		include: TS,
		scan(src, code) {
			const out: Hit[] = [];
			for (const m of matches(/\bspawn\s*\(/g, src, code)) {
				// spawnSync throws rather than emitting; it is a different lifecycle.
				if (/\bspawnSync\s*\($/.test(code.slice(0, m.index + 6))) continue;
				const before = code.slice(0, m.index);
				// Capture the handle the call is assigned to, so the listener check is about
				// this spawn rather than any spawn in the file.
				const target = before.match(/(?:const|let|var)\s+([\w$]+)\s*=\s*$|([\w$.#]+)\s*=\s*$/);
				const name = target?.[1] ?? target?.[2];
				if (!name) {
					// Nothing holds the handle, so no listener can ever be attached to it.
					out.push({ line: m.line, evidence: m.evidence });
					continue;
				}
				const bare = name.replace(/^this\./, "");
				// audit-ok(regexp-non-literal-source): the interpolated handle name is regex-escaped inline
				const listener = new RegExp(`(?:this\\.)?${bare.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\s*[?]?\\.on\\s*\\(\\s*["']error["']`);
				// Matched against the ORIGINAL source: stripped() blanks string bodies, so the
				// "error" in .on("error") is exactly the text it erases.
				if (!listener.test(src)) out.push({ line: m.line, evidence: m.evidence });
			}
			return out;
		},
	},

	{
		id: "fetch-no-timeout",
		severity: "medium",
		message:
			"fetch() with no AbortSignal. A hung remote leaves the caller pending forever; every other fetch in this codebase passes AbortSignal.timeout.",
		include: TS,
		scan(src, code) {
			const out: Hit[] = [];
			for (const m of matches(/(^|[^.\w$])(?:\w+\.)?fetch\s*\(/g, src, code)) {
				const open = code.indexOf("(", m.index);
				const args = sliceCall(code, open);
				if (/\bsignal\s*:|AbortSignal/.test(args)) continue;
				out.push({ line: m.line, evidence: m.evidence });
			}
			return out;
		},
	},

	{
		id: "ws-server-no-origin-check",
		severity: "high",
		message:
			"WebSocketServer with no origin or upgrade check. WebSocket handshakes are exempt from the same-origin policy, so binding to loopback does not stop a web page from connecting.",
		include: TS,
		scan(src, code) {
			const hits = matches(/new\s+WebSocket(?:\.)?Server\s*\(/g, src, code);
			if (!hits.length) return [];
			// Assert only that the concept is present in the file. Whether the check is
			// *correct* is unknowable by string analysis and belongs to a test.
			if (/verifyClient|headers\.origin|headers\[["']origin["']\]|handleUpgrade/i.test(code)) return [];
			return hits.map((h) => ({ line: h.line, evidence: h.evidence }));
		},
	},

	{
		id: "promise-timer-never-cleared",
		severity: "low",
		message:
			"This file arms a promise deadline with setTimeout and never calls clearTimeout, so the timer outlives every settled call.",
		include: TS,
		scan(src, code) {
			// Deliberately coarse: proving *this* timer is cleared on every resolution path
			// needs control-flow analysis, which is 200 fragile lines to catch one bug. But
			// scoping to timers armed inside a `new Promise` executor is the difference
			// between firing here and firing on every fire-and-forget UI retry, so it is
			// worth the extra few lines.
			if (/\bclearTimeout\s*\(/.test(code)) return [];
			const out: Hit[] = [];
			for (const p of matches(/new\s+Promise\s*\(/g, src, code)) {
				const open = code.indexOf("(", p.index);
				const body = sliceCall(code, open);
				const rel = body.search(/\bsetTimeout\s*\(/);
				if (rel < 0) continue;
				const line = lineOf(code, open + 1 + rel);
				out.push({ line, evidence: normalizeEvidence(lineText(src, line)) });
			}
			return out;
		},
	},

	{
		id: "download-without-integrity",
		severity: "medium",
		message:
			"This file downloads to disk with no integrity check. A tampered or truncated response is written and then used as if it were trusted.",
		include: TS,
		scan(src, code) {
			const downloads = /\bfetch\s*\(|createWriteStream|Downloader/.test(code);
			const writes = /writeFileSync\s*\(|\bwriteFile\s*\(|createWriteStream/.test(code);
			if (!downloads || !writes) return [];
			if (/sha256|createHash|integrity|checksum/i.test(code)) return [];
			return matches(/\bfetch\s*\(/g, src, code)
				.slice(0, 1)
				.map((h) => ({ line: h.line, evidence: h.evidence }));
		},
	},

	{
		id: "tool-handler-unjailed-fs",
		severity: "high",
		message:
			"Tool handler reaches the filesystem without resolveInRoot. Model-supplied paths are untrusted input, and a cwd option is not a jail — a pattern can climb out of it or ignore it entirely.",
		include: /^src\/native\/tools\/tools\.ts$/,
		scan(src, code) {
			const out: Hit[] = [];
			// Scoped to the one file where model input meets the filesystem. Scoping is what
			// turns an impossible general rule into a reliable specific one.
			for (const m of matches(/async handler\s*\(/g, src, code)) {
				const start = code.indexOf("{", code.indexOf(")", m.index));
				if (start < 0) continue;
				let depth = 0;
				let end = code.length;
				for (let i = start; i < code.length; i++) {
					if (code[i] === "{") depth++;
					else if (code[i] === "}" && --depth === 0) {
						end = i;
						break;
					}
				}
				const body = code.slice(start, end);
				const touchesFs = /\b(readFile|writeFile|readdir|mkdir|rm|rename|glob|stat|open)\s*\(/.test(body);
				if (touchesFs && !/resolveInRoot\s*\(/.test(body)) {
					out.push({ line: m.line, evidence: m.evidence });
				}
			}
			return out;
		},
	},

	{
		id: "layering-violation",
		severity: "medium",
		message:
			"Layering violation. core/ is UI-agnostic and must not import preview/ or electron/; native/ may reference core/ only as `import type`, which erases at compile time.",
		include: TS,
		scan(src, code, path) {
			const out: Hit[] = [];
			for (const m of matches(/^\s*import\s[^;]*?from\s*["'][^"']+["']/gm, src, code)) {
				const line = lineText(src, m.line);
				const spec = line.match(/from\s*["']([^"']+)["']/)?.[1] ?? "";
				const isType = /^\s*import\s+type\b/.test(line);
				if (path.startsWith("src/core/") && /(^|\/)\.\.\/(preview|electron)\//.test(spec)) {
					out.push({ line: m.line, evidence: normalizeEvidence(line) });
				} else if (path.startsWith("src/native/") && /(^|\/)\.\.\/core\//.test(spec) && !isType) {
					out.push({ line: m.line, evidence: normalizeEvidence(line) });
				}
			}
			return out;
		},
	},

	{
		id: "regexp-non-literal-source",
		severity: "medium",
		message:
			"RegExp built from a non-literal source. Review what feeds it: an unbounded pattern from the wire or the model can backtrack catastrophically and hang the single-threaded server.",
		include: TS,
		scan(src, code, path) {
			// A test constructing a pattern from its own constants is not an untrusted-input
			// path, and the rule exists to flag untrusted input.
			if (path.endsWith(".test.ts")) return [];
			const out: Hit[] = [];
			for (const m of matches(/new\s+RegExp\s*\(/g, src, code)) {
				const open = code.indexOf("(", m.index);
				const blanked = sliceCall(code, open);
				// stripped() preserves every offset, so the same span of the ORIGINAL text is
				// the real first argument — needed because a template's `${}` is exactly what
				// gets blanked, and an interpolated pattern is the case worth flagging.
				const first = src.slice(open + 1, open + 1 + blanked.length).split(",")[0]!.trim();
				// A plain string literal, or a template with no interpolation, is fixed at
				// author time and safe. Anything else is a review trigger — the rule cannot
				// detect catastrophic backtracking and does not claim to.
				// A bare `$` is fine in a pattern (it anchors); only `${` makes it non-fixed.
				if (/^(["'])(?:[^\\]|\\.)*\1$/.test(first) || /^`(?:[^`\\$]|\$(?!\{)|\\.)*`$/.test(first)) continue;
				out.push({ line: m.line, evidence: m.evidence });
			}
			return out;
		},
	},

	{
		id: "innerhtml-interpolated",
		severity: "medium",
		message:
			"Interpolated value assigned to innerHTML without escaping. Model output, server error strings and Hub metadata all reach the DOM here.",
		include: /\.(html|ts)$/,
		scan(src, _code, path) {
			if (path.endsWith(".test.ts")) return [];
			const out: Hit[] = [];
			// Scans the ORIGINAL text: the interpolations are the subject, and stripped()
			// blanks exactly what needs reading here.
			// Captures to end of line rather than to the first `;` — an HTML entity contains
			// one (`&lt;`), and truncating there cut escaped interpolations in half, which then
			// read as unescaped and fired on the lines that were already correct.
			const re = /\.(?:innerHTML|outerHTML)\s*\+?=\s*([^\n]*)|insertAdjacentHTML\s*\(([^\n]*)/g;
			for (const m of src.matchAll(re)) {
				const rhs = m[1] ?? m[2] ?? "";
				if (!/\$\{|\s\+\s/.test(rhs)) continue; // a literal or a clear-to-empty
				const interpolations = [...rhs.matchAll(/\$\{([^}]*)\}/g)].map((x) => x[1]!);
				// Every interpolation escaped is fine; one raw value is the finding.
				if (interpolations.length && interpolations.every((x) => /escHtml\s*\(|\.replace\s*\(/.test(x))) continue;
				const line = src.slice(0, m.index!).split("\n").length;
				out.push({ line, evidence: normalizeEvidence(lineText(src, line)) });
			}
			return out;
		},
	},
];

const RULE_IDS = new Set(RULES.map((r) => r.id));

/** Rule ids referenced by `audit-ok` comments that do not exist — a typo silently suppresses nothing. */
export function unknownSuppressions(src: string): string[] {
	return [...src.matchAll(/audit-ok\(([a-z0-9-]+)\)/g)].map((m) => m[1]!).filter((id) => !RULE_IDS.has(id));
}

/**
 * Run every rule over the tree. Reuses `walkFiles` from the tool layer — it already
 * skips node_modules/.git/dist and is already tested, so there is no second walker.
 */
export async function runRules(root: string, rules: Rule[] = RULES): Promise<Finding[]> {
	const out: Finding[] = [];
	await walkFiles(root, root, async (abs, rel) => {
		const path = rel.split(sep).join("/");
		const applicable = rules.filter((r) => r.include.test(path));
		if (!applicable.length) return;

		let src: string;
		try {
			src = await readFile(abs, "utf8");
		} catch {
			return;
		}
		if (src.includes("\0")) return; // binary
		const code = stripped(src);

		for (const rule of applicable) {
			for (const hit of rule.scan(src, code, path)) {
				if (suppressed(src, hit.line, rule.id)) continue;
				out.push({
					rule: rule.id,
					severity: rule.severity,
					file: path,
					line: hit.line,
					message: rule.message,
					evidence: hit.evidence,
					origin: "rule",
				});
			}
		}
	});
	return out;
}

/** Repo-relative, forward-slashed. Exported so the CLI and tests agree on finding paths. */
export const relPath = (root: string, abs: string): string => relative(root, abs).split(sep).join("/");
