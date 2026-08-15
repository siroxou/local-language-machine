// Every rule gets a hit AND a near-miss. The near-miss is the important half: a rule
// that cannot tell a compliant call from a defective one is noise with extra steps.
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { RULES, runRules, sliceCall, stripped, unknownSuppressions } from "./rules.ts";
import { validate } from "./findings.ts";

// fileURLToPath, not .pathname: a file URL percent-encodes, so a checkout under a path
// containing a space resolves to a directory that does not exist. walkFiles swallows an
// unreadable root, so the three repo-level tests below passed by scanning nothing at all —
// green on a dev machine at "/Users/…/Local first IDE" and red on CI, which is the wrong
// way round for a test whose whole job is to look at the real tree.
const REPO = fileURLToPath(new URL("../..", import.meta.url));
const rule = (id: string) => RULES.filter((r) => r.id === id);

/** Write files into a temp root and run one rule over it. */
async function scan(id: string, files: Record<string, string>) {
	const root = await mkdtemp(join(tmpdir(), "llm-rules-"));
	for (const [rel, body] of Object.entries(files)) {
		await mkdir(dirname(join(root, rel)), { recursive: true });
		await writeFile(join(root, rel), body);
	}
	return runRules(root, rule(id));
}

// ————— source preparation —————

test("stripped blanks comments and string bodies but preserves offsets and lines", () => {
	const src = `const a = "fetch(";\n// fetch( in a comment\n/* fetch( */ const b = 1;\n`;
	const out = stripped(src);
	assert.equal(out.length, src.length);
	assert.equal(out.split("\n").length, src.split("\n").length);
	assert.ok(!out.includes("fetch("), "no occurrence should survive inside a string or comment");
	assert.ok(out.includes("const a ="), "code outside the literal is untouched");
	assert.ok(out.includes("const b = 1;"));
});

test("stripped survives a regex literal containing a quote", () => {
	const src = `const re = /["']/g;\nconst real = fetch(url);\n`;
	const out = stripped(src);
	// If the regex desynced the scanner, everything after it would be treated as a string.
	assert.ok(out.includes("fetch(url)"), "code after a quote-bearing regex must stay visible");
});

test("sliceCall spans nested parens and braces", () => {
	const code = `fetch(url, { headers: h(1), signal: s })`;
	assert.equal(sliceCall(code, code.indexOf("(")), `url, { headers: h(1), signal: s }`);
});

// ————— individual rules —————

test("spawn-no-error-listener: fires without a listener, silent with one", async () => {
	const bad = await scan("spawn-no-error-listener", {
		"src/bad.ts": `import { spawn } from "node:child_process";\nconst p = spawn("ls", []);\np.on("exit", () => {});\n`,
	});
	assert.equal(bad.length, 1);
	assert.equal(bad[0]!.line, 2);

	const good = await scan("spawn-no-error-listener", {
		"src/good.ts": `import { spawn } from "node:child_process";\nconst p = spawn("ls", []);\np.on("error", () => {});\n`,
	});
	assert.equal(good.length, 0, "an attached error listener must silence the rule");

	const priv = await scan("spawn-no-error-listener", {
		"src/priv.ts": `class C {\n#proc = spawn("ls", []);\ninit() { this.#proc.on("error", () => {}); }\n}\n`,
	});
	assert.equal(priv.length, 0, "a private-field handle is still a handle");
});

test("fetch-no-timeout: fires bare, silent with AbortSignal", async () => {
	const bad = await scan("fetch-no-timeout", { "src/bad.ts": `const r = await fetch(url, { headers: h });\n` });
	assert.equal(bad.length, 1);

	const good = await scan("fetch-no-timeout", {
		"src/good.ts": `const r = await fetch(url, { headers: h, signal: AbortSignal.timeout(20) });\n`,
	});
	assert.equal(good.length, 0, "a signal in a multi-key options object must be seen");
});

test("ws-server-no-origin-check: fires bare, silent with verifyClient", async () => {
	const bad = await scan("ws-server-no-origin-check", { "src/bad.ts": `const wss = new WebSocketServer({ server });\n` });
	assert.equal(bad.length, 1);

	const good = await scan("ws-server-no-origin-check", {
		"src/good.ts": `const wss = new WebSocketServer({ server, verifyClient: (i) => ok(i) });\n`,
	});
	assert.equal(good.length, 0);
});

test("promise-timer-never-cleared: only fires for timers armed inside a promise executor", async () => {
	const bad = await scan("promise-timer-never-cleared", {
		"src/bad.ts": `function f() {\nreturn new Promise((res, rej) => {\nsetTimeout(() => rej(new Error("late")), 20);\n});\n}\n`,
	});
	assert.equal(bad.length, 1);
	assert.equal(bad[0]!.line, 3);

	const cleared = await scan("promise-timer-never-cleared", {
		"src/ok.ts": `function f() {\nreturn new Promise((res) => {\nconst t = setTimeout(res, 20);\nclearTimeout(t);\n});\n}\n`,
	});
	assert.equal(cleared.length, 0);

	// A fire-and-forget retry alongside an unrelated promise is not a leak.
	const unrelated = await scan("promise-timer-never-cleared", {
		"src/retry.ts": `const probe = () => new Promise((res) => server.on("x", res));\nsetTimeout(() => reload(), 250);\n`,
	});
	assert.equal(unrelated.length, 0, "a timer outside any promise executor must not fire");
});

test("tool-handler-unjailed-fs: fires on a handler that skips resolveInRoot", async () => {
	const bad = await scan("tool-handler-unjailed-fs", {
		"src/native/tools/tools.ts": `export function buildTools(ctx) { return {\nglob: { async handler({ pattern }) {\nreturn await glob(pattern, { cwd: ctx.root });\n} },\n}; }\n`,
	});
	assert.equal(bad.length, 1);

	const good = await scan("tool-handler-unjailed-fs", {
		"src/native/tools/tools.ts": `export function buildTools(ctx) { return {\nread: { async handler({ path }) {\nconst abs = resolveInRoot(ctx.root, path);\nreturn await readFile(abs);\n} },\n}; }\n`,
	});
	assert.equal(good.length, 0, "cwd is not a jail, but resolveInRoot is");
});

test("layering-violation: value imports across layers fire, type imports do not", async () => {
	const bad = await scan("layering-violation", {
		"src/core/a.ts": `import { send } from "../preview/server.ts";\n`,
		"src/native/b.ts": `import { loadSettings } from "../core/settings.ts";\n`,
	});
	assert.equal(bad.length, 2);

	const good = await scan("layering-violation", {
		"src/native/b.ts": `import type { McpServerConfig } from "../core/settings.ts";\n`,
	});
	assert.equal(good.length, 0, "a type-only import erases at compile time and is not a layering edge");
});

test("regexp-non-literal-source: literals pass, interpolation and identifiers fire", async () => {
	const good = await scan("regexp-non-literal-source", {
		"src/good.ts": "const a = new RegExp(\"^ok$\");\nconst b = new RegExp(`^plain$`);\n",
	});
	assert.equal(good.length, 0);

	const bad = await scan("regexp-non-literal-source", {
		"src/bad.ts": "const a = new RegExp(userInput);\nconst b = new RegExp(`^${x}$`);\n",
	});
	assert.equal(bad.length, 2, "an interpolated template is not a fixed pattern");
});

test("innerhtml-interpolated: raw interpolation fires, escaped and literal do not", async () => {
	const bad = await scan("innerhtml-interpolated", {
		"ui.html": "<script>el.innerHTML = `<b>${m.id}</b>`;</script>\n",
	});
	assert.equal(bad.length, 1);

	const good = await scan("innerhtml-interpolated", {
		"ui.html": "<script>el.innerHTML = `<b>${escHtml(m.id)}</b>`;\nother.innerHTML = \"\";</script>\n",
	});
	assert.equal(good.length, 0);
});

// ————— the exception convention —————

test("audit-ok suppresses only its own rule, and only with a reason", async () => {
	const suppressed = await scan("fetch-no-timeout", {
		"src/a.ts": `// audit-ok(fetch-no-timeout): the caller aborts this one\nconst r = await fetch(url);\n`,
	});
	assert.equal(suppressed.length, 0);

	const wrongRule = await scan("fetch-no-timeout", {
		"src/b.ts": `// audit-ok(some-other-rule): unrelated\nconst r = await fetch(url);\n`,
	});
	assert.equal(wrongRule.length, 1, "an annotation for another rule must not suppress this one");

	const noReason = await scan("fetch-no-timeout", {
		"src/c.ts": `// audit-ok(fetch-no-timeout):\nconst r = await fetch(url);\n`,
	});
	assert.equal(noReason.length, 1, "a bare annotation with no reason suppresses nothing");
});

test("unknownSuppressions catches a typo'd rule id", () => {
	assert.deepEqual(unknownSuppressions("// audit-ok(fetch-no-timout): typo\n"), ["fetch-no-timout"]);
	assert.deepEqual(unknownSuppressions("// audit-ok(fetch-no-timeout): correct\n"), []);
});

// ————— against the real tree —————
// These two assert properties that stay true as defects get fixed, so they never churn.

test("every finding on the real repo cites a line that exists", async () => {
	const findings = await runRules(REPO);
	const { dropped } = await validate(REPO, findings);
	assert.deepEqual(dropped, [], "a rule that reports a bad citation is broken");
});

test("the layering invariant holds: core/ is UI-agnostic and native/ imports core only as types", async () => {
	const findings = await runRules(REPO, rule("layering-violation"));
	assert.deepEqual(
		findings.map((f) => `${f.file}:${f.line}`),
		[],
		"README claims nothing in the core knows about HTTP or webviews — keep it true",
	);
});

test("no audit-ok annotation in the repo names a rule that does not exist", async () => {
	const { readFile } = await import("node:fs/promises");
	const { walkFiles } = await import("../native/tools/tools.ts");
	const bad: string[] = [];
	await walkFiles(REPO, REPO, async (abs, rel) => {
		// Test files carry deliberately bogus ids as fixtures — including the typo this very
		// rule exists to catch — so scanning them would report the suite's own inputs.
		if (!/\.(ts|html)$/.test(rel) || rel.endsWith(".test.ts")) return;
		for (const id of unknownSuppressions(await readFile(abs, "utf8"))) bad.push(`${rel}: ${id}`);
	});
	assert.deepEqual(bad, []);
});
