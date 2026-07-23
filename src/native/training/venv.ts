// Local Language Machine — one-time trainer environment bootstrap.
//
// Training needs a Python toolchain the inference app doesn't ship, so the first run
// lazily builds a dedicated venv under the app data dir and pip-installs the backend.
// Install output is streamed to the UI (via onLine) so the user sees setup progress.
// First-run setup + base-model pull need network — the same controlled bend the model
// download already makes; steady-state training is on-device.

import { spawn, execFileSync } from "node:child_process";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { appHome } from "../paths.ts";
import type { Backend } from "./train.ts";

const venvDir = () => join(appHome(), "training", "venv");
const pythonBin = () =>
	process.platform === "win32" ? join(venvDir(), "Scripts", "python.exe") : join(venvDir(), "bin", "python");

// MLX runs on 3.9+; Unsloth needs 3.10+. System python3 here is 3.9.6, so Unsloth
// requires a newer interpreter to be installed (e.g. via Homebrew).
const MIN_MINOR: Record<Backend, number> = { mlx: 9, unsloth: 10 };

/** First interpreter on PATH whose 3.x minor is ≥ `min`. Throws with guidance if none. */
function findBasePython(min: number): string {
	const candidates = ["python3.13", "python3.12", "python3.11", "python3.10", "python3.9", "python3"];
	for (const c of candidates) {
		try {
			const out = execFileSync(c, ["--version"], { encoding: "utf8", timeout: 3000 });
			const m = out.match(/Python 3\.(\d+)/);
			if (m && +m[1]! >= min) return c;
		} catch {}
	}
	throw new Error(`Need Python 3.${min}+ on PATH for this backend — install a newer Python (e.g. \`brew install python@3.12\`).`);
}

function run(cmd: string, args: string[], onLine: (line: string) => void): Promise<void> {
	return new Promise((resolve, reject) => {
		const p = spawn(cmd, args, { env: process.env });
		let buf = "";
		const pump = (chunk: string) => {
			buf += chunk;
			let i: number;
			while ((i = buf.indexOf("\n")) >= 0) { onLine(buf.slice(0, i)); buf = buf.slice(i + 1); }
		};
		p.stdout?.on("data", (d) => pump(d.toString()));
		p.stderr?.on("data", (d) => pump(d.toString()));
		p.on("error", reject);
		p.on("close", (code) => { if (buf) onLine(buf); code === 0 ? resolve() : reject(new Error(`\`${cmd} ${args[0]}\` exited ${code}.`)); });
	});
}

/**
 * Ensure the trainer venv exists with `backend` installed; return the venv's python.
 * A per-backend marker file makes subsequent runs a no-op.
 */
export async function ensureVenv(backend: Backend, onLine: (line: string) => void): Promise<string> {
	const dir = venvDir();
	const py = pythonBin();
	const marker = join(dir, `.${backend}-ready`);
	if (existsSync(marker) && existsSync(py)) return py;

	mkdirSync(dir, { recursive: true });
	if (!existsSync(py)) {
		const base = findBasePython(MIN_MINOR[backend]);
		onLine(`▸ Creating trainer venv (${base}) at ${dir} …`);
		await run(base, ["-m", "venv", dir], onLine);
	}

	onLine("▸ Upgrading pip …");
	await run(py, ["-m", "pip", "install", "-U", "pip"], onLine);

	// default pip install. CUDA users may need Unsloth's hardware-specific
	// install line — swap the package list here if a run fails to find GPU kernels.
	const pkgs = backend === "mlx" ? ["mlx-lm"] : ["unsloth", "trl", "peft"];
	onLine(`▸ Installing ${pkgs.join(", ")} … (first run only, may take a few minutes)`);
	await run(py, ["-m", "pip", "install", "-U", ...pkgs], onLine);

	writeFileSync(marker, new Date().toISOString());
	onLine("▸ Trainer environment ready.");
	return py;
}
