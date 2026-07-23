// Local Language Machine — "use the fine-tuned model in the Assistant".
//
// The Assistant runs GGUF via node-llama-cpp, but training produces a LoRA adapter. MLX's
// own --export-gguf can't serialize llama-family weights ("row-major arrays"), so the
// reliable path is: mlx_lm.fuse --dequantize → a standalone f16 HF model → llama.cpp's
// convert_hf_to_gguf.py → GGUF → register so the existing engine loads it unchanged.
// Best-effort by design: needs the converter's Python deps (torch/gguf) and network on the
// first run; every failure surfaces a clear message and leaves the rest of the app intact.

import { existsSync, mkdirSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { appHome } from "../paths.ts";
import { ensureVenv } from "./venv.ts";
import { runManaged, withBusy } from "./train.ts";
import { readMeta, writeMeta, runDirOf } from "./runs.ts";
import { addUserModel, cacheDir, type ModelEntry } from "../models/registry.ts";

type Emit = (o: any) => void;
const log = (emit: Emit, line: string) => emit({ type: "train-log", line });

// Pinned to a llama.cpp tag whose converter is a single self-contained file (master split
// it into a `conversion` package). b4400 needs only the `gguf` pip package + torch.
const CONVERTER_URL = "https://raw.githubusercontent.com/ggml-org/llama.cpp/b4400/convert_hf_to_gguf.py";
const toolingDir = () => join(appHome(), "tooling");

async function ensureConvertDeps(python: string, emit: Emit): Promise<void> {
	const have = await runManaged(python, ["-c", "import torch, gguf"], appHome(), () => {});
	if (have === 0) return;
	log(emit, "▸ Installing GGUF-convert deps (torch, gguf) — first run only, ~1GB …");
	const code = await runManaged(python, ["-m", "pip", "install", "-U", "torch", "gguf"], appHome(), (l) => log(emit, l));
	if (code !== 0) throw new Error("Could not install the GGUF converter dependencies (torch, gguf).");
}

async function ensureConverter(emit: Emit): Promise<string> {
	const dir = toolingDir();
	mkdirSync(dir, { recursive: true });
	const script = join(dir, "convert_hf_to_gguf.py");
	if (!existsSync(script)) {
		log(emit, "▸ Fetching llama.cpp GGUF converter …");
		const res = await fetch(CONVERTER_URL);
		if (!res.ok) throw new Error(`Could not download the GGUF converter (${res.status}).`);
		writeFileSync(script, await res.text());
	}
	return script;
}

/** Fuse the LoRA adapter into the base model, convert to GGUF, and register it. */
export function startConvert(runId: string, emit: Emit): Promise<void> {
	return withBusy(async () => {
		const meta = readMeta(runId);
		if (!meta) throw new Error(`Unknown run: ${runId}`);
		if (!meta.baseModel) throw new Error("This run has no recorded base model — retrain to convert.");
		if (!existsSync(meta.adapterDir)) throw new Error("Adapter not found for this run.");

		const python = await ensureVenv(meta.backend, (l) => log(emit, l));
		const fusedDir = join(runDirOf(runId), "fused");
		mkdirSync(fusedDir, { recursive: true });
		mkdirSync(cacheDir(), { recursive: true });
		const ggufName = `trained-${runId.slice(0, 19)}.gguf`;
		const outFile = join(cacheDir(), ggufName);

		// Fusing copies tokenizer/config from the base repo, which needs the COMPLETE snapshot
		// (training only pulled the weights). Complete it first (argv, not interpolated).
		log(emit, `▸ Completing base-model snapshot for ${meta.baseModel} …`);
		const dl = await runManaged(python, ["-c", "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1])", meta.baseModel], appHome(), (l) => log(emit, l));
		if (dl !== 0) throw new Error("Could not complete the base-model snapshot (needed to fuse).");

		log(emit, "▸ Fusing adapter into a standalone f16 model …");
		const fuse = await runManaged(python, ["-m", "mlx_lm.fuse", "--model", meta.baseModel, "--adapter-path", meta.adapterDir, "--save-path", fusedDir, "--dequantize"], appHome(), (l) => log(emit, l));
		if (fuse !== 0) throw new Error(`Fuse failed (exit ${fuse}).`);

		await ensureConvertDeps(python, emit);
		const script = await ensureConverter(emit);

		log(emit, "▸ Converting fused model → GGUF (f16) …");
		const code = await runManaged(python, [script, fusedDir, "--outfile", outFile, "--outtype", "f16"], appHome(), (l) => log(emit, l));
		if (code !== 0) throw new Error(`GGUF conversion failed (exit ${code}). This architecture may be unsupported by llama.cpp's converter.`);
		if (!existsSync(outFile)) throw new Error("Converter reported success but produced no .gguf.");

		const shortBase = meta.baseModel.split("/").pop() ?? meta.baseModel;
		const entry: ModelEntry = {
			id: `trained:${runId.slice(0, 19)}`,
			label: `Fine-tuned ${shortBase}`,
			modelUri: `file:${outFile}`, // never fetched — the file already lives in the cache dir
			fileName: ggufName,
			quant: "F16",
			sizeGB: +(statSync(outFile).size / 1e9).toFixed(2),
		};
		addUserModel(entry);
		meta.ggufModelId = entry.id;
		writeMeta(meta);
		emit({ type: "convert-done", runId, modelId: entry.id, label: entry.label });
	}).catch((e) => emit({ type: "train-error", message: e instanceof Error ? e.message : String(e) }));
}
