// Local Language Machine — curated model registry + local cache.
//
// Replaces backend port auto-detection: models resolve to a file on disk under
// ~/.local-language-machine/models. Present → load. Absent → download once
// (resumable) and verify. No localhost probing, no external service.

import { createHash } from "node:crypto";
import { createReadStream, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { mkdir, stat } from "node:fs/promises";
import { join } from "node:path";
import { createModelDownloader } from "node-llama-cpp";
import { appHome } from "../paths.ts";
import { getHfToken } from "./config.ts";
import type { HubFile } from "./hub.ts";

export interface ModelEntry {
	/** Stable id shown in the picker, e.g. "qwen2.5-coder:7b". */
	id: string;
	label: string;
	/** node-llama-cpp resolvable URI (hf:owner/repo:quant). Only used on cache miss. */
	modelUri: string;
	/** Local file name in the cache dir. */
	fileName: string;
	quant: string;
	sizeGB: number;
	/**
	 * Pinned SHA-256 of the GGUF. Verified after download — the trust boundary
	 * for "this is the model we curated". Left undefined until curated: a wrong
	 * pinned hash would reject a good download, so we only pin hashes we have
	 * actually computed and verified. See tools/pin-hashes when adding a model.
	 */
	sha256?: string;
}

export const REGISTRY: ModelEntry[] = [
	{
		id: "qwen2.5-coder:7b",
		label: "Qwen2.5-Coder 7B (Instruct)",
		// Explicit file path (not the :quant shorthand) → resolves straight to
		// HF's public CDN, no auth-gated manifest lookup.
		modelUri: "hf:Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/qwen2.5-coder-7b-instruct-q4_k_m.gguf",
		fileName: "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
		quant: "Q4_K_M",
		sizeGB: 4.7,
	},
	{
		id: "qwen2.5-coder:1.5b",
		label: "Qwen2.5-Coder 1.5B (Instruct) — fast, tool-capable",
		modelUri: "hf:Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
		fileName: "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
		quant: "Q4_K_M",
		sizeGB: 1.0,
	},
	{
		// Tiny model for smoke tests / low-RAM machines. Fast on CPU.
		id: "smollm2:135m",
		label: "SmolLM2 135M (Instruct)",
		modelUri: "hf:bartowski/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q4_K_M.gguf",
		fileName: "SmolLM2-135M-Instruct-Q4_K_M.gguf",
		quant: "Q4_K_M",
		sizeGB: 0.1,
	},
];

export function cacheDir(): string {
	return join(appHome(), "models");
}

const userModelsPath = () => join(cacheDir(), "user-models.json");

/** Models the user added from Hugging Face (persisted next to the cache). */
export function loadUserModels(): ModelEntry[] {
	try {
		const data = JSON.parse(readFileSync(userModelsPath(), "utf8"));
		return Array.isArray(data) ? data : [];
	} catch {
		return [];
	}
}

export function addUserModel(entry: ModelEntry): void {
	const models = loadUserModels().filter((m) => m.id !== entry.id); // dedupe by id
	models.push(entry);
	mkdirSync(cacheDir(), { recursive: true });
	writeFileSync(userModelsPath(), JSON.stringify(models, null, 2));
}

/** Curated defaults + user-added Hugging Face models. */
export function allModels(): ModelEntry[] {
	return [...REGISTRY, ...loadUserModels()];
}

/** Build a registry entry from a Hugging Face repo + chosen GGUF file. */
export function entryFromHub(repoId: string, file: HubFile): ModelEntry {
	const basename = file.path.split("/").pop()!;
	const quant = basename.match(/(IQ\d[\w]*|Q\d[\w_]*|F16|BF16|F32)/i)?.[1].toUpperCase() ?? "GGUF";
	const repoName = repoId.split("/").pop()!.replace(/-?GGUF$/i, "");
	return {
		id: `${repoId}:${quant.toLowerCase()}`,
		label: `${repoName} · ${quant}`,
		modelUri: `hf:${repoId}/${file.path}`,
		// Prefix with the owner so basenames from different repos don't collide in the flat cache dir.
		fileName: `${repoId.replace(/\//g, "__")}__${basename}`,
		quant,
		sizeGB: +(file.sizeBytes / 1e9).toFixed(2),
		sha256: file.sha256,
	};
}

export function resolve(id: string): ModelEntry | undefined {
	return allModels().find((m) => m.id === id);
}

export function modelPathFor(entry: ModelEntry): string {
	return join(cacheDir(), entry.fileName);
}

async function fileExists(path: string): Promise<boolean> {
	try {
		return (await stat(path)).isFile();
	} catch {
		return false;
	}
}

export async function sha256File(path: string): Promise<string> {
	const hash = createHash("sha256");
	for await (const chunk of createReadStream(path)) hash.update(chunk);
	return hash.digest("hex");
}

/** Known models that are already present in the cache. */
export async function listCached(): Promise<Array<{ id: string; path: string }>> {
	const out: Array<{ id: string; path: string }> = [];
	for (const entry of allModels()) {
		const path = modelPathFor(entry);
		if (await fileExists(path)) out.push({ id: entry.id, path });
	}
	return out;
}

export interface EnsureProgress {
	downloadedBytes: number;
	totalBytes: number;
}

/**
 * Resolve a model id to a local GGUF path, downloading it once if missing.
 * Verifies the pinned sha256 when present. Returns the on-disk path.
 */
export async function ensureModel(
	id: string,
	opts: { onProgress?: (p: EnsureProgress) => void } = {},
): Promise<string> {
	const entry = resolve(id);
	if (!entry) throw new Error(`Unknown model id: ${id}`);

	const dir = cacheDir();
	const path = modelPathFor(entry);

	if (await fileExists(path)) {
		await verify(entry, path);
		return path;
	}

	await mkdir(dir, { recursive: true });
	const token = getHfToken();
	const downloader = await createModelDownloader({
		modelUri: entry.modelUri,
		dirPath: dir,
		fileName: entry.fileName,
		headers: token ? { Authorization: `Bearer ${token}` } : undefined,
		onProgress: opts.onProgress
			? ({ totalSize, downloadedSize }) =>
					opts.onProgress!({ downloadedBytes: downloadedSize, totalBytes: totalSize })
			: undefined,
	});

	const downloadedPath = await downloader.download();
	await verify(entry, downloadedPath);
	return downloadedPath;
}

async function verify(entry: ModelEntry, path: string): Promise<void> {
	if (!entry.sha256) return; // not yet pinned — nothing to check against
	const actual = await sha256File(path);
	if (actual !== entry.sha256) {
		throw new Error(
			`Checksum mismatch for ${entry.id}: expected ${entry.sha256}, got ${actual}`,
		);
	}
}
