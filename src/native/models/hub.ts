// Local Language Machine — Hugging Face Hub REST client (search + file tree).
// Uses Node's global fetch (no dependency). Only reached when *adding* a model;
// runtime inference stays fully offline. Sends the user's HF token when present.

import { getHfToken } from "./config.ts";

const HF = "https://huggingface.co";

export interface HubModel {
	id: string; // "owner/repo"
	downloads: number;
	likes: number;
}

export interface HubFile {
	path: string; // path within the repo
	sizeBytes: number;
	sha256?: string; // from lfs.oid — a single-file GGUF only
	sharded: boolean;
}

function authHeaders(): Record<string, string> {
	const t = getHfToken();
	return t ? { Authorization: `Bearer ${t}` } : {};
}

// The model manager runs these on a click, so a stalled connection would leave the panel
// spinning with no error and no way out. fetch has no default timeout — give it one.
const HF_TIMEOUT_MS = 20_000;

async function hfGet(path: string): Promise<unknown> {
	const res = await fetch(`${HF}${path}`, { headers: authHeaders(), signal: AbortSignal.timeout(HF_TIMEOUT_MS) });
	if (res.status === 429) throw new Error("Hugging Face rate limit — try again shortly.");
	if (res.status === 401 || res.status === 403) throw new Error("Access denied — this repo may be gated or private. Add your Hugging Face token.");
	if (res.status === 404) throw new Error("Not found on Hugging Face.");
	if (!res.ok) throw new Error(`Hugging Face error ${res.status}.`);
	return res.json();
}

/** Search the Hub for public GGUF models, most-downloaded first. */
export async function searchGguf(query: string, limit = 20): Promise<HubModel[]> {
	const q = encodeURIComponent(query.trim());
	const data = await hfGet(`/api/models?filter=gguf&search=${q}&sort=downloads&direction=-1&limit=${limit}`);
	const arr = Array.isArray(data) ? data : [];
	return arr
		.filter((m: any) => !m.private)
		.map((m: any) => ({ id: m.id ?? m.modelId, downloads: m.downloads ?? 0, likes: m.likes ?? 0 }))
		.filter((m: HubModel) => typeof m.id === "string");
}

/** List a repo's GGUF files, collapsing multi-part (sharded) GGUF into one entry. */
export async function listGgufFiles(repoId: string): Promise<HubFile[]> {
	const data = await hfGet(`/api/models/${repoId}/tree/main?recursive=true`);
	const arr = Array.isArray(data) ? data : [];
	const ggufs = arr.filter((e: any) => e.type === "file" && typeof e.path === "string" && e.path.toLowerCase().endsWith(".gguf"));
	return collapseShards(ggufs);
}

// `…-00001-of-000NN.gguf` → one entry (entrypoint = first shard, size = Σ of parts).
// node-llama-cpp downloads the remaining shards from the first automatically.
export function collapseShards(files: any[]): HubFile[] {
	const shardRe = /-(\d{5})-of-\d{5}\.gguf$/i;
	const groups = new Map<string, { firstPath?: string; total: number }>();
	const out: HubFile[] = [];
	for (const f of files) {
		const size = Number(f.size ?? f.lfs?.size ?? 0);
		const m = shardRe.exec(f.path);
		if (!m) {
			const oid = f.lfs?.oid;
			out.push({ path: f.path, sizeBytes: size, sha256: /^[a-f0-9]{64}$/i.test(oid) ? oid : undefined, sharded: false });
			continue;
		}
		const base = f.path.replace(shardRe, "");
		const g = groups.get(base) ?? { total: 0 };
		g.total += size;
		if (m[1] === "00001") g.firstPath = f.path;
		groups.set(base, g);
	}
	for (const g of groups.values()) {
		if (g.firstPath) out.push({ path: g.firstPath, sizeBytes: g.total, sharded: true });
	}
	return out.sort((a, b) => a.sizeBytes - b.sizeBytes);
}
