// Local Language Machine — local app config (~/.local-language-machine/config.json).
// Currently holds the user's optional Hugging Face token for gated/private repos.
// The user pastes their own token; the app never generates or requests one.

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { appHome } from "../paths.ts";

/**
 * Per-model tuning preset (LM-Studio-shaped). Every field optional/unset = "let the
 * runtime auto-pick" — a model with no saved preset loads exactly as it did before.
 * `load.*` needs a model reload to take effect; `inference.*` applies live.
 */
export interface TuningConfig {
	inference?: {
		temperature?: number;
		topK?: number;
		topP?: number;
		minP?: number;
		repeatPenalty?: number;
		maxTokens?: number;
		systemPrompt?: string;
	};
	load?: {
		contextSize?: number;
		gpuLayers?: number;
		batchSize?: number;
		threads?: number;
		flashAttention?: boolean;
		useMmap?: boolean;
		useMlock?: boolean;
		seed?: number;
	};
}

interface Config {
	hfToken?: string;
	/** Tuning presets keyed by model id. */
	tuning?: Record<string, TuningConfig>;
}

const configPath = () => join(appHome(), "config.json");

function read(): Config {
	try {
		return JSON.parse(readFileSync(configPath(), "utf8"));
	} catch {
		return {};
	}
}

function write(cfg: Config): void {
	mkdirSync(dirname(configPath()), { recursive: true });
	writeFileSync(configPath(), JSON.stringify(cfg, null, 2));
}

export function getHfToken(): string | undefined {
	const t = read().hfToken;
	return t && t.trim() ? t.trim() : undefined;
}

export function setHfToken(token: string | undefined): void {
	const cfg = read();
	if (token && token.trim()) cfg.hfToken = token.trim();
	else delete cfg.hfToken;
	write(cfg);
}

/** A model's saved tuning preset, or {} (all-auto) when none is stored. */
export function getTuning(modelId: string): TuningConfig {
	return read().tuning?.[modelId] ?? {};
}

/** Persist a model's tuning preset. An empty cfg drops the entry (reset to defaults). */
export function setTuning(modelId: string, cfg: TuningConfig): void {
	const c = read();
	const empty = !cfg.inference && !cfg.load;
	c.tuning = { ...c.tuning };
	if (empty) delete c.tuning[modelId];
	else c.tuning[modelId] = cfg;
	write(c);
}
