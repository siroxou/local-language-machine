// Local Language Machine — bundled inference engine.
//
// Owned abstraction over the vendored runtime. The rest of the product talks to
// `InferenceEngine` / `ChatSession` only and never names the underlying library,
// so swapping the backend later touches this file alone.
//
// Design note: the engine is *session*-based, not a stateless `stream(messages)`
// call. A session holds a live context + KV cache, so each user turn only
// evaluates the new tokens instead of recomputing the whole conversation — the
// reason to bundle a real engine in the first place.

import {
	getLlama,
	LlamaChatSession,
	defineChatSessionFunction,
	type Llama,
	type LlamaModel,
	type GbnfJsonSchema,
	type ChatHistoryItem,
} from "node-llama-cpp";

/** Serializable conversation history — persisted for session resume + spliced during compaction. */
export type { ChatHistoryItem };

/** A tool the model may call. `params` is a JSON schema for the arguments. */
export interface ToolDef {
	description?: string;
	params?: GbnfJsonSchema;
	/** Executed when the model calls the tool. Return value is fed back to the model. */
	handler: (args: any) => Promise<unknown> | unknown;
}

export interface PromptOptions {
	tools?: Record<string, ToolDef>;
	/** Called with each streamed text chunk as it is generated. */
	onText?: (chunk: string) => void;
	maxTokens?: number;
}

export interface ChatSession {
	/** Chat turn. Streams via `onText`, runs any tool handlers, resolves to the final text. */
	prompt(text: string, opts?: PromptOptions): Promise<string>;
	/**
	 * Grammar-constrained JSON output — generation is forced to match `jsonSchema`,
	 * so the result always parses. Backs deterministic tool routing / structured
	 * extraction (and the future Hermes fallback for models without native tool calls).
	 */
	structured<T = unknown>(text: string, jsonSchema: GbnfJsonSchema): Promise<T>;
	/** Tokens currently held in the KV cache — i.e. how much of the context window is in use. */
	usedTokens?(): number;
	/** Current conversation history (for persistence + compaction). */
	getHistory(): ChatHistoryItem[];
	/** Replace the conversation history (session resume, or splicing in a compaction summary). */
	setHistory(history: ChatHistoryItem[]): void;
	dispose(): void;
}

export interface LoadOptions {
	modelPath: string;
	// Calibration knobs — real hardware varies. Auto-detect first, override when needed.
	// Undefined = let the runtime auto-pick from the GGUF (never silently override the model's own defaults).
	gpuLayers?: "auto" | "max" | number;
	contextSize?: "auto" | number;
	batchSize?: number;
	threads?: number;
	flashAttention?: boolean;
	useMmap?: boolean;
	useMlock?: boolean;
	/** Context seed — makes generation reproducible when set. */
	seed?: number;
}

/**
 * Per-generation sampling. Held on the engine (not passed per prompt) so a tuning
 * change applies to the very next turn without threading through the agent loop.
 * Every field undefined = the runtime's own default.
 */
export interface SamplingOptions {
	temperature?: number;
	topK?: number;
	topP?: number;
	minP?: number;
	/** Single knob → mapped to node-llama-cpp's `{ penalty }`. */
	repeatPenalty?: number;
	/** Response length cap. 0/undefined = unlimited. */
	maxTokens?: number;
}

export interface InferenceEngine {
	readonly gpu: string;
	/** Resolved context window (tokens) of the current session, or undefined if no model loaded. */
	readonly contextSize: number | undefined;
	load(opts: LoadOptions): Promise<void>;
	/** `contextSize` overrides the model's own window for this session only — a short-lived
	 *  helper session (summarization) must not allocate a second full-size KV cache. */
	createSession(opts?: { systemPrompt?: string; contextSize?: number }): Promise<ChatSession>;
	/** Set the sampling defaults applied to every subsequent prompt. Live — no reload. */
	setSampling(opts: SamplingOptions): void;
	unload(): Promise<void>;
}

export class LlamaCppEngine implements InferenceEngine {
	#llama?: Llama;
	#model?: LlamaModel;
	// Context-level load opts, applied in every createContext(). Refreshed on load().
	// `seed` is deliberately NOT in here: it is not a LlamaContextOptions field, so createContext
	// silently ignores it. It belongs to the per-prompt sampler — see the prompt() call below.
	#contextOpts: Pick<LoadOptions, "contextSize" | "batchSize" | "threads" | "flashAttention"> = { contextSize: "auto" };
	#seed?: number;
	#sampling: SamplingOptions = {};
	#contextSize?: number;
	readonly #gpuPref: "auto" | "cuda" | "vulkan" | "metal" | false;

	constructor(opts: { gpu?: "auto" | "cuda" | "vulkan" | "metal" | false } = {}) {
		this.#gpuPref = opts.gpu ?? "auto";
	}

	setSampling(opts: SamplingOptions): void {
		this.#sampling = opts ?? {};
	}

	get gpu(): string {
		return this.#llama ? String(this.#llama.gpu) : "unloaded";
	}

	get contextSize(): number | undefined {
		return this.#contextSize;
	}

	async load(opts: LoadOptions): Promise<void> {
		await this.unload();
		this.#llama ??= await getLlama({ gpu: this.#gpuPref });
		this.#model = await this.#llama.loadModel({
			modelPath: opts.modelPath,
			gpuLayers: opts.gpuLayers ?? "auto",
			useMmap: opts.useMmap,
			useMlock: opts.useMlock,
		});
		// Context opts take effect when a session's context is created, below.
		this.#contextOpts = {
			contextSize: opts.contextSize ?? "auto",
			batchSize: opts.batchSize,
			threads: opts.threads,
			flashAttention: opts.flashAttention,
		};
		this.#seed = opts.seed;
	}

	async createSession(opts: { systemPrompt?: string; contextSize?: number } = {}): Promise<ChatSession> {
		const model = this.#model;
		const llama = this.#llama;
		if (!model || !llama) throw new Error("Engine has no model loaded; call load() first.");
		const self = this; // the prompt wrapper reads self.#sampling at call time, so live tuning changes apply next turn

		const context = await model.createContext(
			opts.contextSize ? { ...this.#contextOpts, contextSize: opts.contextSize } : this.#contextOpts,
		);
		// Only the main session defines the window the status bar reports — a sized helper
		// session would otherwise overwrite it with its own much smaller number.
		if (!opts.contextSize) this.#contextSize = context.contextSize;
		const sequence = context.getSequence();
		const session = new LlamaChatSession({
			contextSequence: sequence,
			systemPrompt: opts.systemPrompt,
			// Our system prompt carries the tool-calling protocol, not just flavour text — on a model
			// whose chat wrapper reports no system-message support it must still be injected, or the
			// agent loop has no instructions at all.
			forceAddSystemPrompt: true,
		});

		return {
			async prompt(text, promptOpts = {}) {
				const functions = promptOpts.tools
					? Object.fromEntries(
							Object.entries(promptOpts.tools).map(([name, t]) => [
								name,
								defineChatSessionFunction({
									description: t.description,
									params: t.params as any,
									handler: t.handler,
								}),
							]),
						)
					: undefined;

				const s = self.#sampling; // read live so a tuning change lands on the next turn
				return session.prompt(text, {
					functions,
					onTextChunk: promptOpts.onText,
					seed: self.#seed, // sampler-level: createContext() has no `seed` and silently drops it
					temperature: s.temperature,
					topK: s.topK,
					topP: s.topP,
					minP: s.minP,
					repeatPenalty: s.repeatPenalty != null ? { penalty: s.repeatPenalty } : undefined,
					maxTokens: promptOpts.maxTokens ?? (s.maxTokens || undefined),
				});
			},

			async structured<T>(text: string, jsonSchema: GbnfJsonSchema): Promise<T> {
				const grammar = await llama.createGrammarForJsonSchema(jsonSchema as any);
				const res = await session.prompt(text, { grammar });
				return grammar.parse(res) as T;
			},

			usedTokens() {
				return sequence.nextTokenIndex; // KV-cache fill = context window in use
			},

			getHistory() {
				return session.getChatHistory();
			},

			setHistory(history) {
				session.setChatHistory(history);
			},

			dispose() {
				context.dispose();
			},
		};
	}

	async unload(): Promise<void> {
		await this.#model?.dispose();
		this.#model = undefined;
		this.#contextSize = undefined;
	}
}
