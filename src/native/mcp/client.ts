// Local Language Machine — minimal MCP stdio client.
//
// hand-rolled newline-delimited JSON-RPC over a spawned stdio server,
// instead of pulling in @modelcontextprotocol/sdk. The whole surface we need is
// initialize → tools/list → tools/call, which is ~a screen of code and needs no
// install (keeps the build fully offline). The SDK + remote SSE/HTTP transport is
// the upgrade if we ever need resources, notifications, or hosted servers.
//
// stdio servers are local processes, so MCP stays offline by construction. Remote (SSE/HTTP)
// transport is not implemented at all — see McpServerConfig in core/settings.ts.

import { spawn, type ChildProcess } from "node:child_process";
import type { ToolDef } from "../inference/engine.ts";
import type { McpServerConfig } from "../../core/settings.ts";

// A stdio server that never answers must not hang the turn behind it.
const REQUEST_TIMEOUT_MS = 20_000;
// Cap on unframed stdout held in memory while waiting for a newline.
const MAX_BUFFERED_BYTES = 8 * 1024 * 1024;

interface RpcTool {
	name: string;
	description?: string;
	inputSchema?: unknown;
}

export class McpClient {
	#proc: ChildProcess;
	#buf = "";
	#nextId = 1;
	#pending = new Map<number, { resolve: (v: any) => void; reject: (e: any) => void }>();

	constructor(private readonly name: string, config: McpServerConfig) {
		this.#proc = spawn(config.command!, config.args ?? [], {
			env: { ...process.env, ...config.env },
			stdio: ["pipe", "pipe", "inherit"],
		});
		this.#proc.stdout!.setEncoding("utf8");
		this.#proc.stdout!.on("data", (chunk: string) => this.#onData(chunk));
		// 'error' is emitted asynchronously — a bad `command` fails after the constructor has
		// already returned, so connectMcpServers' try/catch cannot see it. With no listener
		// attached, Node turns that into an uncaught exception and the whole app goes down
		// because one configured server was misspelled. A failing server must never do more
		// than disable itself.
		this.#proc.on("error", (e) => this.#fail(`MCP server ${name} failed to start: ${e.message}`));
		// stdin can also break independently: writing to a server that has just died raises
		// EPIPE on the stream, not at the call site.
		this.#proc.stdin!.on("error", (e) => this.#fail(`MCP server ${name} stdin closed: ${e.message}`));
		this.#proc.on("exit", () => this.#fail(`MCP server ${name} exited`));
	}

	/** Reject everything in flight with one reason. Safe to call more than once. */
	#fail(reason: string): void {
		for (const p of this.#pending.values()) p.reject(new Error(reason));
		this.#pending.clear();
	}

	#onData(chunk: string): void {
		this.#buf += chunk;
		// Framing is newline-delimited, so a server that streams without ever emitting one
		// would grow this string until the process runs out of memory. Nothing legitimate
		// sends a single 8 MB frame; drop the buffer and resynchronise at the next newline.
		if (this.#buf.length > MAX_BUFFERED_BYTES) {
			const nl = this.#buf.lastIndexOf("\n");
			this.#buf = nl >= 0 ? this.#buf.slice(nl + 1) : "";
			console.warn(`  MCP server ${this.name} sent an oversized frame; buffer reset`);
		}
		let nl: number;
		while ((nl = this.#buf.indexOf("\n")) >= 0) {
			const line = this.#buf.slice(0, nl).trim();
			this.#buf = this.#buf.slice(nl + 1);
			if (!line) continue;
			let msg: any;
			try {
				msg = JSON.parse(line);
			} catch {
				continue; // some servers log non-JSON to stdout; skip it
			}
			if (msg.id != null && this.#pending.has(msg.id)) {
				const p = this.#pending.get(msg.id)!;
				this.#pending.delete(msg.id);
				msg.error ? p.reject(new Error(msg.error.message ?? "MCP error")) : p.resolve(msg.result);
			}
		}
	}

	#request(method: string, params?: unknown): Promise<any> {
		const id = this.#nextId++;
		return new Promise((resolve, reject) => {
			// Register before writing. The write can fail synchronously, and a server that
			// answers instantly must find its pending entry already in place.
			const timer = setTimeout(() => {
				if (this.#pending.delete(id)) reject(new Error(`MCP ${this.name} timed out on ${method}`));
			}, REQUEST_TIMEOUT_MS);
			// Cleared on every settled path, not just the timeout — an uncleared deadline holds
			// the event loop open for 20s after each answered call, so a session that made a few
			// hundred tool calls kept a few hundred live timers and delayed process exit.
			const done = (fn: (v: any) => void) => (v: any) => {
				clearTimeout(timer);
				fn(v);
			};
			this.#pending.set(id, { resolve: done(resolve), reject: done(reject) });
			try {
				this.#proc.stdin!.write(JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n");
			} catch (e) {
				this.#pending.delete(id);
				clearTimeout(timer);
				reject(e);
			}
		});
	}

	#notify(method: string): void {
		this.#proc.stdin!.write(JSON.stringify({ jsonrpc: "2.0", method }) + "\n");
	}

	async initialize(): Promise<void> {
		await this.#request("initialize", {
			protocolVersion: "2024-11-05",
			capabilities: {},
			clientInfo: { name: "local-language-machine", version: "0.1" },
		});
		this.#notify("notifications/initialized");
	}

	async listTools(): Promise<RpcTool[]> {
		const res = await this.#request("tools/list");
		return Array.isArray(res?.tools) ? res.tools : [];
	}

	callTool(name: string, args: unknown): Promise<unknown> {
		return this.#request("tools/call", { name, arguments: args ?? {} });
	}

	dispose(): void {
		this.#proc.kill();
	}
}

/**
 * Connect the given stdio MCP servers and return their tools, namespaced `server__tool`
 * to avoid collisions with native tools. Servers that fail to start are skipped (best-effort).
 */
export async function connectMcpServers(
	servers: Record<string, McpServerConfig>,
): Promise<{ tools: Record<string, ToolDef>; clients: McpClient[] }> {
	const tools: Record<string, ToolDef> = {};
	const clients: McpClient[] = [];
	for (const [name, config] of Object.entries(servers)) {
		if (!config.command) {
			// Say so rather than skipping in silence: a config with no command connects to nothing,
			// and the failure was previously indistinguishable from a server that started fine.
			console.warn(`MCP ${name}: no "command" — only stdio servers are supported, so it was skipped.`);
			continue;
		}
		try {
			const client = new McpClient(name, config);
			await client.initialize();
			clients.push(client);
			for (const t of await client.listTools()) {
				tools[`${name}__${t.name}`] = {
					description: t.description ?? `${name} tool ${t.name}`,
					params: (t.inputSchema as any) ?? { type: "object", properties: {} },
					handler: (args: unknown) => client.callTool(t.name, args),
				};
			}
		} catch {
			// server unavailable — skip it rather than failing the whole session
		}
	}
	return { tools, clients };
}
