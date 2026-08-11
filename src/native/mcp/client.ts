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
		this.#proc.on("exit", () => {
			for (const p of this.#pending.values()) p.reject(new Error(`MCP server ${name} exited`));
			this.#pending.clear();
		});
	}

	#onData(chunk: string): void {
		this.#buf += chunk;
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
		this.#proc.stdin!.write(JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n");
		return new Promise((resolve, reject) => {
			this.#pending.set(id, { resolve, reject });
			setTimeout(() => {
				if (this.#pending.delete(id)) reject(new Error(`MCP ${this.name} timed out on ${method}`));
			}, 20_000);
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
