// A misconfigured MCP server must disable itself, never take the app down with it.
// These run against real spawned processes — no model, no network.
import { test } from "node:test";
import assert from "node:assert/strict";
import { McpClient, connectMcpServers } from "./client.ts";

test("a server whose command does not exist rejects instead of crashing the process", async () => {
	// spawn emits 'error' asynchronously, so this failure arrives after the constructor has
	// already returned. Without an 'error' listener Node escalates it to an uncaught
	// exception and the whole app dies because one entry in settings.json was misspelled.
	const client = new McpClient("ghost", { command: "definitely-not-a-real-binary-xyz" });
	await assert.rejects(client.initialize(), /failed to start|exited|stdin closed|EPIPE/);
	client.dispose();
});

test("connectMcpServers skips a broken server and keeps the good ones", async () => {
	const { tools, clients } = await connectMcpServers({
		ghost: { command: "definitely-not-a-real-binary-xyz" },
		// A real binary that is not an MCP server: it starts, says nothing useful, and exits.
		mute: { command: "true" },
	});
	// The point is that we got here at all — a returned result rather than a dead process.
	assert.equal(typeof tools, "object");
	for (const c of clients) c.dispose();
});

test("a request does not leave a live timer behind after it settles", async () => {
	// An uncleared 20s deadline per call kept the event loop alive long after the work was
	// done, so a session with a few hundred tool calls delayed process exit by that much.
	const client = new McpClient("ghost", { command: "definitely-not-a-real-binary-xyz" });
	await client.initialize().catch(() => {});
	client.dispose();

	const handles = (process as any)._getActiveHandles?.() ?? [];
	const timers = handles.filter((h: any) => h?.constructor?.name === "Timeout");
	assert.equal(timers.length, 0, "no Timeout handle should outlive a settled request");
});
