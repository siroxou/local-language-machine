// The control socket's admission check. This is the boundary that stops a web page the
// user happens to visit from driving the app, so each case gets an explicit assertion.
import { test } from "node:test";
import assert from "node:assert/strict";
import { isAllowedOrigin, localOrigins } from "./origin.ts";

const PORT = 7433;

test("both local spellings are allowed on the configured port", () => {
	assert.ok(isAllowedOrigin("http://127.0.0.1:7433", PORT));
	assert.ok(isAllowedOrigin("http://localhost:7433", PORT));
	assert.ok(isAllowedOrigin("http://[::1]:7433", PORT));
});

test("a foreign origin is rejected", () => {
	// The drive-by case: a page the user merely visited, opening the socket in the
	// background. WebSocket handshakes are not subject to the same-origin policy, so
	// without this check the page reaches every message type the app handles.
	assert.equal(isAllowedOrigin("https://evil.example", PORT), false);
	assert.equal(isAllowedOrigin("http://evil.example", PORT), false);
	assert.equal(isAllowedOrigin("null", PORT), false, "a sandboxed iframe sends the string \"null\"");
});

test("a local origin on a different port is rejected", () => {
	// Another app on this machine is still another origin.
	assert.equal(isAllowedOrigin("http://127.0.0.1:3000", PORT), false);
	assert.equal(isAllowedOrigin("http://localhost:8080", PORT), false);
});

test("an origin that merely contains the local one is rejected", () => {
	// Guards against a substring or prefix check being introduced later.
	assert.equal(isAllowedOrigin("http://127.0.0.1:7433.evil.example", PORT), false);
	assert.equal(isAllowedOrigin("http://evil.example/http://127.0.0.1:7433", PORT), false);
	assert.equal(isAllowedOrigin("https://127.0.0.1:7433", PORT), false, "scheme is part of the origin");
});

test("a missing Origin is allowed, because only a non-browser client can omit it", () => {
	// A browser cannot be made to drop the header on a WebSocket handshake, so absence
	// means a local script — which already has the user's privileges either way.
	assert.ok(isAllowedOrigin(undefined, PORT));
	assert.ok(isAllowedOrigin("", PORT));
});

test("the allow-list tracks the configured port", () => {
	assert.ok(isAllowedOrigin("http://127.0.0.1:9999", 9999));
	assert.equal(isAllowedOrigin("http://127.0.0.1:7433", 9999), false);
	assert.equal(localOrigins(9999).length, 3);
});
