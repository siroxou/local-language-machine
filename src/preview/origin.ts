// Local Language Machine — who is allowed to open the control socket.
//
// The WebSocket carries the whole app: it can run a terminal command, re-root the
// workspace, read and write files, and read the stored Hub token. Binding to 127.0.0.1
// keeps it off the network, but it is NOT an authentication boundary — the same-origin
// policy does not apply to WebSocket handshakes, so any page in any browser on this
// machine can open ws://127.0.0.1:7433 and start sending messages. The browser does
// attach an Origin header to that handshake; checking it is what closes the gap.
//
// A separate module from server.ts so it can be tested without starting a listener.

/**
 * The two spellings of this machine. They are distinct browser origins — which is why
 * electron/main.ts and the printed URL both use 127.0.0.1, since localStorage under one
 * spelling is invisible to the other — but both are genuinely local, so both are allowed.
 */
export function localOrigins(port: number): string[] {
	return [`http://127.0.0.1:${port}`, `http://localhost:${port}`, `http://[::1]:${port}`];
}

/**
 * Decide whether a handshake may proceed.
 *
 * A missing Origin is allowed. Browsers always send one on a WebSocket handshake and
 * cannot be made to omit it, so absence means a non-browser client — a script or a CLI
 * on this machine, which already runs with the user's own privileges and has no need to
 * go through here to do damage. Rejecting it would break local tooling while closing
 * nothing, and the drive-by case this guards against is exactly the one that always
 * carries an Origin.
 */
export function isAllowedOrigin(origin: string | undefined, port: number): boolean {
	if (origin === undefined || origin === "") return true;
	return localOrigins(port).includes(origin.toLowerCase());
}
