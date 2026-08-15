// Local Language Machine — desktop shell.
//
// The runtime is unchanged: this process starts the same preview server the CLI
// starts, and the window loads the same page a browser would. No preload, no IPC —
// the renderer already talks to the backend over a WebSocket. The only work here is
// repairing the two things a Finder/Dock launch breaks (PATH and cwd), and checking
// GitHub for a newer release.

import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { connect } from "node:net";
import { homedir } from "node:os";
import { app, dialog, net, shell, BrowserWindow } from "electron";
import { recentFolders } from "../core/recents.ts";
import { pickUpdate, type GithubRelease } from "./version.ts";

const PORT = 7433;
// 127.0.0.1, never "localhost": they are different origins, and the UI keeps theme,
// background and sound settings in localStorage, which is scoped per origin.
const APP_URL = `http://127.0.0.1:${PORT}`;
const RELEASES = "https://api.github.com/repos/siroxou/local-language-machine/releases/latest";
const UPDATE_CHECK_TIMEOUT_MS = 10_000;

// A Finder or Dock launch inherits PATH=/usr/bin:/bin:/usr/sbin:/sbin rather than the
// login shell's. That silently breaks Python discovery for training, `git clone` for
// skill imports, the git and run_terminal agent tools, and the integrated terminal.
// All of them read process.env, so one fix here covers every caller.
function inheritShellPath(): void {
	if (process.platform === "win32") return; // GUI processes there already get the user environment
	const MARK = "__LLM_PATH__"; // interactive rc files print banners; anchor past the noise
	try {
		const out = execFileSync(process.env.SHELL || "/bin/zsh", ["-ilc", `printf '${MARK}%s' "$PATH"`], {
			encoding: "utf8",
			timeout: 4000,
			stdio: ["ignore", "pipe", "ignore"],
		});
		const shellPath = out.split(MARK).pop()?.trim();
		if (!shellPath) return;
		const merged = `${shellPath}:${process.env.PATH ?? ""}`.split(":").filter(Boolean);
		process.env.PATH = [...new Set(merged)].join(":");
	} catch {
		// Keep the stripped PATH. Terminal and training degrade; the app still starts.
	}
}

// Launched from the Dock the process starts in "/", and the server takes its workspace
// root from process.cwd() when the module first evaluates — so the whole filesystem
// would become the workspace. Launched from a terminal the cwd is a deliberate choice,
// so leave it alone.
function fixCwd(): void {
	if (process.cwd() !== "/") return;
	try {
		process.chdir(recentFolders().find((p) => existsSync(p)) ?? homedir());
	} catch {
		// Fall through with whatever cwd we have.
	}
}

// Something already serving on 7433 (a `npm run preview` left running) would make the
// server throw an unhandled EADDRINUSE — it installs no 'error' handler. Attach to it.
const portTaken = (): Promise<boolean> =>
	new Promise((resolve) => {
		const probe = connect(PORT, "127.0.0.1")
			.on("connect", () => {
				probe.destroy();
				resolve(true);
			})
			.on("error", () => resolve(false));
	});

async function checkForUpdate(): Promise<void> {
	// Bounded like every other request in the app: the update check runs on launch, and a
	// hung endpoint would otherwise leave this promise pending for the life of the process.
	const res = await net.fetch(RELEASES, {
		headers: { accept: "application/vnd.github+json" },
		signal: AbortSignal.timeout(UPDATE_CHECK_TIMEOUT_MS),
	});
	if (!res.ok) return; // no releases yet, offline, or rate limited — stay quiet
	const update = pickUpdate((await res.json()) as GithubRelease, app.getVersion());
	if (!update) return;

	const { response } = await dialog.showMessageBox({
		type: "info",
		buttons: ["Download", "Later"],
		defaultId: 0,
		cancelId: 1,
		message: `Version ${update.version} is available.`,
		detail: update.notes,
	});
	if (response === 0) await shell.openExternal(update.url);
}

function createWindow(): BrowserWindow {
	// webPreferences is deliberately omitted: Electron already defaults to
	// nodeIntegration:false, contextIsolation:true and sandbox:true, which is exactly
	// what this page needs — it imports nothing from Node.
	const win = new BrowserWindow({
		width: 1440,
		height: 900,
		minWidth: 900,
		minHeight: 600,
		title: "Local Language Machine",
		backgroundColor: "#eceadf", // the UI's --bg, so there is no flash of a different colour on launch
	});

	// Links and the browser panel's window.open go to the real browser, not a new shell.
	win.webContents.setWindowOpenHandler(({ url }) => {
		if (/^https?:/.test(url)) void shell.openExternal(url);
		return { action: "deny" };
	});

	// The window is created before the server finishes binding, so the first load can
	// lose the race. isMainFrame skips the UI's two iframes; -3 is ERR_ABORTED, which
	// fires on ordinary navigation.
	// Retries indefinitely; add a ceiling if a permanently dead server ever matters.
	win.webContents.on("did-fail-load", (_event, code, _desc, _url, isMainFrame) => {
		if (isMainFrame && code !== -3) setTimeout(() => void win.loadURL(APP_URL), 250);
	});

	void win.loadURL(APP_URL);
	return win;
}

if (!app.requestSingleInstanceLock()) {
	app.quit();
} else {
	app.on("second-instance", () => {
		const [win] = BrowserWindow.getAllWindows();
		if (!win) return;
		if (win.isMinimized()) win.restore();
		win.focus();
	});

	app.whenReady().then(async () => {
		// Paint first: reading the login shell's PATH can take a moment on a heavy rc file,
		// and the Dock icon should not bounce while that happens.
		createWindow();
		app.on("activate", () => {
			if (BrowserWindow.getAllWindows().length === 0) createWindow();
		});

		// Both must land before the server module evaluates — it captures cwd on import.
		inheritShellPath();
		fixCwd();
		if (!(await portTaken())) await import("../preview/server.ts");

		if (app.isPackaged) void checkForUpdate().catch(() => {});
	});

	// macOS convention: closing the window leaves the app (and the loaded model) running.
	app.on("window-all-closed", () => {
		if (process.platform !== "darwin") app.quit();
	});
}
