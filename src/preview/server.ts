// Local Language Machine — localhost preview server.
//
// Bridges the browser UI to the shared Orchestrator over a WebSocket. Serves one
// static page. No external services: the model runs in-process via the engine.
// The message protocol here is what the VS Code webview will speak via postMessage.

import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { spawn, execFile, type ChildProcess } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve, isAbsolute } from "node:path";
import { homedir } from "node:os";
import { randomUUID } from "node:crypto";
import { promisify } from "node:util";
import { WebSocketServer } from "ws";
import { LlamaCppEngine } from "../native/inference/engine.ts";
import { Orchestrator } from "../core/orchestrator.ts";
import { Session } from "../core/session.ts";
import { tree, read, write, createFile, createDir, rename, remove, listDirs, searchWorkspace, escapeRegex } from "../core/workspace.ts";
import { addRecentFolder, recentFolders } from "../core/recents.ts";
import { errMsg } from "../core/loop.ts";
import { searchGguf, listGgufFiles } from "../native/models/hub.ts";
import { setHfToken } from "../native/models/config.ts";
import { startTraining, stopTraining, isTraining } from "../native/training/train.ts";

const PORT = Number(process.env.PORT ?? 7433);
const HERE = dirname(fileURLToPath(import.meta.url));
const INDEX = join(HERE, "public", "index.html");

const execFileP = promisify(execFile);
// Best-effort live GPU utilization (0–100). NVIDIA via nvidia-smi on any OS; else macOS
// ioreg. Returns null when the platform doesn't expose it (e.g. Apple Silicon — ioreg has
// no "Device Utilization %"), in which case the status bar just shows the backend name.
async function gpuUsage(): Promise<number | null> {
	try {
		const { stdout } = await execFileP("nvidia-smi", ["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], { timeout: 1500 });
		const n = parseInt(stdout.trim().split("\n")[0]!, 10);
		if (Number.isFinite(n)) return n;
	} catch {}
	if (process.platform === "darwin") {
		try {
			const { stdout } = await execFileP("ioreg", ["-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"], { timeout: 1500 });
			const m = stdout.match(/"Device Utilization %"\s*=\s*(\d+)/);
			if (m) return parseInt(m[1]!, 10);
		} catch {}
	}
	return null;
}

// One engine (model lives in memory once) + orchestrator for the whole server.
// single shared chat session — fine for a single-user localhost preview.
let ROOT = process.cwd(); // mutable: the Open Folder handler reassigns it
const engine = new LlamaCppEngine();
const orch = new Orchestrator(engine, ROOT);

// Replace every match of `query` across the workspace, snapshotting each file first so
// Esc-Esc can undo the whole operation. Returns how many files/occurrences changed.
async function replaceAll(query: string, replacement: string, opts: { regex?: boolean; caseSensitive?: boolean }) {
	const { matches } = await searchWorkspace(ROOT, query, opts);
	const files = [...new Set(matches.map((m) => m.path))];
	const pattern = opts.regex ? query : escapeRegex(query);
	const re = new RegExp(pattern, "g" + (opts.caseSensitive ? "" : "i"));
	let count = 0;
	for (const rel of files) {
		await orch.snapshotBeforeWrite(rel).catch(() => {});
		const content = await read(ROOT, rel);
		count += (content.match(re) ?? []).length;
		await write(ROOT, rel, content.replace(re, replacement));
	}
	return { files: files.length, count };
}

const MIME: Record<string, string> = {
	html: "text/html", js: "text/javascript", mjs: "text/javascript", css: "text/css",
	json: "application/json", svg: "image/svg+xml", txt: "text/plain",
};
const contentType = (p: string) => `${MIME[p.split(".").pop()?.toLowerCase() ?? ""] ?? "text/plain"}; charset=utf-8`;

const http = createServer(async (req, res) => {
	const path = decodeURIComponent((req.url ?? "/").split("?")[0]);
	if (path === "/" || path === "/index.html") {
		try {
			res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
			res.end(await readFile(INDEX));
		} catch {
			res.writeHead(500);
			res.end("index.html not found");
		}
		return;
	}
	// Serve a workspace file (path-safe) so generated pages run in the Browser panel.
	// Read first, THEN write headers — otherwise a 404 after writeHead(200) crashes the server.
	try {
		const content = await read(ROOT, path.replace(/^\/+/, ""));
		res.writeHead(200, { "content-type": contentType(path) });
		res.end(content);
	} catch {
		res.writeHead(404);
		res.end("Not found");
	}
});

const wss = new WebSocketServer({ server: http });

// Push a message to every connected client. Training is an app-wide singleton, so its
// live logs/progress fan out to all clients rather than the one that started the run.
const broadcast = (o: unknown) => {
	const s = JSON.stringify(o);
	for (const client of wss.clients) if (client.readyState === client.OPEN) client.send(s);
};

// Poll GPU utilization and push it to every client. shells out every 2.5s —
// fine for a single-user localhost preview. Gives up after a few nulls so we don't spawn
// a subprocess forever on hardware that never reports GPU % (Apple Silicon, headless CPU).
let gpuNulls = 0;
const gpuTimer = setInterval(async () => {
	if (wss.clients.size === 0) return;
	const usage = await gpuUsage();
	if (usage == null) {
		if (++gpuNulls >= 5) { clearInterval(gpuTimer); console.log("  ▸ GPU utilization not exposed on this platform — status bar shows the backend only."); }
		return;
	}
	gpuNulls = 0;
	const msg = JSON.stringify({ type: "gpu-usage", usage });
	for (const client of wss.clients) if (client.readyState === client.OPEN) client.send(msg);
}, 2500);

wss.on("connection", (ws) => {
	const pending = new Map<string, (ok: boolean) => void>();
	const send = (o: unknown) => ws.readyState === ws.OPEN && ws.send(JSON.stringify(o));
	const sendStatus = async () => send({ type: "status", ...(await orch.status()) });

	// Integrated terminal: a persistent cwd + at most one running child per connection.
	// user-typed commands run directly (no PTY) — real interactive shells (vim/top) are the node-pty upgrade.
	let termCwd = ROOT;
	let child: ChildProcess | null = null;

	const confirm = (message: string) =>
		new Promise<boolean>((resolve) => {
			const id = randomUUID();
			pending.set(id, resolve);
			send({ type: "confirm-request", id, message });
		});

	void sendStatus();
	send({ type: "term-cwd", cwd: termCwd });

	ws.on("close", () => child?.kill());

	ws.on("message", async (raw) => {
		let msg: any;
		try {
			msg = JSON.parse(String(raw));
		} catch {
			return;
		}

		switch (msg.type) {
			case "chat": {
				let touched: string | undefined; // a file path the agent read/wrote → open it in the editor
				try {
					const text = await orch.chat(String(msg.text ?? ""), {
						onToken: (t) => send({ type: "token", text: t }),
						onToolCall: (name, args) => {
							send({ type: "tool-call", name, args });
							if ((name === "write_file" || name === "read_file") && (args as any)?.path) {
								touched = String((args as any).path);
							}
						},
						onToolResult: (name, result) => {
							send({ type: "tool-result", name, result });
							if (touched) {
								const p = touched;
								touched = undefined;
								read(ROOT, p).then((content) => send({ type: "file", path: p, content })).catch(() => {});
							}
						},
						onCodeOpen: (path, lang) => send({ type: "code-open", path, lang }),
						onCodeToken: (t) => send({ type: "code-token", text: t }),
						onCodeClose: (path) => send({ type: "code-close", path }),
						confirm,
					});
					send({ type: "assistant-done", text });
					await sendStatus(); // refresh context-window usage now the turn added tokens
				} catch (e) {
					send({ type: "error", message: errMsg(e) });
				}
				break;
			}

			case "confirm-response":
				pending.get(msg.id)?.(Boolean(msg.ok));
				pending.delete(msg.id);
				break;

			case "download":
				try {
					await orch.download(String(msg.modelId), (p) =>
						send({
							type: "download-progress",
							modelId: msg.modelId,
							downloaded: p.downloadedBytes,
							total: p.totalBytes,
						}),
					);
					await orch.load(String(msg.modelId));
					send({ type: "download-done", modelId: msg.modelId });
					await sendStatus();
				} catch (e) {
					send({ type: "error", message: errMsg(e) });
				}
				break;

			case "load":
				try {
					await orch.load(String(msg.modelId));
					await sendStatus();
				} catch (e) {
					send({ type: "error", message: errMsg(e) });
				}
				break;

			case "status":
				await sendStatus();
				break;

			case "tree":
				try {
					send({ type: "tree", nodes: await tree(ROOT) });
				} catch (e) {
					send({ type: "error", message: errMsg(e) });
				}
				break;

			case "open":
				try {
					send({ type: "file", path: msg.path, content: await read(ROOT, String(msg.path)) });
				} catch (e) {
					send({ type: "error", message: errMsg(e) });
				}
				break;

			case "save":
				try {
					// Model-generated (streamed) code is checkpointed before the write so Esc-Esc can rewind it.
					if (msg.generated) await orch.snapshotBeforeWrite(String(msg.path)).catch(() => {});
					await write(ROOT, String(msg.path), String(msg.content ?? ""));
					send({ type: "saved", path: msg.path });
					if (msg.generated) await sendStatus(); // checkpoint count changed
				} catch (e) {
					send({ type: "error", message: errMsg(e) });
				}
				break;

			case "set-mode":
				orch.setMode(msg.mode);
				await sendStatus();
				break;

			case "undo":
				try {
					const undone = await orch.undo();
					if (!undone) { send({ type: "undo-none" }); break; }
					send({ type: "undone", path: undone.path, deleted: undone.deleted });
					send({ type: "tree", nodes: await tree(ROOT) });
					if (!undone.deleted) send({ type: "file", path: undone.path, content: await read(ROOT, undone.path) });
					await sendStatus();
				} catch (e) {
					send({ type: "error", message: errMsg(e) });
				}
				break;

			case "sessions":
				send({ type: "sessions", list: Session.list(ROOT) });
				break;

			case "resume-session":
				try {
					await orch.resumeSession(String(msg.id));
					send({ type: "resumed", id: msg.id, events: Session.open(ROOT, String(msg.id)).readEvents() });
					await sendStatus();
				} catch (e) {
					send({ type: "error", message: errMsg(e) });
				}
				break;

			case "import-skills":
				try {
					const result = msg.url ? orch.importSkillsFromGit(String(msg.url)) : orch.importSkills(String(msg.path));
					send({ type: "imported", ...result });
					await sendStatus();
				} catch (e) {
					send({ type: "import-error", message: errMsg(e) });
				}
				break;

			// ————— Open Folder —————
			case "browse-dirs":
				try {
					send({ type: "dirs", ...(await listDirs(msg.path ? String(msg.path) : undefined)), recents: recentFolders() });
				} catch (e) {
					send({ type: "dirs-error", message: errMsg(e) });
				}
				break;

			case "open-folder":
				try {
					const target = String(msg.path ?? "");
					if (!(await stat(target)).isDirectory()) throw new Error("Not a folder");
					ROOT = target;
					await orch.setRoot(ROOT);
					termCwd = ROOT;
					send({ type: "folder-opened", root: ROOT, recents: addRecentFolder(ROOT) });
					send({ type: "term-cwd", cwd: termCwd });
					send({ type: "tree", nodes: await tree(ROOT) });
					await sendStatus();
				} catch (e) {
					send({ type: "error", message: errMsg(e) });
				}
				break;

			// ————— file management —————
			case "create-file":
				try {
					await createFile(ROOT, String(msg.path));
					send({ type: "tree", nodes: await tree(ROOT) });
					send({ type: "file", path: msg.path, content: "" });
				} catch (e) {
					send({ type: "error", message: errMsg(e) });
				}
				break;

			case "create-dir":
				try {
					await createDir(ROOT, String(msg.path));
					send({ type: "tree", nodes: await tree(ROOT) });
				} catch (e) {
					send({ type: "error", message: errMsg(e) });
				}
				break;

			case "rename":
				try {
					await rename(ROOT, String(msg.from), String(msg.to));
					send({ type: "renamed", from: msg.from, to: msg.to });
					send({ type: "tree", nodes: await tree(ROOT) });
				} catch (e) {
					send({ type: "error", message: errMsg(e) });
				}
				break;

			case "delete":
				try {
					await orch.snapshotBeforeWrite(String(msg.path)).catch(() => {}); // a deleted file is undoable via Esc-Esc
					await remove(ROOT, String(msg.path));
					send({ type: "deleted", path: msg.path });
					send({ type: "tree", nodes: await tree(ROOT) });
					await sendStatus();
				} catch (e) {
					send({ type: "error", message: errMsg(e) });
				}
				break;

			// ————— global search / replace —————
			case "search":
				try {
					send({ type: "search-results", ...(await searchWorkspace(ROOT, String(msg.query ?? ""), { regex: !!msg.regex, caseSensitive: !!msg.caseSensitive })) });
				} catch (e) {
					send({ type: "search-error", message: errMsg(e) });
				}
				break;

			case "replace-all":
				try {
					const res = await replaceAll(String(msg.query ?? ""), String(msg.replacement ?? ""), { regex: !!msg.regex, caseSensitive: !!msg.caseSensitive });
					send({ type: "replaced", ...res });
					send({ type: "tree", nodes: await tree(ROOT) });
					await sendStatus();
				} catch (e) {
					send({ type: "search-error", message: errMsg(e) });
				}
				break;

			case "term": {
				const cmd = String(msg.cmd ?? "").trim();
				if (!cmd) { send({ type: "term-exit", code: 0 }); break; }
				const cd = /^cd(?:\s+(.*))?$/.exec(cmd);
				if (cd) {
					const arg = (cd[1] ?? "~").trim().replace(/^~/, homedir());
					const target = resolve(termCwd, arg);
					try {
						if (!(await stat(target)).isDirectory()) throw new Error();
						termCwd = target;
						send({ type: "term-cwd", cwd: termCwd });
						send({ type: "term-exit", code: 0 });
					} catch {
						send({ type: "term-out", data: `cd: no such directory: ${cd[1] ?? ""}\n` });
						send({ type: "term-exit", code: 1 });
					}
					break;
				}
				if (child) { send({ type: "term-out", data: "[a command is already running — press Ctrl-C]\n" }); break; }
				try {
					child = spawn(cmd, { shell: true, cwd: termCwd, env: process.env });
					child.stdout?.on("data", (d) => send({ type: "term-out", data: d.toString() }));
					child.stderr?.on("data", (d) => send({ type: "term-out", data: d.toString() }));
					child.on("close", (code) => { send({ type: "term-exit", code: code ?? 0 }); child = null; });
					child.on("error", (e) => { send({ type: "term-out", data: errMsg(e) + "\n" }); send({ type: "term-exit", code: 1 }); child = null; });
				} catch (e) {
					send({ type: "term-out", data: errMsg(e) + "\n" });
					send({ type: "term-exit", code: 1 });
					child = null;
				}
				break;
			}

			case "term-kill":
				child?.kill("SIGINT");
				break;

			case "hf-search":
				try {
					send({ type: "hf-results", models: await searchGguf(String(msg.query ?? "")) });
				} catch (e) {
					send({ type: "hf-error", message: errMsg(e) });
				}
				break;

			case "hf-files":
				try {
					send({ type: "hf-files", repoId: msg.repoId, files: await listGgufFiles(String(msg.repoId)) });
				} catch (e) {
					send({ type: "hf-error", message: errMsg(e) });
				}
				break;

			case "hf-add":
				try {
					const entry = orch.addModel(String(msg.repoId), msg.file);
					await orch.download(entry.id, (p) =>
						send({ type: "download-progress", modelId: entry.id, downloaded: p.downloadedBytes, total: p.totalBytes }),
					);
					await orch.load(entry.id);
					send({ type: "download-done", modelId: entry.id });
					await sendStatus();
				} catch (e) {
					send({ type: "hf-error", message: errMsg(e) });
				}
				break;

			case "hf-token":
				setHfToken(msg.token ? String(msg.token) : undefined);
				send({ type: "hf-token-ok" });
				break;

			case "get-tuning":
				send({ type: "tuning", modelId: orch.loadedModelId, config: orch.tuning() });
				break;

			case "set-tuning":
				try {
					await orch.saveTuning(msg.config ?? {}); // persist + apply live (Inference); Load knobs stage for reload
					send({ type: "tuning", modelId: orch.loadedModelId, config: orch.tuning() });
				} catch (e) {
					send({ type: "error", message: errMsg(e) });
				}
				break;

			case "reload-model":
				try {
					await orch.reloadModel();
					send({ type: "reloaded" });
					await sendStatus();
				} catch (e) {
					send({ type: "error", message: errMsg(e) });
				}
				break;

			// ————— LoRA fine-tuning (train-only v1) —————
			case "train-start":
				try {
					if (isTraining()) { send({ type: "train-error", message: "A training run is already active." }); break; }
					const c = msg.config ?? {};
					const datasetPath = isAbsolute(String(c.datasetPath ?? "")) ? String(c.datasetPath) : resolve(ROOT, String(c.datasetPath ?? ""));
					await startTraining({ ...c, datasetPath }, broadcast); // resolves once spawned; done/error arrive via broadcast
				} catch (e) {
					broadcast({ type: "train-error", message: errMsg(e) });
				}
				break;

			case "train-stop":
				stopTraining();
				break;
		}
	});
});

// Open the port first, then load the model in the background — a big model can
// take several seconds and shouldn't delay the server being reachable.
http.listen(PORT, async () => {
	console.log(`\n  ▸ Preview ready:  http://localhost:${PORT}`);
	console.log("  Local Language Machine — loading cached model…");
	const loaded = await orch.autoload().catch((e) => {
		console.error("  autoload failed:", errMsg(e));
		return undefined;
	});
	console.log(loaded ? `  ▸ Model loaded:   ${loaded} (gpu=${engine.gpu})\n` : "  ▸ No model cached — use the Download button.\n");
	// Tell any already-connected clients that the model is now ready.
	const status = JSON.stringify({ type: "status", ...(await orch.status()) });
	for (const ws of wss.clients) if (ws.readyState === ws.OPEN) ws.send(status);
});
