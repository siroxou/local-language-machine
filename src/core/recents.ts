// Local Language Machine — recently opened folders (for the Open Folder picker).
// Persisted at appHome()/recent-folders.json. Newest first, capped.

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { appHome } from "../native/paths.ts";

const MAX = 10;
const file = () => join(appHome(), "recent-folders.json");

export function recentFolders(): string[] {
	try {
		const data = JSON.parse(readFileSync(file(), "utf8"));
		return Array.isArray(data) ? data.filter((p) => typeof p === "string") : [];
	} catch {
		return [];
	}
}

export function addRecentFolder(path: string): string[] {
	const list = [path, ...recentFolders().filter((p) => p !== path)].slice(0, MAX);
	mkdirSync(appHome(), { recursive: true });
	writeFileSync(file(), JSON.stringify(list, null, 2));
	return list;
}
