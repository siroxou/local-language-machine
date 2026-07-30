// Local Language Machine — release comparison for the update check.

export interface GithubRelease {
	tag_name?: string;
	body?: string;
	html_url?: string;
}

export interface AvailableUpdate {
	version: string;
	url: string;
	notes: string;
}

// Plain x.y.z compare; a semver dependency is only worth it if prerelease tags ship.
export function isNewer(candidate: string, current: string): boolean {
	if (!/^\d+\.\d+\.\d+$/.test(candidate) || !/^\d+\.\d+\.\d+$/.test(current)) return false;
	const a = candidate.split(".").map(Number);
	const b = current.split(".").map(Number);
	for (let i = 0; i < 3; i++) if (a[i] !== b[i]) return a[i] > b[i];
	return false;
}

// Returns what to show the user, or null to stay quiet. Release tags carry a leading
// "v" (electron-builder's default), the running version does not.
export function pickUpdate(release: GithubRelease, current: string): AvailableUpdate | null {
	const version = String(release.tag_name ?? "").replace(/^v/, "");
	const url = String(release.html_url ?? "");
	if (!url || !isNewer(version, current)) return null;
	return {
		version,
		url,
		// Release bodies are HTML; the dialog takes plain text.
		notes: String(release.body ?? "")
			.replace(/<[^>]*>/g, "")
			.slice(0, 800),
	};
}
