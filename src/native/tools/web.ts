// Local Language Machine — optional web tools (opt-in online only).
//
// Registered by the orchestrator ONLY when settings.online is true, so the offline
// promise holds by default. Uses Node's global fetch — no dependency, same as hub.ts.

import type { ToolDef } from "../inference/engine.ts";

const MAX_TEXT = 20_000;
// fetch has no default timeout: a host that accepts the connection and then stalls would
// hang the agent turn for good, with nothing in the UI able to cancel it.
const FETCH_TIMEOUT_MS = 20_000;

/** Strip a fetched HTML document down to readable text. regex strip, not a DOM parse. */
function htmlToText(html: string): string {
	return html
		.replace(/<script[\s\S]*?<\/script>/gi, " ")
		.replace(/<style[\s\S]*?<\/style>/gi, " ")
		.replace(/<[^>]+>/g, " ")
		.replace(/&nbsp;/g, " ")
		.replace(/&amp;/g, "&")
		.replace(/&lt;/g, "<")
		.replace(/&gt;/g, ">")
		.replace(/\s+/g, " ")
		.trim();
}

export function webTools(): Record<string, ToolDef> {
	return {
		web_fetch: {
			description: "Fetch a URL and return its readable text content (online).",
			params: { type: "object", properties: { url: { type: "string" } }, required: ["url"] },
			async handler({ url }: { url: string }) {
				const res = await fetch(url, { headers: { "user-agent": "LocalLanguageMachine/0.1" }, signal: AbortSignal.timeout(FETCH_TIMEOUT_MS) });
				if (!res.ok) return { error: `HTTP ${res.status}` };
				const body = await res.text();
				const text = /html/i.test(res.headers.get("content-type") ?? "") ? htmlToText(body) : body;
				return { url, text: text.slice(0, MAX_TEXT), truncated: text.length > MAX_TEXT };
			},
		},

		web_search: {
			description: "Search the web and return result titles and links (online).",
			params: { type: "object", properties: { query: { type: "string" } }, required: ["query"] },
			async handler({ query }: { query: string }) {
				// keyless DuckDuckGo HTML scrape — good enough, no API key. Swap for a
				// real search API if results get flaky.
				const res = await fetch(`https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`, {
					headers: { "user-agent": "Mozilla/5.0" },
					signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
				});
				if (!res.ok) return { error: `HTTP ${res.status}` };
				const html = await res.text();
				const results: Array<{ title: string; url: string }> = [];
				const re = /<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/gi;
				let m: RegExpExecArray | null;
				while ((m = re.exec(html)) && results.length < 10) {
					results.push({ url: decodeURIComponent(m[1]!.replace(/^\/\/duckduckgo\.com\/l\/\?uddg=/, "").split("&")[0]!), title: htmlToText(m[2]!) });
				}
				return { results };
			},
		},
	};
}
