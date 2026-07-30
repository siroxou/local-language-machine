import { test } from "node:test";
import assert from "node:assert/strict";
import { isNewer, pickUpdate } from "./version.ts";

test("isNewer compares release tags segment by segment", () => {
	assert.equal(isNewer("0.0.2", "0.0.1"), true);
	assert.equal(isNewer("0.1.0", "0.0.9"), true);
	assert.equal(isNewer("1.0.0", "0.9.9"), true);
	assert.equal(isNewer("0.0.1", "0.0.1"), false);
	assert.equal(isNewer("0.0.1", "0.0.2"), false);
	// string compare would call "0.0.10" older than "0.0.9"
	assert.equal(isNewer("0.0.10", "0.0.9"), true);
	assert.equal(isNewer("0.10.0", "0.9.0"), true);
});

test("isNewer rejects anything that is not a plain x.y.z tag", () => {
	assert.equal(isNewer("", "0.0.1"), false);
	assert.equal(isNewer("v0.0.2", "0.0.1"), false); // the caller strips the leading v
	assert.equal(isNewer("0.0.2-beta.1", "0.0.1"), false);
	assert.equal(isNewer("nightly", "0.0.1"), false);
	assert.equal(isNewer("0.2", "0.0.1"), false);
});

const RELEASE = { tag_name: "v0.2.0", body: "<h2>What's new</h2>\n<p>Faster startup.</p>", html_url: "https://example.test/r/v0.2.0" };

test("pickUpdate strips the tag prefix and flattens the notes to text", () => {
	const up = pickUpdate(RELEASE, "0.1.0");
	assert.equal(up?.version, "0.2.0");
	assert.equal(up?.url, "https://example.test/r/v0.2.0");
	assert.equal(up?.notes, "What's new\nFaster startup.");
});

test("pickUpdate stays quiet unless there is a strictly newer, usable release", () => {
	assert.equal(pickUpdate(RELEASE, "0.2.0"), null); // same version
	assert.equal(pickUpdate(RELEASE, "0.3.0"), null); // running ahead of the release
	assert.equal(pickUpdate({}, "0.1.0"), null); // empty payload
	assert.equal(pickUpdate({ tag_name: "v0.2.0" }, "0.1.0"), null); // no download page
	assert.equal(pickUpdate({ tag_name: "nightly", html_url: "u" }, "0.1.0"), null);
});

test("pickUpdate caps release notes so the dialog stays readable", () => {
	const up = pickUpdate({ ...RELEASE, body: "x".repeat(2000) }, "0.1.0");
	assert.equal(up?.notes.length, 800);
});
