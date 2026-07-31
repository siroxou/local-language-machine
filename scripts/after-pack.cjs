// Apple Silicon refuses to open a bundle that carries no signature at all — it reports
// the app as damaged. This project ships without a signing certificate, so apply an
// ad-hoc signature, which is enough for the app to launch.
//
// Ad-hoc is not notarization: `spctl -a` still rejects the bundle, so a copy downloaded
// from the web carries com.apple.quarantine and Gatekeeper blocks the first launch with
// "Apple could not verify this app is free of malware". Clearing the flag once is what
// gets past it — see the macOS section of the README. Right-click → Open does NOT work
// any more; Apple removed that bypass in macOS 15.

const { execFileSync } = require("node:child_process");
const path = require("node:path");

exports.default = async function afterPack(context) {
	if (context.electronPlatformName !== "darwin") return;
	const app = path.join(context.appOutDir, `${context.packager.appInfo.productFilename}.app`);
	execFileSync("codesign", ["--force", "--deep", "--sign", "-", app], { stdio: "inherit" });
};
