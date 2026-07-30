// Apple Silicon refuses to open a bundle that carries no signature at all — it reports
// the app as damaged. This project ships without a signing certificate, so apply an
// ad-hoc signature, which is enough for the app to launch. Anyone downloading it still
// clears Gatekeeper once with right-click → Open.

const { execFileSync } = require("node:child_process");
const path = require("node:path");

exports.default = async function afterPack(context) {
	if (context.electronPlatformName !== "darwin") return;
	const app = path.join(context.appOutDir, `${context.packager.appInfo.productFilename}.app`);
	execFileSync("codesign", ["--force", "--deep", "--sign", "-", app], { stdio: "inherit" });
};
