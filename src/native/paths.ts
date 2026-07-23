// Local Language Machine — where the app stores its data.
//
// One place so models, config, settings, sessions, skills, and agents all agree.
// Honors $LLM_HOME (lets the user relocate data, and lets tests point at a temp dir
// instead of polluting the real home directory).

import { homedir } from "node:os";
import { join } from "node:path";

export function appHome(): string {
	return process.env.LLM_HOME || join(homedir(), ".local-language-machine");
}
