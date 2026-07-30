// tsc emits only .js — these three assets are resolved at runtime via import.meta.url
// and must sit next to the emitted files: the editor UI, and the two Python trainer
// scripts that get handed to spawn().

import { cpSync } from "node:fs";

cpSync("src/preview/public", "dist/preview/public", { recursive: true });
cpSync("src/native/training", "dist/native/training", { recursive: true, filter: (s) => !s.endsWith(".ts") });
