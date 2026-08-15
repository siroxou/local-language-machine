# Project memory

An offline Electron IDE in TypeScript. Inference is bundled and runs in-process; nothing
about a user's code leaves the device.

This file is prepended to **every** request, so every line costs tokens on every turn.
Keep it short — reference material belongs in a skill, not here.

## Commands

```bash
npm test          # node:test, colocated *.test.ts
npm run typecheck # strict, no emit
npm run audit     # invariant scan; --check fails on anything not baselined
npm run preview   # serve the UI at http://127.0.0.1:7433
npm run build     # tsc + copy assets into dist/
```

## Layout

- `src/core/` — UI-agnostic brain. Orchestrator is the composition root.
- `src/native/` — offline building blocks: inference, tools, models, training, MCP.
- `src/preview/` — the localhost server and the single-page UI.
- `src/electron/` — the desktop shell. No preload, no IPC; the renderer uses the WebSocket.
- `src/audit/` — the project's own source audit. Excluded from the build.

## Layering

`core/` must not import from `preview/` or `electron/`. `native/` may reference `core/`
only as `import type`, which erases at compile time. Enforced by `npm run audit`.

## Conventions

- Tabs. ESM only. `node:` prefixes on builtins. `.ts` extensions in relative imports.
- Comments explain **why**, including approaches that were tried and rejected. A comment
  restating what the next line does is noise; one naming a constraint is the point.
- No new runtime dependency without a written reason. There are currently two.
- Mark a deliberate simplification with `// ponytail:` and name its ceiling.

## Tests

`node:test` + `node:assert/strict`, in a `*.test.ts` beside the source. Test names are
sentences describing the behaviour. A bug fix ships with a test that was **seen failing**
on the parent commit — a regression test never observed red proves nothing.

## Invariants that must not break

- Every workspace path resolves through `resolveInRoot` — physically, not lexically.
- Every `fetch` carries an `AbortSignal`.
- Every `spawn` handle gets an `'error'` listener; `'error'` fires asynchronously.
- Every RegExp built from wire or model input goes through `safeRegExp`.
- Anything reaching `innerHTML` is wrapped in `escHtml`.
- A limit that is hit must say so. Never return a falsy default for "budget exhausted".
- The control WebSocket checks `Origin`. Loopback is not an authentication boundary.

## Commits

Imperative, naming the behaviour change rather than the code change:

- `Stop the release runners from clobbering each other's installers`
- `Make every settings knob actually do what it says`
