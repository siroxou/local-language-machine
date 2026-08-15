# Testing Stateful Agent Applications

A methodology for finding the bugs that unit tests structurally cannot see: the ones that
only appear after an application has been *running* for a while.

This document is project-agnostic. It was written from a session that took a local-first AI
IDE from "82/82 tests green" to "82/82 tests green, and now it also survives an hour of
real use" — the numbers throughout are the measured evidence from that session, kept because
concrete magnitudes are what make the failure classes recognisable.

---

## 0. The premise

A stateful application — anything holding a model in memory, a session, a connection pool, a
cache — has two populations of bugs:

**Population A** is what a unit test suite catches. Wrong return value, bad branch, unhandled
input. Pure functions, single calls, deterministic.

**Population B** never appears in a unit test, because every unit test starts from a fresh
process and makes one call. Population B needs *accumulation*: the tenth model switch, the
turn after a large paste, the request whose client went away. These bugs are invisible to
assertion-style testing and highly visible to users, because users are the only ones who run
the process for hours.

The entire method below is about population B.

> A green suite is not evidence of a healthy application. It is evidence that the first call
> to each function behaves. Test the hundredth.

---

## 1. Failure taxonomy

Seven classes. Each entry gives the shape to grep for, the detection recipe, and the fix
shape. These recur across languages and runtimes.

### 1.1 Cumulative resource leak

**Symptom.** Application is fine for an hour, then everything is slow. Restarting fixes it.

**Shape.** A handle is replaced without disposing the old one.

```
this.#session = await createSession(...)   // where did the old #session go?
```

The tell is an assignment to a field that owns an external resource — a context, a file
handle, a socket, a GPU buffer — with no `dispose`/`close`/`free` on the outgoing value.

**Detection.** Repeat the triggering action N times and sample RSS. Do not sample once
before and once after; sample *per iteration*, because the shape of the curve tells you
whether it is a leak (monotonic) or fragmentation (noisy plateau).

**Measured example.** Replacing a chat session without disposing its KV cache:

| restarts | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| RSS (GB) | 11.9 | 18.9 | 25.9 | 28.0 | 28.2 | 28.1 |

~7 GB per restart, plateauing only because the OS began compressing. Downstream effect: an
operation that normally took 900 ms took over three minutes. **After the fix: flat at 11.8 GB
across all five.**

**Fix shape.** Dispose before allocating the replacement, so peak never holds two:

```ts
try { this.#session?.dispose(); } catch {}   // the resource may already be gone
this.#session = undefined;
this.#session = await createSession(...);
```

The `try/catch` matters: if the parent resource was already torn down, disposal throws, and
an exception here would break the replacement path.

### 1.2 Non-convergent state reduction

**Symptom.** A cleanup step runs, then runs again on the next operation, and again, forever.
Each run is expensive. The application gets slower the longer it runs.

**Shape.** A reducer whose output can still exceed its own trigger threshold.

```ts
if (overThreshold(state)) state = reduce(state);   // but is reduce(state) under threshold?
```

Any `compact`, `evict`, `prune`, `truncate`, `summarize`, or `gc` step. The bug is that the
reducer preserves some region verbatim — "keep the most recent N" — and a single oversized
element inside that region defeats the whole reduction.

**Detection.** Assert the *post-condition*, not that the step ran:

```ts
const after = reduce(state);
assert(!overThreshold(after), "one pass must clear the threshold");
```

Then simulate several rounds in a loop and confirm it converges in one, not three.

**Measured example.** Compaction keeping the last 4 turns verbatim, after one large paste:

| turn | before fix | after fix |
|---|---|---|
| the large input | 69 s (compacts) | 67 s (compacts) |
| next trivial prompt | 82 s (compacts again) | **1.8 s** |
| the one after | 116 s (compacts again) | **0.13 s** |

Each of those re-runs also allocated a second full-size context — so this class compounds
with 1.1.

**Fix shape.** Bound each preserved element, not just the count:

```ts
const KEEP_ITEM_CHARS = 3_000;
const recent = rest.slice(-keepRecent).map(capItem);
```

Size the cap against the *floor* — the part that survives every reduction. Measure that floor
before picking the number (here: a ~5 KB system prompt against a 24 KB threshold, so
4 × 3 KB + summary clears it with margin).

### 1.3 Unbounded wait

**Symptom.** One operation hangs forever. No error, no timeout, no way to cancel. Often the
whole application is stuck behind it.

**Shape.** Every one of these, with no timeout:

```ts
exec(cmd)                    // no timeout option
execSync(cmd)                // worse: blocks the entire event loop
fetch(url)                   // fetch has NO default timeout
new Promise((resolve) => { pending.set(id, resolve) })   // never rejected
```

**Detection.** This one is a grep, not an experiment:

```bash
grep -rnE "exec(File)?(Sync)?\(|fetch\(|new Promise\(" src \
  | grep -v "timeout\|signal\|AbortSignal"
```

Then for each hit ask: *what happens if this never returns?* If the answer is "the user's
turn hangs with no cancel button", it needs a bound.

**Real instances found in one pass:** shell tool (a dev server the model starts never
returns), lifecycle hooks (run inline on every tool call), two HTTP clients, and a
`execFileSync` git clone — that last one froze the entire server, not just its own feature,
because sync exec blocks the event loop.

**Fix shape.**

```ts
execAsync(cmd, { timeout: 120_000, killSignal: "SIGKILL" })
fetch(url, { signal: AbortSignal.timeout(20_000) })
execFileSync(cmd, args, { timeout: 60_000, env: { ...process.env, GIT_TERMINAL_PROMPT: "0" } })
```

Return a result the caller can *reason about*, not a bare failure:

```ts
if (e?.killed) return { exitCode: 124, stderr: `Timed out after ${MS/1000}s and was killed. …` };
```

### 1.4 Unsettled promise on disconnect

**Symptom.** A request awaiting a client response is orphaned when that client vanishes. The
server stays up; one logical operation is pinned forever.

**Shape.** A correlation map with no teardown path:

```ts
const pending = new Map<string, (ok: boolean) => void>();
ws.on("close", () => child?.kill());   // pending is never touched
```

**Detection.** Connect, provoke the round-trip, then `terminate()` the socket without
answering. Confirm the server still serves other clients — and read the source to confirm the
promise is settled, because from outside, "leaked" and "fine" look identical.

**Fix shape.** Settle everything outstanding in the safe direction:

```ts
ws.on("close", () => {
  child?.kill();
  for (const resolve of pending.values()) resolve(false);   // deny, not hang
  pending.clear();
});
```

### 1.5 Static rule outranking explicit user intent

**Symptom.** A control does nothing. Often it still *looks* active.

**Shape.** In CSS, a media query with equal specificity placed later in the sheet:

```css
.app.hide-files { grid-template-columns: 0 1fr 384px; }         /* the toggle */
@media (max-width: 1180px) { .app { grid-template-columns: 0 1fr 360px; } }  /* always wins */
```

The general form is any declarative default that cannot be overridden by an explicit choice.

**Detection.** Exercise every toggle at several viewport widths, and assert on *rendered
geometry* (`getBoundingClientRect().width`), never on the class list. A class list says what
the code intended; the rect says what the user sees.

**Fix shape.** Collapse the combinatorics into one variable per axis, then let an explicit
opt-in outrank the default:

```css
.app { --col-files: 258px; grid-template-columns: var(--col-files) 1fr var(--col-agent); }
.app.hide-files { --col-files: 0; }
@media (max-width: 1180px) {
  .app:not(.show-files) { --col-files: 0; }   /* default folds; show-files wins */
}
```

And keep the control's lit state derived from actual visibility, not from a toggle counter —
otherwise it lies during the auto-folded state.

### 1.6 Handler scoped too narrowly

**Symptom.** A keyboard shortcut works sometimes. Users report it "randomly" failing.

**Shape.** A document-level concern bound to one element:

```js
editor.addEventListener("keydown", e => { if (mod && e.key === "s") save(); });
```

Save is a *document* concern — the file is still the open file when focus is in a sidebar, a
chat box, or a terminal.

**Detection.** For each shortcut, move focus to every other focusable region and re-fire.

**Fix shape.** Bind at `document`, read state from the app model rather than the focused DOM
node:

```js
send({ type: "save", path: active, content: files.get(active)?.content ?? code.value });
```

### 1.7 Capped operation reported as complete

**Symptom.** A bulk operation silently does part of the work and reports success.

**Shape.** A consumer that discards the truncation flag its producer returns:

```ts
const { matches } = await search(query);        // search also returns `truncated`
return { files: files.length, count };          // caller never learns
```

**Detection.** Find every producer returning a `truncated` / `hasMore` / `nextCursor` and
check each consumer forwards it to the surface the user reads.

**Fix shape.** Thread it through and say so in the UI:

```ts
return { files: files.length, count, truncated };
```
```js
toast(`Replaced ${m.count} in ${m.files} files` +
      (m.truncated ? " — hit the search cap, run again to catch the rest" : ""));
```

---

## 2. Harness architecture

Four layers, cheapest first. Each catches a different population; none substitutes for
another.

```
Layer 1  protocol surface   every message/endpoint, no model     seconds    ~50 assertions
Layer 2  stress             real model, long session, hostile    ~10 min    ~45 assertions
Layer 3  UI drive           real browser, real geometry          ~5 min     ~40 checks
Layer 4  targeted probe     measure ONE thing precisely          minutes    1 number
```

### Layer 1 — protocol surface

One assertion per message type or endpoint, including the error paths. This is where you
learn the protocol's real shapes, and it is fast enough to re-run after every change.

Cover, for each operation: the happy path, a malformed argument, a missing argument, a path
that escapes the sandbox, and a resource that does not exist. Finish with a robustness block:

```js
ws.send("this is not json");
ws.send(JSON.stringify({ type: "totally-unknown-message" }));
ws.send(JSON.stringify({}));
// then prove the server still answers
```

### Layer 2 — stress

The layer that finds population B. It must include:

- **Concurrency.** Two operations in flight on one connection; two clients at once.
- **Rapid-fire.** N sequential operations, asserting all N succeed.
- **Hostile input.** Empty, whitespace-only, unicode, control characters, and input shaped
  like the system's own control protocol.
- **Oversized input.** Something near the capacity limit — this is what triggers class 1.2.
- **Slash/meta commands.** Every one, including the unknown-command path.
- **Permission and gating.** Each mode, asserting both the allow and the deny path.
- **Mid-operation disconnect.** Yank the socket during a stream.
- **Persistence and resume.** Then keep operating after the resume.

Critically: **after each destructive test, assert the system still works.** A test that
proves an oversized input is handled is worth little; a test that proves the *next* operation
still completes is what caught the compaction loop.

```js
const t = await turn(ws, hugeInput);
ok("oversized input handled", !!t.done);
const after = await turn(ws, "Say exactly: RECOVERED");
ok("session usable afterwards", !after.error && after.text.length > 0);  // ← this one failed
```

### Layer 3 — UI drive

Assert on rendered geometry and real content, not on class names or internal flags.

- Panels: `getBoundingClientRect().width`, at several viewport widths.
- Editor: gutter line count, syntax-span count, tab list, dirty markers.
- Every overlay: open, read its populated content, close.
- Round-trip anything that writes: perform the action, then verify **on disk**.

Two mechanical cautions, both of which produced false failures in practice:

- **Coordinate spaces lie.** Screenshot pixels, CSS pixels and device pixels can all differ.
  Establish the mapping empirically before trusting a click:
  ```js
  document.addEventListener('click', e => window.__clicks.push({x: e.clientX, y: e.clientY}), true);
  ```
  Then click a known point and read back where it landed.
- **Synthetic events are not always trusted.** Some handlers only respond to real input.
  When a synthetic `KeyboardEvent` fails, retry with real key injection before filing a bug.
  In this session, `Esc Esc` undo appeared broken under synthetic dispatch and worked
  perfectly under real keys.

### Layer 4 — targeted probe

Once a layer above says "this got slow" or "this grew", write a probe that measures exactly
one quantity and prints a table. Do not try to make it a pass/fail test; make it produce a
number you can put in a commit message.

The two that matter most:

**Leak probe** — repeat the trigger, sample RSS per iteration:

```js
const rss = () => +execSync(`ps -o rss= -p ${pid}`).toString().trim() / 1048576;
console.log(`baseline RSS ${rss().toFixed(2)} GB`);
for (let i = 1; i <= 5; i++) {
  await triggerTheReplacement(i);
  await sleep(2500);                       // let disposal settle
  console.log(`  after #${i}: RSS ${rss().toFixed(2)} GB`);
}
```

Pick a trigger that does *not* also reload the underlying resource, so you isolate the leak
from ordinary allocation. (Here: changing the system prompt restarts the session on the
already-loaded model.)

**Convergence probe** — pure, no process needed, runs in milliseconds:

```js
let hist = [systemItem, small(), small(), oversized(), small()];
for (let turn = 1; turn <= 8; turn++) {
  if (needsCompaction(hist)) hist = spliceSummary(hist, "summary");
  hist = [...hist, small(), small()];
  console.log(`turn ${turn}: ${historyChars(hist)} chars  compactsAgain=${needsCompaction(hist)}`);
}
```

If `compactsAgain` is true for more than one turn, you have class 1.2.

---

## 3. Discipline: separating harness bugs from application bugs

Most "failures" a new harness reports are the harness's fault. Filing them wastes the
session and, worse, "fixing" working code breaks it. The rule:

> Before filing a finding, verify the expectation against the source. If the source disagrees
> with your assumption, the assumption was wrong.

False positives from this session, all of which initially looked like real defects:

| Symptom | Actual cause |
|---|---|
| `sessions` returned nothing | Reply field is `.list`, not `.sessions` |
| `replace-all` did nothing | Parameter is `replacement`, not `replace` |
| Terminal never reported exit | Wait helper scanned from *after* the send, missing a fast reply |
| Clicks on the file tree did nothing | Screenshot coordinate space ≠ CSS pixels (3.57× off) |
| `Esc Esc` undo broken | Synthetic key events not trusted; real keys worked |
| Palette wouldn't open | Binding is `Cmd+Shift+P`, not `Cmd+K` |
| Model ignored a tool in one mode | Model nondeterminism — identical in all three modes |

That last row is the subtle one and deserves its own rule:

> **A model declining to do something is not a gate failure.** Before concluding a permission
> layer is broken, run the *same prompt* through every mode. If the behaviour is identical
> across modes, the mode is not the variable — the model is.

Verify gates through the deterministic layer instead: a unit test over the
`(mode × tool) → decision` matrix, plus one end-to-end case that actually provokes the call.

Corollary for the wait helper, since this bites everyone writing an async harness:

```js
// WRONG — a reply that already arrived is invisible
send(msg); await waitFor(pred);

// RIGHT — mark the position before sending
const mark = inbox.length;
send(msg);
await waitFor(pred, ms, label, mark);
```

---

## 4. Complexity audit

Separate pass, separate mindset. Correctness and simplicity reviews interfere with each
other — run them apart.

Scan the whole tree, not a diff. Tag each finding and rank biggest cut first:

- `delete:` dead code, unused flexibility, speculative feature → replacement: nothing
- `stdlib:` hand-rolled thing the standard library ships → name the function
- `native:` code doing what the platform already does → name the feature
- `yagni:` abstraction with one implementation, config nobody sets, layer with one caller
- `shrink:` same logic, fewer lines → show the shorter form

Mechanical hunts that pay:

```bash
# exports nobody outside the defining file references
for s in $(grep -rhoE "^export (async )?(function|const|class|interface|type) [A-Za-z_][A-Za-z0-9_]*" src | awk '{print $NF}' | sort -u); do
  def=$(grep -rlE "^export (async )?(function|const|class|interface|type) $s\b" src | head -1)
  others=$(grep -rl "\b$s\b" src | grep -v "^$def$")
  [ -z "$others" ] && echo "NEVER-IMPORTED  $s  [$def]"
done

# the same hand-rolled primitive written more than once
grep -rn 'indexOf("\\n")' src        # stream→lines, three times over → readline.createInterface

# files nothing imports
# dead DOM ids: declared in markup, never queried in script
```

Two guards that keep this honest:

- **Separate "exported for tests" from "exported for nobody."** Count test references
  independently. A symbol with 14 test uses and 2 production uses is a deliberate seam, not
  dead surface.
- **Verify platform claims before recommending them.** `RegExp.escape` looks like the obvious
  replacement for a hand-rolled escaper — and is absent on Node 22. One line confirms it:
  ```bash
  node -e "console.log(typeof RegExp.escape)"
  ```

Interfaces with a single production implementation are the classic YAGNI flag, but check
whether the *tests* implement them too. Two implementations, one of which keeps a test
hermetic, is a working seam. Leave it.

---

## 5. Checklist

Run in order. Stop and fix before continuing when a layer goes red.

**Baseline**
- [ ] Unit suite green, typecheck clean — recorded as the starting number
- [ ] Fresh scratch workspace, never the real project (tests mutate files)
- [ ] Fixed sampling settings (low temperature, capped output) so timings are comparable

**Layer 1 — protocol**
- [ ] Every message type / endpoint, happy path
- [ ] Every error path: malformed, missing arg, path traversal, missing resource
- [ ] Malformed and unknown messages, then prove the process still answers

**Layer 2 — stress**
- [ ] Two operations concurrent on one connection
- [ ] Two clients concurrent
- [ ] N sequential operations, all succeed
- [ ] Empty / whitespace / unicode / control-character input
- [ ] Input shaped like the system's own control protocol
- [ ] Oversized input **and the operation after it**
- [ ] Every meta/slash command, including unknown
- [ ] Each permission mode, allow path and deny path
- [ ] Disconnect mid-stream
- [ ] Persist, resume, then keep operating

**Layer 3 — UI**
- [ ] Every panel toggle, at ≥2 viewport widths, asserting rendered width
- [ ] Every overlay: open → populated → close
- [ ] Every keyboard shortcut, from every focus region
- [ ] Every write round-trips to disk
- [ ] Console clean

**Layer 4 — probes**
- [ ] RSS sampled per iteration across ≥5 repetitions of each resource-replacing action
- [ ] Every reduction step asserted to converge in one pass
- [ ] Grep: every `exec`/`fetch`/`new Promise` has a timeout or a settle-on-close path
- [ ] Every `truncated`/`hasMore` flag reaches the user-visible surface

**Close**
- [ ] Re-run all layers against the fixed build
- [ ] Regression test added for each fix that has a pure core
- [ ] Findings reported with measured before/after numbers, not adjectives

---

## 6. Packaging this as a skill

Directory layout:

```
.claude/skills/agent-app-testing/
  SKILL.md            # frontmatter + the method, kept short
  taxonomy.md         # section 1, loaded on demand
  harness/
    protocol.mjs      # layer 1 template
    stress.mjs        # layer 2 template
    probe-leak.mjs    # layer 4 leak probe
    probe-converge.mjs
```

`SKILL.md` frontmatter:

```markdown
---
name: agent-app-testing
description: >
  Find the bugs unit tests structurally cannot see in a long-running stateful
  application — resource leaks, non-convergent cleanup, unbounded waits, orphaned
  promises. Use when asked to stress-test, harden, audit reliability, or verify an
  app "actually works" rather than "passes its tests".
---
```

Keep `SKILL.md` under ~200 lines: the premise, the four layers, the checklist, and pointers
into `taxonomy.md` and the harness templates. Detail belongs in the loaded-on-demand files,
not the entry point.

**What to parameterise per project.** Only these change:

- Transport (WebSocket / HTTP / IPC / stdio) — layer 1's send-and-wait helper
- The resource-replacing action that drives the leak probe
- The reduction step that drives the convergence probe
- The UI selectors, if there is a UI

Everything else — the taxonomy, the checklist, the discipline rules — is portable as written.

**What not to parameterise.** Do not turn the checklist into config. Its value is that it is
run in full every time; a project that skips half of it silently skips the half where its own
bugs live.
