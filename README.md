<div align="center">

# ◆ Local Language Machine

### All‑in‑one, fully offline AI coding — self‑contained inference, zero external services.

A complete AI coding IDE that runs **entirely on your own machine**. The model is bundled and
runs in‑process; your code never leaves the device; there is no API key, no account, and no
network dependency. Open a folder, load a model, and start pairing.

![Node](https://img.shields.io/badge/node-%E2%89%A522-3C873A?logo=node.js&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-5.9-3178C6?logo=typescript&logoColor=white)
![Offline‑first](https://img.shields.io/badge/offline--first-100%25-6E56CF)
![Inference](https://img.shields.io/badge/inference-llama.cpp-orange)
![Tests](https://img.shields.io/badge/tests-65%20passing-2ea44f)
![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen)

<img src="docs/screenshots/overview.png" alt="Local Language Machine — the main IDE: file explorer, editor, and AI assistant" width="900" />

</div>

---

## Why

Most "AI IDEs" are thin clients for someone else's cloud. Every keystroke of context, every file
you open, and every prompt you send is shipped to a remote server. **Local Language Machine flips
that around:**

- 🔒 **Private by construction.** Inference runs in‑process. Your files, prompts, and generated
  code stay on disk. The network switch is **off** by default and gates *every* online feature.
- ⚡ **Session‑based inference.** A live context + KV cache means each turn only evaluates new
  tokens — not the whole conversation — so multi‑turn coding stays fast.
- 🧩 **Real agent, real tools.** The assistant actually reads and edits your files, runs commands,
  and searches the tree — through a permissioned tool loop, not copy‑paste.
- 🪶 **No lock‑in.** Curated models out of the box, or bring any GGUF from Hugging Face. Swap the
  backend without touching the rest of the app.

---

## Features

| | |
|---|---|
| 🧠 **Bundled inference** | `llama.cpp` runs in‑process with automatic GPU detection (CUDA · Vulkan · Metal) and a clean fallback to CPU. |
| 🛠️ **Agentic tool loop** | Read, write, `grep`, `glob`, run a terminal command, and drive `git` — every path sandboxed to the workspace root. |
| 🎚️ **Permission modes** | `manual`, `accept‑edits`, `plan`, and `auto` — with a dangerous‑command guard and an explicit allow‑list. |
| ⏪ **Checkpoints** | Every model write is snapshotted first, so a single undo rewinds file edits — independent of git. |
| 🗜️ **Context compaction** | Long sessions auto‑summarize older turns to stay inside the model's context window. |
| 📎 **Project memory** | Drop an `AGENTS.md` at the root and its conventions ride along in every request. |
| 🧭 **Skills & subagents** | Reusable, `/slash`‑invocable skills and isolated read‑only research subagents. |
| 🔌 **MCP & hooks** | Connect Model Context Protocol servers and fire shell hooks on lifecycle events. |
| 🤗 **Model manager** | Search the Hugging Face Hub or paste an `owner/repo` id; resumable download + SHA‑256 verify. |
| 🎓 **On‑device fine‑tuning** | Built‑in LoRA training — MLX on Apple Silicon, Unsloth on NVIDIA — with live loss streamed to the UI, plus a runs history. |
| 📊 **Benchmark & evaluate** | Measure base vs fine‑tuned on real held‑out data — perplexity, generation speed, and side‑by‑side samples — rendered as charts. Convert a run to GGUF and load it right back into the Assistant. |
| 🎨 **Liquid‑glass UI** | A themeable interface with color presets, custom backgrounds, effect tuning, and a performance mode. |
| 💻 **Integrated dock** | A built‑in terminal and a live browser panel for previewing what you build. |
| 🔀 **Sessions** | Resume, fork, and clear conversations — history is persisted per project. |

<div align="center">
<img src="docs/screenshots/theming.png" alt="Customize panel — theme presets, per‑token colors, and a live preview" width="820" />
<br/><em>Customize everything — theme presets, per‑token syntax colors, and a live glass preview.</em>
</div>

---

## Architecture

A UI‑agnostic core sits between any front‑end and a bundled native layer. The **Orchestrator** is
the composition root; nothing in the core knows about HTTP or webviews.

```mermaid
flowchart TB
    subgraph UI["🖥️  Interface — UI-agnostic"]
      WEB["Web preview<br/>index.html ⇄ WebSocket"]
    end

    subgraph CORE["🧠  Core — Orchestrator (composition root)"]
      direction LR
      LOOP["Agent loop<br/>generate → tool → repeat"]
      SESS["Sessions<br/>resume · fork"]
      CKPT["Checkpoints<br/>undo"]
      PERM["Permissions<br/>& hooks"]
      CTX["Context<br/>compaction"]
      MEM["Memory<br/>AGENTS.md"]
      SKILL["Skills &<br/>subagents"]
    end

    subgraph NATIVE["⚙️  Native — 100% offline"]
      direction LR
      ENGINE["Inference engine<br/>llama.cpp · GPU auto"]
      TOOLS["Workspace tools<br/>read · write · grep · git · shell"]
      MODELS["Model registry<br/>+ Hugging Face hub"]
      MCP["MCP client"]
      TRAIN["Fine-tuning<br/>LoRA · MLX / Unsloth"]
    end

    WEB --> LOOP
    WEB --> TRAIN
    LOOP --> ENGINE
    LOOP --> TOOLS
    LOOP --> SKILL
    LOOP --> MCP
    ENGINE --> MODELS

    style CORE fill:#6E56CF22,stroke:#6E56CF
    style NATIVE fill:#2ea44f22,stroke:#2ea44f
    style UI fill:#3178C622,stroke:#3178C6
```

### How a turn works

Tool calling uses a **JSON‑in‑text protocol** rather than native function calling — local models
are inconsistent about native call formats, and JSON‑in‑text works with anything that can emit JSON.

```mermaid
sequenceDiagram
    autonumber
    participant U as You
    participant L as Agent loop
    participant M as Local model
    participant T as Workspace tools

    U->>L: prompt
    loop up to 6 steps
        L->>M: generate
        M-->>L: prose · code · or a JSON tool call
        alt tool call
            L->>T: run tool (gated by permission mode)
            T-->>L: result
            L->>M: feed result back
        else final answer
            L-->>U: stream prose → chat, fenced code → editor
        end
    end
```

---

## Quick start

> **Requirements:** Node ≥ 22. A GPU is used automatically when present, but everything runs on CPU too.

```bash
# 1. Install dependencies (the inference engine is bundled — no extra services)
npm install

# 2. Launch the app (opens the local preview server)
npm run preview

# 3. Open the printed http://localhost:… URL, then load a model from the Assistant panel.
#    The first load downloads the model once into ~/.local-language-machine and caches it.
```

Other scripts:

```bash
npm test        # run the test suite (node --test)
npm run typecheck   # strict TypeScript, no emit
```

---

## Models

Curated, coding‑focused defaults ship in the registry. The best cached model auto‑loads on startup.

| Model | Params | Quant | Size | Notes |
|-------|:------:|:-----:|:----:|-------|
| **Qwen2.5‑Coder** | 7B | Q4_K_M | ~4.7 GB | Default — best quality |
| **Qwen2.5‑Coder** | 1.5B | Q4_K_M | ~1.0 GB | Fast, tool‑capable |
| **SmolLM2** | 135M | Q4_K_M | ~0.1 GB | Smoke test / low‑RAM |

…or add **any GGUF** from the Hugging Face Hub — search inside the app or paste an `owner/repo` id.
Downloads are resumable and verified against a pinned SHA‑256 when one is set.

<div align="center">
<img src="docs/screenshots/models.png" alt="Add a model from Hugging Face — search the Hub or paste an owner/repo id" width="820" />
</div>

---

## Slash commands

| Command | Does |
|---------|------|
| `/help` | List commands and available skills |
| `/model [id]` | Show or switch the loaded model |
| `/context` | Token / context‑window usage |
| `/compact [focus]` | Summarize older turns to free context |
| `/clear` | Start a fresh session |
| `/init` | Scaffold an `AGENTS.md` for project memory |
| `/resume [id]` | List or resume a saved session |
| `/fork` | Branch the current session |
| `/<skill>` | Invoke any installed skill by name |

---

## The agent's tools

| Tool | Purpose |
|------|---------|
| `read_file` · `list_dir` | Read files and browse the tree |
| `grep` · `glob` | Search contents by regex, or find files by pattern |
| `write_file` | Create/overwrite a file (checkpointed first) |
| `run_terminal` | Run a shell command in the workspace root |
| `git` | Read‑only subcommands run directly; mutating ones ask first |
| `task` | Delegate a focused, read‑only job to an isolated subagent |
| `web_search` · `web_fetch` | Online research — **only** when online mode is enabled |

Every file path is resolved **inside** the workspace root; escapes are rejected. Mutating actions
pass through the permission gate before they run.

<div align="center">
<img src="docs/screenshots/terminal.png" alt="Integrated terminal and browser dock beneath the editor" width="900" />
<br/><em>An integrated terminal and live browser panel sit right under the editor.</em>
</div>

---

## Extending it

- **Skills** — a folder with a `SKILL.md` (frontmatter + body). Descriptions are injected into the
  system prompt; the body loads on demand when the skill is invoked or matched. Import from a local
  folder (offline) or a git URL (online).
- **Subagents** — Markdown agent definitions that run the same loop in an isolated context with a
  safe, read‑only toolset by default.
- **Hooks** — shell commands fired on `SessionStart`, `UserPromptSubmit`, `PreToolUse`,
  `PostToolUse`, and `Stop`.
- **MCP** — point the app at Model Context Protocol servers to add tools; a failing server never
  blocks the session.

---

## Fine‑tuning & evaluation (LoRA)

Teach a model your codebase or house style **locally**. The built‑in trainer runs LoRA fine‑tuning
with the right backend for your hardware — **MLX** on Apple Silicon, **Unsloth** on NVIDIA — and
streams live loss into the UI. Point it at a `.jsonl` dataset (rows of `text`, `messages`, or
`prompt`+`completion`) **or paste a Hugging Face dataset id** (fetched and converted for you),
choose a base model, and start. Every run is saved with its metadata under a **Runs** history, and
the first run bootstraps a dedicated Python venv automatically (the one online moment, just like the
initial model download).

<div align="center">
<img src="docs/screenshots/finetune.png" alt="LoRA fine-tuning panel — base model, dataset, hyperparameters, and a live training log" width="860" />
</div>

### Benchmark it — real data, real charts

The **Evaluate** tab measures your fine‑tuned model against the base on a held‑out test split, all
on‑device, and renders it as inline SVG (no chart library). Example: a SmolLM2‑135M LoRA on a slice
of `tatsu-lab/alpaca` — **held‑out perplexity dropped from 12.99 → 3.61** while the training and
validation loss fell together:

<div align="center">
<img src="docs/screenshots/bench-loss.svg" alt="Training and validation loss falling from iter 1 to iter 30" width="560" />
<br/>
<img src="docs/screenshots/bench-perplexity.svg" alt="Held-out perplexity: base 12.99 vs fine-tuned 3.61 (lower is better)" width="300" />
<img src="docs/screenshots/bench-speed.svg" alt="Generation speed: base 536 vs fine-tuned 291 tokens/sec" width="300" />
</div>

It also generates **side‑by‑side samples** from the same prompts so you can eyeball the behaviour
change — e.g. after training on Alpaca, the tuned model adopts its terse numbered‑list style:

| Prompt | Base | Fine‑tuned |
|---|---|---|
| *Give three tips for staying healthy.* | "Here are three tips for staying healthy:\n\n1. **Stay Hydrated**: Drink plenty of water…" | "1. Eat a balanced diet: A healthy diet provides the necessary nutrients…\n2. Get enough sleep: Aim for 7‑9…" |

When you're happy with a run, **Convert & add** fuses the adapter and exports a GGUF that loads
straight back into the Assistant — so you can chat with the model you just trained.

---

## Project layout

```
src/
├─ core/           UI-agnostic brain
│  ├─ orchestrator.ts   composition root — wires everything to the agent loop
│  ├─ loop.ts           the reusable generate → tool → repeat loop
│  ├─ session.ts        persisted conversation history
│  ├─ checkpoints.ts    file-edit snapshots (undo)
│  ├─ settings.ts       permission modes, allow-list, hooks, MCP
│  ├─ context.ts        token accounting + compaction
│  ├─ memory.ts         AGENTS.md / MEMORY.md loading
│  ├─ skills.ts         skill discovery + frontmatter parsing
│  ├─ subagent.ts       isolated-context workers
│  └─ commands.ts       slash-command router
├─ native/         offline building blocks
│  ├─ inference/        the bundled engine abstraction
│  ├─ tools/            file / grep / git / terminal / web tools
│  ├─ models/           curated registry + Hugging Face hub
│  ├─ training/         LoRA trainer, benchmarker, and GGUF export
│  └─ mcp/              Model Context Protocol client
└─ preview/        localhost web UI (server + single-page app)
```

---

## Roadmap

- [ ] VS Code webview front‑end (the core already speaks a postMessage‑style protocol)
- [ ] Nested project‑memory discovery
- [ ] Per‑folder MCP configuration
- [ ] HTTP / prompt / subagent hook types
- [ ] Mutating subagents with per‑agent model overrides
- [ ] Full PTY terminal (interactive `vim` / `top`)

---

## License

No license has been chosen yet — all rights reserved by default. Add a `LICENSE` file if you want
others to reuse the code.
