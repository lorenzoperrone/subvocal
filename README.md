# Subvocal

[![macOS Apple Silicon](https://img.shields.io/badge/macOS-Apple_Silicon-black?logo=apple)](#-macos-apple-silicon)
[![Node.js 22+](https://img.shields.io/badge/Node.js-%3E%3D22.19-green?logo=node.js)](#-download-and-installation)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

<img width="2048" height="1152" alt="Gemini_Generated_Image_af87h9af87h9af87 (1)" src="https://github.com/user-attachments/assets/1a07a9c9-709a-4075-9b26-add683f2191b" />

> **⚠️ Disclaimer: This is a massive experiment (not for production use) ⚠️**
>
> Let's be clear upfront: this is **not** a "production-ready" tool. Subvocal is essentially a huge playground where I stitched together a bunch of ideas regarding local coding agents.
> Some of these ideas have definitely been implemented by others (and probably better!), while others I hope are fairly novel. To be honest, many of these experiments yielded only marginal gains, but they were built purely for the joy of building them, pushing the limits, and having fun with the code.
> 
> If you are looking for a stable tool for your day-to-day work, this is not it. If, instead, you want to poke around a "tensor-native", slightly crazy experimental agent... welcome aboard!

## What is Subvocal (and why)?

Subvocal is a **tensor-native local developer agent**. 

Most local AI coding agents today are built as wrappers around an HTTP API (like Ollama or LM Studio). They serialize prompts to JSON, send requests over a local port, wait for text, parse it, and repeat. 

Subvocal was born from a completely different philosophy: **human-readable text is only required at the boundaries** (what the user types in, and what gets shown on screen). *Everywhere in between, human language is a cost, not a requirement.*

Why should the agent and the model communicate via JSON or HTTP when they are running on the same machine? Subvocal strips away these "text-for-convenience" layers. By running the inference engine directly in-process and interacting natively with token IDs, pointers, and tensors, we unlock capabilities that are normally impossible over an HTTP boundary.

In short, it's an attempt to build a coding agent that doesn't just "talk" to an LLM, but is physically wired into its brain.

### Why the name "Subvocal"?

In cognitive science, **subvocalization** is the internal speech you make when reading silently or thinking verbally. The brain activates the exact same language circuits and even the micro-muscles used for speaking, but without ever producing actual sound. It is a silent, internal process that helps the brain retain words in working memory while processing them.

This perfectly captures the philosophy of this architecture. Traditional coding agents constantly "vocalize" their thoughts out loud, serializing heavy JSON blobs and streaming text over HTTP APIs back and forth. Subvocal, instead, operates *silently* and internally: its dual-brain loop processes intent, routes logic, and drafts speculative tokens directly within the same shared memory space, communicating via native token IDs and tensors. It's an internal voice without a voice.

> **A dose of reality**: Let's not overpromise. We haven't actually managed to remove *all* intermediate text layers yet, and achieving a 100% tensor-native pipeline is probably impossible with just our solo efforts. But this is exactly the direction we want to experiment in!
> 
> It's also crucial to clarify *why* we are doing this. We don't remove the HTTP API to gain raw generation speed (in our tests, the REST network overhead is ~30 milliseconds, while the GPU decode takes 98% of the total time). Instead, we remove it to unlock architectural capabilities that a REST server structurally prevents: direct manipulation of the KV Cache, memory tiering on SSD, and dual-brain speculative execution in the exact same memory space.

---

## What's inside the toy box?

Unlike traditional agents that communicate with an LLM via standard HTTP API calls, Subvocal runs GGUF inference directly *in-process*, leveraging native hardware acceleration. The project consists of three main modules:

- **`synapse`** — *The muscle*. A C++ N-API binding that exposes llama.cpp's native backend directly to Node. It gives us access to decoding, KV cache control, and hooks for logit-level steering, all without the overhead of an HTTP server.
- **`encode`** — *The brain*. The preprocessing and orchestration layer. This is where the weirdest stuff happens: "AST-aware" source tagging, a registry of single-token "ideograms" for tool calls, KV cache tiering, and a "dual-brain" Agent Loop (pairing a small, fast model for routing/drafting with a large one for generation).
- **`utter`** — *The face*. The terminal frontend. It's a heavily modified fork of [pi agent harness](https://github.com/earendil-works/pi), wired directly into our in-process `AgentLoop`.

To start everything, you just need a single command: `subvocal`.

## Architectural Highlights

Here are the top 5 most experimental (and hopefully original) ideas we've been playing with that differentiate Subvocal from standard API-wrapper agents:

1. **In-Process Tensor-Native Wiring (Zero HTTP/JSON overhead)**: Unlike standard tools that serialize prompts over HTTP to a local server, Subvocal physically embeds the `llama.cpp` inference engine inside the Node.js process via `synapse` N-API. This allows direct manipulation of the KV Cache, mid-layer state, and logits.
2. **Multi-Tier KV Cache with "Prefill Ladders" on SSD**: Instead of keeping the entire conversation in active VRAM, Subvocal uses an OS-like memory tiering system. It leverages SWA (Sliding Window Attention) to cap the RAM footprint, while asynchronously dumping "prefill ladders" (checkpoints compressed with `zstd`) to the SSD. If the session is interrupted, it resumes from the deepest SSD block seamlessly.
3. **Dual-Brain Heterogeneous Orchestration in Shared Memory**: The `AgentLoop` dynamically routes tasks between a massive 12B generator and a tiny 2.6B E2B drafter/classifier. Because they share the exact same tokenizer and run in the same memory space, the small model can distill massive 128k-token contexts or draft speculative tokens without ever serializing data to text.
4. **Ideogram Compressions & Tag Registry**: To save tokens and reduce parsing errors, verbose system markers (like unchanged file breadcrumbs, bash pass/fail signals, or intent labels) are mapped "on the fly" into single-token Unicode characters (Ideograms) injected from a static Tag Registry.
5. **Direct Compiler API Telemetry (`lspShim`)**: Instead of spawning heavy `tsc` subprocesses or running a full LSP server over IPC, Subvocal loads the TypeScript compiler API directly into memory via `ts.createLanguageService`. It maintains an incremental, persistent AST cache and natively extracts structured `ts.Diagnostic` objects.

*(This is just a sneak peek. For the complete, exhaustive list of all implemented features—and all the discarded explorations!—make sure to read [ARCHITECTURE.md](ARCHITECTURE.md)).*

## Model Choice: Why Gemma?

While the architecture technically supports any GGUF model through `llama.cpp`, this entire project, its memory budget, and its native interaction protocol were built and aggressively optimized around the **Gemma 4** family (specifically `gemma-4-12B-it-qat` as the large generator, and a smaller E2B-class variant as the drafter/router).

The choice was not random. We chose Gemma for four crucial reasons:
1. **Memory Footprint**: The 12B QAT (Quantization-Aware Training) model at Q4_K_XL takes about 6.7 GB. When paired with a smaller 2.6 GB model for dual-brain routing, it leaves just enough room for the macOS system, the application overhead, and our custom multi-tier KV Cache, fitting perfectly within the strict 16 GB Apple Silicon unified memory limit.
2. **Hybrid SWA Architecture**: Gemma 4 natively uses a hybrid **SWA (Sliding Window Attention)** mechanism alongside global attention. This architectural quirk severely caps the growth of the KV Cache in RAM, making it possible to keep an otherwise impossible 16k context window resident without crashing the unified memory.
3. **Byte-Identical Tokenizers**: Both the large 12B model and the small E2B model share the exact same 262k-vocabulary tokenizer. This guarantees that token IDs mean the exact same thing to both models, making our cross-model Speculative Decoding and native tool-calling protocol possible without expensive text re-serialization.
4. **Native Tool Calling**: The model is trained to emit native special tokens (e.g., `<|tool_call|>`) instead of relying on fragile JSON blocks for tool interaction, virtually eliminating parsing errors and malformed edits.

## Download and Installation

Currently, we are releasing Subvocal exclusively for the Apple ecosystem. The architecture is designed to be cross-platform, but in this experimental phase we prefer to focus on a single hardware target to guarantee a pre-tested, super-optimized package right out of the box.

### macOS (Apple Silicon)

**Minimum requirements:** M-series processor, at least 16GB of RAM, and **[Homebrew](https://brew.sh)** installed (the script will use it to install Node.js or cmake if missing).

To install, simply clone this repo and run:
```bash
./install.sh
```

This script does all the dirty work for you:
1. If it finds a prebuilt bundle (including `node_modules`) for your exact Node version in the latest release, it downloads and extracts it (**no compilation needed**).
2. If it doesn't find it (e.g., you use a different Node version), it will download the toolchain and build everything from source.
3. It will ask for confirmation before downloading the GGUF models from Hugging Face into the `models/` folder (they weigh several GB).
4. Finally, it registers the `subvocal` command globally (`npm link`). 

*(The script is safe to re-run; it will automatically skip steps that have already been completed).*

#### Memory Allocation Profile (Default Dual-Brain)

Subvocal is designed to squeeze into 16GB of Unified Memory by aggressively capping RAM and tiering state to the SSD. If you use the default Dual-Brain configuration (12B + E2B drafter), here is exactly what to expect from your Mac's resources during execution:

#### 1. Unified Memory (RAM) Breakdown

Subvocal requires **~11.7 GB** of your Mac's Unified Memory. Here is exactly where every byte goes:

- **Models Weights (mmap): ~9.3 GB**
  - **Gemma 12B (Main):** ~6.7 GB
  - **Gemma E2B (Drafter):** ~2.6 GB

- **KV Cache (RAM): ~0.9 GB**
  - **12B SWA Cache:** 480 MB (Fixed size sliding window)
  - **12B Global KV:** ~256 MB (Linearly scaling up to 16k context)
  - **E2B Shadow KV:** ~150 MB 
  *(This is the active memory used by the engine to retain the immediate context of the current turn).*

- **Application & Compute Overhead: ~1.5 GB**
  - **Compute Buffers (GPU):** ~1.0 GB (Required by the inference engine to store intermediate matrix multiplications for the two models. This is pure physics, not UI bloat).
  - **Node.js & TUI:** ~0.5 GB (The React-based terminal UI, N-API bridging, and JavaScript engine).

*(When you close Subvocal, only the KV Cache and App Overhead drop instantly. The 9.3 GB of weights remain parked in macOS's inactive memory until needed by other apps).*

#### 2. Disk Storage (SSD) Breakdown

To prevent Out-of-Memory crashes on 16GB machines, Subvocal heavily relies on your disk.

- **KV Cold Store Budget: 4.0 GB**
  Historical context and old branches of the conversation are compressed asynchronously in the background (using `zstd` with a byte-plane split) and written to `~/.cache/subvocal/kv-cold`. Thanks to the ~1.18x compression ratio, this 4 GB budget safely holds **~6 complete 16k-token contexts** (from both models combined) ready to be swapped in seamlessly without touching your RAM.

> **⚠️ Note on 16GB Macs:** The agent requires almost 12 GB of Unified Memory footprint to run smoothly. If you run heavy apps (like Chrome with many tabs, or other Electron desktop apps) alongside Subvocal, you will force macOS to heavily swap to disk, slowing down generation. For the best experience on a 16GB machine, we recommend closing other RAM-hungry apps before starting the agent.

## Usage & Advanced Flags

Once installed, simply open your terminal in the directory of the project you want to work on and type:
```bash
subvocal
```
This will launch the interactive terminal UI.

### Advanced Environment Variables
The `subvocal` launcher supports a few hidden "power-user" flags:

- **Custom Models (`SUBVOCAL_LOCAL_MODEL`)**: By default, the script looks for the Gemma models downloaded by `install.sh`. If you want to test a different GGUF model, pass its absolute path:
  ```bash
  SUBVOCAL_LOCAL_MODEL=/path/to/my-custom-model.gguf subvocal
  ```
- **The "Monitor" Window (`SUBVOCAL_MONITOR`)**: Since the main TUI hides the raw generation to provide a clean chat interface, you can launch Subvocal with the monitor flag. It will automatically open a second read-only terminal window showing the real-time activity feed (what the agent is writing and the exact files it is modifying).
  ```bash
  SUBVOCAL_MONITOR=1 subvocal
  ```
  *(Tip: Use `SUBVOCAL_MONITOR=raw` to see the literal token-by-token decode stream).*

### Linux x86-64 (TBD)

*Support coming soon. (The architecture will leverage native CUDA/Vulkan backends).*

### Windows x86-64 (TBD)

*Support coming soon.*

## Manual Build (For the curious)

If you prefer not to use the magic script and want to understand how things work under the hood:

```bash
# 1. Clone llama.cpp next to synapse/ and build it with hardware support (e.g., Metal)
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && cmake -B build -DGGML_METAL=ON && cmake --build build --target llama -j  # (Use GGML_CUDA=ON etc. on other OS)
cd ..

# 2. Apply the tracked patches (see subvocal-patches/INDEX.md)
cd llama.cpp
git apply ../subvocal-patches/llama-cpp-upstream/101-partial-forward.patch
git apply ../subvocal-patches/llama-cpp-upstream/102-compact-swa-iswa.patch
cmake --build build --target llama -j
cd ..

# 3. Install workspace dependencies and build
npm install
npm run build

# 4. Optional: register the global command
npm link

# 5. Run
subvocal
```

## Release Management (Maintainers)

To publish prebuilt artifacts to a GitHub Release:
- **`scripts/package-full-bundle.sh`** — Packages everything (dist/, compiled addon, `node_modules`) for a specific Node version into a zip archive. This is the only prebuilt artifact we distribute (it allows for a zero-build installation for users with the exact same Node version).

## Bug Reports & Feedback

Since this is an experimental project pushing the limits of the hardware, **you will encounter bugs**.
If you experience a crash, an out-of-memory (OOM) error, or the model gets stuck, please open an Issue on GitHub. 

To help us debug effectively, make sure to include:
- **Your exact Mac model and Unified Memory size** (e.g., Mac Studio M2 Max 32GB, Mac Mini M4 16GB).
- **The Node.js version** you are running (`node -v`).
- **The crash logs**, if any are present (usually dumped in the terminal output or your standard error).

## Inspirations and Acknowledgments

As mentioned, this project is a mosaic of ideas and stands on the shoulders of giants. Massive thanks go to:
- **[llama.cpp](https://github.com/ggml-org/llama.cpp) (Georgi Gerganov & co.) — MIT License**: The formidable inference engine that made the local LLM explosion possible, on which our entire native backend is based.
- **[pi agent harness](https://github.com/earendil-works/pi) (Mario Zechner) — MIT License**: The starting point for our `utter/` frontend, a fantastic terminal UI that we forked (which was originally designed for external APIs).
- **[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) (I. Kawrakow) — MIT License**: Whose excellent work on the *suffix-tree* was fundamental in developing our drafter for speculative decoding.
- **[ds4](https://github.com/antirez/ds4) (Salvatore Sanfilippo "antirez") — MIT License**: His excellent work on this engine and his attitude toward experimentation were a major source of inspiration.
- A general thank you to the amazing open source AI community and to all the researchers constantly experimenting with new architectures (routing, dual-brain, etc.), providing the spark for all these experiments.
- **AI Pair Programmers**: A significant portion of this project's code, architectural validation, and documentation was written and debugged collaboratively with the help of various AI coding assistants.

## License

The source code of Subvocal is released under the **MIT** license — see the [LICENSE](LICENSE) file.

This project includes third-party code and libraries. For full details on original copyrights (pi harness, llama.cpp, ik_llama.cpp, etc.) and their respective licenses, please refer to the [NOTICE.md](NOTICE.md) file.

---
*From Turin, with love 🍫 and a lot of trial & error.*
