# Architecture

## Guiding idea

Human-readable text is only required at two boundaries: what the user types in, and what gets
shown or written back out. Everywhere in between — prompts, tool calls, file re-reads, session
restores — natural-language text is a convenience with a token cost, not a requirement. Subvocal
is built by repeatedly asking, at every layer that currently passes data as text/JSON purely out
of convenience, whether it can instead be a direct pointer, token ID, or tensor. The pieces below
(the tag registry, KV-native session state, the native tool-calling protocol) are each an
instance of that same move, not separate features.

## Layers

```
utter (terminal UI)
   │  AgentLoop.start() / .continue() / .followUp()
   ▼
encode (@subvocal/encode)
   │  preprocessing, KV cache policy, dual-brain orchestration, tool-call protocol
   ▼
synapse (@subvocal/synapse)
   │  N-API binding: decode, KV control, logits/hidden-state access, steering
   ▼
llama.cpp (Native GPU backend)
```

### synapse — native inference binding

A C++ N-API addon exposing llama.cpp's native GPU backend directly to Node: no HTTP server, no
subprocess, no JSON serialization between the model and the orchestration layer. It exposes:
decode (batched, KV-cache-aware), explicit KV-cache control (per-sequence remove/reset, needed
for isolated auxiliary sequences), logits and mid-layer hidden-state access, and steering hooks
(activation-space bias) for future model-behavior shaping.

llama.cpp is not vendored — it's cloned separately and built with hardware acceleration enabled (specifically, a **Metal backend** for Apple Silicon, allowing full use of unified memory). Local
modifications are tracked as a small patch set (see [subvocal-patches/](subvocal-patches/README.md))
re-applied to a fresh checkout after each upstream sync, rather than committing a full engine
fork's history to this repo.

### encode — preprocessing and orchestration

- **Dual-brain model orchestration.** The system routes tasks dynamically: a small model (E2B) acts as a fast generator for easy, mechanical edits, and escalates to a large model (12B) for complex tasks. When dual-brain is active, a dedicated small model instance is loaded for generation to keep its KV context isolated from the continuous intent-classification instance. *(Note on MacProfile: by default, the small model is disabled to fit the 12B model within the 16GB unified memory budget, falling back to a single-model approach).*
- **KV cache tiering.** Conversation state lives as KV cache, not re-serialized text. A hot tier (SWA) and warm tier (global) stay resident in RAM/unified memory. A cold tier checkpoints to disk (async write, zstd compressed) incrementally at ~4k-token boundaries mid-session (prefill ladders), so restoring an interrupted session replays from the deepest checkpoint block instead of re-running the whole transcript.
- **Tag registry ("ideograms").** A curated set of single-token Unicode
  characters used as structural markers: AST node tags for surgical edits, per-session
  path references, and CRC-style block anchors. *(Update: This feature was verified and kept. Ideograms are actively used by the legacy REPL frontend as they proved ~20% cheaper in tokens, while the TUI uses the native tool-calling protocol).*
- **TextlessRAG / Embedding Bridge.** A native embedding path via `TensorArena` that runs an embedding model (e.g. `embeddinggemma-300m`) directly through the Metal backend (`ModelGPU`) for semantically-correct code retrieval without external services.
- **Native tool-calling protocol.** Tool declarations, calls, responses, and quoting are each a
  single reserved token extracted from the model's own chat template, instead of a JSON
  tool-call block round-tripped through text parsing. This includes an **Anchor-based edit system (`edit_lines`)** that relies on a line-number gutter rather than exact text matching to avoid whitespace/indentation fragility.
- **Direct Compiler API (`lspShim`).** Replaces terminal-oriented `tsc` subprocess calls and regex parsing with direct `ts.createLanguageService` memory access, maintaining a persistent incremental cache and returning structured syntax diagnostics natively without text round-tripping.
- **Speculative decoding.** Employs two mechanisms: a suffix-tree self-speculation path that drafts from the model's own recent output, and an independent small-model drafter (E2B) running on Metal which provides up to ~1.9x speedup for the large model.
- **Predictive prefill & Prewarming.** Runs prompt-independent phases (AST tagging, intent classification, suffix-tree seeding) incrementally while the user is still typing in the REPL, hiding their latency. Intent classification specifically runs in a dedicated off-main-thread worker (using mmap-shared weights) to guarantee zero UI blocking.
- **Proactive context budgeting & Distillation.** `estimateIncomingTokens()` predicts KV overflow before generation. To handle massive histories without OOM, a small model (E2B) can read up to 128k tokens and "distill" the relevant facts into a compact briefing for the large model (which runs with a 16k context limit).
- **Ideogram Compressions.** To save tokens, recurring textual markers (e.g. intent labels, bash success/fail signals, file breadcrumbs) are mapped to single-token Unicode ideograms from a static Tag Registry.
- **Tool & Context Compaction.** Repetitive context is aggressively minimized. Re-reads of unchanged files are replaced with a single-line breadcrumb, and verbose tool outputs (e.g., bash logs) are heuristically truncated to preserve only the head, tail, and error lines.

### utter — terminal frontend

A heavily modified fork of the [pi agent harness](https://github.com/earendil-works/pi) (now fully absorbed into the monorepo), wired
directly to `encode`'s `AgentLoop` in place of pi's original provider abstraction. The entire system boots via a single unified CLI command (`subvocal`). Model decode
runs in a worker thread so the UI event loop stays responsive (streaming tokens, mid-turn abort),
and turn boundaries map onto `AgentLoop`'s native start/continue/followUp calls so KV state
persists across a session instead of re-prefilling the growing transcript on every turn.

## Where to look next

- `encode/src/agentLoop.ts` — the orchestration entry point.
- `encode/src/modelProfile.ts` — model paths, memory budgets, per-model chat templates.
- `synapse/src/native/binding_gpu.cpp` — the hardware-accelerated N-API binding.
- `utter/packages/coding-agent/src/core/local-provider/` — the frontend↔AgentLoop wiring.
- `subvocal-patches/` — tracked local modifications to the vendored llama.cpp checkout.

## Discarded Explorations

Over the course of development, several architectural paths were explicitly investigated, tested, and ultimately discarded. They are documented here to prevent repeating dead ends:

> [!NOTE]
> **MLX Native Bridge**  
> An MLX-based engine was investigated as an alternative to `llama.cpp/Metal` to potentially improve throughput. Empirical tests demonstrated that MLX offered no generation speedup, because the real bottleneck is the physical unified memory bandwidth (~200 GB/s on M2 Pro), not the dispatch framework.

> [!NOTE]
> **Live Paging of KV Cache to SSD**  
> Paging the KV cache to SSD for contexts larger than RAM was dropped. Reading KV from SSD each token dropped throughput to an unacceptable ~0.6 t/s. Furthermore, after applying a "compact SWA" patch, the memory footprint was reduced enough that a full 128k context fits entirely in the 16 GB unified RAM, making SSD paging unnecessary.

> [!NOTE]
> **Cross-Model KV Token Injection & State Transfer**  
> Directly injecting token IDs or transferring KV state from the small model to the large model was investigated but discarded. Token injection disrupted the human-readable prompt template structure. Direct KV state transfer is architecturally impossible between different models (E2B vs 12B). Instead, escalation is handled by speculatively re-prefilling the large model in the background.

> [!NOTE]
> **KV-q8 Quantization**  
> Quantizing the KV cache to q8_0 to save memory bandwidth was tested but discarded. At the project's typical context sizes, the bandwidth savings were entirely negated by the dequantization overhead on Apple Silicon, resulting in break-even performance at best. The KV cache remains f16.

> [!NOTE]
> **Speculative Full-Answer Prefill**  
> Attempting to guess the full prompt and speculatively decode a GPU answer while the user types was ruled out. It had a high miss rate for open-ended coding tasks and pointlessly competed for memory bandwidth.

> [!NOTE]
> **MTP & CPU-Draft Speculative Decoding**  
> Multi-Token Prediction (MTP) was fully implemented but resulted in net-negative speed on non-repetitive text due to low acceptance rates. Similarly, a classic *CPU-draft* pipeline (independent models where the small model ran on CPU) proved slower due to cross-bus sequential bottlenecks. (However, moving the independent small-model drafter to *Metal* completely resolved this, yielding a 1.9x speedup, and was subsequently adopted).

> [!NOTE]
> **Grammar-Constrained Tool Calls (Logit Masks)**  
> Forcing tool calls into a fixed JSON grammar using logit masks (e.g. `setASTTokenMask`) was initially investigated to fix malformed JSON outputs. It was discarded when we discovered the model had a built-in, native special-token protocol (`<|tool_call>`) which reduced malformations to 0% intrinsically, making manual grammar constraints and regex parsing unnecessary.

> [!NOTE]
> **Resident Intent Classifier in TUI**  
> Attempted to run a continuous background small-model (E2B) instance for intent classification alongside the generation models. It was discarded because keeping three Metal instances resident simultaneously (12B + E2B classifier + E2B generator) exceeded the 16 GB Apple Silicon unified memory limit, causing OOMs. Intent classification currently falls back to regex in the TUI.

> [!NOTE]
> **Action Tokens & Edit Micro-ops**  
> Advanced ideogram features like V9 action-token routing (single-token tool calls) and fine-grained edit opcodes (rename/delete/insert) were fully built into the engine. However, they remain dormant and unused because the 12B model, when only taught via prompts, could not reliably comply with such strict, specialized grammars without dedicated fine-tuning or LoRA adaptation. For the same reason, **Tool-Argument Key Ideograms** (replacing `path`, `newText` keys with single tokens) were rejected to avoid breaking the model's trained native JSON schemas.

> [!NOTE]
> **Model-based Tool Output Distillation & Auto-Single Model Fallback**  
> Using the E2B model to distill every verbose tool output (like bash logs) was discarded in favor of a fast regex-based heuristic (head/tail/errors) which proved sufficient at zero cost. Additionally, automatically downgrading to a single-model setup when editing large files was rejected; the dual-brain setup remains the default, with users able to manually opt out if needed.
