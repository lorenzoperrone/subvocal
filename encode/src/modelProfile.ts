/**
 * modelProfile.ts
 *
 * Model profiles for Subvocal — one profile per model family.
 * Each profile captures everything that is model-specific:
 *   - model paths
 *   - chat template format
 *   - EOT / stop token IDs
 *   - intent anchor token IDs (verified against that family's vocab)
 *   - TagRegistry file (mono-token chars differ per vocab)
 *   - default init options
 *
 * Active profile: SUBVOCAL_MODEL_PROFILE env var ('gemma4' | 'qwen3' | 'mac').
 * Default: 'mac' -- this is the Mac-only port repo; 'gemma4'/'qwen3' are the
 * Linux/CUDA dual-brain profiles this was forked from and use a different chat
 * template (<start_of_turn>/<end_of_turn>) than this repo's actual 12B-Unified
 * checkpoint, which needs <|turn>role\n...<turn|> (see project memory
 * project_12b_chat_template.md). Defaulting to gemma4 here silently fed every
 * caller that didn't pass `profile: MacProfile` explicitly the wrong template.
 */

import path from 'path';
import os from 'os';
import { fileURLToPath } from 'url';
import type { Intent } from './intentRouter.js';
import type { ToolDefinition } from './toolParse.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/** Wraps a string in the native protocol's reserved quote-marker token (never JSON's `"`). */
function nativeQuote(s: string): string {
	return `<|"|>${s}<|"|>`;
}

// ── Profile interface ─────────────────────────────────────────────────────────

export interface ModelProfile {
  readonly name: 'gemma4' | 'qwen3' | 'mac' | 'mac-e2b';

  // ── Model paths ───────────────────────────────────────────────────────────
  readonly smallModelPath: string;
  readonly largeModelPath: string;

  // ── Tokenizer ─────────────────────────────────────────────────────────────
  readonly vocabSize: number;
  /** Token IDs that signal end-of-turn / stop generation. */
  readonly eotTokenIds: readonly number[];
  /** String stop sequences for generation (complement to eotTokenIds). */
  readonly stopStrings: readonly string[];

  // ── Chat template ─────────────────────────────────────────────────────────
  /**
   * Build the raw prompt string fed to tokenize().
   * BOS is added by tokenize(addBos=true) — do NOT include it here.
   * @param prefill  Optional string to pre-fill the model turn with.
   */
  buildPrompt(opts: {
    systemPrompt: string;
    userPrompt: string;
    prefill?: string;
  }): string;

  /**
   * Like buildPrompt, but for the SMALL model when its checkpoint's chat template differs
   * from the large model's. Optional: callers fall back to buildPrompt when undefined.
   * MacProfile needs this because the E2B generation prompt is bare '<|turn>model\n' — the
   * 12B's empty-thought-channel insertion ('<|channel>thought\n<channel|>') is absent from
   * the E2B chat_template (verified by diffing both GGUFs' tokenizer.chat_template, 2026-07-01;
   * feeding it anyway is what produced the '<channel|>' artifact in the dual-brain E2B spike).
   */
  buildSmallPrompt?(opts: {
    systemPrompt: string;
    userPrompt: string;
    prefill?: string;
  }): string;

  /**
   * Build the tool-response turn format for feeding observation back to the model.
   * Injected via decodeAppend() into the KV cache after the model's turn.
   * Must end with the opening token(s) that start the next model turn.
   * @param toolName  Name of the tool that produced `observation` (native-protocol profiles
   *                  render this into the response block; profiles without a native
   *                  tool-response convention may ignore it).
   */
  buildToolResponse(observation: string, toolName?: string): string;

  /**
   * Render tool declarations using this checkpoint's own native special-token protocol
   * (confirmed from the GGUF's tokenizer.chat_template -- see
   * doc/epics/EPIC-M8-grammar-constrained-tool-calls.md), so the model can use its trained-in
   * tool-calling behavior instead of being asked to imitate a different model family's text
   * convention. Optional: only implemented by profiles whose checkpoint actually has this
   * mechanism (MacProfile). Callers should fall back to describing tools in free-form system
   * text when this is undefined.
   */
  buildToolDeclarations?(tools: readonly ToolDefinition[]): string;

  /**
   * Build a follow-up user turn for appending to KV cache.
   * Used when the user sends a second prompt without switching files.
   */
  buildFollowUp(userPrompt: string): string;

  // ── Intent routing ────────────────────────────────────────────────────────
  /**
   * Per-class anchor token IDs for logit sniffing.
   * Verified with model.tokenize(word, false, false) on this family's vocab.
   */
  readonly intentAnchors: Readonly<Record<Intent, readonly number[]>>;

  /**
   * Pre-fill string appended to the model turn before logit sniffing.
   * Steers the first logit away from chat-template markers.
   */
  readonly intentPrefill: string;

  // ── TagRegistry ───────────────────────────────────────────────────────────
  /** Absolute path to TagRegistry.<profile>.json for this model family. */
  readonly tagRegistryPath: string;

  // ── KV cache tier config ──────────────────────────────────────────────────
  /**
   * Multi-tier KV layout for the large model. Two DISTINCT concepts share the word "tier":
   *
   *  hot  — SWA KV in VRAM (fixed by model architecture, always via LLAMA_KV_SWA_OFFLOAD=1).
   *  warm — the live global KV of the CURRENT context, in system RAM (noKvOffload=true,
   *         contextSize=warm.contextSize). This is the default operating mode.
   *  cold — a disk-resident KV *checkpoint cache* on SSD (encode/src/kvColdStore.ts, the ds4
   *         design — doc/research/ds4-kvstore-findings.md). It stores full KV snapshots of
   *         prompt prefixes so a re-seen prompt skips its (expensive) re-prefill; it does NOT
   *         page a single live context to SSD (that would be a separate llama.cpp feature,
   *         LLAMA_KV_DISK_PATH, which is not ported anywhere). `budgetBytes` caps the on-disk
   *         cache — SSD is cheap, so this is the tier to make LARGE. Optional; omit to disable.
   *
   * Physical-tier count differs by platform (both use the same `kvTiers` shape):
   *   - discrete-GPU PC (Gemma4Profile): 3 tiers — VRAM (hot) + system RAM (warm) + SSD (cold).
   *   - Apple Silicon (MacProfile): 2 tiers — hot and warm share the ONE unified 16 GB pool
   *     (no separate VRAM), so only unified-RAM + SSD (cold) are physically distinct.
   *
   * hot/warm approxMib are PER-PROFILE measured/derived values — do not assume one KV dtype
   * across profiles: Gemma4Profile (Linux) sizes with its q8_0 formulas; MacProfile runs KV
   * at f16 (2 B/elem — q8 was measured net-negative on Metal, see project_kv_q8_negative)
   * with numbers read from the actual runtime cache logs (M3.5).
   */
  readonly kvTiers: {
    readonly hot: {
      readonly location: 'vram';
      readonly approxMib: number;   // fixed; does not scale with contextSize
    };
    readonly warm: {
      readonly location: 'ram';
      readonly contextSize: number;
      readonly approxMib: number;
    };
    readonly cold?: {
      readonly location: 'ssd';
      /** Directory holding the SHA1-text-keyed KV checkpoint files. */
      readonly diskPath: string;
      /** On-disk budget in bytes; the store evicts lowest-value checkpoints when over it. */
      readonly budgetBytes: number;
    };
  };

  // ── Default init options ──────────────────────────────────────────────────
  /**
   * Which synapse binding loads the small model. 'cpu' (default) = ModelCPU/ik_llama.cpp
   * static; 'gpu' = ModelGPU via the platform GPU binding. MacProfile must say 'gpu':
   * the CPU addon is deliberately not built on Mac (see
   * feedback_avoid_ik_llama_cpu_backend) — everything model-shaped routes through Metal.
   */
  readonly smallBackend?: 'cpu' | 'gpu';
  readonly smallOpts: { contextSize: number; threads: number; gpuLayers: number };
  /**
   * Cap on the LARGE model's context when the small model is ALSO loaded (dual-brain mode,
   * --cpu-model on). Applied by initSession() unless SUBVOCAL_CONTEXT_SIZE is set explicitly
   * (an explicit env var always wins). Only meaningful on unified memory: both models share
   * one pool, so the warm KV is the lever that keeps the machine out of swap — measured
   * 2026-07-01 on the 16 GB M2 Pro: 12B@128k + E2B together = 12.1 GiB wired, 64 MB unused,
   * 2.2 GB swap in use with a normal desktop session open. Undefined = no cap (discrete-GPU
   * profiles keep their full warm context; the small model lives in system RAM there).
   */
  readonly dualBrainMaxCtx?: number;
  /**
   * The E2B's OWN context cap when it's resident alongside the large model (drafter or
   * dual-brain generator role) — independent of dualBrainMaxCtx (2026-07-07 owner call: the
   * pair no longer shares one window; the cheap/fast E2B gets a generous one, the expensive
   * 12B gets a tight one). Undefined = the E2B falls back to its own smallOpts.contextSize.
   */
  readonly e2bMaxCtx?: number;
  /**
   * largeOpts drives the default (warm tier) initialization.
   * contextSize is overridden by initLargeModel when a tier is specified explicitly.
   */
  readonly largeOpts: { contextSize: number; threads: number; gpuLayers: number; noKvOffload?: boolean };

  /** Default sampling temperature. 0 = greedy. Use >0 for low-quant models to break loops. */
  readonly defaultTemperature?: number;
  /** Default nucleus-sampling cutoff. Strongly recommended whenever defaultTemperature > 0 on
   * large-vocab models -- raw temperature sampling with no truncation can draw from the long
   * tail (garbage/foreign-script tokens), see feedback_agentloop_sampling_missing_topp_topk. */
  readonly defaultTopP?: number;
  /** Default top-K cutoff, applied alongside or instead of defaultTopP. */
  readonly defaultTopK?: number;
}

// ── Gemma 4 profile ───────────────────────────────────────────────────────────
//
// CPU small : Gemma 4 E2B (4.6B dense, 35 layers, hidden 1536, vocab 262144)
// GPU large : Gemma 4 26B-A4B (MoE 128 experts top-8, 30 layers)
// EOT       : 106 '<turn|>'  (also 1 '<eos>', 50 '<|tool_response|>', 212 '</s>')
// Template  : system embedded in user turn; <start_of_turn>model\n{prefill}

const MODEL_DIR = '/mnt/dati_cachy/LLM/lmstudio-community';

// Configurable context size: SUBVOCAL_CONTEXT_SIZE env var or default 64k (65536).
// Set to any value >= 4096. The model supports up to 256k (n_ctx_train = 262144).
// Higher values use more system RAM for the global KV cache (~20 MB per 1k tokens).
const DEFAULT_CTX = 65536;
const CTX = (() => {
  const v = process.env.SUBVOCAL_CONTEXT_SIZE ? Number(process.env.SUBVOCAL_CONTEXT_SIZE) : DEFAULT_CTX;
  if (isNaN(v) || v < 4096) return DEFAULT_CTX;
  return Math.min(v, 262144);
})();

export const subvocalContextSize = CTX;

export const Gemma4Profile: ModelProfile = {
  name: 'gemma4',

  smallModelPath: `${MODEL_DIR}/unsloth-gemma-4-E2B-it-qat-GGUF/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf`,
  largeModelPath: `${MODEL_DIR}/unsloth-gemma-4-26B-A4B-it-qat-GGUF/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf`,

  vocabSize: 262144,
  eotTokenIds: [106, 1, 50, 212],  // <turn|>, <eos>, <|tool_response|>, </s>
  // String stop sequences for generation (use in server/decoder alongside token IDs).
  // <|channel> appears when the model emits thinking tokens — treat as end of response.
  stopStrings: ['<end_of_turn>', '<|channel>', '<turn|>'],

  buildPrompt({ systemPrompt, userPrompt, prefill = '' }) {
    // Gemma 4 has no separate system role — system is embedded in user turn.
    return (
      `<start_of_turn>user\n${systemPrompt}\n\n${userPrompt}<end_of_turn>\n` +
      `<start_of_turn>model\n${prefill}`
    );
  },

  buildToolResponse(observation: string) {
    return `\n<start_of_turn>tool\n<tool_response>\n${observation}\n</tool_response><end_of_turn>\n` +
      '<start_of_turn>model\n';
  },

  buildFollowUp(userPrompt: string) {
    // Leading <end_of_turn>: closes the model's previous turn — same missing-turn-close
    // gap measured on MacProfile (see its buildFollowUp comment); structurally identical
    // here. NOT re-measured on this profile (no CUDA on this machine) — verify on Linux.
    return `<end_of_turn>\n<start_of_turn>user\n${userPrompt}<end_of_turn>\n` +
      '<start_of_turn>model\n';
  },

  // Intent anchor token IDs — verified with debugTokenIds.ts on Gemma 4 E2B QAT.
  // "RE" (1357) excluded from REFACTOR: first subword of REST/REPO/RENAME too.
  // "Write/TEST/Test/test" excluded from WRITE_TEST: too noisy for ADD_FEATURE.
  intentAnchors: {
    //               BUG    Bug    bug
    BUGFIX:      [23173, 74379,  9618],
    //               Ref   REF    ref   Rewrite
    REFACTOR:    [ 7166, 43209,  1811, 107191],
    //               Explain  explain  EXPL
    EXPLAIN:     [155122,  70351, 215383],
    //               ADD    Add    add
    ADD_FEATURE: [20704,  3218,  1282],
    //               WRITE   write
    WRITE_TEST:  [59212,  5986],
    //               UNKNOWN   Unknown  unknown  UNK
    UNKNOWN:     [146233,  41387, 20774, 86957],
  },

  intentPrefill: 'Intent:\n',

  tagRegistryPath: path.join(__dirname, 'TagRegistry.gemma4.json'),

   // ── Hardware: RTX 4070 Ti SUPER 16 GiB VRAM, 32 GiB DDR5 RAM, /mnt/cache-llm 24 GiB XFS ──────
   // ISWA split: LLAMA_KV_SWA_OFFLOAD=1 keeps SWA KV hot in VRAM; global KV goes to RAM.
   // Global KV formula (q8_0): 5 layers × 2 (K+V) × 8 heads × 256 dim × ctx / 1024²
   //   → ≈20 MB per 1k tokens  →  64k = ~5.1 GiB RAM
   // Patch 102: compact SWA cache (n_swa sized, ~288 MiB fixed in VRAM).
   // Bench (2026-06-27): ISWA@64k = 136.9 t/s TG.
   kvTiers: {
     hot: {
       location: 'vram',
       // SWA: 1024 tokens × 25 layers × 2 × 8 × 256 × q8_0 → ~288 MiB (patch 102).
       // Fixed by Gemma4 architecture + our patch. VRAM headroom ~1.1 GiB at configured ctx.
       approxMib: 288,
     },
     warm: {
       location: 'ram',
       // 64k context: 64k × 20 KB/token ≈ 5.1 GiB system RAM. Default operating mode.
      contextSize: CTX,
      approxMib: 2560,
    },
    cold: {
      location: 'ssd',
      // Disk KV checkpoint cache (kvColdStore.ts). PC = 3 physical tiers: VRAM(hot) +
      // RAM(warm) + SSD(cold). /mnt/cache-llm is a 24 GiB XFS scratch — leave headroom.
      diskPath: '/mnt/cache-llm/subvocal-kv-cold',
      budgetBytes: 20 * 1024 * 1024 * 1024, // 20 GiB
    },
  },

  smallOpts: { contextSize: 4096, threads: 8, gpuLayers: 0 },
  // largeOpts reflects the warm tier default (64k in RAM, ISWA split active).
  // llama_init_from_model() directly — no CUDA warmup pass, so large ctx works without --no-warmup.
  largeOpts:  { contextSize: CTX, threads: 4, gpuLayers: 30, noKvOffload: true },
};

// ── Mac profile (Apple Silicon, Metal backend) ────────────────────────────────
//
// Small     : Gemma 4 E2B QAT (2.4 GiB Q4_K_XL), loaded via ModelGPU/Metal (no CPU addon on
//             Mac — smallBackend: 'gpu'). Off by default (cpuOff=true in utter.ts, same as
//             Linux); --cpu-model on enables it for intent routing (M11.2).
// GPU large : Gemma 4 12B "Unified" QAT (dense, encoder-free multimodal architecture),
//             downloaded to subvocal-mac/models/. NOT a quantization of Gemma4Profile's
//             26B-A4B MoE — a different point in the model family, chosen because the
//             26B-A4B doesn't fit a 16 GB unified-memory Mac (see doc/epics/EPIC-M2-*.md).
// EOT       : 106 '<turn|>' (also 1 '<eos>', 50 '<|tool_response>', 212 '</s>')
// Template  : confirmed from this GGUF's own tokenizer.chat_template (dumped via
//             llama.cpp's gguf-py, 2026-06-28) — '<|turn>role\n...<turn|>' markers,
//             NOT Gemma4Profile's '<start_of_turn>'/'<end_of_turn>' (those strings do not
//             appear anywhere in this model's actual chat template — would have been a
//             silent correctness bug to copy Gemma4Profile's template here unverified).

const MAC_MODEL_DIR = path.join(__dirname, '../../models');

// Same SUBVOCAL_CONTEXT_SIZE convention as Gemma4Profile. Default 128k: KV-per-token here is
// far cheaper than the 26B-A4B's (global layers use head_count_kv=1, "unified" KV per the
// model card), so 128k fits comfortably — see kvTiers comment below for the math.
const MAC_DEFAULT_CTX = 131072;
const MAC_CTX = (() => {
  const v = process.env.SUBVOCAL_CONTEXT_SIZE ? Number(process.env.SUBVOCAL_CONTEXT_SIZE) : MAC_DEFAULT_CTX;
  if (isNaN(v) || v < 4096) return MAC_DEFAULT_CTX;
  return Math.min(v, 262144);
})();

export const MacProfile: ModelProfile = {
  name: 'mac',

  // Real, downloaded checkpoint (same E2B QAT as Gemma4Profile's small). Loaded only when
  // --cpu-model on (cpuOff defaults true in utter.ts); on Mac it loads via ModelGPU/Metal —
  // see smallBackend below. M11.2: this is what makes incremental intent (mechanism B) pay.
  smallModelPath: `${MAC_MODEL_DIR}/unsloth-gemma-4-E2B-it-qat-GGUF/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf`,
  largeModelPath: `${MAC_MODEL_DIR}/unsloth-gemma-4-12B-it-qat-GGUF/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf`,

  vocabSize: 262144,
  eotTokenIds: [106, 1, 50, 212],  // <turn|>, <eos>, <|tool_response>, </s>
  // '<end_of_turn>' deliberately NOT included — confirmed absent from this model's vocab
  // and chat template (Gemma4Profile's Gemma-3-style markers do not apply here).
  stopStrings: ['<turn|>', '<|channel>'],

  buildPrompt({ systemPrompt, userPrompt, prefill = '' }) {
    // Mirrors the official chat_template's non-tool, non-thinking path: a system turn,
    // a user turn, then an opened model turn with the empty thinking channel the template
    // inserts whenever enable_thinking is false (confirmed in the template source, not
    // decorative — the model expects to see '<|channel>thought\n<channel|>' there).
    return (
      `<|turn>system\n${systemPrompt}<turn|>\n` +
      `<|turn>user\n${userPrompt}<turn|>\n` +
      `<|turn>model\n<|channel>thought\n<channel|>${prefill}`
    );
  },

  buildToolResponse(observation: string, toolName = 'tool') {
    // format_tool_response_block(tool_name, response) from the official chat_template. Two
    // bugs fixed here vs. the prior version (Epic M8 native-protocol work): (1) `toolName`
    // was hardcoded to the literal word "tool" instead of the actual tool that was called —
    // wrong per the macro, which interpolates the real name; (2) the scalar response value
    // needs the same `<|"|>` quote-wrapping format_argument() applies to strings, not raw
    // interpolation. The template injects the response inline within the model's own turn
    // (no separate 'tool'-role turn, unlike Gemma4Profile's Gemma-3-style assumption), then
    // closes and reopens a fresh model turn.
    return `<|tool_response>response:${toolName}{value:${nativeQuote(observation)}}<tool_response|><turn|>\n` +
      `<|turn>model\n<|channel>thought\n<channel|>`;
  },

  // Epic M8: renders this checkpoint's own native tool-declaration protocol, confirmed
  // byte-for-byte from the GGUF's tokenizer.chat_template (format_function_declaration /
  // format_parameters macros -- see doc/epics/EPIC-M8-grammar-constrained-tool-calls.md).
  // Only STRING-typed parameters are exercised by AgentLoop's tools today; other types render
  // their `type:` tag correctly but skip the template's enum/items/nested-properties extras,
  // since nothing defined in this codebase needs them yet.
  buildToolDeclarations(tools: readonly ToolDefinition[]): string {
    return tools.map(tool => {
      const { properties, required } = tool.parameters;
      const propsRendered = Object.keys(properties).sort().map(key => {
        const p = properties[key];
        const desc = p.description ? `description:${nativeQuote(p.description)},` : '';
        return `${key}:{${desc}type:${nativeQuote(p.type)}}`;
      }).join(',');
      const requiredRendered = required.map(nativeQuote).join(',');
      return (
        `<|tool>declaration:${tool.name}{description:${nativeQuote(tool.description)},` +
        `parameters:{properties:{${propsRendered}},required:[${requiredRendered}],type:${nativeQuote('OBJECT')}}}<tool|>`
      );
    }).join('');
  },

  buildFollowUp(userPrompt: string) {
    // Leading <turn|>\n closes the MODEL's previous turn: the decode loop stops AT the
    // model's <turn|> without committing it to the KV, so without this the KV reads
    // "...model output<|turn>user..." — off-template at every follow-up boundary. Measured
    // (2026-07-03 A/B, greedy 12B): the open-turn variant degenerated on turn 3 of a
    // memory-chain conversation (hallucinated user turns, <|channel> junk, no stop) while
    // the closed variant answered cleanly; no regression anywhere else. See
    // doc/research/turn-close-ab.md.
    return `<turn|>\n<|turn>user\n${userPrompt}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>`;
  },

  // E2B small-model prompt. Same '<|turn>' protocol as the 12B EXCEPT the generation prompt:
  // E2B's chat_template opens the model turn bare ('<|turn>model\n'), with no empty thought
  // channel — the two GGUFs' templates were diffed directly (2026-07-01) and that insertion
  // plus multimodal tool-response parts are the ONLY differences. Do not reuse the 12B
  // buildPrompt for E2B: the spurious '<|channel>thought\n<channel|>' is what caused the
  // '<channel|>' artifact seen in the dual-brain E2B spike (project_dual_brain_e2b_spike).
  buildSmallPrompt({ systemPrompt, userPrompt, prefill = '' }) {
    return (
      `<|turn>system\n${systemPrompt}<turn|>\n` +
      `<|turn>user\n${userPrompt}<turn|>\n` +
      `<|turn>model\n${prefill}`
    );
  },

  // Intent anchors: reused from Gemma4Profile — those were sniffed with debugTokenIds.ts
  // against this SAME Gemma 4 E2B QAT checkpoint (see Gemma4Profile.intentAnchors comment),
  // which is what routeIntent() runs on here too, so they apply as-is. The 12B caveat
  // remains for anything ideogram/TagRegistry-shaped driving the LARGE model: those values
  // are still unverified against the 12B — Epic M4 is deferred, re-run debugTokenIds/
  // intentBench against the 12B before trusting them there.
  intentAnchors: Gemma4Profile.intentAnchors,
  intentPrefill: Gemma4Profile.intentPrefill,
  tagRegistryPath: Gemma4Profile.tagRegistryPath,

  // ── Hardware: Apple Silicon unified memory — no separate VRAM/RAM pool, both KV tiers
  // and the model weights share the same 16 GB. Derived from this GGUF's own metadata
  // (dumped via llama.cpp's gguf-py, 2026-06-28): 48 layers total, 40 SWA (sliding window
  // 1024 tokens, 8 kv heads × 256 dim) + 8 global (head_count_kv=1 "unified" KV × 512 dim —
  // far cheaper per-token than Gemma4Profile's 26B-A4B global layers).
  //
  // Per-token KV bytes — MEASURED from the actual runtime cache, NOT the q8_0 formula that
  // an earlier version of this comment assumed. Two corrections from reading llama.cpp's own
  // `llama_kv_cache: size = …` logs (2026-07):
  //   (a) KV runs at f16 (2 B/elem), not q8_0 — kv-q8 was a net-negative result, left off.
  //   (b) The compact-SWA path (subvocal patch 102, llama-kv-cache-iswa.cpp) is what keeps the
  //       40 SWA layers to the 1024-token window; WITHOUT it they allocate the full context and
  //       the KV explodes to ~43 GiB at 128k. Patch 102 is applied to this Mac llama.cpp build
  //       (verify: the load log must say "creating SWA KV cache, size = 1536 cells", NOT
  //       "using full-size SWA cache"). These numbers assume it stays applied.
  //   SWA (hot):   40 layers × 1536-cell window × f16 → 480 MiB, FIXED (does not grow with ctx).
  //   Global (warm): 8 layers × 2(K+V) × 1 kv-head × 512 dim × 2 B (f16) = 16,384 B/token,
  //                  scales with context: 128k × 16,384 B = 2048 MiB.
  //
  // Memory budget @ 128k ctx (MEASURED process RSS, not estimated): 12B weights + hot SWA
  // (480 MiB) + compute buffers (~1.7 GiB) ≈ 8.2 GiB right after a forward, rising toward
  // ~10 GiB as the 2 GiB warm/global KV commits while the context fills. Plus macOS's own
  // ~3 GiB baseline (separate from this process) → ~13 GiB of 16 in use at full context, with
  // headroom. Without patch 102 the SWA cache alone would be ~40 GiB and 128k unusable — see
  // doc/epics/EPIC-M3-*.md.
  kvTiers: {
    hot: {
      location: 'vram',
      approxMib: 480, // measured: 40 SWA layers × 1536 cells × f16 (compact SWA, patch 102)
    },
    warm: {
      location: 'ram',
      contextSize: MAC_CTX,
      // 16384 B/token (f16 global KV) × MAC_CTX, MiB-rounded. Recompute if MAC_CTX changes.
      approxMib: Math.round((16384 * MAC_CTX) / (1024 * 1024)),
    },
    // Cold (SSD) KV checkpoint cache — the disk-resident tier (kvColdStore.ts, ds4 design).
    // Mac = 2 physical tiers: unified RAM (hot SWA + warm global share the ONE 16 GB pool,
    // there is no separate VRAM) + SSD (cold). This is a checkpoint cache (skip re-prefill),
    // NOT live KV paging — see doc/epics/EPIC-M3-kv-tiering-unified-memory.md for the full
    // three-axis analysis and why live SSD paging (Axis 3) is out of scope for the 12B.
    // Sized for the dual-brain trade (2026-07-01): dualBrainMaxCtx shrinks the RAM-resident
    // warm KV, which means more contexts get evicted/re-prefilled — the checkpoint cache on
    // SSD is the tier that absorbs that (this IS the "KV on SSD" in this design: whole
    // checkpoints, Axis 2; live per-token paging stays ruled out by bandwidth physics,
    // Axis 3 — see the epic). A realistic coding-session checkpoint is tens of MiB, so 4 GiB
    // still holds well over a hundred session states; eviction (hit-decayed score,
    // kvColdStore.ts) absorbs the rest. 2026-07-07: unified back to the same 4 GiB as
    // agentLoop.ts's generic default (was 8 GiB) — owner call, one number to reason about.
    cold: {
      location: 'ssd',
      diskPath: path.join(os.homedir(), '.cache', 'subvocal', 'kv-cold'),
      budgetBytes: 4 * 1024 * 1024 * 1024, // 4 GiB
    },
  },

  // gpuLayers: 999 (offload-everything convention, matches Qwen3Profile) rather than
  // copying Gemma4Profile's CUDA-specific gpuLayers: 30 split — unified memory means there's
  // no separate VRAM pool to size the split against.
  //
  // Small model on Metal too (no CPU addon on Mac): E2B weights are 2.4 GiB Q4 + a 4k-ctx
  // KV on top of the 12B's budget. MEASURED 2026-07-01 (live REPL, --cpu-model on, default
  // 128k ctx, M2 Pro 16 GB): both models fit but saturate the machine — 12.1 GiB wired
  // (Metal residency sets), 64 MB unused, compressor ~1.1 GiB, 2.2/3 GB swap in use with a
  // normal desktop session (browser etc.) open. Single-model keeps the full MAC_CTX default.
  //
  // 2026-07-07 (owner call): the pair no longer shares one window. The E2B is cheap
  // (drafter/fast-path generator) and gets a generous 32k so it can hold a long effective
  // conversation; the 12B is the expensive, rarely-invoked escalation path and gets a tight
  // 8k — the KV savings from shrinking the 12B's window pay for the E2B's bigger one. Was
  // a single shared dualBrainMaxCtx=16384; now asymmetric (dualBrainMaxCtx / e2bMaxCtx).
  smallBackend: 'gpu',
  smallOpts: { contextSize: 4096, threads: 4, gpuLayers: 999 },
  dualBrainMaxCtx: 8192,
  e2bMaxCtx: 32768,
  largeOpts: { contextSize: MAC_CTX, threads: 4, gpuLayers: 999, noKvOffload: true },

  // Per the model card's "Sampling Parameters" section: temperature=1.0, top_p=0.95,
  // top_k=64 standardized across all use cases (not just a low-quant loop-breaker, unlike
  // Gemma4Profile's comment on defaultTemperature).
  defaultTemperature: 1.0,
  defaultTopP: 0.95,
  defaultTopK: 64,
};

// ── Mac E2B generation profile (Epic M12 — dual-brain routing) ────────────────
//
// MacProfile variant for driving the E2B checkpoint AS THE GENERATOR (easy-edit routing,
// doc/epics/EPIC-M12-*.md). The two GGUFs' chat templates were diffed directly (2026-07-01,
// M11.2): identical '<|turn>' protocol INCLUDING all tool macros, except E2B never inserts the
// empty thought channel ('<|channel>thought\n<channel|>') when opening a model turn — feeding
// it anyway is what produced the '<channel|>' artifact and 0-parse tool calls in the original
// dual-brain spike. Every model-turn opener here is therefore bare '<|turn>model\n'.
export const MacE2BProfile: ModelProfile = {
  ...MacProfile,
  name: 'mac-e2b',

  // E2B is the generator in this profile; smallModelPath stays E2B too (irrelevant — no
  // second, even smaller brain below it).
  largeModelPath: MacProfile.smallModelPath,

  buildPrompt({ systemPrompt, userPrompt, prefill = '' }) {
    return (
      `<|turn>system\n${systemPrompt}<turn|>\n` +
      `<|turn>user\n${userPrompt}<turn|>\n` +
      `<|turn>model\n${prefill}`
    );
  },

  buildToolResponse(observation: string, toolName = 'tool') {
    // Same format_tool_response_block macro as the 12B (verified identical in E2B's template),
    // minus the thought-channel insertion when the fresh model turn opens.
    return `<|tool_response>response:${toolName}{value:${nativeQuote(observation)}}<tool_response|><turn|>\n` +
      `<|turn>model\n`;
  },

  buildFollowUp(userPrompt: string) {
    // Leading <turn|>\n: same missing-turn-close fix as MacProfile (see its comment).
    return `<turn|>\n<|turn>user\n${userPrompt}<turn|>\n<|turn>model\n`;
  },

  // largeOpts sized for the small model: E2B KV is cheap but its trained context is what it
  // is — 16k is plenty for the easy-edit band this profile exists for.
  largeOpts: { contextSize: 16384, threads: 4, gpuLayers: 999, noKvOffload: true },
};

// ── Qwen 2.5 Coder profile ─────────────────────────────────────────────────────
//
// CPU small : Qwen3.5-0.8B IQ2_XXS (vocab 151936) — intent routing only
// GPU large : Qwen2.5-Coder-7B Q4_K_M (~4.7 GB, tool-calling native)
// EOT       : 151645 '<|im_end|>'  (also 151643 '<|endoftext|>')
// Template  : ChatML — <|im_start|>role\ncontent<|im_end|>\n

const QWEN_MODEL_DIR = '/mnt/dati_cachy/LLM/lmstudio-community';

export const Qwen3Profile: ModelProfile = {
  name: 'qwen3',

  smallModelPath: `${QWEN_MODEL_DIR}/unsloth-Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-UD-IQ2_XXS.gguf`,
  largeModelPath: `${QWEN_MODEL_DIR}/Qwen2.5-Coder-7B-Instruct-GGUF/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf`,

  vocabSize: 151936,
  eotTokenIds: [151645, 151643, 151644],  // <|im_end|>, <|endoftext|>, <|im_start|>
  // Qwen at low quants may emit "assistant" / "user" as text instead of control tokens.
  // These stop-strings catch the degenerate ChatML looping pattern.
  stopStrings: ['<|im_end|>', '<|endoftext|>', '<|im_start|>'],

  buildPrompt({ systemPrompt, userPrompt, prefill = '' }) {
    return (
      `<|im_start|>system\n${systemPrompt}<|im_end|>\n` +
      `<|im_start|>user\n${userPrompt}<|im_end|>\n` +
      `<|im_start|>assistant\n${prefill}`
    );
  },

  buildToolResponse(observation: string) {
    return `\n<|im_start|>tool\n${observation}<|im_end|>\n` +
      '<|im_start|>assistant\n';
  },

  buildFollowUp(userPrompt: string) {
    // Leading <|im_end|>: closes the assistant's previous turn — same missing-turn-close
    // gap measured on MacProfile (see its buildFollowUp comment). Not re-measured on Qwen.
    return `<|im_end|>\n<|im_start|>user\n${userPrompt}<|im_end|>\n` +
      '<|im_start|>assistant\n';
  },

  // Intent anchor token IDs — verified against Qwen3.5-0.8B vocab (151k).
  // NOTE: these were not fully tuned (Qwen had BUGFIX bias before model switch).
  // Re-tune if reverting to Qwen: run intentBench with Qwen3Profile active.
  intentAnchors: {
    BUGFIX:      [   33, 13937,   257],   // B, Bug, bug  (approx — re-verify)
    REFACTOR:    [
      /* TODO: re-run debugTokenIds with Qwen model */
      19884,  // Ref (approx)
    ],
    EXPLAIN:     [14470, 95765],          // Explain, explain (approx)
    ADD_FEATURE: [ 2082,  2753],          // Add, add (approx)
    WRITE_TEST:  [19505,  3234],          // Write, write (approx)
    UNKNOWN:     [29952, 19216],          // UNKNOWN, unknown (approx)
  },

  intentPrefill: '',

  tagRegistryPath: path.join(__dirname, 'TagRegistry.json'),  // original Qwen registry

  // Qwen2.5 7B dense — 4.7 GB fits easily in 16 GB VRAM with 32k context.
  // KV tiers not tuned — using single VRAM tier at 32k ctx.
  kvTiers: {
    hot: { location: 'vram', approxMib: 0 },
    warm: { location: 'ram', contextSize: 32768, approxMib: 512 },
  },

  smallOpts: { contextSize: 4096, threads: 8,  gpuLayers: 0 },
  largeOpts:  { contextSize: 32768, threads: 4, gpuLayers: 999 },
};

// ── Active profile ────────────────────────────────────────────────────────────

const PROFILE_MAP: Record<string, ModelProfile> = {
  gemma4: Gemma4Profile,
  qwen3:  Qwen3Profile,
  mac:    MacProfile,
};

const profileName = (process.env.SUBVOCAL_MODEL_PROFILE ?? 'mac').toLowerCase();

export const activeProfile: ModelProfile = PROFILE_MAP[profileName] ?? MacProfile;

if (!(profileName in PROFILE_MAP)) {
  console.warn(
    `[modelProfile] Unknown SUBVOCAL_MODEL_PROFILE="${profileName}", falling back to mac`,
  );
}
