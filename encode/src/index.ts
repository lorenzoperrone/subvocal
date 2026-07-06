/**
 * preprocessor/index.ts
 *
 * Public API of the Subvocal Engine Preprocessor (Phase 1 & 2).
 *
 * Orchestrates the full ingestion + context assembly pipeline:
 *   1. routeIntent()      — subvocal-small zero-latency intent classification
 *   2. injectASTTags()    — Tree-sitter AST tagging with ideogram tokens
 *   3. assembleTensorPayload() — Produce binary TensorPayload for subvocal-large
 *
 * Usage:
 *   import { preprocess } from './preprocessor/index.js';
 *   const payload = await preprocess({ prompt, fileContent, filePath });
 */

export { ContinuousPrefillSession, type ContinuousPrefillConfig, type GravityContext } from './continuousPrefill.js';
export { routeIntent, routeIntentRegex, classifyIntentOnAuxSeq, classifyIntentWith, type Intent, type IntentResult } from './intentRouter.js';
export { IncrementalIntentClassifier, type IncrementalIntentOptions } from './incrementalIntent.js';
export { injectASTTags, computeBlockAnchors, detectLanguage, type ASTTagResult, type SupportedLanguage, type TagInjection } from './astTagger.js';
export { PathRegistry, stripRegistryTags, nextFreeAnchor, crcAnchorIndex, intentChar, intentLegend, UNCHANGED_FILE_CHAR, BASH_PASS_CHAR, BASH_FAIL_CHAR, NODE_TAG_RANGE, CRC_ANCHOR_RANGE, PATH_TOKEN_RANGE, STATIC_TOKEN_RANGE, type RegistryEntry } from './ideogramAllocator.js';
export { prewarmFile, FilePrewarmCache, type FilePrewarmResult, type PrewarmOptions } from './filePrewarm.js';
export { generateLogitMaskForASTNodes, type LogitMaskConfig } from './logitMaskGenerator.js';
export { assembleTensorPayload, tokenize, loadSystemRules, type TensorPayload } from './tensorPayload.js';
export { murmurHash3, buildLineHashSnapshot, computeLineDelta, type LineHashSnapshot } from './lineLevelCRC.js';
export { LineCRCCache } from './lineCRCCache.js';
export { KVCacheManager, type TokenRange, type TokenBoundaries, type InvalidationResult } from './kvCacheManager.js';
export { initSmallModel, freeSmallModel, getSmallModel, type SmallModelConfig } from './smallModel.js';
export { initLargeModel, freeLargeModel, getLargeModel, type LargeModelConfig } from './largeModel.js';
export { AgentLoop, type AgentLoopConfig, type AgentStep, type ParsedToolCall, type ReplayTurn } from './agentLoop.js';
export { activeProfile, MacProfile, MacE2BProfile, Gemma4Profile, type ModelProfile } from './modelProfile.js';
export { parseAssistantOutput, type ParsedAssistantOutput, type ToolDefinition, type ToolParameterSchema } from './toolParse.js';
export { detagCandidates, routeTask, shouldEscalate, type RouteDecision, type Brain } from './dualBrainRouter.js';
export { editASTNode, insertAfterNode, renameASTNode, deleteASTNode, type ASTEditInput, type ASTEditResult, type ASTRenameInput, type ASTDeleteInput, type SyntaxError } from './astEditor.js';
export { validateSyntax, validateSyntaxForPath, type SyntaxValidationResult } from './syntaxValidator.js';
export { StateSnapshotter, type ASTSnapshot } from './stateSnapshotter.js';
export { ShadowVFS, type VFSFile } from './shadowVFS.js';
export { createEphemeralWorktree, removeEphemeralWorktree, sandboxedCommand, airGapPrefix, mountTmpfs, unmountTmpfs, deadMansSwitch, computeDiff, type SandboxConfig, type DiffResult } from './osSandbox.js';
export { buildDependencyGraph, findAffectedFiles, type DepNode, type DependencyGraph } from './dependencyGraph.js';
export { parsePayloadBlocks, wrapPayload, PAYLOAD_START, PAYLOAD_END, type PayloadBlock, type ParsedPayloadText } from './bytePayload.js';
export { buildSandbox, checkCommand, SAFE_COMMANDS, type SandboxRule, type SandboxResult } from './tokenSandbox.js';
export { typescriptDiagnostics } from './lspShim.js';
export { compileCheck, type CompileResult } from './compileValidator.js';
export { distillObservation, heuristicDistill, type DistilledObservation } from './observationDistiller.js';
export { proxyFilter, buildProxyPrompt, type TensorProxyConfig } from './tensorProxy.js';
export {
	initSession,
	freeSession,
	runTask,
	type UtterConfig,
	type UtterSession,
	type UtterTask,
	type UtterResult,
	type UtterLogEntry,
} from './utter.js';
export { compressPrompt, type CompressedPrompt } from './promptCompressor.js';
export { ShadowFileSystem, type ShadowFSOptions } from './shadowFileSystem.js';
export { TensorArena, type ArenaEntry, type ArenaQueryHit } from './tensorArena.js';
export { retrieveContext, packRAGTokens, type RAGResult } from './textlessRAG.js';
export { KVColdStore, type CheckpointReason, type LoadResult as KVColdLoadResult } from './kvColdStore.js';
export {
  buildFoldedContext,
  foldSavings,
  FOLD_START_TOKEN_ID,
  FOLD_END_TOKEN_ID,
  type PrunedRange,
} from './foldTokens.js';
export { resolveMultiFileContext, type MultiFileBlock } from './multiFileContext.js';
export { distillContext, type DistillInput, type DistillResult } from './contextDistiller.js';
export { PipelineController, precomputeNextTurn } from './dualBrainPipeline.js';
export {
  computeResultDiffs,
  buildCompilerStats,
  formatCompilerStats,
  formatDiffSummaries,
  ghostTextHints,
  formatTaskSummaryPrompt,
  generateNLReport,
  type DiffSummary,
  type CompilerStats,
} from './outputEmitter.js';
export {
  extractMutation,
  computeStructuralDiff,
  renderUtterFrame,
  renderFull,
  renderDiff,
  createBuffer,
  type TuiFrame,
  type TuiBuffer,
  type FileMutation,
} from './tuiRenderer.js';

import { dlog } from './debugLog.js';
import { routeIntent, routeIntentRegex, type Intent, type IntentResult } from './intentRouter.js';
import { generateLogitMaskForASTNodes } from './logitMaskGenerator.js';
import { assembleTensorPayload, tokenize } from './tensorPayload.js';
import { LineCRCCache } from './lineCRCCache.js';
import { KVCacheManager, type InvalidationResult } from './kvCacheManager.js';
import { compressPrompt } from './promptCompressor.js';
import { prewarmFile, FilePrewarmCache } from './filePrewarm.js';

// ── Full pipeline entry point ─────────────────────────────────────────────────

export interface PreprocessInput {
  /** Raw user prompt (natural language). */
  prompt: string;
  /** Raw source code of the target file. */
  fileContent: string;
  /** File path (used for language detection and targeting). */
  filePath: string;
  /**
   * Substory 1.3: KVCacheManager for dirty-state check + targeted KV invalidation.
   * When provided, `payloadCache` enables early-return on cache hit (no re-tokenization).
   * Supersedes the legacy `crcCache` param.
   */
  kvCacheManager?: KVCacheManager;
  /**
   * Substory 1.3: Per-file payload cache. Required alongside `kvCacheManager` to
   * enable early-return when a file is unchanged since the last GPU prefill.
   * Managed by the caller (persist across preprocess() calls for the same session).
   */
  payloadCache?: Map<string, PreprocessResult>;
  /** @deprecated Use kvCacheManager instead. Kept for backward compat. */
  crcCache?: LineCRCCache;
  /**
   * Substory 17: Disable CPU small model. When true:
   *   - Intent classification uses regex fallback (fast, ~80% accuracy)
   *   - Tensor payload assembly is skipped (no CPU tokenizer)
   *   - Observation distillation falls back to truncation
   */
  cpuOff?: boolean;
  /** Substory 14: Resolve and include imported dependency files in context. */
  multiFile?: boolean;
  /**
   * Mechanism A (doc/research/predictive-prefill-while-typing.md): per-session cache of
   * AST-tagging + multi-file-context results, keyed by filePath. Pass the same
   * FilePrewarmCache instance the caller used with `.warm(filePath, fileContent)` ahead of
   * submit time (e.g. at file-open) to skip recomputing that file-dependent work here.
   * Managed by the caller, like `payloadCache` -- persist across preprocess() calls.
   */
  filePrewarmCache?: FilePrewarmCache;
  /**
   * Mechanism B (doc/research/predictive-prefill-while-typing.md): an intent already
   * classified incrementally while the user was still typing this prompt (via
   * IncrementalIntentClassifier, running on the idle CPU during the compose gap). When set,
   * preprocess() reuses it instead of re-running routeIntent()/routeIntentRegex() at submit
   * time. The caller is responsible for the staleness guard -- only pass a result that was
   * computed from this exact `prompt` text.
   */
  precomputedIntent?: IntentResult;
}

export interface PreprocessResult {
  /** The classified intent. Always one of routeIntent/routeIntentRegex's Intent union at
   *  runtime (both IntentResult.intent's real type) — typed precisely so intentChar() can
   *  take it directly (M15.6). */
  intent: Intent;
  /** Intent classification latency in ms. */
  intentLatencyMs: number;
  /** Source code with ideogram tags injected. */
  taggedCode: string;
  /** Map of TokenID -> AST node label for subvocal-large output resolution. */
  tagMap: Map<number, string>;
  /** Number of ideogram tags injected. */
  tagCount: number;
  /** Ordered list of tag injections with positional data for AST editing. */
  injections: import('./astTagger.js').TagInjection[];
  /** The assembled binary payload ready for subvocal-large. */
  payload: {
    system_rules: Int32Array;
    context_window: Int32Array;
    directives: Int32Array;
  };
  /** Logit mask (valid token IDs) for AST-Forced speculation. */
  logitMask?: Int32Array;
  /** Total pipeline latency in ms. */
  totalLatencyMs: number;
  /**
   * Substory 1.3: KV cache invalidation result.
   * Present when `kvCacheManager` is provided in PreprocessInput.
   * If `dirty === false`, this result was served from cache (no re-tokenization).
   * If `dirty === true`, call `model.kvCacheSeqRemove(0, r.start, r.end)` for each
   * range in `tokenRanges` before re-prefilling with the new payload.
   * Granularity: full-file (token boundaries not yet populated — see task.md 1.3).
   */
  invalidationResult?: InvalidationResult;
  /** Substory 16: Compressed request label (for prompt text). */
  compressedLabel?: string;
  /** Substory 14: Resolved dependency file blocks with tagged code. */
  multiFileBlocks?: import('./multiFileContext.js').MultiFileBlock[];
}

/**
 * Run the complete Phase 1 + Phase 2 pipeline.
 *
 * Phase 1 (Ingestion):
 *   - routeIntent: classify user prompt into a structured intent in ~15ms
 *
 * Phase 2 (Context Assembly):
 *   - injectASTTags: parse source + inject ideogram pointers
 *   - assembleTensorPayload: produce binary payload for GPU injection
 *
 * Substory 1.3: when `kvCacheManager` + `payloadCache` are provided, returns
 * immediately from cache if the file is unchanged (full-file granularity —
 * fine-grained boundary population is deferred until Shadow FS, substory 2.3).
 */
export async function preprocess(input: PreprocessInput): Promise<PreprocessResult> {
  const t0 = performance.now();

  // ── Phase 0 (Substory 1.3): Dirty-state check + early-return on cache hit ─
  let invalidationResult: InvalidationResult | undefined;

  if (input.kvCacheManager) {
    const inv = await input.kvCacheManager.invalidateIfDirty(input.filePath, input.fileContent);
    if (inv) {
      invalidationResult = inv;
      if (!inv.dirty) {
        // File unchanged since last GPU prefill — return cached payload without re-tokenising.
        const cached = input.payloadCache?.get(input.filePath);
        if (cached) {
          console.log(`🧊 CRC cache hit — ${input.filePath} unchanged, skipping pipeline`);
          return { ...cached, invalidationResult, totalLatencyMs: performance.now() - t0 };
        }
      } else {
        console.log(`🔥 CRC dirty — ${input.filePath}: ${inv.changedLineCount} line(s) changed`);
      }
    }
    // inv === null: first time we see this file — run full pipeline to establish baseline
  } else if (input.crcCache) {
    // Legacy path: log-only dirty check (no early-return, no payload cache)
    const delta = input.crcCache.checkDirty(input.filePath, input.fileContent);
    if (delta && !delta.isStructureChanged && delta.changedRanges.length === 0) {
      console.log(`🧊 CRC hit for ${input.filePath} (legacy path — no payload cache wired)`);
    } else if (delta) {
      console.log(`🔥 CRC dirty for ${input.filePath}: ${delta.changedRanges.length} range(s)`);
    }
    input.crcCache.set(input.filePath, input.fileContent);
  }

  // ── Phase 1: Intent classification ───────────────────────────────────────
  // Mechanism B (doc/research/predictive-prefill-while-typing.md): reuse an intent already
  // classified while the user was typing, if the caller vouched it matches this prompt.
  const intentResult = input.precomputedIntent
    ?? (input.cpuOff ? routeIntentRegex(input.prompt) : routeIntent(input.prompt));
  const intentSource = input.precomputedIntent ? ' [prefetched]' : input.cpuOff ? ' [regex]' : '';
  dlog(`🎯 Intent: ${intentResult.intent} (${intentResult.latencyMs.toFixed(1)}ms)${intentSource}`);

  // ── Phase 2a + Substory 14: AST tagging + multi-file context ────────────
  // Mechanism A (doc/research/predictive-prefill-while-typing.md): both depend only on
  // the file, not the prompt, so a caller may have already run this via
  // FilePrewarmCache.warm() ahead of submit time (e.g. at file-open). Reuse it if valid.
  const cachedPrewarm = input.filePrewarmCache?.get(input.filePath, input.fileContent, input.multiFile);
  const prewarmed = cachedPrewarm ?? prewarmFile(input.filePath, input.fileContent, { multiFile: input.multiFile });
  if (cachedPrewarm) {
    dlog(`⚡ File prewarm cache hit — ${input.filePath} (lang=${prewarmed.lang})`);
  } else {
    dlog(`🏷  AST tags injected: ${prewarmed.tagCount} (lang=${prewarmed.lang})`);
    if (prewarmed.multiFileBlocks && prewarmed.multiFileBlocks.length > 0) {
      dlog(`📎 Multi-file context: ${prewarmed.multiFileBlocks.length} dependency file(s)`);
    }
    input.filePrewarmCache?.set(prewarmed);
  }
  const { lang, taggedCode, tagMap, tagCount, injections, multiFileBlocks } = prewarmed;

  // ── Phase 2b-d: Tensor payload + directives (skip when CPU model off) ──
  let payload: PreprocessResult['payload'] = { system_rules: new Int32Array(0), context_window: new Int32Array(0), directives: new Int32Array(0) };
  let logitMask: Int32Array | undefined;
  let compressedLabel: string | undefined;

  if (!input.cpuOff) {
    const filePathTokens = tokenize(input.filePath);
    const intentTokens = tokenize(intentResult.intent);
    const intentTokenId = intentTokens[0];
    payload = assembleTensorPayload(taggedCode, intentTokenId, filePathTokens);

    logitMask = generateLogitMaskForASTNodes({
      allowedNodeTypes: new Set([
        'function_declaration', 'function_expression', 'class_declaration',
        'class_definition', 'method_definition', 'function_definition',
        'decorated_definition', 'for_statement', 'for_in_statement',
        'for_of_statement', 'while_statement', 'do_statement',
      ]),
      contextTokens: payload.context_window,
      tagMap,
    });

    const compressed = compressPrompt(input.prompt, input.filePath);
    if (compressed) {
      compressedLabel = compressed.label;
    }
  }

  const totalLatencyMs = performance.now() - t0;

  const result: PreprocessResult = {
    intent: intentResult.intent,
    intentLatencyMs: intentResult.latencyMs,
    taggedCode,
    tagMap,
    tagCount,
    injections,
    payload,
    logitMask,
    totalLatencyMs,
    invalidationResult,
    compressedLabel,
    multiFileBlocks,
  };

  // ── Substory 1.3: Establish / update per-file baseline ────────────────────
  // Boundaries not populated (full-file granularity). Fine-grained mapping
  // deferred to substory 2.3 when the Shadow FS provides stable tokenised
  // representations outside the tag-injected context window.
  if (input.kvCacheManager && !input.cpuOff) {
    await input.kvCacheManager.setFile(input.filePath, input.fileContent, payload.context_window);
  }
  if (input.payloadCache) {
    input.payloadCache.set(input.filePath, result);
  }

  return result;
}
