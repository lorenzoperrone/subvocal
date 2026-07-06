/**
 * tensorPayload.ts
 *
 * Substory 2.2 + 3.2: Tensor Payload assembler — fully in-process.
 *
 * Replaces the REST /tokenize bridge with direct N-API FFI calls via
 * subvocal-small (ModelCPU). All tokenization is now synchronous, zero-copy,
 * and in-process — no HTTP, no JSON, no serialization overhead.
 *
 * The resulting Int32Arrays are passed directly to subvocal-large via the
 * GPU FFI binder (ModelGPU.forward / decodeAppend) in the next pipeline stage.
 */

import { getSmallModel } from './smallModel.js';

// ── Types ─────────────────────────────────────────────────────────────────────

/**
 * The binary payload that travels from the Harness to subvocal-large (GPU).
 *
 * system_rules   - Pre-tokenised compressed DSL rules (e.g. [ERR]->[EDIT]).
 *                  Loaded once at boot, cached in RAM forever.
 * context_window - The tagged source code, serialised as token IDs.
 *                  Contains ideogram pointers from TagRegistry.
 * directives     - Intent + target info: [INTENT_TOKEN_ID, FILE_TOKEN_ID, …]
 */
export interface TensorPayload {
  system_rules: Int32Array;
  context_window: Int32Array;
  directives: Int32Array;
}

// ── Tokenisation helper ───────────────────────────────────────────────────────

/**
 * Tokenize a text fragment using subvocal-small in-process.
 *
 * Does NOT add BOS or special tokens — suitable for tokenizing code fragments,
 * file paths, intent words, and DSL rules that are assembled into larger sequences.
 * The GPU model's full context is constructed by the caller from these pieces.
 *
 * Requires initSmallModel() to have been called at harness boot.
 */
export function tokenize(text: string): Int32Array {
  return getSmallModel().tokenize(text, false, false);
}

// ── Compressed DSL system rules ───────────────────────────────────────────────

/**
 * Compressed DSL system prompt (machine-readable, not human-readable).
 * Pre-tokenised once at boot and cached in RAM for the session lifetime.
 */
const COMPRESSED_DSL_RULES = [
  '[EDIT]→diff',     // Output must be a unified diff
  '[TAG]→resolve',   // When a tag ID appears, resolve it via tagMap
  '[REASON]:0',      // Disable chain-of-thought verbosity in output
  '[FORMAT]:patch',  // Output format: patch/diff only
].join(' ');

let _cachedSystemRules: Int32Array | null = null;

/**
 * Load and cache the pre-tokenised system rules.
 * Called once at harness boot (or lazily on first assembleTensorPayload call).
 */
export function loadSystemRules(): Int32Array {
  if (_cachedSystemRules) return _cachedSystemRules;
  _cachedSystemRules = tokenize(COMPRESSED_DSL_RULES);
  console.log(`📐 System rules tokenised: ${_cachedSystemRules.length} tokens`);
  return _cachedSystemRules;
}

// ── Payload assembly ──────────────────────────────────────────────────────────

/**
 * Assemble a complete TensorPayload ready for subvocal-large.
 *
 * All operations are synchronous and in-process — no await needed.
 *
 * @param taggedCode     Source code string with ideogram tags already injected.
 * @param intentTokenId  The intent Token ID returned by the Intent Router.
 * @param filePathTokens Optional pre-tokenised file path for targeting.
 */
export function assembleTensorPayload(
  taggedCode: string,
  intentTokenId: number,
  filePathTokens?: Int32Array,
): TensorPayload {
  const contextTokens = tokenize(taggedCode);
  const systemRules = loadSystemRules();

  const directiveParts: number[] = [intentTokenId];
  if (filePathTokens) {
    for (const t of filePathTokens) directiveParts.push(t);
  }
  const directives = new Int32Array(directiveParts);

  console.log('📦 TensorPayload assembled:');
  console.log(`   system_rules   : ${systemRules.length} tokens`);
  console.log(`   context_window : ${contextTokens.length} tokens`);
  console.log(`   directives     : ${directives.length} tokens → intent=${intentTokenId}`);

  return {
    system_rules: systemRules,
    context_window: contextTokens,
    directives,
  };
}
