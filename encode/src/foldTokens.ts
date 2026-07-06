/**
 * foldTokens.ts — Substory 2.4: Breadcrumb-based AST fold encoding.
 *
 * Instead of dropping pruned tokens (which would leave a positional gap causing
 * OOD attention), each pruned range is replaced with a 3-token breadcrumb:
 *   [FOLD_START_TOKEN_ID] [tagTokenId] [FOLD_END_TOKEN_ID]
 *
 * Positions remain sequential. The model sees a structural marker ("a block
 * was here, specifically the one tagged with this ideogram") without ever
 * encountering an attention-breaking positional gap.
 *
 * The FOLD_START and FOLD_END characters are reserved from the active profile's
 * TagRegistry (indices 64 and 65) — outside the default maxTags=64 range used
 * by astTagger, so they never collide with AST-tag injections.
 *   ⊂ (U+2282, idx=64) — opening fold marker
 *   ⊃ (U+2283, idx=65) — closing fold marker
 *
 * M15.5: three op markers reuse the SAME reserved gap (indices 66-68, between the fold
 * markers and ⊕ at 69) — verified single-token, never claimed by astTagger's default
 * maxTags=64 node-tag allocation. Chosen from registry entries that already sit there
 * (⊆/⊇/⊔) rather than the story's originally-sketched ⇋/⊘/⊚, which aren't in the
 * registry at all — same lesson as M15.6's bash markers: pick reserved slots, don't
 * hand-pick "meaningful-looking" symbols without checking they're actually available.
 *   ⊆ (U+2286, idx=66) — rename operator
 *   ⊇ (U+2287, idx=67) — delete operator
 *   ⊔ (U+2294, idx=68) — wrap operator (reserved for a future story; unused today)
 */

import { readFileSync } from 'fs';
import { activeProfile } from './modelProfile.js';

interface TagRegistryEntry { char: string; tokenId: number; }

function loadFoldTokenIds(): { start: number; end: number; insert: number; rename: number; del: number; wrap: number } {
  try {
    const raw = readFileSync(activeProfile.tagRegistryPath, 'utf-8');
    const registry: TagRegistryEntry[] = JSON.parse(raw);
    const start  = registry.find(e => e.char === '\u2282')?.tokenId;  // ⊂
    const end    = registry.find(e => e.char === '\u2283')?.tokenId;  // ⊃
    const insert = registry.find(e => e.char === '\u2295')?.tokenId;  // ⊕
    const rename = registry.find(e => e.char === '\u2286')?.tokenId;  // ⊆
    const del    = registry.find(e => e.char === '\u2287')?.tokenId;  // ⊇
    const wrap   = registry.find(e => e.char === '\u2294')?.tokenId;  // ⊔
    if (start && end && insert && rename && del && wrap) return { start, end, insert, rename, del, wrap };
  } catch { /* fall through to defaults */ }
  return { start: 245272, end: 252107, insert: 245337, rename: 0, del: 0, wrap: 0 };
}

const _foldIds = loadFoldTokenIds();

/** Token ID for the opening fold marker ⊂ (U+2282). Loaded from active TagRegistry. */
export const FOLD_START_TOKEN_ID = _foldIds.start;

/** Token ID for the closing fold marker ⊃ (U+2283). Loaded from active TagRegistry. */
export const FOLD_END_TOKEN_ID = _foldIds.end;

/** Token ID for the insert marker ⊕ (U+2295). Loaded from active TagRegistry. */
export const INSERT_MARKER_TOKEN_ID = _foldIds.insert;

/** M15.5: token ID for the rename operator ⊆ (U+2286). */
export const RENAME_OP_TOKEN_ID = _foldIds.rename;

/** M15.5: token ID for the delete operator ⊇ (U+2287). */
export const DELETE_OP_TOKEN_ID = _foldIds.del;

/** M15.5: token ID for the wrap operator ⊔ (U+2294). Reserved, not yet consumed by the
 *  steering state machine (wrap needs a payload-template convention not yet designed). */
export const WRAP_OP_TOKEN_ID = _foldIds.wrap;

/**
 * A contiguous range of tokens to replace with a fold breadcrumb.
 * Ranges must be sorted ascending by `startTokenIdx` and non-overlapping.
 */
export interface PrunedRange {
  /** Zero-based index into the token array where the pruned range starts (inclusive). */
  startTokenIdx: number;
  /** Number of tokens to replace (must be ≥ 1). */
  length: number;
  /**
   * Token ID of the ideogram tag for this node (as injected by astTagger and
   * recorded in tagMap). This becomes the middle token of the breadcrumb:
   *   [FOLD_START] [tagTokenId] [FOLD_END]
   */
  tagTokenId: number;
}

/**
 * Replace pruned AST ranges with 3-token fold breadcrumbs.
 *
 * Each pruned range is replaced by:
 *   [FOLD_START_TOKEN_ID] [range.tagTokenId] [FOLD_END_TOKEN_ID]
 *
 * The output array has sequential positions (no gaps), allowing safe use with
 * `decodeAppend()` without triggering OOD attention due to position jumps.
 *
 * @param tokens  Full token sequence (e.g., the tagged context_window).
 * @param ranges  Sorted, non-overlapping ranges to prune. May be empty.
 * @returns       New token array with breadcrumbs substituted; same reference if ranges is empty.
 */
export function buildFoldedContext(tokens: Int32Array, ranges: PrunedRange[]): Int32Array {
  if (ranges.length === 0) return tokens;

  for (let i = 1; i < ranges.length; i++) {
    const prev = ranges[i - 1];
    const curr = ranges[i];
    if (curr.startTokenIdx < prev.startTokenIdx + prev.length) {
      throw new Error(
        `PrunedRange[${i}] (start=${curr.startTokenIdx}) overlaps with range[${i - 1}] (start=${prev.startTokenIdx}, len=${prev.length})`,
      );
    }
    if (curr.startTokenIdx < 0 || curr.length < 1) {
      throw new Error(`PrunedRange[${i}] has invalid startTokenIdx or length`);
    }
  }

  // Validate first range
  if (ranges[0].startTokenIdx < 0 || ranges[0].length < 1) {
    throw new Error('PrunedRange[0] has invalid startTokenIdx or length');
  }

  // Output length: subtract pruned tokens, add 3 breadcrumb tokens per range
  let outLen = tokens.length;
  for (const r of ranges) {
    outLen -= r.length;
    outLen += 3;
  }

  const out = new Int32Array(outLen);
  let outIdx = 0;
  let inIdx = 0;

  for (const range of ranges) {
    // Validate range bounds
    if (range.startTokenIdx + range.length > tokens.length) {
      throw new Error(
        `PrunedRange [${range.startTokenIdx}, ${range.startTokenIdx + range.length}) exceeds token array length ${tokens.length}`,
      );
    }

    // Copy tokens before this range
    while (inIdx < range.startTokenIdx) {
      out[outIdx++] = tokens[inIdx++];
    }

    // Insert breadcrumb
    out[outIdx++] = FOLD_START_TOKEN_ID;
    out[outIdx++] = range.tagTokenId;
    out[outIdx++] = FOLD_END_TOKEN_ID;

    // Skip the pruned range
    inIdx += range.length;
  }

  // Copy remaining tokens after all ranges
  while (inIdx < tokens.length) {
    out[outIdx++] = tokens[inIdx++];
  }

  return out;
}

/**
 * Compute token savings from folding: how many tokens are eliminated.
 * Each range of length L becomes 3 tokens, so savings = L - 3 (can be negative for L < 3).
 */
export function foldSavings(ranges: PrunedRange[]): number {
  let saved = 0;
  for (const r of ranges) saved += r.length - 3;
  return saved;
}
