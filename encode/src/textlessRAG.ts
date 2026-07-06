/**
 * textlessRAG.ts
 *
 * Substory 2.3: Textless Retrieval — matrix-multiply query embedding against
 * the TensorArena to find the most semantically relevant functions without
 * relying on keyword matching or BM25.
 *
 * "Textless" = retrieval happens entirely in hidden-state space (Float32Array
 * dot products), bypassing the vocabulary/token layer. The retrieved function
 * snippets are then injected as pre-tokenized Int32Array blocks into the GPU
 * context window, skipping re-tokenization.
 *
 * Integration point for preprocess():
 *   const hits = retrieveContext(arena, promptText, 5);
 *   // Prepend hits[i].tokens to the context_window payload before GPU prefill.
 */

import type { TensorArena } from './tensorArena.js';
import type { ArenaQueryHit } from './tensorArena.js';

// ── Types ─────────────────────────────────────────────────────────────────────

export interface RAGResult {
  /** Composite key: "<absolutePath>:<funcName>" */
  key: string;
  /** Cosine similarity score [0, 1]. */
  score: number;
  /** Absolute path of the source file. */
  filePath: string;
  /** Function/class name. */
  funcName: string;
  /** AST node type. */
  nodeType: string;
  /** Source snippet (capped at MAX_SNIPPET_CHARS). */
  snippet: string;
  /** Pre-tokenized form of `snippet` — ready for GPU prefill without re-tokenization. */
  tokens: Int32Array;
}

// ── Main API ──────────────────────────────────────────────────────────────────

/**
 * Retrieve the `topN` most relevant codebase functions for a given query text.
 *
 * Steps:
 *   1. Embed `queryText` using the arena's dedicated embedding model.
 *   2. Compute cosine similarity against every indexed function.
 *   3. Return the top-N hits sorted by descending score.
 *
 * Returns [] if the arena is empty or the model is not initialized.
 *
 * @param arena      Populated TensorArena (must have called init() + indexFile()).
 * @param queryText  The user prompt or a concise summary of the current task.
 * @param topN       Number of results to return. Default: 5.
 * @param minScore   Minimum cosine similarity threshold. Default: 0.0 (no filter).
 */
export function retrieveContext(
  arena: TensorArena,
  queryText: string,
  topN = 5,
  minScore = 0.0,
): RAGResult[] {
  if (!arena.isReady || arena.size === 0) return [];

  const queryState = arena.getQueryEmbedding(queryText);
  if (queryState.length === 0) return [];

  const hits: ArenaQueryHit[] = arena.query(queryState, topN);

  return hits
    .filter((h) => h.score >= minScore)
    .map((h) => ({
      key: h.key,
      score: h.score,
      filePath: h.entry.filePath,
      funcName: h.entry.funcName,
      nodeType: h.entry.nodeType,
      snippet: h.entry.snippet,
      tokens: h.entry.tokens,
    }));
}

/**
 * Concatenate the pre-tokenized RAG results into a single Int32Array suitable
 * for prepending to the GPU context window payload.
 *
 * Each hit is separated by a single space token (ID 235 in Gemma 4 vocab)
 * to give the model a clean boundary between retrieved snippets.
 *
 * @param results  Output of retrieveContext().
 * @param sepToken Token ID used as separator between snippets. Default: 235 (space).
 */
export function packRAGTokens(results: RAGResult[], sepToken = 235): Int32Array {
  if (results.length === 0) return new Int32Array(0);

  const parts: Int32Array[] = [];
  for (let i = 0; i < results.length; i++) {
    if (i > 0) parts.push(new Int32Array([sepToken]));
    parts.push(results[i].tokens);
  }

  const totalLen = parts.reduce((s, p) => s + p.length, 0);
  const out = new Int32Array(totalLen);
  let offset = 0;
  for (const p of parts) {
    out.set(p, offset);
    offset += p.length;
  }
  return out;
}
