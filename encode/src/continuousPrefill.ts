/**
 * continuousPrefill.ts
 *
 * Substory 1.5: Continuous Prefill & Latent Gravity
 *
 * Streams each keystroke into the model KV cache in real-time via
 * decodeAppend(), and rolls back the KV cache on Backspace via
 * kvCacheSeqRemove() + resetNPast().
 *
 * BPE safety: the session re-tokenizes the full accumulated text on each
 * change and reconciles against what is actually committed in the KV cache.
 * This avoids BPE divergence (where "ab"+"c" tokenizes differently than "abc")
 * at the cost of one tokenize() call per keystroke (~microseconds on CPU).
 *
 * Deferred (not yet implemented):
 *   - Async speculative thread for target resolution (needs separate model
 *     context — sharing one llama_context across threads corrupts KV state).
 */

import type { BaseModel } from '@subvocal/synapse';

// ── Types ──────────────────────────────────────────────────────────────────────

export interface ContinuousPrefillConfig {
  /**
   * Number of "attention sink" tokens at the start of the sequence that are
   * never evicted by the session's backspace handler.  Matches the speculative
   * decoder default.  Default: 4.
   */
  attentionSinks?: number;
  /**
   * Strength of the Latent Cursor Gravity logit bias applied when
   * applyGravityBias() is called.  Default: 3.0.
   */
  gravityStrength?: number;
}

export interface GravityContext {
  /** AST node type at the cursor (e.g. 'function_definition', 'class_declaration'). */
  nodeType: string;
  /** The full tagMap from injectASTTags() — maps tokenId → node label. */
  tagMap: Map<number, string>;
}

// ── ContinuousPrefillSession ───────────────────────────────────────────────────

/**
 * Manages a real-time typing session against a live model KV cache.
 *
 * Usage:
 *   // 1. Prefill context (system prompt + file content) via model.forward()
 *   const baseNPast = model.nPast; // capture position after prefill
 *   const session = new ContinuousPrefillSession(model, baseNPast);
 *
 *   // 2. Wire to UI events
 *   editor.on('keypress', (char) => session.onKeystroke(char));
 *   editor.on('backspace', () => session.onBackspace());
 *
 *   // 3. Read completion logits after each keystroke
 *   const topK = model.getLogitsTopK(20);
 */
export class ContinuousPrefillSession {
  private model: BaseModel;
  private readonly baseNPast: number;
  private readonly attentionSinks: number;
  private readonly gravityStrength: number;

  /** Full text typed by the user so far in this session. */
  private _text = '';

  /**
   * Token IDs currently committed to the KV cache past baseNPast.
   * Length = number of tokens currently in the KV cache for this session.
   */
  private committedTokens: number[] = [];

  /** Special-key registry: UI key name → tokenId to inject directly. */
  private specialKeys = new Map<string, number>();

  constructor(model: BaseModel, baseNPast: number, config?: ContinuousPrefillConfig) {
    this.model = model;
    this.baseNPast = baseNPast;
    this.attentionSinks = config?.attentionSinks ?? 4;
    this.gravityStrength = config?.gravityStrength ?? 3.0;
  }

  // ── Accessors ──────────────────────────────────────────────────────────────

  /** Current accumulated text typed by the user. */
  get text(): string { return this._text; }

  /**
   * Current KV position = baseNPast + committed token count.
   * Use this to resume decodeAppend() after an external forward() call.
   */
  get nPast(): number { return this.baseNPast + this.committedTokens.length; }

  /** Number of tokens currently committed to the KV cache by this session. */
  get committedCount(): number { return this.committedTokens.length; }

  // ── Keystroke API ──────────────────────────────────────────────────────────

  /**
   * Feed a printable character (or multi-byte grapheme) into the model.
   * Internally re-tokenizes the accumulated text and commits only the new tokens
   * appended since the last call.
   */
  onKeystroke(char: string): void {
    this._text += char;
    this._reconcile();
  }

  /**
   * Remove the last grapheme from the accumulated text and roll back the KV
   * cache to match.  Attention sinks (first `attentionSinks` positions of the
   * full sequence) are never touched — they anchor attention stability.
   */
  onBackspace(): void {
    if (this._text.length === 0) return;
    // Remove last Unicode grapheme cluster (handles multi-byte chars correctly)
    const segs = [...new Intl.Segmenter().segment(this._text)];
    if (segs.length === 0) return;
    this._text = this._text.slice(0, segs[segs.length - 1]!.index);
    this._reconcile();
  }

  // ── Special-key hardwiring ─────────────────────────────────────────────────

  /**
   * Map a UI key name (e.g. 'Tab', 'Enter', 'F1') to a token ID that gets
   * injected directly into the KV cache without going through the tokenizer.
   * Useful for shortcutting known control tokens or intent anchors.
   */
  registerSpecialKey(key: string, tokenId: number): void {
    this.specialKeys.set(key, tokenId);
  }

  /**
   * Inject the token registered for `key` directly into the model's KV cache.
   * No-op if the key is not registered.
   */
  onSpecialKey(key: string): void {
    const tokenId = this.specialKeys.get(key);
    if (tokenId === undefined) return;
    this.model.decodeAppend(new Int32Array([tokenId]));
    this.committedTokens.push(tokenId);
    // Special tokens are NOT added to _text — they live only in the KV cache.
  }

  // ── Latent Cursor Gravity ──────────────────────────────────────────────────

  /**
   * Apply a positive logit bias to all tokens whose AST label starts with
   * `ctx.nodeType`, pulling the model toward continuing within the current
   * syntactic block.  Call after each keystroke for cursor-aware completion.
   *
   * The bias is persistent — call clearGravityBias() when the cursor moves to
   * a different node type.
   */
  applyGravityBias(ctx: GravityContext): void {
    const biases: Array<{ tokenId: number; bias: number }> = [];
    for (const [tokenId, label] of ctx.tagMap) {
      if (label === ctx.nodeType || label.startsWith(`${ctx.nodeType}:`)) {
        biases.push({ tokenId, bias: this.gravityStrength });
      }
    }
    if (biases.length > 0) {
      this.model.applyLogitBias(biases);
    }
  }

  /** Remove all pending logit biases set by applyGravityBias(). */
  clearGravityBias(): void {
    this.model.clearLogitBiases();
  }

  // ── Session reset ──────────────────────────────────────────────────────────

  /**
   * Clear all text and KV state added by this session, restoring the model to
   * `baseNPast`.  Does NOT free the model or touch the pre-session context.
   */
  reset(): void {
    if (this.committedTokens.length > 0) {
      this.model.kvCacheSeqRemove(0, this.baseNPast, -1);
      this.model.resetNPast(this.baseNPast);
    }
    this._text = '';
    this.committedTokens = [];
  }

  // ── BPE-safe reconciliation ────────────────────────────────────────────────

  /**
   * Re-tokenize the full current text and reconcile with what is in the KV cache.
   *
   * Algorithm (O(n) in text length):
   *  1. Tokenize currentText → newTokenIds
   *  2. Find longest common prefix with committedTokens
   *  3. Roll back KV to commonLen (kvCacheSeqRemove + resetNPast)
   *  4. Append newTokenIds[commonLen..] via decodeAppend
   *
   * This is correct for all BPE edge cases: if typing 'c' after 'ab' causes
   * 'abc' to tokenize differently than ['ab','c'], the reconcile detects the
   * divergence at the first changed token and re-decodes from there.
   */
  private _reconcile(): void {
    const newIds = Array.from(this.model.tokenize(this._text, false, false));
    const oldIds = this.committedTokens;

    // Find where the two token sequences first diverge
    let commonLen = 0;
    const minLen = Math.min(newIds.length, oldIds.length);
    while (commonLen < minLen && newIds[commonLen] === oldIds[commonLen]) {
      commonLen++;
    }

    // Roll back if KV has tokens beyond the common prefix
    if (commonLen < oldIds.length) {
      const rollbackTo = this.baseNPast + commonLen;
      // Attention sinks protect the first N positions of the GLOBAL sequence.
      // rollbackTo is always >= baseNPast, so we only cap against
      // min(attentionSinks, baseNPast) to avoid protecting session-owned tokens
      // when baseNPast < attentionSinks (e.g. in tests; never happens in practice).
      const safeRollback = Math.max(rollbackTo, Math.min(this.attentionSinks, this.baseNPast));
      this.model.kvCacheSeqRemove(0, safeRollback, -1);
      this.model.resetNPast(safeRollback);
    }

    // Commit new tokens that extend beyond the common prefix
    if (commonLen < newIds.length) {
      const toAppend = new Int32Array(newIds.slice(commonLen));
      this.model.decodeAppend(toAppend);
    }

    this.committedTokens = newIds;
  }
}
