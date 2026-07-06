import { BaseModel, sampleGreedy, topK } from "./index.js";

export interface DraftResult {
  tokens: Int32Array;
  accepted: number;
  correctedToken: number | null;
}

export interface DecoderConfig {
  draftLength: number;
  /**
   * Acceptance threshold: draft token t_i is accepted if
   *   P_gpu(t_i | context) >= alpha * P_cpu(t_i | context)
   * Range [0, 1]. Lower = more permissive (faster but lower quality).
   */
  alpha: number;
  temperature: number;
  maxSteps: number;
  stopTokens: Set<number>;
  useTreeSpeculation: boolean;
  treeBranches: number;
  /**
   * Number of initial KV positions to protect as attention sinks.
   * These positions are NEVER evicted by kvCacheSeqRemove, preserving
   * the mathematical stability of the attention distribution across
   * long sessions. Typically the first 4-10 tokens of the system prompt.
   */
  attentionSinks: number;
  /** Optional per-step logit mask applied on the CPU draft model. */
  logitMask?: (logits: Float32Array, step: number) => void;
}

export interface DecodeStats {
  totalTokens: number;
  acceptedTokens: number;
  rejectedTokens: number;
  totalSteps: number;
  acceptanceRate: number;
}

export const DEFAULT_DECODER_CONFIG: DecoderConfig = {
  draftLength: 7,
  alpha: 0.85,
  temperature: 0.7,
  maxSteps: 2048,
  stopTokens: new Set(),
  useTreeSpeculation: false,
  treeBranches: 3,
  attentionSinks: 4,
};

function softmaxSingle(logits: Float32Array, tokenId: number): number {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > max) max = logits[i];
  }
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    sum += Math.exp(logits[i] - max);
  }
  return Math.exp(logits[tokenId] - max) / sum;
}

function concatInt32(a: Int32Array, b: Int32Array | readonly number[]): Int32Array {
  const out = new Int32Array(a.length + b.length);
  out.set(a);
  out.set(b instanceof Int32Array ? b : Int32Array.from(b), a.length);
  return out;
}

/**
 * Build a `logitMask` callback for AST-Forced Speculation from the preprocessor
 * result. Call this once after `preprocess()` and pass the returned function
 * into `DecoderConfig.logitMask`.
 *
 * How it works:
 *   - `tagMap` (from preprocess) maps token IDs → AST node labels.
 *     These are the ~93 structural ideogram tokens injected into the context.
 *   - `allowedIdeogramIds` (from preprocess logitMask) is the SUBSET of
 *     ideogram tokens whose node types are currently allowed.
 *   - Everything in tagMap that is NOT in allowedIdeogramIds is DISALLOWED:
 *     its logit is set to -Infinity before the CPU draft model samples.
 *
 * This prevents the CPU draft from generating structure-change tokens
 * (e.g., class_declaration anchors) when only function-body edits are allowed.
 * Regular vocabulary tokens (identifiers, keywords, punctuation) are never touched.
 *
 * The disallowed set is computed once at construction time; the callback itself
 * is a simple scalar loop over ≤93 token IDs — negligible overhead per draft step.
 */
export function buildASTLogitMask(
  tagMap: Map<number, string>,
  allowedIdeogramIds: Int32Array,
): (logits: Float32Array, step: number) => void {
  const allowedSet = new Set(allowedIdeogramIds);
  const disallowed = Int32Array.from([...tagMap.keys()].filter(id => !allowedSet.has(id)));

  if (disallowed.length === 0) {
    return () => {};
  }

  return (logits: Float32Array) => {
    for (let k = 0; k < disallowed.length; k++) {
      const id = disallowed[k];
      if (id >= 0 && id < logits.length) logits[id] = -Infinity;
    }
  };
}

export class SpeculativeDecoder {
  private cpuModel: BaseModel;
  private gpuModel: BaseModel;
  private cfg: DecoderConfig;

  constructor(cpuModel: BaseModel, gpuModel: BaseModel, config: Partial<DecoderConfig> = {}) {
    this.cpuModel = cpuModel;
    this.gpuModel = gpuModel;
    this.cfg = { ...DEFAULT_DECODER_CONFIG, ...config };
  }

  /**
   * Generate tokens using incremental speculative decoding:
   *
   * Round 0 (init):
   *   CPU: forward(contextTokens)   → KV filled, logits ready at last context pos
   *   GPU: forward(contextTokens)   → KV filled, logits ready at last context pos
   *   cpuPast = gpuPast = contextTokens.length
   *
   * Each round:
   *   1. CPU drafts K tokens via decodeAppend([t]) in a loop — O(K) per round
   *   2. GPU captures logits at current position (for predicting draft[0])
   *   3. GPU appends all K draft tokens with allLogits=true  — O(K) per round
   *   4. GPU getLogitsBatch([0..K-2]) gives logits predicting draft[1..K-1]
   *   5. Accept/reject: compare GPU vs CPU probs for each draft position
   *   6. Evict rejected tokens from both KV caches via kvCacheSeqRemove + resetNPast
   *   7. Append correction token (if any) via decodeAppend
   *   8. Repeat
   *
   * Result: GPU KV cache is NEVER rebuilt after round 0 — O(n) total GPU work.
   */
  generate(
    contextTokens: Int32Array,
    onToken?: (token: number) => void
  ): { tokens: number[]; stats: DecodeStats } {
    const cfg = this.cfg;
    const generated: number[] = [];
    let totalAccepted = 0;
    let totalRejected = 0;
    let totalSteps = 0;

    // ── Round 0: prefill both models once ─────────────────────────────────────
    let status = this.cpuModel.forward(contextTokens);
    if (status !== 0) throw new Error(`CPU forward() failed: ${status}`);
    status = this.gpuModel.forward(contextTokens);
    if (status !== 0) throw new Error(`GPU forward() failed: ${status}`);

    // Track n_past_ manually: avoids a C++ getter round-trip on every step.
    let cpuPast = contextTokens.length;
    let gpuPast = contextTokens.length;

    while (generated.length < cfg.maxSteps) {
      const K = cfg.draftLength;

      let draft: { tokens: number[]; cpuLogits: Float32Array[] };
      if (cfg.useTreeSpeculation) {
        draft = this.draftTree(K);
      } else {
        draft = this.draftLinear(K);
      }
      cpuPast += K;

      // GPU verify: capture logits for draft[0] BEFORE appending
      const gpuLogitD0 = new Float32Array(this.gpuModel.getLogitsFast());

      // Append all K draft tokens; request logits at every position so
      // getLogitsBatch can retrieve them for draft[1..K-1].
      const draftInt32 = Int32Array.from(draft.tokens);
      status = this.gpuModel.decodeAppend(draftInt32, /* allLogits */ true);
      if (status !== 0) {
        // GPU decode error: rebuild both KV caches from the committed sequence.
        this.resync(contextTokens, generated);
        cpuPast = contextTokens.length + generated.length;
        gpuPast = cpuPast;
        continue;
      }
      gpuPast += K;

      // Retrieve GPU logits for predicting draft[1..K-1] (batch indices 0..K-2)
      const gpuLogitsAll: Float32Array[] = [gpuLogitD0];
      if (K > 1) {
        const batchIndices = new Int32Array(K - 1);
        for (let i = 0; i < K - 1; i++) batchIndices[i] = i;
        const rest = this.gpuModel.getLogitsBatch(batchIndices);
        for (const l of rest) gpuLogitsAll.push(l);
      }

      const { accepted: M, corrected } = this.accept(draft.tokens, gpuLogitsAll, draft.cpuLogits);
      totalSteps++;

      // ── Commit accepted tokens ─────────────────────────────────────────────
      for (let i = 0; i < M; i++) {
        const tok = draft.tokens[i];
        generated.push(tok);
        totalAccepted++;
        if (onToken) onToken(tok);
        if (cfg.stopTokens.has(tok)) {
          this.evictTail(cpuPast - K + M, cpuPast, gpuPast - K + M, gpuPast);
          return { tokens: generated, stats: this.buildStats(generated.length, totalAccepted, totalRejected, totalSteps) };
        }
      }

      // ── Evict rejected draft tokens from both KV caches ───────────────────
      // Positions to remove: [cpuPast-K+M, cpuPast) — the M accepted survive.
      const cpuKeepEnd = cpuPast - K + M;
      const gpuKeepEnd = gpuPast - K + M;
      if (M < K) {
        this.evictTail(cpuKeepEnd, cpuPast, gpuKeepEnd, gpuPast);
        cpuPast = cpuKeepEnd;
        gpuPast = gpuKeepEnd;
      }

      // ── Commit correction token (sampled by GPU at first mismatch) ─────────
      if (corrected !== undefined) {
        this.cpuModel.decodeAppend(Int32Array.from([corrected]));
        this.gpuModel.decodeAppend(Int32Array.from([corrected]));
        cpuPast++;
        gpuPast++;
        generated.push(corrected);
        totalRejected++;
        if (onToken) onToken(corrected);
        if (cfg.stopTokens.has(corrected)) {
          return { tokens: generated, stats: this.buildStats(generated.length, totalAccepted, totalRejected, totalSteps) };
        }
      }

      // All K accepted + no correction → harvest the bonus token predicted by
      // the GPU at the last position of the draft batch.
      if (M === K && corrected === undefined) {
        const bonusLogits = this.gpuModel.getLogitsFast();
        const bonus = sampleGreedy(bonusLogits);
        this.cpuModel.decodeAppend(Int32Array.from([bonus]));
        this.gpuModel.decodeAppend(Int32Array.from([bonus]));
        cpuPast++;
        gpuPast++;
        generated.push(bonus);
        totalAccepted++;
        if (onToken) onToken(bonus);
        if (cfg.stopTokens.has(bonus)) {
          return { tokens: generated, stats: this.buildStats(generated.length, totalAccepted, totalRejected, totalSteps) };
        }
      }

      if (M === 0 && corrected === undefined) break;
    }

    return { tokens: generated, stats: this.buildStats(generated.length, totalAccepted, totalRejected, totalSteps) };
  }

  // ── Private helpers ──────────────────────────────────────────────────────────

  /**
   * Linear draft: CPU autoregressively generates K tokens using decodeAppend.
   * O(K) evaluations — the model already has the accepted context in KV from
   * the previous round. Logits are captured BEFORE each decodeAppend call:
   * they represent the distribution over the NEXT token from the current state.
   */
  private draftLinear(K: number): { tokens: number[]; cpuLogits: Float32Array[] } {
    const tokens: number[] = [];
    const cpuLogits: Float32Array[] = [];

    for (let i = 0; i < K; i++) {
      const logits = this.cpuModel.getLogitsFast();
      if (this.cfg.logitMask) this.cfg.logitMask(logits, i);
      cpuLogits.push(new Float32Array(logits));
      const token = sampleGreedy(logits);
      tokens.push(token);
      this.cpuModel.decodeAppend(Int32Array.from([token]));
    }

    return { tokens, cpuLogits };
  }

  /**
   * Tree draft: generate nBranches candidate continuations from the top-N first
   * tokens, score each branch by cumulative log-probability, and commit the best
   * branch to the CPU KV cache for GPU verification.
   *
   * Uses kvSave/kvRestore to branch and backtrack cheaply. Full tree-attention
   * (sending all branches to GPU in one batch) is a future optimization.
   */
  private draftTree(draftLen: number): { tokens: number[]; cpuLogits: Float32Array[] } {
    const nBranches = Math.min(this.cfg.treeBranches, 8);
    const firstLogits = this.cpuModel.getLogitsFast();
    const candidates = topK(firstLogits, nBranches);

    let bestBranch: { tokens: number[]; cpuLogits: Float32Array[]; logProb: number } | null = null;

    for (const candidate of candidates) {
      const snapshot = this.cpuModel.kvSave();
      const tokens: number[] = [candidate.id];
      const logitsCopy: Float32Array[] = [new Float32Array(firstLogits)];

      this.cpuModel.decodeAppend(Int32Array.from([candidate.id]));

      for (let i = 1; i < draftLen; i++) {
        const logits = this.cpuModel.getLogitsFast();
        if (this.cfg.logitMask) this.cfg.logitMask(logits, i);
        logitsCopy.push(new Float32Array(logits));
        const token = sampleGreedy(logits);
        tokens.push(token);
        this.cpuModel.decodeAppend(Int32Array.from([token]));
      }

      let logProb = 0;
      for (let i = 0; i < tokens.length; i++) {
        logProb += Math.log(softmaxSingle(logitsCopy[i], tokens[i]) + 1e-30);
      }

      if (bestBranch === null || logProb > bestBranch.logProb) {
        bestBranch = { tokens, cpuLogits: logitsCopy, logProb };
      }

      // Restore CPU KV to before this branch, then advance along the winner below.
      this.cpuModel.kvRestore(snapshot);
    }

    // Advance CPU KV along the winning branch (kvRestore left us at pre-branch state).
    const best = bestBranch!;
    for (const tok of best.tokens) {
      this.cpuModel.decodeAppend(Int32Array.from([tok]));
    }

    return { tokens: best.tokens, cpuLogits: best.cpuLogits };
  }

  /**
   * Acceptance criterion (standard speculative decoding):
   *   Accept draft[i] if P_gpu(draft[i] | ...) >= alpha * P_cpu(draft[i] | ...)
   * On first rejection, sample a correction from the GPU's distribution.
   */
  private accept(
    draft: number[],
    gpuLogits: Float32Array[],
    cpuLogits: Float32Array[]
  ): { accepted: number; corrected?: number } {
    const alpha = this.cfg.alpha;

    for (let i = 0; i < draft.length; i++) {
      const gpuProb = softmaxSingle(gpuLogits[i], draft[i]);
      const cpuProb = softmaxSingle(cpuLogits[i], draft[i]);

      if (gpuProb >= alpha * cpuProb) continue;

      const corrected = sampleGreedy(gpuLogits[i]);
      return { accepted: i, corrected };
    }

    return { accepted: draft.length };
  }

  /**
   * Evict a tail range from both CPU and GPU KV caches using
   * kvCacheSeqRemove + resetNPast. Attention sink positions are never touched.
   */
  private evictTail(
    cpuKeepEnd: number,
    cpuEvictEnd: number,
    gpuKeepEnd: number,
    gpuEvictEnd: number
  ): void {
    const sinks = this.cfg.attentionSinks;
    if (cpuEvictEnd > cpuKeepEnd) {
      this.cpuModel.kvCacheSeqRemove(0, Math.max(cpuKeepEnd, sinks), cpuEvictEnd);
      this.cpuModel.resetNPast(cpuKeepEnd);
    }
    if (gpuEvictEnd > gpuKeepEnd) {
      this.gpuModel.kvCacheSeqRemove(0, Math.max(gpuKeepEnd, sinks), gpuEvictEnd);
      this.gpuModel.resetNPast(gpuKeepEnd);
    }
  }

  /**
   * Emergency resync after a GPU decode error: rebuild both KV caches by
   * re-prefilling the full committed sequence. This should not happen in
   * normal operation — it is the fallback guard for hardware / context errors.
   */
  private resync(context: Int32Array, generated: readonly number[]): void {
    const full = concatInt32(context, generated);
    this.cpuModel.forward(full);
    this.gpuModel.forward(full);
  }

  private buildStats(
    totalTokens: number,
    acceptedTokens: number,
    rejectedTokens: number,
    totalSteps: number
  ): DecodeStats {
    return {
      totalTokens,
      acceptedTokens,
      rejectedTokens,
      totalSteps,
      acceptanceRate: totalTokens > 0 ? acceptedTokens / totalTokens : 0,
    };
  }
}
