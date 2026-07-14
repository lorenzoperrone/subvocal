import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const here = dirname(fileURLToPath(import.meta.url));

const CPU_PATH = resolve(here, "..", "build-cpu", "Release", "subvocal_ffi_cpu.node");
// This Mac port's GPU-accelerated backend is Metal, built into build-metal/ (addon name
// subvocal_ffi_metal per CMakeLists.txt's SUBVOCAL_BACKEND=metal branch) -- there is no CUDA
// backend in this repo at all. "build-gpu/subvocal_ffi_gpu.node" never exists here; it was a
// leftover path from the Linux/CUDA port this was copied from. ModelGPU/getGpuBinding() still
// mean "the GPU-accelerated backend" generically -- on this platform that's always Metal.
const GPU_PATH = process.platform === "darwin"
	? resolve(here, "..", "build-metal", "subvocal_ffi_metal.node")
	: resolve(here, "..", "build-gpu", "Release", "subvocal_ffi_gpu.node");

interface NativeModel {
	tokenize(text: string, addSpecial?: boolean, parseSpecial?: boolean): Int32Array;
	forward(tokens: Int32Array): number;
	forwardAsync(tokens: Int32Array): Promise<number>;
	decodeAppend(tokens: Int32Array, allLogits?: boolean): number;
	decodeAppendAsync(tokens: Int32Array, allLogits?: boolean): Promise<number>;
	// M11.3 variant (b): decode onto an explicit seq/position instead of always seq 0 /
	// n_past_. Requires ModelOptions.auxSeq at construction.
	decodeAppendSeq(tokens: Int32Array, seqId: number, pos: number, allLogits?: boolean): Promise<number>;
	resetNPast(n: number): void;
	forwardPartial(tokens: Int32Array, layerLimit: number): number;
	forwardEmbedding?(embd: Float32Array, tokens?: Int32Array): number;
	getLogits(): Float32Array;
	getLogitsFast(): Float32Array;
	getLogitsUnsafe(): Float32Array;
	getLogitsBatch(indices: Int32Array): Float32Array[];
	getHiddenState(): Float32Array;
	getHiddenStateLayer(layer: number): Float32Array;
	getKVState(): Uint8Array;
	setKVState(state: Uint8Array): number;
	detokenize(tokens: Int32Array): string;
	vocabSize(): number;
	contextSize(): number;
	embeddingSize(): number;
	layerCount(): number;
	// V6.1: logit bias / steering
	applyLogitBias(biases: Array<{ tokenId: number; bias: number }>): void;
	setPersistentBiases(biases: Array<{ tokenId: number; bias: number }>): void;
	clearLogitBiases(): void;
	setSteeringVector(vector: Float32Array, strength: number): void;
	clearSteeringVector(): void;
	// V6.2: KV cache manipulation
	kvCacheSeqRemove(seqId: number, p0: number, p1: number): void;
	kvCacheSeqCopy(srcSeq: number, dstSeq: number, p0: number, p1: number): void;
	kvCacheSeqKeep(seqId: number): void;
	kvCacheSeqShift(seqId: number, p0: number, p1: number, delta: number): void;
	kvCacheClear(): void;
	kvCacheFork(seqId: number): number;
	kvCacheEvict(keepLastN: number, seqId?: number): void;
	kvCacheView(): { maxSize: number; used: number; cells: Array<{ pos: number; seqId: number; hasValue: boolean }> };
	// V6.3: AST-aware logit masking
	setASTTokenMask(allowed: Int32Array): void;
	clearASTMask(): void;
	// V6.4: sparse logits — extract only top-K token ID + logit pairs (~80 bytes vs ~600KB)
	// Layout: [id0..idK-1, logitBits0..logitBitsK-1] where logitBits are IEEE 754 float
	// bits packed as int32. Read floats: new Float32Array(result.buffer, k*4, k)
	getLogitsTopK(k?: number): Int32Array;
	// applyMaskSIMD: scalar loop masking disallowed token IDs in the logit buffer.
	// (Name is kept for API compat; SIMD will be wired in when AVX2 is enabled.)
	applyMaskSIMD(disallowedTokenIds: Int32Array): void;
	forkContext(): NativeModel;
	// V9: token-to-pointer routing — action token callbacks
	registerActionToken(tokenId: number, callback: (tokenId: number) => void): void;
	removeActionToken(tokenId: number): void;
	handleActionToken(tokenId: number): boolean;
	// Suffix-tree (prompt-lookup) drafter — both backends. See
	// doc/research/suffix-tree-speculative-decoding.md: free drafter (no model, no GPU
	// cost), net-positive on edit/refactor workloads (1.2-1.3x), negative on from-scratch
	// generation. extend() appends only NEW tokens (not cumulative re-passing of history).
	suffixTreeExtend(tokens: Int32Array): void;
	suffixTreeClear(): void;
	suffixTreeSpeculate(
		context: Int32Array,
		maxSpecTokens?: number,
		minTokenProb?: number,
		minMatchCount?: number,
		minMatchLen?: number,
	): Int32Array;
	suffixTreeTokenCount(): number;
	suffixTreeMaxDepth(): number;
	// GPU-only: drafter-agnostic batched verify (only touches the target context, no
	// draft_ctx_ needed -- verifies suffix-tree drafts directly, see binding_gpu.cpp).
	// Returns Int32Array [acceptedCount, ...emittedTokenIds].
	mtpVerifyBatch?(draftTokens: Int32Array): Int32Array;
	free(): void;
}
interface NativeBinding {
	Model: new (
		path: string,
		opts?: {
			contextSize?: number;
			threads?: number;
			threadsBatch?: number;
			gpuLayers?: number;
			embeddings?: boolean;
			captureLayerHidden?: boolean;
			noKvOffload?: boolean;
			kvType?: "f16" | "q8_0" | "q5_1" | "q4_0";
			loraPath?: string;
			loraScale?: number;
			nUbatch?: number;
		},
	) => NativeModel;
	/** Substory 1.4: AVX2 byte-scan. Returns offsets of needle in haystack. */
	scanBytes(haystack: Uint8Array, needle: Uint8Array): Int32Array;
	/** Substory 2.4: Off-heap Int32Array backed by malloc. GC calls free() via finalizer. */
	createPinnedBuffer(sizeInInts: number): Int32Array;
	/** libgit2: open a repository for subsequent git ops. */
	gitInit(repoPath: string, opts?: { authorName?: string; authorEmail?: string }): boolean;
	/** libgit2: stage all changes (git add -A). */
	gitStageAll(): void;
	/** libgit2: commit staged changes. Returns commit SHA. */
	gitCommit(message: string, authorName?: string, authorEmail?: string): string;
	/** libgit2: diff of staged changes vs working tree. */
	gitDiffStaged(): string;
	/** libgit2: diff of working tree vs HEAD. */
	gitDiffWorking(): string;
	/** libgit2: release the git repository handle. */
	gitFree(): void;
}

let _cpuBinding: NativeBinding | undefined;
let _gpuBinding: NativeBinding | undefined;
function getCpuBinding(): NativeBinding {
	if (!_cpuBinding) _cpuBinding = require(CPU_PATH) as NativeBinding;
	return _cpuBinding;
}
function getGpuBinding(): NativeBinding {
	if (!_gpuBinding) _gpuBinding = require(GPU_PATH) as NativeBinding;
	return _gpuBinding;
}

export interface ModelOptions {
	/** Max context size in tokens. Default 2048. */
	contextSize?: number;
	/** CPU threads for single-token generation. Default 4. */
	threads?: number;
	/** CPU threads for batch/prefill processing. Default = same as `threads`. */
	threadsBatch?: number;
	/** Layers to offload to GPU. NEVER use 999 — pass the model's exact n_layer. */
	gpuLayers?: number;
	/**
	 * V3: enable hidden-state extraction (cparams.embeddings = true).
	 * Allows getHiddenState() after forward(). Header says embeddings are extracted
	 * "together with logits" so no second forward pass is required.
	 */
	embeddings?: boolean;
	/**
	 * V3.2: capture per-layer hidden state of the last token via cb_eval.
	 * Allows getHiddenStateLayer(layer) after forward(). Unlocks logit-lens and
	 * cross-layer alignment studies. Memory: n_layer × n_embd floats (e.g. ~100KB for small dense model).
	 */
	captureLayerHidden?: boolean;
	/**
	 * V8: disable KV cache offload to GPU (sets offload_kqv=false in cparams).
	 * Use with LLAMA_KV_SWA_OFFLOAD=1 env var for the 2-tier ISWA split:
	 *   SWA KV → VRAM (hot, ~100 MiB fixed), global KV → RAM (cold, scales with ctx).
	 * Allows ctx=65536 on a 16GB GPU that would OOM with full KV offload at that context.
	 */
	noKvOffload?: boolean;
	/**
	 * KV cache element type (Metal binding only; CPU binding ignores it). Default "f16".
	 * Measured on Metal: q8_0 is ~45% SLOWER at 4-6k ctx (per-token dequant outweighs the
	 * bandwidth saved), so never set this for speed — only to fit an otherwise-OOM context
	 * in the unified-memory budget, accepting the decode hit. Needs flash attention
	 * (auto-on) for the quantized V path.
	 */
	kvType?: "f16" | "q8_0" | "q5_1" | "q4_0";
	/**
	 * M15.7: path to a GGUF LoRA adapter to load on top of the base at construction and apply to
	 * this context. Preserves the base's QAT (the adapter is a delta, not a re-quantized merge).
	 *
	 * EXPERIMENTAL / OFF BY DEFAULT. The dialect-LoRA experiment is PARKED — neither trained
	 * adapter was deployable (dialect vs tool-call tradeoff), see doc/research/m15.7-lora-parked.md.
	 * This option is inert unless a path is explicitly passed; no product code path sets it (only
	 * training/eval-lora.mjs does). The binding support is kept so the experiment can be resumed.
	 * Produced by the dialect QLoRA train — see doc/substories/M15.7-stage2-lora-recipe.md.
	 * The Metal binding must be rebuilt (make in build-metal) for this to take effect.
	 */
	loraPath?: string;
	/** M15.7: scale applied to the LoRA adapter (default 1.0). Inert unless loraPath is set. */
	loraScale?: number;
	/**
	 * M11.3 variant (b): reserve a second, attention-isolated KV sequence (seq_id=1) alongside
	 * the main conversation (seq_id=0) — for out-of-band work (e.g. intent classification) via
	 * `decodeAppendSeq()` that must not see or affect the live conversation. Sets
	 * cparams.n_seq_max=2 and cparams.kv_unified=true under the hood so BOTH sequences get the
	 * FULL contextSize (llama.cpp partitions n_ctx across sequences unless kv_unified is set —
	 * verified against llama-context.cpp before shipping this option). OFF by default; every
	 * model construction before this option existed is unaffected (n_seq_max stays 1).
	 */
	auxSeq?: boolean;
	/**
	 * Micro-batch size (cparams.n_ubatch; 0/unset = library default, 512). The worst-case
	 * compute-graph reserve holds an n_vocab × n_ubatch fp32 logits tensor — at Gemma's 262144
	 * vocab the default costs ~512 MiB of anonymous memory PER CONTEXT (measured 2026-07-08),
	 * and the compact-SWA cache is sized n_swa + n_ubatch cells, so smaller values shrink both.
	 * Decode and specdec verify batches (K ≤ 32) are unaffected; only prefill throughput trades
	 * off. Clamped to n_batch (2048) natively. The SUBVOCAL_UBATCH env var sets a process-wide
	 * default; this option overrides it per instance.
	 */
	nUbatch?: number;
}

export abstract class BaseModel {
	protected native: NativeModel;
	protected freed = false;
	constructor(native: NativeModel) {
		this.native = native;
	}
	tokenize(text: string, addSpecial = true, parseSpecial = true): Int32Array {
		this.assertAlive();
		return this.native.tokenize(text, addSpecial, parseSpecial);
	}
	forward(tokens: Int32Array): number {
		this.assertAlive();
		return this.native.forward(tokens);
	}
	forwardAsync(tokens: Int32Array): Promise<number> {
		this.assertAlive();
		if (this.native.forwardAsync) return this.native.forwardAsync(tokens);
		return Promise.resolve(this.native.forward(tokens));
	}
	/**
	 * V7: incremental decode. Appends `tokens` onto the KV cache left by the
	 * previous forward()/decodeAppend() without clearing or re-prefilling.
	 * Produces logits for the last appended token by default.
	 * When allLogits=true, logits are computed for EVERY position in the batch
	 * — use this for GPU speculative verification (then call getLogitsBatch).
	 * Returns the llama_decode status (0 = ok).
	 */
	decodeAppend(tokens: Int32Array, allLogits = false): number {
		this.assertAlive();
		return this.native.decodeAppend(tokens, allLogits);
	}
	decodeAppendAsync(tokens: Int32Array, allLogits = false): Promise<number> {
		this.assertAlive();
		if (this.native.decodeAppendAsync) return this.native.decodeAppendAsync(tokens, allLogits);
		return Promise.resolve(this.native.decodeAppend(tokens, allLogits));
	}
	/**
	 * M11.3 variant (b): decode `tokens` onto sequence `seqId` at position `pos`, instead of
	 * always seq 0 at n_past_. Requires the model to have been constructed with
	 * ModelOptions.auxSeq — otherwise llama_decode rejects seq_id=1 as out of range. Does NOT
	 * touch n_past_ (that tracks seq 0's position only); the caller tracks the aux sequence's
	 * own position and cleans it up with kvCacheSeqRemove(seqId, ...).
	 */
	decodeAppendSeq(tokens: Int32Array, seqId: number, pos: number, allLogits = false): Promise<number> {
		this.assertAlive();
		if (this.native.decodeAppendSeq) return this.native.decodeAppendSeq(tokens, seqId, pos, allLogits);
		throw new Error("decodeAppendSeq is not implemented on CPU backend");
	}
	/**
	 * V7.1: reposition the internal n_past_ counter without touching the KV cache.
	 * Call AFTER kvCacheSeqRemove() to evict tail tokens from a sequence:
	 * the slots are gone but n_past_ still points past them. resetNPast(newLen)
	 * corrects the pointer so the next decodeAppend() places tokens at the
	 * right sequence position.
	 */
	resetNPast(n: number): void {
		this.assertAlive();
		this.native.resetNPast(n);
	}
	/**
	 * V4: forward but stop after layer `layerLimit` is computed. Hidden state of
	 * that layer is available via `getHiddenStateLayer(layerLimit)`.
	 * Requires `captureLayerHidden: true` at construction.
	 * Implementation: uses public cb_eval + abort_callback APIs, no engine patch.
	 * The graph is still built fully, but compute aborts mid-way — saving most
	 * of the work for deep models.
	 */
	forwardPartial(tokens: Int32Array, layerLimit: number): number {
		this.assertAlive();
		return this.native.forwardPartial(tokens, layerLimit);
	}
	getLogits(): Float32Array {
		this.assertAlive();
		return this.native.getLogits();
	}
	/**
	 * V6: zero-alloc logits — copies into a pre-allocated shadow buffer.
	 * No per-call malloc overhead. Safe to use until the next forward() call
	 * overwrites the shadow buffer.
	 */
	getLogitsFast(): Float32Array {
		this.assertAlive();
		return this.native.getLogitsFast();
	}
	/**
	 * V6: true zero-copy logits — points directly into llama's internal buffer.
	 * @unsafe Invalid after the next llama_decode() call.
	 */
	getLogitsUnsafe(): Float32Array {
		this.assertAlive();
		return this.native.getLogitsUnsafe();
	}
	/**
	 * V6: multi-position logits extraction.
	 * Returns an array of Float32Array, one per position index.
	 */
	getLogitsBatch(indices: Int32Array): Float32Array[] {
		this.assertAlive();
		return this.native.getLogitsBatch(indices);
	}
	/** V3: returns hidden-state vector of the last decoded token (dim = embeddingSize). Requires `embeddings: true` in constructor. */
	getHiddenState(): Float32Array {
		this.assertAlive();
		return this.native.getHiddenState();
	}
	detokenize(tokens: Int32Array | readonly number[]): string {
		this.assertAlive();
		const arr = tokens instanceof Int32Array ? tokens : Int32Array.from(tokens);
		return this.native.detokenize(arr);
	}
	get vocabSize(): number {
		this.assertAlive();
		return this.native.vocabSize();
	}
	get contextSize(): number {
		this.assertAlive();
		return this.native.contextSize();
	}
	/** V3: hidden state dimension (Gemma 4 E2B = 1536, Gemma 4 26B-A4B = 2816). */
	get embeddingSize(): number {
		this.assertAlive();
		return this.native.embeddingSize();
	}
	/** Number of transformer layers (Gemma 4 E2B = 35, Gemma 4 26B-A4B = 30). */
	get layerCount(): number {
		this.assertAlive();
		return this.native.layerCount();
	}
	/** V3.2: hidden state of the last token after the N-th layer. Requires `captureLayerHidden: true`. */
	getHiddenStateLayer(layer: number): Float32Array {
		this.assertAlive();
		return this.native.getHiddenStateLayer(layer);
	}
	/**
	 * V5: snapshot full KV cache + context state. Returns binary blob.
	 * Agent use cases: branching, backtracking, multi-turn memory without re-tokenize.
	 * Only valid for the exact same model (same arch, same context size).
	 */
	kvSave(): Uint8Array {
		this.assertAlive();
		return this.native.getKVState();
	}
	/** V5: restore a previously saved KV state. Returns number of bytes read. */
	kvRestore(state: Uint8Array): number {
		this.assertAlive();
		return this.native.setKVState(state);
	}
	// ========================================================================
	// V6.1: Logit Bias / Steering
	// ========================================================================
	/**
	 * Apply per-call logit biases. These are applied once and cleared after
	 * the next forward() call. Each bias adds to the logit of the specified token.
	 */
	applyLogitBias(biases: Array<{ tokenId: number; bias: number }>): void {
		this.assertAlive();
		this.native.applyLogitBias(biases);
	}
	/**
	 * Set persistent biases that are applied on every forward() call.
	 * Useful for "always avoid token X" or "always prefer token Y".
	 */
	setPersistentBiases(biases: Array<{ tokenId: number; bias: number }>): void {
		this.assertAlive();
		this.native.setPersistentBiases(biases);
	}
	/** Clear all biases (persistent and pending). */
	clearLogitBiases(): void {
		this.assertAlive();
		this.native.clearLogitBiases();
	}
	/**
	 * Set a dense steering vector applied to logits on every forward() call.
	 * Formula: logits[i] += vector[i] * strength
	 * The vector must have length equal to vocabSize.
	 */
	setSteeringVector(vector: Float32Array, strength: number): void {
		this.assertAlive();
		this.native.setSteeringVector(vector, strength);
	}
	/** Clear the steering vector. */
	clearSteeringVector(): void {
		this.assertAlive();
		this.native.clearSteeringVector();
	}
	// ========================================================================
	// V6.2: KV Cache Manipulation
	// ========================================================================
	/**
	 * Remove positions [p0, p1] from a sequence in the KV cache.
	 * Use p0=-1, p1=-1 to remove the entire sequence.
	 */
	kvCacheSeqRemove(seqId: number, p0: number, p1: number): void {
		this.assertAlive();
		this.native.kvCacheSeqRemove(seqId, p0, p1);
	}
	/** Copy positions [p0, p1] from srcSeq to dstSeq. */
	kvCacheSeqCopy(srcSeq: number, dstSeq: number, p0: number, p1: number): void {
		this.assertAlive();
		this.native.kvCacheSeqCopy(srcSeq, dstSeq, p0, p1);
	}
	/** Remove all sequences except the specified one. */
	kvCacheSeqKeep(seqId: number): void {
		this.assertAlive();
		this.native.kvCacheSeqKeep(seqId);
	}
	/** Shift positions [p0, p1] by delta (e.g., for insertion). */
	kvCacheSeqShift(seqId: number, p0: number, p1: number, delta: number): void {
		this.assertAlive();
		this.native.kvCacheSeqShift(seqId, p0, p1, delta);
	}
	/** Clear the entire KV cache. */
	kvCacheClear(): void {
		this.assertAlive();
		this.native.kvCacheClear();
	}
	/**
	 * Fork a sequence: copy srcSeq to a new sequence ID.
	 * Returns the new sequence ID.
	 */
	kvCacheFork(seqId: number): number {
		this.assertAlive();
		return this.native.kvCacheFork(seqId);
	}
	/**
	 * Evict old positions keeping only the last keepLastN tokens.
	 * Optionally specify a seqId (default: 0).
	 */
	kvCacheEvict(keepLastN: number, seqId?: number): void {
		this.assertAlive();
		this.native.kvCacheEvict(keepLastN, seqId);
	}
	/**
	 * Get a snapshot of the KV cache state, including cell positions,
	 * sequence IDs, and usage statistics.
	 */
	kvCacheView(): { maxSize: number; used: number; cells: Array<{ pos: number; seqId: number; hasValue: boolean }> } {
		this.assertAlive();
		return this.native.kvCacheView();
	}
	// ========================================================================
	// V6.3: AST-Aware Logit Masking
	// ========================================================================
	/** Set the whitelist of allowed token IDs. All others are masked to -Infinity on every forward(). */
	setASTTokenMask(allowed: Int32Array): void {
		this.assertAlive();
		this.native.setASTTokenMask(allowed);
	}
	/** Remove the AST token mask (re-enable all tokens). */
	clearASTMask(): void {
		this.assertAlive();
		this.native.clearASTMask();
	}
	// ========================================================================
	// V6.4: Sparse Logits
	// ========================================================================
	/**
	 * Extract the top-K (tokenId, logit) pairs as a packed Int32Array.
	 * Layout: [id0..idK-1, logitBits0..logitBitsK-1]
	 * Read the float logits: `new Float32Array(result.buffer, k * 4, k)`
	 * Much cheaper than getLogits() (e.g. ~80 bytes vs ~1MB for 262k vocab).
	 */
	getLogitsTopK(k = 20): Int32Array {
		this.assertAlive();
		return this.native.getLogitsTopK(k);
	}
	/** Zero a set of token IDs in the logit buffer (scalar loop; SIMD pending). */
	applyMaskSIMD(disallowedTokenIds: Int32Array): void {
		this.assertAlive();
		this.native.applyMaskSIMD(disallowedTokenIds);
	}
	// ========================================================================
	// V9: Token-to-Pointer Routing — Action Token Callbacks
	// ========================================================================
	/**
	 * Register a zero-latency callback for an action token ID.
	 * When the model samples this token, the callback fires immediately
	 * (before the token enters the decode history).
	 */
	registerActionToken(tokenId: number, callback: (tokenId: number) => void): void {
		this.assertAlive();
		this.native.registerActionToken(tokenId, callback);
	}
	/**
	 * Remove a previously registered action token callback.
	 */
	removeActionToken(tokenId: number): void {
		this.assertAlive();
		this.native.removeActionToken(tokenId);
	}
	/**
	 * Check if the sampled token should be routed as an action.
	 * Returns true if handled by a registered callback (skip adding to history).
	 * Returns false if it's a normal token (add to history as usual).
	 */
	handleActionToken(tokenId: number): boolean {
		this.assertAlive();
		return this.native.handleActionToken(tokenId);
	}
	// ========================================================================
	// Suffix-tree (prompt-lookup) drafter — see doc/research/suffix-tree-speculative-decoding.md
	// ========================================================================
	/** Append newly-seen tokens to the suffix trie. NOT cumulative -- pass only what's new. */
	suffixTreeExtend(tokens: Int32Array): void {
		this.assertAlive();
		this.native.suffixTreeExtend(tokens);
	}
	/** Reset the trie (e.g. at the start of a new session/file). */
	suffixTreeClear(): void {
		this.assertAlive();
		this.native.suffixTreeClear();
	}
	/**
	 * Draft a candidate continuation by matching the tail of `context` against
	 * previously-extended suffixes. Returns an empty array on a cold/no-match trie.
	 */
	suffixTreeSpeculate(
		context: Int32Array,
		maxSpecTokens = 8,
		minTokenProb = 0.05,
		minMatchCount = 1,
		minMatchLen = 3,
	): Int32Array {
		this.assertAlive();
		return this.native.suffixTreeSpeculate(context, maxSpecTokens, minTokenProb, minMatchCount, minMatchLen);
	}
	suffixTreeTokenCount(): number {
		this.assertAlive();
		return this.native.suffixTreeTokenCount();
	}
	suffixTreeMaxDepth(): number {
		this.assertAlive();
		return this.native.suffixTreeMaxDepth();
	}
	free(): void {
		if (this.freed) return;
		this.native.free();
		this.freed = true;
	}
	protected assertAlive(): void {
		if (this.freed) throw new Error("Model has been freed");
	}
}

/** CPU-only model — linked statically against ik_llama.cpp (Zen 3 optimized). */
export class ModelCPU extends BaseModel {
	constructor(path: string, opts: ModelOptions = {}) {
		super(new (getCpuBinding().Model)(path, opts));
	}

	forkContext(): ModelCPU {
		this.assertAlive();
		const fork = new ModelCPU("__subvocal_fork__", {});
		(fork as unknown as { native: NativeModel }).native = this.native.forkContext();
		return fork;
	}
}

/** GPU model — linked statically against llama.cpp upstream + CUDA. */
export class ModelGPU extends BaseModel {
	constructor(path: string, opts: ModelOptions = {}) {
		super(new (getGpuBinding().Model)(path, opts));
	}

	forkContext(): ModelGPU {
		this.assertAlive();
		throw new Error("GPU context forking not supported");
	}
	/**
	 * V3: forward with a prefix of raw embedding vectors followed by optional token IDs.
	 *
	 * This is the GPU-side "write" primitive of Subvocal's hidden-state passing.
	 * The small brain produces a hidden state, an adapter projects it to this model's
	 * embeddingSize, and we feed it here as if it were a token embedding.
	 *
	 * @param embd Float32Array of length k * embeddingSize (k = number of prefix slots, typically 1).
	 * @param tokens Optional Int32Array of token IDs appended after the prefix.
	 */
	forwardEmbedding(embd: Float32Array, tokens?: Int32Array): number {
		this.assertAlive();
		if (embd.length % this.embeddingSize !== 0) {
			throw new Error(`embd length (${embd.length}) must be multiple of embeddingSize (${this.embeddingSize})`);
		}
		return this.native.forwardEmbedding!(embd, tokens);
	}
	/**
	 * Drafter-agnostic batched verify: submits `draftTokens` to the target in one decode
	 * call and accepts up to the first mismatch with the target's own greedy prediction.
	 * Works with ANY draft source (suffix-tree, MTP, etc.) -- only touches this model's own
	 * context, no second model needed. GREEDY ONLY: the accept rule is a hard equality
	 * check against the target's argmax, so this is not valid under temperature>0 sampling.
	 * Returns Int32Array [acceptedCount, ...emittedTokenIds] (emittedTokenIds has
	 * acceptedCount+1 entries: the accepted draft tokens, then one correction token).
	 */
	mtpVerifyBatch(draftTokens: Int32Array): Int32Array {
		this.assertAlive();
		return this.native.mtpVerifyBatch!(draftTokens);
	}
}

/** Legacy alias for the previous single-binder API. */
export const Model = ModelCPU;

/** Returns the indices of the top-k logits, sorted descending. */
export function argmaxK(logits: Float32Array, k: number): number[] {
	const indices = Array.from({ length: logits.length }, (_, i) => i);
	indices.sort((a, b) => logits[b] - logits[a]);
	return indices.slice(0, k);
}

// ============================================================================
// V5: Agent-side sampling primitives (operate on raw logits/probs Float32Array)
// Subvocal philosophy: sampling lives in TS, not buried inside the engine.
// The agent has full control to mutate logits / sample / re-sample step-by-step.
// ============================================================================

/** Greedy argmax — equivalent to argmaxK(logits, 1)[0]. Returns the top token id. */
export function sampleGreedy(logits: Float32Array): number {
	let best = 0;
	let bestVal = logits[0];
	for (let i = 1; i < logits.length; i++) {
		if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
	}
	return best;
}

/** In-place temperature scaling. temp=1 is no-op; <1 sharpens, >1 flattens; ≤0 means "greedy" downstream. */
export function applyTemperature(logits: Float32Array, temperature: number): Float32Array {
	if (temperature === 1 || temperature <= 0) return logits;
	const inv = 1 / temperature;
	for (let i = 0; i < logits.length; i++) logits[i] *= inv;
	return logits;
}

/** In-place additive bias: logits[token_id] += value for each entry in `biases`. */
export function applyLogitBias(logits: Float32Array, biases: ReadonlyMap<number, number>): Float32Array {
	for (const [id, v] of biases) {
		if (id >= 0 && id < logits.length) logits[id] += v;
	}
	return logits;
}

/** Compute softmax over the full logits vector. Returns a new Float32Array of probabilities (sum=1). */
export function softmax(logits: Float32Array): Float32Array {
	let max = -Infinity;
	for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
	const out = new Float32Array(logits.length);
	let sum = 0;
	for (let i = 0; i < logits.length; i++) { out[i] = Math.exp(logits[i] - max); sum += out[i]; }
	const inv = 1 / sum;
	for (let i = 0; i < logits.length; i++) out[i] *= inv;
	return out;
}

/** Top-K filter: returns the K best (token_id, logit) pairs, sorted descending by logit. */
export function topK(logits: Float32Array, k: number): { id: number; logit: number }[] {
	const ids = argmaxK(logits, k);
	return ids.map((id) => ({ id, logit: logits[id] }));
}

/**
 * Top-P (nucleus) filter on logits: returns the smallest set of token ids whose
 * cumulative softmax probability ≥ p, sorted by descending logit.
 * Standard nucleus sampling: discard low-probability tail to avoid garbage tokens.
 */
export function topP(logits: Float32Array, p: number): { id: number; logit: number; prob: number }[] {
	const idx = Array.from({ length: logits.length }, (_, i) => i);
	idx.sort((a, b) => logits[b] - logits[a]);
	// Compute softmax over the sorted-by-logit array (numerically stable)
	const max = logits[idx[0]];
	const exp: number[] = new Array(idx.length);
	let sum = 0;
	for (let i = 0; i < idx.length; i++) { exp[i] = Math.exp(logits[idx[i]] - max); sum += exp[i]; }
	const out: { id: number; logit: number; prob: number }[] = [];
	let cum = 0;
	for (let i = 0; i < idx.length; i++) {
		const prob = exp[i] / sum;
		out.push({ id: idx[i], logit: logits[idx[i]], prob });
		cum += prob;
		if (cum >= p) break;
	}
	return out;
}

/**
 * Sample one token from a (id, logit) candidate list using categorical sampling
 * weighted by exp(logit). Pass a seeded `rng()` for determinism in tests.
 */
export function sampleCategorical(
	candidates: { id: number; logit: number }[],
	rng: () => number = Math.random,
): number {
	if (candidates.length === 0) throw new Error("empty candidates");
	if (candidates.length === 1) return candidates[0].id;
	// Stable softmax over the candidates
	let max = -Infinity;
	for (const c of candidates) if (c.logit > max) max = c.logit;
	let sum = 0;
	const w: number[] = new Array(candidates.length);
	for (let i = 0; i < candidates.length; i++) { w[i] = Math.exp(candidates[i].logit - max); sum += w[i]; }
	let r = rng() * sum;
	for (let i = 0; i < candidates.length; i++) {
		r -= w[i];
		if (r <= 0) return candidates[i].id;
	}
	return candidates[candidates.length - 1].id;
}

/** Convenience sampler combining the common path: bias → temperature → top-K → top-P → sample. */
export interface SampleOptions {
	temperature?: number; // default 1.0 (no scaling). Set to 0 or negative for greedy.
	topK?: number;        // default 0 = disabled
	topP?: number;        // default 1.0 = disabled (no truncation)
	biases?: ReadonlyMap<number, number>; // optional per-token logit bias
	rng?: () => number;   // optional seeded RNG
}
export function sample(logits: Float32Array, opts: SampleOptions = {}): number {
	const temp = opts.temperature ?? 1.0;
	if (temp <= 0) {
		// Greedy with optional bias
		if (opts.biases && opts.biases.size > 0) {
			const copy = new Float32Array(logits);
			applyLogitBias(copy, opts.biases);
			return sampleGreedy(copy);
		}
		return sampleGreedy(logits);
	}
	// Work on a copy so callers' raw logits stay intact
	const work = new Float32Array(logits);
	if (opts.biases && opts.biases.size > 0) applyLogitBias(work, opts.biases);
	applyTemperature(work, temp);
	let candidates: { id: number; logit: number }[];
	if (opts.topP !== undefined && opts.topP > 0 && opts.topP < 1) {
		candidates = topP(work, opts.topP);
	} else if (opts.topK !== undefined && opts.topK > 0) {
		candidates = topK(work, opts.topK);
	} else {
		// Full categorical (slow on big vocabs — caller should usually set topK or topP)
		candidates = Array.from(work, (logit, id) => ({ id, logit }));
	}
	return sampleCategorical(candidates, opts.rng);
}

/** Shannon entropy of a softmax distribution — useful agent signal for "uncertainty". */
export function entropy(logits: Float32Array): number {
	const probs = softmax(logits);
	let h = 0;
	for (let i = 0; i < probs.length; i++) {
		if (probs[i] > 0) h -= probs[i] * Math.log(probs[i]);
	}
	return h;
}

/** Margin between top-1 and top-2 logit — useful agent signal for "confidence". */
export function top1Margin(logits: Float32Array): number {
	const t = argmaxK(logits, 2);
	return logits[t[0]] - logits[t[1]];
}

/** L2-normalize in place. Returns the same array for chaining. Embedding models always use unit vectors. */
export function normalizeL2(v: Float32Array): Float32Array {
	let n = 0;
	for (let i = 0; i < v.length; i++) n += v[i] * v[i];
	const norm = Math.sqrt(n);
	if (norm === 0) return v;
	for (let i = 0; i < v.length; i++) v[i] /= norm;
	return v;
}

/** Cosine similarity between two equal-length float vectors. */
export function cosineSim(a: Float32Array, b: Float32Array): number {
	if (a.length !== b.length) throw new Error(`length mismatch: ${a.length} vs ${b.length}`);
	let dot = 0;
	let na = 0;
	let nb = 0;
	for (let i = 0; i < a.length; i++) {
		dot += a[i] * b[i];
		na += a[i] * a[i];
		nb += b[i] * b[i];
	}
	const denom = Math.sqrt(na) * Math.sqrt(nb);
	return denom === 0 ? 0 : dot / denom;
}

/**
 * Placeholder projection from a `srcDim` hidden state to a `dstDim` target embedding.
 * Used as identity-with-zero-pad/truncate when no trained adapter is available yet.
 * Replace with a real linear adapter (trained) for semantically meaningful round-trips.
 */
export function placeholderProject(src: Float32Array, dstDim: number): Float32Array {
	const out = new Float32Array(dstDim);
	const n = Math.min(src.length, dstDim);
	for (let i = 0; i < n; i++) out[i] = src[i];
	return out;
}

// V6.3: Speculative Decoding — dual-brain CPU draft → GPU verify
export {
	SpeculativeDecoder,
	buildASTLogitMask,
	type DecoderConfig,
	type DecodeStats,
	type DraftResult,
	DEFAULT_DECODER_CONFIG,
} from "./speculative-decoder.js";

// V6.3: Suffix Tree — self-speculative draft generation
interface NativeBindingWithSuffixTree {
	Model: new (path: string, opts?: object) => NativeModel;
	SuffixTree: new (maxDepth?: number) => {
		extend(tokens: Int32Array): void;
		clear(): void;
		speculate(context: Int32Array, maxSpecTokens?: number, minTokenProb?: number, minMatchCount?: number, minMatchLen?: number): Int32Array;
		tokenCount(): number;
		maxDepth(): number;
	};
}

export class SuffixTree {
	private native: InstanceType<NativeBindingWithSuffixTree["SuffixTree"]>;

	constructor(maxDepth?: number) {
		const binding = ((): NativeBindingWithSuffixTree => {
			try {
				return require(CPU_PATH) as unknown as NativeBindingWithSuffixTree;
			} catch {
				return require(GPU_PATH) as unknown as NativeBindingWithSuffixTree;
			}
		})();
		this.native = new binding.SuffixTree(maxDepth);
	}

	extend(tokens: Int32Array): void {
		this.native.extend(tokens);
	}

	clear(): void {
		this.native.clear();
	}

	speculate(context: Int32Array, maxSpecTokens = 7, minTokenProb = 0.1, minMatchCount = 1, minMatchLen = 5): Int32Array {
		return this.native.speculate(context, maxSpecTokens, minTokenProb, minMatchCount, minMatchLen);
	}

	get tokenCount(): number {
		return this.native.tokenCount();
	}

	get maxDepth(): number {
		return this.native.maxDepth();
	}
}

// ── Substory 1.4: AVX2 full-text byte scan ─────────────────────────────────────

/**
 * Search for all byte-exact occurrences of `needle` in `haystack` using AVX2
 * SIMD acceleration (scalar fallback on non-AVX2 hosts).
 *
 * Returns an Int32Array of byte offsets (one entry per match).  Zero-length
 * needle or needle longer than haystack returns an empty array.
 *
 * Typical use: full-text search over in-RAM codebase content.
 *
 * @param haystack  Source code bytes (e.g. Buffer.from(fileContent)).
 * @param needle    Search pattern bytes (e.g. Buffer.from("function foo")).
 * @returns         Sorted byte offsets of every occurrence.
 */
export function scanBytes(haystack: Uint8Array, needle: Uint8Array): Int32Array {
	return getCpuBinding().scanBytes(haystack, needle);
}

/**
 * Substory 2.4: Allocate an off-heap Int32Array backed by C++ malloc.
 * V8 cannot relocate this buffer (no GC moves). The address is stable for
 * the lifetime of the returned Int32Array — safe to pass back to C++ without pinning.
 * Memory is freed automatically via NAPI finalizer when the GC collects the buffer.
 */
export function createPinnedBuffer(sizeInInts: number): Int32Array {
	return getCpuBinding().createPinnedBuffer(sizeInInts);
}

// ── libgit2: in-process Git operations ──────────────────────────────────────

/**
 * Initialize a Git repository for subsequent operations.
 * Must be called before gitStageAll / gitCommit / gitDiff*.
 */
export function gitInit(repoPath: string, opts?: { authorName?: string; authorEmail?: string }): void {
	getCpuBinding().gitInit(repoPath, opts);
}

/** Stage all changes (equivalent to git add -A). */
export function gitStageAll(): void {
	getCpuBinding().gitStageAll();
}

/**
 * Create a commit with the given message.
 * @returns The commit SHA hash.
 */
export function gitCommit(message: string, authorName?: string, authorEmail?: string): string {
	return getCpuBinding().gitCommit(message, authorName, authorEmail);
}

/** Get the diff of staged/index changes vs working tree as a unified patch string. */
export function gitDiffStaged(): string {
	return getCpuBinding().gitDiffStaged();
}

/** Get the diff of working tree vs HEAD as a unified patch string. */
export function gitDiffWorking(): string {
	return getCpuBinding().gitDiffWorking();
}

/** Release the Git repository handle. Call when done. */
export function gitFree(): void {
	getCpuBinding().gitFree();
}
