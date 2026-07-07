/**
 * kvColdStore.ts
 *
 * Epic M3 (KV tiering) — cold tier: disk-persistent KV checkpoints.
 *
 * Design ported from antirez/ds4's kvstore (see doc/research/ds4-kvstore-findings.md):
 *   - Checkpoint files are keyed by the SHA1 hex of the *rendered prompt text*, not the
 *     token sequence — robust to tokenizer/model changes, and a directory listing alone
 *     reconstructs the whole index (no separate manifest to keep in sync).
 *   - Loading finds the longest stored-text prefix of the incoming prompt and restores
 *     that checkpoint, instead of requiring an exact match — a session's running
 *     transcript naturally grows by appending, so its earlier states are valid prefixes.
 *   - Eviction score = (hit-decayed effective hits + 1) × tokens / fileSize, with a
 *     stickiness bonus for "cold" (session-start) checkpoints over "continued"
 *     (mid-session) ones — the lowest-scoring entries are evicted first when over budget.
 *
 * This is intentionally CPU-only and low-bandwidth: it reads/writes the (relatively small)
 * KV blob to disk, not the multi-GB model weights, so it doesn't compete with the GPU's
 * memory-bandwidth budget during generation (see doc/research/m1-metal-benchmark.md).
 *
 * Usage:
 *   const store = new KVColdStore('/path/to/cache/dir', 4 * 1024 * 1024 * 1024); // 4 GiB budget
 *   store.store(model, renderedPromptText, tokenCount);              // after a forward()/prefill
 *   const matchedChars = store.tryLoad(model, renderedPromptText);   // before a forward()
 *   if (matchedChars > 0) { / only re-decode promptText.slice(matchedChars) onward / }
 */

import { createHash } from 'node:crypto';
import {
  closeSync, existsSync, mkdirSync, openSync, readFileSync, readSync, readdirSync, renameSync,
  statSync, unlinkSync, write as fsWrite, writeSync,
} from 'node:fs';
import { join } from 'node:path';
import { zstdCompress, zstdCompressSync, zstdDecompressSync } from 'node:zlib';
import type { BaseModel } from '@subvocal/synapse';

// ── On-disk format ───────────────────────────────────────────────────────────────
//
// [0:4)   magic "SVKV"
// [4]     version (1 = raw KV payload; 2 = adds the compression byte at [6])
// [5]     reason: 0 = cold (session start), 1 = continued (mid-session checkpoint)
// [6]     v2+: KV payload compression: 0 = none, 1 = zstd (M3.6), 2 = shuffle2+zstd (M3.6
//         follow-up). v1 files: reserved/0.
// [7]     reserved
// [8:12)  tokens (u32 LE)
// [12:16) hits (u32 LE)
// [16:24) createdAt, ms since epoch (f64 LE)
// [24:32) lastUsed, ms since epoch (f64 LE)
// [32:36) textBytes (u32 LE)
// [36:...) rendered prompt text (textBytes), then the KV blob (model.kvSave() output,
//          transformed per byte [6] on v2 files; raw on v1 — both remain readable)

const HEADER_MAGIC = 'SVKV';
const HEADER_VERSION = 2;
const HEADER_FIXED = 36;
const OFF_COMPRESSION = 6;
const COMPRESSION_NONE = 0;
/** Legacy (pre byte-plane-split): plain zstd on the raw KV bytes. Still readable; no
 *  longer written — COMPRESSION_ZSTD_SHUFFLE beats it at the same CPU cost (see below). */
const COMPRESSION_ZSTD = 1;
/** M3.6 follow-up (2026-07-06): 2-byte "shuffle" (HDF5/blosc-style byte-plane split) applied
 *  to the raw KV bytes BEFORE zstd. Measured on real 12B + E2B checkpoints (12B: 690 MiB/
 *  583 MiB @ 13.5k/6.6k tokens; E2B: 27/26 MiB @ ~2.5k tokens): consistent -7.7% on-disk size
 *  vs plain zstd (ratio 1.09x → 1.18x), byte-identical round-trip verified. `llama_state`'s
 *  buffer isn't a single homogeneous f16 array (RNG state, per-layer size headers, cell
 *  metadata are mixed in with the bulk KV tensor bytes) — this was measured, not derived from
 *  the format, precisely because reasoning about "should this compress better" without
 *  reading the whole state layout would have been a guess. See doc/substories/
 *  M3.6-cold-checkpoint-compression.md. */
const COMPRESSION_ZSTD_SHUFFLE = 2;
const SHA_NAME_RE = /^[0-9a-f]{40}$/;
/** Suffix for in-progress checkpoint writes (see the write-then-rename note on writeSnapshot). */
const TMP_SUFFIX = '.tmp';

/**
 * 2-byte "shuffle" (byte-plane split): treats `buf` as a stream of u16 elements and gathers
 * all LOW bytes into the first half of the output, all HIGH bytes into the second half.
 * Structured/low-entropy bytes (sign+exponent-like, in the common case of f16-heavy KV state)
 * end up contiguous instead of interleaved with noisier bytes, which is what lets zstd find
 * more redundancy. Byte-identical round-trip via `unshuffle2`; a trailing odd byte (buf.length
 * is odd) is copied through unchanged by both directions.
 */
function shuffle2(buf: Buffer): Buffer {
  const n = buf.length;
  const pairs = n >> 1;
  const out = Buffer.allocUnsafe(n);
  for (let i = 0; i < pairs; i++) {
    out[i] = buf[i * 2];
    out[pairs + i] = buf[i * 2 + 1];
  }
  if (n & 1) out[n - 1] = buf[n - 1];
  return out;
}

/** Inverse of `shuffle2`. */
function unshuffle2(buf: Buffer): Buffer {
  const n = buf.length;
  const pairs = n >> 1;
  const out = Buffer.allocUnsafe(n);
  for (let i = 0; i < pairs; i++) {
    out[i * 2] = buf[i];
    out[i * 2 + 1] = buf[pairs + i];
  }
  if (n & 1) out[n - 1] = buf[n - 1];
  return out;
}

/**
 * Async, YIELDING variant of `shuffle2` for the write path only. `writeSnapshot()` runs
 * concurrently with the current turn's decode (that's the whole point of the async write —
 * see the class doc), so a plain synchronous `shuffle2()` here would stall the JS main thread
 * for the shuffle's full duration (measured ~0.6-0.9s on a 600-700 MiB 12B checkpoint) in one
 * uninterruptible block — starving the event loop of the very callbacks that report decode
 * progress and directly adding to this turn's generation latency. Chunked with a `setImmediate`
 * yield between slices so pending callbacks (a decode step's completion, most importantly) get
 * to run between chunks instead of queuing behind the whole shuffle. `tryLoad()`'s restore-path
 * `unshuffle2` stays synchronous on purpose: restore runs once, before decode starts, with
 * nothing else in flight to starve — see `store()`'s doc-comment for the same reasoning applied
 * to its (unused-in-production, but kept correct) fully-synchronous path.
 */
async function shuffle2Chunked(buf: Buffer): Promise<Buffer> {
  const n = buf.length;
  const pairs = n >> 1;
  const out = Buffer.allocUnsafe(n);
  const CHUNK_PAIRS = 4 * 1024 * 1024; // ~8 MiB of input per slice before yielding
  for (let start = 0; start < pairs; start += CHUNK_PAIRS) {
    const end = Math.min(start + CHUNK_PAIRS, pairs);
    for (let i = start; i < end; i++) {
      out[i] = buf[i * 2];
      out[pairs + i] = buf[i * 2 + 1];
    }
    if (end < pairs) await new Promise<void>(resolve => setImmediate(resolve));
  }
  if (n & 1) out[n - 1] = buf[n - 1];
  return out;
}

// Same constants as ds4_kvstore.c (see doc/research/ds4-kvstore-findings.md).
const HIT_HALF_LIFE_MS = 6 * 60 * 60 * 1000;
const MIN_EFFECTIVE_HITS = 0.01;
const ANCHOR_REASON_SCORE_FACTOR = 2.0;

export type CheckpointReason = 'cold' | 'continued';

/**
 * A KV checkpoint captured synchronously (via `snapshot()`) but not yet written to disk. The
 * buffers are JS-owned copies (getKVState() allocates a fresh ArrayBuffer), fully independent of
 * the model's live KV cache — so `writeSnapshot()` can flush them to SSD *asynchronously, while
 * the GPU keeps decoding*, with no data race. See AgentLoop.prefillOrResume().
 */
export interface ColdSnapshot {
  sha: string;
  buffers: Buffer[];
  totalSize: number;
}

export interface LoadResult {
  /** Length, in chars, of the matched stored-text prefix. 0 = no checkpoint matched. */
  matchedChars: number;
  /** Authoritative token count for the restored KV state (from the checkpoint's own header,
   *  set by the caller at `store()` time) — use this, not a re-derivation from matchedChars. */
  matchedTokens: number;
}

interface ColdStoreEntry {
  sha: string;
  path: string;
  tokens: number;
  hits: number;
  createdAt: number;
  lastUsed: number;
  fileSize: number;
  reason: CheckpointReason;
}

export class KVColdStore {
  constructor(
    private readonly dir: string,
    /** Total disk budget in bytes. 0 or negative disables eviction (unbounded). */
    private readonly budgetBytes: number,
  ) {
    if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
    // 2026-07 KV audit: sweep stale *.tmp leftovers from writes interrupted by a process
    // exit/crash. Writes go tmp-then-rename (see writeSnapshot), so an interrupted write can
    // only ever leave a .tmp — never a truncated checkpoint at its final SHA path (a truncated
    // v1/raw payload would make llama_state_set_data read garbage; a truncated v2 payload used
    // to make tryLoad() THROW on zstd decompress, failing every later start() that matched it).
    for (const name of readdirSync(this.dir)) {
      if (!name.endsWith(TMP_SUFFIX)) continue;
      try { unlinkSync(join(this.dir, name)); } catch { /* best-effort sweep */ }
    }
  }

  /** On-disk budget in bytes (0 or negative = unbounded). Read-only view for callers that
   *  size their own write policy against it (see AgentLoop's ladder-vs-budget guard). */
  get budget(): number {
    return this.budgetBytes;
  }

  private pathForSha(sha: string): string {
    return join(this.dir, sha);
  }

  private static shaOf(text: string): string {
    return createHash('sha1').update(text, 'utf-8').digest('hex');
  }

  /**
   * Capture `model`'s current KV state into JS-owned buffers, keyed by the rendered prompt text.
   * Synchronous and cheap-ish (one `getKVState()` memory copy) — MUST run before any further
   * decode mutates the KV. The returned snapshot can then be flushed to disk with
   * `writeSnapshot()` asynchronously, overlapping GPU decode. Does not touch disk itself.
   */
  snapshot(model: BaseModel, promptText: string, tokens: number, reason: CheckpointReason = 'cold'): ColdSnapshot {
    const sha = KVColdStore.shaOf(promptText);
    const kv = model.kvSave();
    const textBuf = Buffer.from(promptText, 'utf-8');

    const header = Buffer.alloc(HEADER_FIXED);
    header.write(HEADER_MAGIC, 0, 'ascii');
    header.writeUInt8(HEADER_VERSION, 4);
    header.writeUInt8(reason === 'cold' ? 0 : 1, 5);
    header.writeUInt32LE(tokens, 8);
    header.writeUInt32LE(0, 12); // hits — fresh checkpoint, never loaded yet
    const now = Date.now();
    header.writeDoubleLE(now, 16);
    header.writeDoubleLE(now, 24);
    header.writeUInt32LE(textBuf.length, 32);

    const kvBuf = Buffer.from(kv.buffer, kv.byteOffset, kv.byteLength);
    return { sha, buffers: [header, textBuf, kvBuf], totalSize: header.length + textBuf.length + kvBuf.length };
  }

  /**
   * Flush a `snapshot()` to disk asynchronously. Safe to run concurrently with model decode —
   * it only reads the snapshot's own (copied) buffers. The KV payload is byte-plane-split then
   * zstd-compressed first (M3.6 + the shuffle follow-up): the shuffle yields between chunks
   * (`shuffle2Chunked`) and zstd runs on libuv's threadpool via the async zlib API, so like the
   * write itself both stages overlap a synchronous decode loop on the main JS thread instead of
   * stalling it. Decompression cost is paid once at restore — the asymmetry that made per-token
   * KV-q8 a negative result is what makes compressing write-once/read-once checkpoints fine. The
   * big buffer is written in a single `fs.write` syscall whenever it fits under the
   * 2^31-1-byte per-call limit — which it always does for the 12B's ≤128k contexts, even more
   * comfortably after compression. Only pathologically large checkpoints fall back to the
   * awaited chunked path.
   */
  async writeSnapshot(snap: ColdSnapshot): Promise<void> {
    const [header, textBuf, kvBuf] = snap.buffers;
    const shuffled = await shuffle2Chunked(kvBuf);
    const compressed = await KVColdStore.zstdAsync(shuffled);
    header.writeUInt8(COMPRESSION_ZSTD_SHUFFLE, OFF_COMPRESSION);
    snap.buffers = [header, textBuf, compressed];
    snap.totalSize = header.length + textBuf.length + compressed.length;

    this.evictIfNeeded(snap.totalSize);
    // 2026-07 KV audit: write to a .tmp sibling, rename into place only when complete. These
    // are multi-hundred-MB background writes that routinely outlive their turn (and the TUI
    // process makes no teardown flush) — writing straight to the final SHA path meant a kill
    // mid-write left a truncated checkpoint that the next session's tryLoad() would match by
    // its (intact) header+text and then blow up decompressing. rename(2) is atomic on the same
    // filesystem: readers only ever see a complete checkpoint or none. A leftover .tmp from a
    // crash is swept by the constructor and skipped by listEntries' SHA-name filter.
    const path = this.pathForSha(snap.sha);
    const tmpPath = path + TMP_SUFFIX;
    const MAX_SINGLE = (1 << 31) - 1; // fs.write rejects a single call larger than this
    const fd = openSync(tmpPath, 'w');
    try {
      let position = 0;
      for (const buf of snap.buffers) {
        if (buf.length <= MAX_SINGLE) {
          await KVColdStore.writeOnce(fd, buf, position); // one syscall, overlaps decode
          position += buf.length;
        } else {
          const CHUNK = 1 << 30;
          for (let off = 0; off < buf.length; off += CHUNK) {
            const len = Math.min(CHUNK, buf.length - off);
            await KVColdStore.writeOnce(fd, buf.subarray(off, off + len), position);
            position += len;
          }
        }
      }
    } catch (err) {
      closeSync(fd);
      try { unlinkSync(tmpPath); } catch { /* best-effort */ }
      throw err;
    }
    closeSync(fd);
    renameSync(tmpPath, path);
  }

  /** Promisified single fs.write (runs the syscall on libuv's threadpool). */
  private static writeOnce(fd: number, buf: Buffer, position: number): Promise<void> {
    return new Promise((resolve, reject) => {
      fsWrite(fd, buf, 0, buf.length, position, err => (err ? reject(err) : resolve()));
    });
  }

  /** Promisified zstd compression (zlib dispatches the work to libuv's threadpool). */
  private static zstdAsync(buf: Buffer): Promise<Buffer> {
    return new Promise((resolve, reject) => {
      zstdCompress(buf, (err, out) => (err ? reject(err) : resolve(out)));
    });
  }

  /**
   * Persist `model`'s current KV state synchronously (snapshot + blocking shuffle + compress +
   * write). Not called anywhere in production (AgentLoop always uses `snapshot()` +
   * `writeSnapshot()` for the overlapped path) — kept as a documented, fully-blocking
   * alternative for a caller with no concurrent decode to protect; the plain (non-chunked)
   * `shuffle2` is fine here for exactly that reason.
   */
  store(model: BaseModel, promptText: string, tokens: number, reason: CheckpointReason = 'cold'): void {
    const snap = this.snapshot(model, promptText, tokens, reason);
    const [header, textBuf, kvBuf] = snap.buffers;
    const compressed = zstdCompressSync(shuffle2(kvBuf));
    header.writeUInt8(COMPRESSION_ZSTD_SHUFFLE, OFF_COMPRESSION);
    this.evictIfNeeded(header.length + textBuf.length + compressed.length);
    // Same tmp-then-rename discipline as writeSnapshot() — atomicity is a property of the
    // store's on-disk format, not of one write path.
    const path = this.pathForSha(snap.sha);
    KVColdStore.writeChunked(path + TMP_SUFFIX, [header, textBuf, compressed]);
    renameSync(path + TMP_SUFFIX, path);
  }

  /**
   * Write `buffers` concatenated to `path`, in chunks under 2^31-1 bytes per call.
   * `fs.writeSync`'s underlying `write()` syscall rejects a single call larger than that —
   * a real limit hit while validating this against a multi-GiB KV state (large context
   * sizes produce checkpoints well past 2 GiB). `Buffer.concat`/`readFileSync` don't have
   * this restriction (Node chunks reads internally), only the write side does.
   */
  private static writeChunked(path: string, buffers: readonly Buffer[]): void {
    const MAX_CHUNK = 1 << 30; // 1 GiB — comfortably under the 2^31-1 syscall limit
    const fd = openSync(path, 'w');
    try {
      for (const buf of buffers) {
        for (let offset = 0; offset < buf.length; offset += MAX_CHUNK) {
          const end = Math.min(offset + MAX_CHUNK, buf.length);
          writeSync(fd, buf, offset, end - offset);
        }
      }
    } finally {
      closeSync(fd);
    }
  }

  /**
   * Find the stored checkpoint whose text is the longest prefix of `promptText` and
   * restore it into `model`. Returns `{ matchedChars: 0, matchedTokens: 0 }` if no usable
   * checkpoint was found (model state is unchanged).
   *
   * `matchedTokens` is the authoritative number to act on — it's the token count the caller
   * passed to `store()` at checkpoint time, not something re-derived from character
   * positions. Char-level prefix matching only decides *which* checkpoint to use; token
   * boundaries in a BPE tokenizer don't reliably align with arbitrary character cut points,
   * so re-tokenizing `promptText.slice(0, matchedChars)` standalone and trusting its length
   * would be unsound. Callers should still re-tokenize the *remaining* text
   * (`promptText.slice(matchedChars)`) and feed only that to `decodeAppend()` — never assume
   * `matchedChars` alone tells you where to split tokens.
   */
  tryLoad(model: BaseModel, promptText: string): LoadResult {
    let bestSha: string | null = null;
    let bestText = '';
    for (const entry of this.listEntries()) {
      // Read ONLY header + prompt text (a few KB) — checkpoints are hundreds of MB, and this
      // scan used to readFileSync() every one of them in full on every tryLoad(). With the
      // M3.7 ladder writing several rungs per prompt that was gigabytes of redundant I/O per
      // start(). (Found by the 2026-07 whole-project audit.)
      const text = KVColdStore.readPromptText(entry.path);
      if (text === null) continue; // evicted/corrupted between listing and reading — skip
      if (promptText.startsWith(text) && text.length > bestText.length) {
        bestSha = entry.sha;
        bestText = text;
      }
    }
    if (!bestSha) return { matchedChars: 0, matchedTokens: 0 };

    // 2026-07 KV audit: the whole load is guarded. Before this, a corrupt/truncated checkpoint
    // (legacy pre-atomic-write leftovers, bitrot, disk-full) made tryLoad THROW — and since the
    // same prompt prefix re-matches the same file every session, ONE bad file failed every
    // subsequent start() until someone deleted it by hand. A checkpoint is a cache entry:
    // unreadable ⇒ unlink it and report no-match (the caller falls back to a full prefill,
    // which clears whatever a partial kvRestore may have left in the KV — see prefillOrResume).
    try {
      const buf = readFileSync(this.pathForSha(bestSha));
      const textBytes = buf.readUInt32LE(32);
      const matchedTokens = buf.readUInt32LE(8);
      let kvBuf: Buffer = buf.subarray(HEADER_FIXED + textBytes);
      // v2 files may carry a compressed payload; v1 files are always raw. Two v2 encodings
      // coexist on disk: legacy plain zstd (M3.6, no longer written) and shuffle2+zstd (the
      // follow-up, written since 2026-07-06) — both stay readable indefinitely, no migration.
      if (buf.readUInt8(4) >= 2) {
        const compression = buf.readUInt8(OFF_COMPRESSION);
        if (compression === COMPRESSION_ZSTD) {
          kvBuf = zstdDecompressSync(kvBuf);
        } else if (compression === COMPRESSION_ZSTD_SHUFFLE) {
          kvBuf = unshuffle2(zstdDecompressSync(kvBuf));
        }
      }
      // Node's zstdDecompressSync does NOT throw on a truncated frame — it returns an EMPTY
      // buffer (verified 2026-07-07). A real KV state is never empty, so treat that as the
      // corruption it is rather than handing llama_state_set_data a zero-byte blob.
      if (kvBuf.length === 0) throw new Error('empty/truncated KV payload');
      model.kvRestore(new Uint8Array(kvBuf.buffer, kvBuf.byteOffset, kvBuf.byteLength));
      this.touch(bestSha, buf);
      return { matchedChars: bestText.length, matchedTokens };
    } catch (err) {
      console.warn(
        `[coldStore] dropping unreadable checkpoint ${bestSha}: ${(err as Error).message}`,
      );
      try { unlinkSync(this.pathForSha(bestSha)); } catch { /* already gone — fine */ }
      return { matchedChars: 0, matchedTokens: 0 };
    }
  }

  /**
   * Bump hit count + lastUsed on a checkpoint that was just loaded. Writes only the two
   * small header fields in place (`pwrite`-style, via a seek+write) rather than rewriting
   * the whole file — `buf` may be a multi-GiB checkpoint; touching it on every load should
   * cost a few bytes of I/O, not a full file rewrite.
   */
  private touch(sha: string, buf: Buffer): void {
    const hits = buf.readUInt32LE(12);
    const hitsBuf = Buffer.alloc(4);
    hitsBuf.writeUInt32LE(hits + 1, 0);
    const lastUsedBuf = Buffer.alloc(8);
    lastUsedBuf.writeDoubleLE(Date.now(), 0);

    const fd = openSync(this.pathForSha(sha), 'r+');
    try {
      writeSync(fd, hitsBuf, 0, 4, 12);
      writeSync(fd, lastUsedBuf, 0, 8, 24);
    } finally {
      closeSync(fd);
    }
  }

  /** Read just the fixed header of a checkpoint file (36 bytes), or null if unreadable. */
  private static readHeader(path: string): Buffer | null {
    let fd: number;
    try {
      fd = openSync(path, 'r');
    } catch {
      return null;
    }
    try {
      const header = Buffer.alloc(HEADER_FIXED);
      if (readSync(fd, header, 0, HEADER_FIXED, 0) !== HEADER_FIXED) return null;
      if (header.toString('ascii', 0, 4) !== HEADER_MAGIC) return null;
      return header;
    } catch {
      return null;
    } finally {
      closeSync(fd);
    }
  }

  /** Read only the stored prompt text of a checkpoint (header + textBytes), or null. */
  private static readPromptText(path: string): string | null {
    let fd: number;
    try {
      fd = openSync(path, 'r');
    } catch {
      return null;
    }
    try {
      const header = Buffer.alloc(HEADER_FIXED);
      if (readSync(fd, header, 0, HEADER_FIXED, 0) !== HEADER_FIXED) return null;
      if (header.toString('ascii', 0, 4) !== HEADER_MAGIC) return null;
      const textBytes = header.readUInt32LE(32);
      const textBuf = Buffer.alloc(textBytes);
      if (readSync(fd, textBuf, 0, textBytes, HEADER_FIXED) !== textBytes) return null;
      return textBuf.toString('utf-8');
    } catch {
      return null;
    } finally {
      closeSync(fd);
    }
  }

  private listEntries(): ColdStoreEntry[] {
    if (!existsSync(this.dir)) return [];
    const out: ColdStoreEntry[] = [];
    for (const name of readdirSync(this.dir)) {
      if (!SHA_NAME_RE.test(name)) continue;
      const path = this.pathForSha(name);
      try {
        // Header-only read (36 bytes) — checkpoints are hundreds of MB and this listing runs
        // on every tryLoad() AND every evictIfNeeded() (i.e. every checkpoint write).
        const buf = KVColdStore.readHeader(path);
        if (!buf) continue;
        out.push({
          sha: name,
          path,
          tokens: buf.readUInt32LE(8),
          hits: buf.readUInt32LE(12),
          createdAt: buf.readDoubleLE(16),
          lastUsed: buf.readDoubleLE(24),
          fileSize: statSync(path).size,
          reason: buf.readUInt8(5) === 0 ? 'cold' : 'continued',
        });
      } catch {
        // Unreadable/corrupt/evicted mid-scan — skip rather than fail the whole listing.
      }
    }
    return out;
  }

  /** Same formula as ds4_kvstore_entry_eviction_score — see the doc cited at the top. */
  private score(e: ColdStoreEntry, now: number): number {
    const elapsedMs = Math.max(0, now - (e.lastUsed || e.createdAt));
    let effectiveHits = e.hits * 2 ** (-elapsedMs / HIT_HALF_LIFE_MS);
    if (effectiveHits < MIN_EFFECTIVE_HITS) effectiveHits = 0;
    let score = ((effectiveHits + 1) * e.tokens) / Math.max(1, e.fileSize);
    if (e.reason === 'cold') score *= ANCHOR_REASON_SCORE_FACTOR;
    return score;
  }

  private evictIfNeeded(extraBytes: number): void {
    if (this.budgetBytes <= 0) return;
    const entries = this.listEntries();
    let total = entries.reduce((sum, e) => sum + e.fileSize, 0);
    const target = this.budgetBytes - extraBytes;
    if (total <= target) return;

    const now = Date.now();
    const byScoreAsc = entries
      .map((e) => ({ e, s: this.score(e, now) }))
      .sort((a, b) => a.s - b.s);

    for (const { e } of byScoreAsc) {
      if (total <= target) break;
      try {
        unlinkSync(e.path);
        total -= e.fileSize;
      } catch {
        // Already gone — fine, keep evicting the next-lowest score.
      }
    }
  }
}
