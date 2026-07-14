/**
 * lineLevelCRC.ts
 *
 * Substory 1.3: Line-Level CRC (MurmurHash3)
 *
 * Provides ultra-fast 32-bit hashing for source-code lines (or AST nodes)
 * so the harness can detect which lines changed between turns and avoid
 * invalidating the entire KV cache.
 */

/**
 * MurmurHash3 32-bit implementation.
 * @param key   Input string to hash.
 * @param seed  Optional seed (default 0).
 * @returns     32-bit unsigned integer hash.
 */
export function murmurHash3(key: string, seed = 0): number {
  let h1 = seed >>> 0;
  const c1 = 0xcc9e2d47;
  const c2 = 0x1b873593;
  const r1 = 15;
  const r2 = 13;
  const m = 5;
  const n = 0xe6546b64;

  let i = 0;
  let k: number;

  // Process 4-byte chunks
  const len = key.length;
  while (i + 4 <= len) {
    k =
      (key.charCodeAt(i) & 0xff) |
      ((key.charCodeAt(i + 1) & 0xff) << 8) |
      ((key.charCodeAt(i + 2) & 0xff) << 16) |
      ((key.charCodeAt(i + 3) & 0xff) << 24);

    k = (k * c1) >>> 0;
    k = ((k << r1) | (k >>> (32 - r1))) >>> 0;
    k = (k * c2) >>> 0;

    h1 ^= k;
    h1 = ((h1 << r2) | (h1 >>> (32 - r2))) >>> 0;
    h1 = (h1 * m + n) >>> 0;

    i += 4;
  }

  // Tail bytes
  k = 0;
  switch (len & 3) {
    case 3:
      k ^= (key.charCodeAt(i + 2) & 0xff) << 16;
    // biome-ignore lint/suspicious/noFallthroughSwitchClause: intentional fallthrough
    case 2:
      k ^= (key.charCodeAt(i + 1) & 0xff) << 8;
    // biome-ignore lint/suspicious/noFallthroughSwitchClause: intentional fallthrough
    case 1:
      k ^= key.charCodeAt(i) & 0xff;
      k = (k * c1) >>> 0;
      k = ((k << r1) | (k >>> (32 - r1))) >>> 0;
      k = (k * c2) >>> 0;
      h1 ^= k;
      break;
  }

  // Finalisation mix
  h1 ^= len;

  h1 ^= h1 >>> 16;
  h1 = (h1 * 0x85ebca6b) >>> 0;
  h1 ^= h1 >>> 13;
  h1 = (h1 * 0xc2b2ae35) >>> 0;
  h1 ^= h1 >>> 16;

  return h1 >>> 0;
}

/**
 * Result of a line-hash snapshot for a file.
 */
export interface LineHashSnapshot {
  /** per-line 32-bit hashes, index 0 = line 1 */
  lineHashes: Uint32Array;
  /** overall file hash (XOR fold) */
  fileHash: number;
}

/**
 * Build a snapshot of 32-bit hashes for every line in a file.
 * Empty lines ARE hashed so that insertion / deletion is detectable.
 *
 * @param content  Raw file content.
 * @returns        Snapshot with line hashes and foldedpresso file hash.
 */
export function buildLineHashSnapshot(content: string): LineHashSnapshot {
  let lineCount = 1;
  for (let i = 0; i < content.length; i++) {
    if (content.charCodeAt(i) === 10) lineCount++;
  }

  const lineHashes = new Uint32Array(lineCount);
  let fileHash = 0;
  let lineStart = 0;
  let lineIdx = 0;

  for (let i = 0; i <= content.length; i++) {
    if (i === content.length || content.charCodeAt(i) === 10) {
      let end = i;
      if (end > lineStart && content.charCodeAt(end - 1) === 13) {
        end--;
      }
      const lineStr = content.slice(lineStart, end);
      const h = murmurHash3(lineStr, lineIdx);
      lineHashes[lineIdx++] = h;
      fileHash ^= h;
      lineStart = i + 1;
    }
  }

  return { lineHashes, fileHash };
}

/**
 * Describe the delta between two snapshots of the *same* file.
 * @returns  ranges of line indices (0-based) that changed.
 */
export function computeLineDelta(
  oldSnap: LineHashSnapshot,
 expanded: LineHashSnapshot,
): { changedRanges: Array<{ start: number; end: number }>; isStructureChanged: boolean } {
  const maxLen = Math.max(oldSnap.lineHashes.length, expanded.lineHashes.length);
  const changedRanges: Array<{ start: number; end: number }> = [];
  let inRange = false;
  let rangeStart = 0;

  for (let i = 0; i < maxLen; i++) {
    const oldH = oldSnap.lineHashes[i] ?? 0;
    const newH = expanded.lineHashes[i] ?? 0;

    if (oldH !== newH) {
      if (!inRange) {
        inRange = true;
        rangeStart = i;
      }
    } else if (inRange) {
      changedRanges.push({ start: rangeStart, end: i - 1 });
      inRange = false;
    }
  }

  if (inRange) {
    changedRanges.push({ start: rangeStart, end: maxLen - 1 });
  }

const isStructureChanged = oldSnap.lineHashes.length !== expanded.lineHashes.length;

	return { changedRanges, isStructureChanged };
}
