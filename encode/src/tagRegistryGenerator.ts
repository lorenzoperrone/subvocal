/**
 * tagRegistryGenerator.ts
 *
 * Generates TagRegistry.<profile>.json — the list of Unicode characters that
 * tokenize as a SINGLE token in a given model family's vocabulary.
 *
 * These single-token "ideogram" characters are used as AST structural markers
 * injected before key nodes (class, function, for, etc.) so the GPU model can
 * refer to code positions by token ID rather than text span.
 *
 * Usage (generates registry for the active profile via FFI):
 *   SUBVOCAL_MODEL_PROFILE=gemma4 npx tsx tagRegistryGenerator.ts
 *   SUBVOCAL_MODEL_PROFILE=qwen3  npx tsx tagRegistryGenerator.ts
 *
 * The output path is determined by activeProfile.tagRegistryPath.
 * For Qwen3 this produces TagRegistry.json (legacy name, kept for compatibility).
 * For Gemma4 this produces TagRegistry.gemma4.json.
 */

import * as fs from 'fs';
import { activeProfile } from './modelProfile.js';
import { initSmallModel, getSmallModel } from './smallModel.js';

// ── Candidate characters ──────────────────────────────────────────────────────

// Build candidate list programmatically — covers all Unicode blocks likely to
// have single-token entries in a multilingual vocab like Gemma 4 (262k tokens).
function buildCandidates(): string[] {
  const chars: string[] = [];
  const seen = new Set<number>();
  function add(cp: number) {
    if (!seen.has(cp)) { seen.add(cp); chars.push(String.fromCodePoint(cp)); }
  }

  // ── Math & symbols (U+2200–U+2BFF) ────────────────────────────────────────
  for (let cp = 0x2200; cp <= 0x2BFF; cp++) add(cp);

  // ── Greek & Coptic (U+0370–U+03FF) ────────────────────────────────────────
  for (let cp = 0x0370; cp <= 0x03FF; cp++) add(cp);

  // ── Latin Extended (U+00C0–U+024F) ────────────────────────────────────────
  for (let cp = 0x00C0; cp <= 0x024F; cp++) add(cp);

  // ── Cyrillic (U+0400–U+04FF) ──────────────────────────────────────────────
  for (let cp = 0x0400; cp <= 0x04FF; cp++) add(cp);

  // ── Hiragana (U+3040–U+309F) ──────────────────────────────────────────────
  for (let cp = 0x3040; cp <= 0x309F; cp++) add(cp);

  // ── Katakana (U+30A0–U+30FF) ──────────────────────────────────────────────
  for (let cp = 0x30A0; cp <= 0x30FF; cp++) add(cp);

  // ── CJK Unified Ideographs — frequenti (U+4E00–U+6FFF) ───────────────────
  for (let cp = 0x4E00; cp <= 0x6FFF; cp++) add(cp);
  // ── CJK Unified — meno comuni (U+7000–U+9FFF) ────────────────────────────
  for (let cp = 0x7000; cp <= 0x9FFF; cp++) add(cp);

  // ── CJK Compatibility & Extension A (U+3400–U+4DBF) ──────────────────────
  for (let cp = 0x3400; cp <= 0x4DBF; cp++) add(cp);

  // ── Hangul Syllables — prime 2000 (U+AC00–U+B3FF) ────────────────────────
  // 11.172 sillabe totali, sample le più comuni
  for (let cp = 0xAC00; cp <= 0xB3FF; cp++) add(cp);

  // ── CJK Symbols & Punctuation (U+3000–U+303F) ────────────────────────────
  for (let cp = 0x3000; cp <= 0x303F; cp++) add(cp);

  // ── Bopomofo (U+02EA–U+02EB, U+3105–U+312F) ──────────────────────────────
  for (let cp = 0x3105; cp <= 0x312F; cp++) add(cp);

  // ── Arrows (U+2190–U+21FF, U+27F0–U+27FF, U+2900–U+297F) ────────────────
  for (let cp = 0x2190; cp <= 0x21FF; cp++) add(cp);
  for (let cp = 0x27F0; cp <= 0x27FF; cp++) add(cp);
  for (let cp = 0x2900; cp <= 0x297F; cp++) add(cp);

  // ── Box Drawing & Blocks (U+2500–U+259F) ─────────────────────────────────
  for (let cp = 0x2500; cp <= 0x259F; cp++) add(cp);

  // ── Geometric Shapes (U+25A0–U+25FF) ─────────────────────────────────────
  for (let cp = 0x25A0; cp <= 0x25FF; cp++) add(cp);

  // ── Misc Symbols (U+2600–U+26FF) ─────────────────────────────────────────
  for (let cp = 0x2600; cp <= 0x26FF; cp++) add(cp);

  // ── Dingbats (U+2700–U+27BF) ─────────────────────────────────────────────
  for (let cp = 0x2700; cp <= 0x27BF; cp++) add(cp);

  // ── Emoji common (U+1F300–U+1F9FF) ───────────────────────────────────────
  for (let cp = 0x1F300; cp <= 0x1F9FF; cp++) add(cp);

  return chars;
}

const CANDIDATE_CHARS = buildCandidates();

interface TagEntry {
  char: string;
  tokenId: number;
}

async function main() {
  console.log(`🚀 TagRegistry generator — profile: ${activeProfile.name}`);
  console.log(`📂 Output: ${activeProfile.tagRegistryPath}`);
  console.log(`🧠 Loading model: ${activeProfile.smallModelPath}\n`);

  initSmallModel({
    modelPath: activeProfile.smallModelPath,
    ...activeProfile.smallOpts,
  });

  const model = getSmallModel();
  const registry: TagEntry[] = [];
  const uniqueTokens = new Set<number>();

  for (const char of CANDIDATE_CHARS) {
    // tokenize(text, addBos=false, parseSpecial=false) — raw char, no BOS
    const tokens = model.tokenize(char, false, false);
    if (tokens.length === 1) {
      const tokenId = tokens[0];
      if (!uniqueTokens.has(tokenId)) {
        uniqueTokens.add(tokenId);
        registry.push({ char, tokenId });
        console.log(`  ✅ '${char}' -> ${tokenId}`);
      } else {
        console.log(`  ⚠️  '${char}' -> ${tokenId} (duplicate, skipped)`);
      }
    } else {
      console.log(`  ❌ '${char}' -> ${tokens.length} tokens (split, skipped)`);
    }
  }

  fs.writeFileSync(activeProfile.tagRegistryPath, JSON.stringify(registry, null, 2), 'utf-8');
  console.log(`\n🎉 Done — ${registry.length} single-token ideograms saved to ${activeProfile.tagRegistryPath}`);

  process.exit(0);
}

main().catch(err => {
  console.error('Fatal:', err);
  process.exit(1);
});
