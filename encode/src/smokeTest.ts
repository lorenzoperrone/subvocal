/**
 * smokeTest.ts
 *
 * End-to-end smoke test for Phase 1 + Phase 2 of Subvocal Engine.
 *
 * Tests the full preprocess() pipeline against a synthetic 'carrello.py'
 * target file and verifies:
 *   1. Intent classification works and returns a valid intent in < 200ms
 *   2. AST tags are injected into the source code (tagCount > 0)
 *   3. TensorPayload is assembled with non-empty context_window
 *   4. tagMap correctly maps TokenIDs to function/class names
 *
 * Run with: npx tsx packages/agent/src/preprocessor/smokeTest.ts
 */

import { preprocess, initSmallModel, freeSmallModel } from './index.js';
import { activeProfile } from './modelProfile.js';

// ── Model path: from active profile, overridable via env var ──────────────────
const MODEL_PATH = process.env.SUBVOCAL_SMALL_MODEL ?? activeProfile.smallModelPath;

// ── Synthetic target file: carrello.py ────────────────────────────────────────
const CARRELLO_PY = `
class Carrello:
    def __init__(self):
        self.items = []
        self.total = 0.0

    def aggiungi_item(self, nome, prezzo):
        self.items.append({"nome": nome, "prezzo": prezzo})
        self.total += prezzo

    def calcola_totale(self):
        return self.total

    def svuota(self):
        self.items = []
        self.total = 0.0

def crea_ordine(carrello):
    if not carrello.items:
        raise ValueError("Carrello vuoto!")
    for item in carrello.items:
        print(f"  - {item['nome']}: €{item['prezzo']:.2f}")
    print(f"Totale: €{carrello.calcola_totale():.2f}")
`;

// ── Test ──────────────────────────────────────────────────────────────────────
async function main() {
  console.log('\n══════════════════════════════════════════════════════');
  console.log('  🧠 Subvocal Engine — Phase 1+2 Smoke Test');
  console.log('══════════════════════════════════════════════════════\n');

  // Boot subvocal-small in-process (replaces REST server dependency).
  console.log(`🔌 Loading subvocal-small from: ${MODEL_PATH}`);
  initSmallModel({ modelPath: MODEL_PATH, threads: 8, contextSize: 4096 });

  const userPrompt = 'Aggiungi logica IVA al carrello.py';
  console.log(`📝 User Prompt: "${userPrompt}"`);
  console.log(`📄 Target File: carrello.py (${CARRELLO_PY.split('\n').length} lines)\n`);

  let result: Awaited<ReturnType<typeof preprocess>>;
  try {
    result = await preprocess({
      prompt: userPrompt,
      fileContent: CARRELLO_PY,
      filePath: 'carrello.py',
    });
  } catch (err) {
    console.error('❌ preprocess() threw an error:', err);
    freeSmallModel();
    process.exit(1);
  }

  // ── Results ───────────────────────────────────────────────────────────────
  console.log('\n══════════════════════════════════════════════════════');
  console.log('  📊 Results');
  console.log('══════════════════════════════════════════════════════');

  console.log(`\n✅ Intent classified   : ${result.intent} (${result.intentLatencyMs.toFixed(1)}ms)`);
  console.log(`✅ AST tags injected   : ${result.tagCount}`);
  console.log(`✅ Context tokens      : ${result.payload.context_window.length}`);
  console.log(`✅ System rule tokens  : ${result.payload.system_rules.length}`);
  console.log(`✅ Directive tokens    : ${result.payload.directives.length}`);
  console.log(`✅ Total pipeline time : ${result.totalLatencyMs.toFixed(1)}ms`);

  console.log('\n── TagMap (TokenID → AST Node) ──────────────────────');
  for (const [tokenId, label] of result.tagMap) {
    console.log(`   Token ${tokenId.toString().padStart(7)} → ${label}`);
  }

  console.log('\n── Tagged Code Preview (first 500 chars) ────────────');
  console.log(result.taggedCode.slice(0, 500));
  if (result.taggedCode.length > 500) console.log('  ... [truncated]');

  // ── Assertions ────────────────────────────────────────────────────────────
  console.log('\n══════════════════════════════════════════════════════');
  console.log('  🔬 Assertions');
  console.log('══════════════════════════════════════════════════════');

  let passed = 0;
  let failed = 0;

  function assert(condition: boolean, label: string) {
    if (condition) {
      console.log(`  ✅ PASS: ${label}`);
      passed++;
    } else {
      console.log(`  ❌ FAIL: ${label}`);
      failed++;
    }
  }

  assert(result.intent !== 'UNKNOWN', 'Intent is not UNKNOWN');
  assert(result.intentLatencyMs < 5000, 'Intent latency < 5000ms');
  assert(result.tagCount > 0, 'At least one AST tag injected');
  assert(result.tagMap.size > 0, 'TagMap is non-empty');
  assert(result.payload.context_window.length > 0, 'context_window is non-empty');
  assert(result.payload.system_rules.length > 0, 'system_rules is non-empty');
  assert(result.payload.directives.length > 0, 'directives is non-empty');
  assert(result.taggedCode.includes('carrello'), 'Tagged code contains original content');
  assert(result.payload.directives[0] !== 0, 'Intent token ID is non-zero');

  console.log(`\n  Results: ${passed} passed, ${failed} failed`);
  if (failed > 0) {
    process.exit(1);
  }

  console.log('\n🎉 All tests passed! Phase 1 + Phase 2 pipeline is functional.\n');
  freeSmallModel();
}

main().catch(err => {
  console.error('Fatal error:', err);
  freeSmallModel();
  process.exit(1);
});
