/**
 * Quick script: show token IDs for all intent word variants.
 *   npx tsx packages/agent/src/preprocessor/debugTokenIds.ts
 */
import { initSmallModel, getSmallModel } from './smallModel.js';

const MODEL_PATH =
  process.env.SUBVOCAL_SMALL_MODEL ??
  '/mnt/dati_cachy/LLM/lmstudio-community/unsloth-gemma-4-E2B-it-qat-GGUF/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf';

const VARIANTS: Record<string, string[]> = {
  BUGFIX:      ['BUGFIX', 'Bugfix', 'bugfix', 'BUG', 'Bug', 'bug'],
  REFACTOR:    ['REFACTOR', 'Refactor', 'refactor', 'REF', 'ref'],
  EXPLAIN:     ['EXPLAIN', 'Explain', 'explain', 'EXPL', 'Expl'],
  ADD_FEATURE: ['ADD_FEATURE', 'Add_feature', 'ADD', 'Add', 'add'],
  WRITE_TEST:  ['WRITE_TEST', 'Write_test', 'WRITE', 'Write', 'write', 'TEST', 'Test', 'test'],
  UNKNOWN:     ['UNKNOWN', 'Unknown', 'unknown', 'UNK', 'Unk', 'unk'],
};

async function main() {
  initSmallModel({ modelPath: MODEL_PATH, contextSize: 512, threads: 4, gpuLayers: 0 });
  const model = getSmallModel();

  for (const [intent, variants] of Object.entries(VARIANTS)) {
    console.log(`\n${intent}:`);
    for (const word of variants) {
      const ids = model.tokenize(word, false, false);
      const back = model.detokenize(ids as unknown as Int32Array);
      const single = ids.length === 1 ? `id=${ids[0]}` : `multi(${ids.join(',')})`;
      console.log(`  "${word}" → [${single}] → detok="${back}"`);
    }
  }

  process.exit(0);
}

main().catch(e => { console.error(e); process.exit(1); });
