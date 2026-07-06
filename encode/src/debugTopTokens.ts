/**
 * Debug script: prints top-20 tokens by logit for a few failing cases.
 * Run from earendil-pi root:
 *   npx tsx packages/agent/src/preprocessor/debugTopTokens.ts
 */
import { initSmallModel, getSmallModel } from './smallModel.js';

const MODEL_PATH =
  process.env.SUBVOCAL_SMALL_MODEL ??
  '/mnt/dati_cachy/LLM/lmstudio-community/unsloth-gemma-4-E2B-it-qat-GGUF/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf';

const INTENT_SYSTEM_PROMPT =
  'Classify the user request into exactly one of these labels:\n' +
  'BUGFIX REFACTOR EXPLAIN ADD_FEATURE WRITE_TEST UNKNOWN\n' +
  'Output only the label, nothing else.';

function buildPrompt(userPrompt: string): string {
  return (
    `<start_of_turn>user\n${INTENT_SYSTEM_PROMPT}\n\n${userPrompt}<end_of_turn>\n` +
    `<start_of_turn>model\nIntent:\n`
  );
}

const FAILING_CASES = [
  { prompt: 'Extract the email validation logic into a separate util', expected: 'REFACTOR' },
  { prompt: 'Rename all occurrences of `usr` to `user` throughout the codebase', expected: 'REFACTOR' },
  { prompt: 'Semplifica questa funzione, è troppo lunga', expected: 'REFACTOR' },
  { prompt: 'Split this 200-line class into smaller components', expected: 'REFACTOR' },
  { prompt: 'Aggiungi la logica di sconto per clienti premium', expected: 'ADD_FEATURE' },
];

async function main() {
  initSmallModel({ modelPath: MODEL_PATH, contextSize: 512, threads: 8, gpuLayers: 0 });
  const model = getSmallModel();

  for (const { prompt, expected } of FAILING_CASES) {
    const tokens = model.tokenize(buildPrompt(prompt), true, true);
    const status = model.forward(tokens);
    if (status !== 0) throw new Error(`forward() returned ${status}`);

    const logits = model.getLogitsFast();

    // Find top-20 by brute force
    const indexed = Array.from(logits).map((v, i) => ({ v, i }));
    indexed.sort((a, b) => b.v - a.v);
    const top20 = indexed.slice(0, 20);

    console.log(`\n=== "${prompt}" (expected: ${expected}) ===`);
    for (let rank = 0; rank < top20.length; rank++) {
      const { v, i } = top20[rank];
      const text = model.detokenize(Int32Array.from([i]));
      const repr = JSON.stringify(text);
      console.log(`  #${rank + 1} id=${i} logit=${v.toFixed(3)} text=${repr}`);
    }
  }

  process.exit(0);
}

main().catch(e => { console.error(e); process.exit(1); });
