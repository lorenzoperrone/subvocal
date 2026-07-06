/**
 * intentBenchSharedContext.ts
 *
 * M11.3 — bench BEFORE any product wiring (see doc/substories/
 * M11.3-intent-classifier-shared-e2b.md). Question: does classifying intent on the SHARED
 * E2B context hold M11.2's measured standalone accuracy, or does causal attention over a live
 * conversation prefix corrupt the intent logits? Two variants, both measured on the real
 * intentBench dataset against the real E2B model:
 *
 * **Variant (a) — prefix-attending** (`decodeAppendAsync` onto seq 0's live prefix, then
 * `kvCacheSeqRemove`/`resetNPast` rollback — the same mechanics AgentLoop's drafter rollback
 * already uses in production). Measured 2026-07-06: **NO-GO**. Standalone (prefix=0) 75.0%,
 * degrading to 62.5% at a 2048-token prefix and 57.5% at 8192 — both far past the ~3pp
 * tolerance this bench was gated on, and WORSENING with conversation length (backwards from
 * what you'd want, since intent matters most on long followUp turns).
 *
 * **Variant (b) — isolated seq_id** (`decodeAppendSeq` on a SEPARATE sequence, seq_id=1,
 * attention-isolated from seq 0 by construction — requires `ModelOptions.auxSeq` and the
 * native binding's `decodeAppendSeq`, both added for this story). Measured 2026-07-06: **GO**.
 * 75.0% / 75.0% / 75.0% at prefix 0 / 2048 / 8192 — byte-identical miss list at every length.
 * A separate isolation test (plant a secret fact in seq 0, classify an unrelated prompt on
 * seq 1, confirm no leakage; confirm seq 0 decodes cleanly afterward) also passed.
 *
 * This closes the story's step 1 (bench). Steps 2-4 (wire into the TUI's `getE2B()` shared
 * instance, flip the classify path off regex, monitor-log the source, live dual-brain+drafter
 * session validation) are separate, further work — not done here.
 *
 * Run with:
 *   npx tsx encode/src/intentBenchSharedContext.ts
 */

import { ModelGPU } from '@subvocal/synapse';
import { activeProfile } from './modelProfile.js';
import { BENCH_DATASET } from './intentBenchDataset.js';
import type { Intent } from './intentRouter.js';

const INTENTS: Intent[] = ['BUGFIX', 'REFACTOR', 'EXPLAIN', 'ADD_FEATURE', 'WRITE_TEST', 'UNKNOWN'];
const PREFIX_LENGTHS = [0, 2048, 8192];
/** GO/NO-GO bar this story set: shared-context accuracy must stay within this many percentage
 *  points of the standalone (prefix=0) baseline. */
const TOLERANCE_PP = 3;

function bestIntentFromLogits(logits: Float32Array): Intent {
  let bestIntent: Intent = 'UNKNOWN';
  let bestLogit = -Infinity;
  for (const intent of INTENTS) {
    const ids = activeProfile.intentAnchors[intent] ?? [];
    const v = ids.reduce((best, id) => Math.max(best, logits[id] ?? -Infinity), -Infinity);
    if (v > bestLogit) { bestLogit = v; bestIntent = intent; }
  }
  return bestIntent;
}

function buildClassifyPrompt(userPrompt: string): string {
  const build = activeProfile.buildSmallPrompt?.bind(activeProfile) ?? activeProfile.buildPrompt;
  return build({
    systemPrompt:
      'Classify the user request into exactly one of these labels:\n' +
      'BUGFIX REFACTOR EXPLAIN ADD_FEATURE WRITE_TEST UNKNOWN\n' +
      'Output only the label, nothing else.',
    userPrompt,
    prefill: activeProfile.intentPrefill,
  });
}

/** Standalone baseline: forward() clears the KV — exactly M11.2's original measured path. */
function classifyStandalone(model: ModelGPU, userPrompt: string): Intent {
  const tokens = model.tokenize(buildClassifyPrompt(userPrompt), true, true);
  const status = model.forward(tokens);
  if (status !== 0) throw new Error(`forward failed: ${status}`);
  return bestIntentFromLogits(model.getLogitsFast());
}

/**
 * Variant (a): classify via decodeAppendAsync onto whatever seq 0's KV holds at `pos`, then
 * roll back to `pos` so the caller's real conversation state is untouched. Mirrors AgentLoop's
 * draftRollback() mechanics exactly (kvCacheSeqRemove + resetNPast). MEASURED NO-GO.
 */
async function classifyPrefixAttending(model: ModelGPU, pos: number, userPrompt: string): Promise<Intent> {
  const tokens = model.tokenize(buildClassifyPrompt(userPrompt), true, true);
  await model.decodeAppendAsync(tokens);
  const intent = bestIntentFromLogits(model.getLogitsFast());
  model.kvCacheSeqRemove(0, pos, -1);
  model.resetNPast(pos);
  return intent;
}

/**
 * Variant (b): classify on an ISOLATED sequence (seq_id=1, position 0 every time — fresh each
 * call, nothing to persist), then wipe it. Requires the model constructed with
 * ModelOptions.auxSeq: true. MEASURED GO — attention isolation holds accuracy regardless of
 * seq 0's content.
 */
async function classifyIsolatedSeq(model: ModelGPU, userPrompt: string): Promise<Intent> {
  const tokens = model.tokenize(buildClassifyPrompt(userPrompt), true, true);
  const status = await model.decodeAppendSeq(tokens, 1, 0);
  if (status !== 0) throw new Error(`decodeAppendSeq failed: ${status}`);
  const intent = bestIntentFromLogits(model.getLogitsFast());
  model.kvCacheSeqRemove(1, 0, -1);
  return intent;
}

/** Synthetic conversation-shaped filler — realistic-ish code+prose repeated to reach a target
 *  token count. Not meant to resemble any real session, just occupy KV positions with content
 *  a causal model would actually attend over (not padding tokens). */
function buildFillerText(model: ModelGPU, targetTokens: number): string {
  const CHUNK =
    'The user asked to review the file utils.ts and fix a bug in the parseConfig function. ' +
    'function parseConfig(raw: string): Config { const obj = JSON.parse(raw); return { ...obj, version: obj.version ?? 1 }; }\n' +
    'Then the assistant read the file, found the issue was a missing null check, and applied an edit. ' +
    'The user confirmed the fix worked and asked for a test to be added covering the empty-input case.\n\n';
  let text = '';
  while (model.tokenize(text, false, false).length < targetTokens) text += CHUNK;
  return text;
}

async function runAtPrefix(model: ModelGPU, prefixTokens: number, variant: 'a' | 'b'): Promise<number> {
  let pos = 0;
  if (prefixTokens > 0) {
    const filler = buildFillerText(model, prefixTokens);
    const tokens = model.tokenize(filler, true, false);
    const status = await model.forwardAsync(tokens); // always on seq 0
    if (status !== 0) throw new Error(`prefix forward failed: ${status}`);
    pos = tokens.length;
  }
  console.log(`\n=== variant (${variant}), prefix=${prefixTokens} (pos=${pos}) ===`);

  let correct = 0;
  const misses: { prompt: string; expected: Intent; got: Intent }[] = [];
  for (const tc of BENCH_DATASET) {
    const pred = prefixTokens === 0 && variant === 'a'
      ? classifyStandalone(model, tc.prompt)
      : variant === 'a'
        ? await classifyPrefixAttending(model, pos, tc.prompt)
        : await classifyIsolatedSeq(model, tc.prompt);
    if (pred === tc.expected) correct++;
    else misses.push({ prompt: tc.prompt.slice(0, 60), expected: tc.expected, got: pred });
  }
  const acc = (correct / BENCH_DATASET.length) * 100;
  console.log(`accuracy: ${correct}/${BENCH_DATASET.length} = ${acc.toFixed(1)}%`);
  for (const m of misses) console.log(`  miss: "${m.prompt}" expected=${m.expected} got=${m.got}`);
  return acc;
}

async function runVariant(model: ModelGPU, variant: 'a' | 'b'): Promise<number[]> {
  const accuracies: number[] = [];
  for (const prefixTokens of PREFIX_LENGTHS) {
    accuracies.push(await runAtPrefix(model, prefixTokens, variant));
  }
  return accuracies;
}

function report(label: string, accuracies: number[]): boolean {
  console.log(`\n=== SUMMARY: variant (${label}) ===`);
  const [baseline, ...rest] = accuracies;
  console.log(`baseline (prefix=${PREFIX_LENGTHS[0]}): ${baseline.toFixed(1)}%`);
  let go = true;
  for (let i = 0; i < rest.length; i++) {
    const delta = rest[i] - baseline;
    const withinTolerance = Math.abs(delta) <= TOLERANCE_PP;
    if (!withinTolerance) go = false;
    console.log(`prefix=${PREFIX_LENGTHS[i + 1]}: ${rest[i].toFixed(1)}%  (delta ${delta.toFixed(1)} pp, ${withinTolerance ? 'within tolerance' : 'OUT OF TOLERANCE'})`);
  }
  console.log(`VERDICT (${label}): ${go ? 'GO' : 'NO-GO'}`);
  return go;
}

async function main(): Promise<void> {
  // variant (a) needs only seq 0 (no auxSeq); variant (b) needs auxSeq:true reserved up front.
  const model = new ModelGPU(activeProfile.smallModelPath, { contextSize: 12288, threads: 4, gpuLayers: 999, auxSeq: true });

  const accA = await runVariant(model, 'a');
  const goA = report('a', accA);

  const accB = await runVariant(model, 'b');
  const goB = report('b', accB);

  console.log(`\n=== FINAL ===`);
  console.log(`variant (a) prefix-attending: ${goA ? 'GO' : 'NO-GO'}`);
  console.log(`variant (b) isolated seq_id:  ${goB ? 'GO' : 'NO-GO'}`);
}

main().catch((err) => {
  console.error('Fatal:', err);
  process.exit(1);
});
