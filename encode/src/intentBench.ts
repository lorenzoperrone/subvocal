/**
 * intentBench.ts
 *
 * Benchmark: Subvocal FFI intent router vs keyword/regex baseline.
 *
 * Measures:
 *   - Accuracy (overall, by intent, by difficulty)
 *   - Latency (p50, p95, cold vs warm)
 *   - Confidence margin (winning logit − runner-up logit, FFI only)
 *   - Confusion matrix
 *
 * Run with:
 *   npx tsx packages/agent/src/preprocessor/intentBench.ts
 */

import { initSmallModel, freeSmallModel } from './smallModel.js';
import { routeIntent, type Intent } from './intentRouter.js';
import { BENCH_DATASET, type BenchCase } from './intentBenchDataset.js';
import { getSmallModel } from './smallModel.js';
import { activeProfile } from './modelProfile.js';

const MODEL_PATH = process.env.SUBVOCAL_SMALL_MODEL ?? activeProfile.smallModelPath;

const INTENTS: Intent[] = ['BUGFIX', 'REFACTOR', 'EXPLAIN', 'ADD_FEATURE', 'WRITE_TEST', 'UNKNOWN'];
const WARMUP_RUNS = 1; // runs before measuring (warms up thread pools and KV alloc)
const BENCH_RUNS = 3;  // timed runs per prompt — take median

// ── Keyword/regex baseline ────────────────────────────────────────────────────

const KEYWORD_RULES: Array<{ pattern: RegExp; intent: Intent }> = [
  { pattern: /\b(fix|bug|broken|crash|error|issue|wrong|fail|doesn'?t work|exception|null|undefined|incorrect|not working|misbehav)\b/i, intent: 'BUGFIX' },
  { pattern: /\b(refactor|clean|extract|rename|reorganize|restructure|simplify|split|decompose|move|decouple|improve|modernize)\b/i, intent: 'REFACTOR' },
  { pattern: /\b(explain|what does|what is|how does|why|understand|describe|tell me|help me|significa|cosa fa|come funziona)\b/i, intent: 'EXPLAIN' },
  { pattern: /\b(add|implement|create|build|develop|integrate|new feature|introduce|support|enable|aggiungi|implementa|crea)\b/i, intent: 'ADD_FEATURE' },
  { pattern: /\b(test|tests|unit test|integration test|coverage|spec|verify|assert|assertion|scrivi.*test|test.*coverage)\b/i, intent: 'WRITE_TEST' },
];

function keywordClassify(prompt: string): Intent {
  // First match wins — order matters (more specific patterns first)
  for (const { pattern, intent } of KEYWORD_RULES) {
    if (pattern.test(prompt)) return intent;
  }
  return 'UNKNOWN';
}

// ── Stats helpers ─────────────────────────────────────────────────────────────

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function percentile(values: number[], p: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.ceil((p / 100) * sorted.length) - 1;
  return sorted[Math.max(0, idx)];
}

// ── Confusion matrix ──────────────────────────────────────────────────────────

class ConfusionMatrix {
  private data = new Map<string, number>();

  record(actual: Intent, predicted: Intent) {
    const key = `${actual}→${predicted}`;
    this.data.set(key, (this.data.get(key) ?? 0) + 1);
  }

  print(label: string) {
    console.log(`\n── Confusion matrix: ${label} ──`);
    const header = ['', ...INTENTS.map(i => i.slice(0, 8).padEnd(8))].join('  ');
    console.log(header);
    for (const actual of INTENTS) {
      const row = INTENTS.map(pred => {
        const n = this.data.get(`${actual}→${pred}`) ?? 0;
        return String(n).padStart(8);
      });
      console.log(`${actual.slice(0, 8).padEnd(10)}  ${row.join('  ')}`);
    }
  }
}

// ── FFI margin extractor ──────────────────────────────────────────────────────
// Reads the raw logits to compute winning margin (winner - runner-up).
// Confidence proxy: large margin = high confidence.

function getMargin(prompt: string): number {
  const model = getSmallModel();
  // Same small-model template rule as intentRouter.buildPrompt (E2B ≠ 12B on Mac).
  const build = activeProfile.buildSmallPrompt?.bind(activeProfile) ?? activeProfile.buildPrompt;
  const fullPrompt = build({
    systemPrompt:
      'Classify the user request into exactly one of these labels:\n' +
      'BUGFIX REFACTOR EXPLAIN ADD_FEATURE WRITE_TEST UNKNOWN\n' +
      'Output only the label, nothing else.',
    userPrompt: prompt,
    prefill: activeProfile.intentPrefill,
  });
  const tokens = model.tokenize(fullPrompt, true, true);
  model.forward(tokens);
  const logits = model.getLogitsFast();

  // Score each intent by max logit across its anchors.
  const anchorLogits = INTENTS.map(intent => {
    const ids = activeProfile.intentAnchors[intent] ?? [];
    return ids.reduce((best, id) => Math.max(best, logits[id] ?? -Infinity), -Infinity);
  }).sort((a, b) => b - a);

  return anchorLogits[0] - anchorLogits[1];
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
  console.log('\n══════════════════════════════════════════════════════════════');
  console.log('  🏎  Subvocal Intent Router — Accuracy & Latency Benchmark');
  console.log('══════════════════════════════════════════════════════════════\n');

  console.log(`📦 Dataset: ${BENCH_DATASET.length} labeled prompts`);
  console.log(`🔁 Runs per prompt: ${WARMUP_RUNS} warmup + ${BENCH_RUNS} timed\n`);

  console.log(`🔌 Loading model: ${MODEL_PATH}`);
  initSmallModel({ modelPath: MODEL_PATH, threads: 8, contextSize: 4096 });
  console.log('✅ Model loaded\n');

  // ── Per-case results ─────────────────────────────────────────────────────────
  interface CaseResult {
    promptShort: string;
    expected: Intent;
    difficulty: 'easy' | 'hard';
    ffiPred: Intent;
    kwPred: Intent;
    ffiLatencies: number[];   // ms per timed run
    margin: number;           // logit margin for FFI
  }

  const results: CaseResult[] = [];

  console.log('Running benchmarks...\n');

  for (const tc of BENCH_DATASET) {
    process.stdout.write(`  ${tc.prompt.slice(0, 55).padEnd(55)} `);

    // Keyword baseline (instant — no timing needed)
    const kwPred = keywordClassify(tc.prompt);

    // FFI warmup
    for (let i = 0; i < WARMUP_RUNS; i++) routeIntent(tc.prompt);

    // FFI timed runs
    const ffiLatencies: number[] = [];
    let ffiPred: Intent = 'UNKNOWN';
    for (let i = 0; i < BENCH_RUNS; i++) {
      const r = routeIntent(tc.prompt);
      ffiPred = r.intent; // last run is canonical (post-warmup)
      ffiLatencies.push(r.latencyMs);
    }

    // Confidence margin (one extra forward pass — amortised over bench runs)
    const margin = getMargin(tc.prompt);

    const ffiOk = ffiPred === tc.expected ? '✅' : '❌';
    const kwOk  = kwPred  === tc.expected ? '✅' : '❌';
    process.stdout.write(`FFI:${ffiOk}${ffiPred.slice(0,10).padEnd(11)} KW:${kwOk}${kwPred.slice(0,10).padEnd(11)} `);
    process.stdout.write(`p50:${median(ffiLatencies).toFixed(1).padStart(6)}ms  margin:${margin.toFixed(2)}\n`);

    results.push({ promptShort: tc.prompt.slice(0, 55), expected: tc.expected, difficulty: tc.difficulty, ffiPred, kwPred, ffiLatencies, margin });
  }

  // ── Aggregate stats ──────────────────────────────────────────────────────────
  const allFfiLatencies = results.flatMap(r => r.ffiLatencies);
  const coldLatency = results[0]?.ffiLatencies[0] ?? 0; // first run ever (fully cold)
  const warmLatencies = results.flatMap(r => r.ffiLatencies.slice(WARMUP_RUNS)); // post-warmup only

  const ffiCorrect = results.filter(r => r.ffiPred === r.expected).length;
  const kwCorrect  = results.filter(r => r.kwPred  === r.expected).length;
  const total = results.length;

  const ffiEasyCorrect = results.filter(r => r.difficulty === 'easy' && r.ffiPred === r.expected).length;
  const ffiHardCorrect = results.filter(r => r.difficulty === 'hard' && r.ffiPred === r.expected).length;
  const easyTotal = results.filter(r => r.difficulty === 'easy').length;
  const hardTotal = results.filter(r => r.difficulty === 'hard').length;

  const kwEasyCorrect = results.filter(r => r.difficulty === 'easy' && r.kwPred === r.expected).length;
  const kwHardCorrect = results.filter(r => r.difficulty === 'hard' && r.kwPred === r.expected).length;

  const avgMarginCorrect = results.filter(r => r.ffiPred === r.expected).reduce((s, r) => s + r.margin, 0) / ffiCorrect;
  const avgMarginWrong   = results.filter(r => r.ffiPred !== r.expected).length > 0
    ? results.filter(r => r.ffiPred !== r.expected).reduce((s, r) => s + r.margin, 0) / (total - ffiCorrect)
    : NaN;

  console.log('\n══════════════════════════════════════════════════════════════');
  console.log('  📊 Results');
  console.log('══════════════════════════════════════════════════════════════\n');

  console.log('── Accuracy ─────────────────────────────────────────────────');
  console.log(`  FFI (subvocal-small)  : ${ffiCorrect}/${total}  = ${(ffiCorrect/total*100).toFixed(1)}%`);
  console.log(`  Keyword/regex baseline: ${kwCorrect}/${total}  = ${(kwCorrect/total*100).toFixed(1)}%`);
  console.log(`  Delta (FFI advantage) : +${((ffiCorrect-kwCorrect)/total*100).toFixed(1)} pp`);

  console.log('\n── Accuracy by difficulty ───────────────────────────────────');
  console.log(`  FFI   easy: ${ffiEasyCorrect}/${easyTotal} (${(ffiEasyCorrect/easyTotal*100).toFixed(1)}%)   hard: ${ffiHardCorrect}/${hardTotal} (${(ffiHardCorrect/hardTotal*100).toFixed(1)}%)`);
  console.log(`  KW    easy: ${kwEasyCorrect}/${easyTotal} (${(kwEasyCorrect/easyTotal*100).toFixed(1)}%)   hard: ${kwHardCorrect}/${hardTotal} (${(kwHardCorrect/hardTotal*100).toFixed(1)}%)`);

  console.log('\n── Accuracy by intent (FFI) ─────────────────────────────────');
  for (const intent of INTENTS) {
    const cases = results.filter(r => r.expected === intent);
    const correct = cases.filter(r => r.ffiPred === intent).length;
    const bar = '█'.repeat(Math.round(correct / cases.length * 20)).padEnd(20, '░');
    console.log(`  ${intent.padEnd(12)} ${bar} ${correct}/${cases.length}`);
  }

  console.log('\n── Latency (FFI) ────────────────────────────────────────────');
  console.log(`  Cold (first ever prompt): ${coldLatency.toFixed(1)}ms`);
  console.log(`  All runs  p50: ${percentile(allFfiLatencies, 50).toFixed(1)}ms   p95: ${percentile(allFfiLatencies, 95).toFixed(1)}ms   min: ${Math.min(...allFfiLatencies).toFixed(1)}ms   max: ${Math.max(...allFfiLatencies).toFixed(1)}ms`);
  if (warmLatencies.length > 0) {
    console.log(`  Warm runs p50: ${percentile(warmLatencies, 50).toFixed(1)}ms   p95: ${percentile(warmLatencies, 95).toFixed(1)}ms`);
  }
  console.log(`  Keyword baseline: ~0.0ms (pure regex, no model)`);

  console.log('\n── Confidence margin (FFI, winner logit − runner-up) ────────');
  console.log(`  Correct predictions: avg margin = ${avgMarginCorrect.toFixed(2)}`);
  if (!isNaN(avgMarginWrong)) {
    console.log(`  Wrong predictions:   avg margin = ${avgMarginWrong.toFixed(2)}`);
    console.log(`  (lower margin on errors → threshold-based fallback possible)`);
  } else {
    console.log(`  No wrong predictions!`);
  }

  // Confusion matrices
  const ffiMatrix = new ConfusionMatrix();
  const kwMatrix  = new ConfusionMatrix();
  for (const r of results) {
    ffiMatrix.record(r.expected, r.ffiPred);
    kwMatrix.record(r.expected, r.kwPred);
  }
  ffiMatrix.print('FFI (subvocal-small)');
  kwMatrix.print('Keyword/regex baseline');

  // Failures detail
  const ffiFailures = results.filter(r => r.ffiPred !== r.expected);
  if (ffiFailures.length > 0) {
    console.log('\n── FFI misclassifications ────────────────────────────────────');
    for (const r of ffiFailures) {
      console.log(`  [${r.difficulty}] "${r.promptShort}"`);
      console.log(`    expected=${r.expected}  got=${r.ffiPred}  margin=${r.margin.toFixed(2)}`);
    }
  }

  console.log('\n══════════════════════════════════════════════════════════════\n');

  freeSmallModel();
}

main().catch(err => {
  console.error('Fatal:', err);
  freeSmallModel();
  process.exit(1);
});
