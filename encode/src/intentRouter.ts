/**
 * intentRouter.ts
 *
 * Two intent classifiers:
 *   1. routeIntent()       — logit sniffing via subvocal-small (precise, ~15ms)
 *   2. routeIntentRegex()  — regex-based fallback (<1ms, ~80% accuracy)
 *
 * The regex classifier is used when --cpu-model off is set.
 */

import { getSmallModel } from './smallModel.js';
import { activeProfile } from './modelProfile.js';

// ── Intent definitions ────────────────────────────────────────────────────────

export type Intent =
  | 'BUGFIX'
  | 'REFACTOR'
  | 'EXPLAIN'
  | 'ADD_FEATURE'
  | 'WRITE_TEST'
  | 'UNKNOWN';

const INTENT_SYSTEM_PROMPT =
  'Classify the user request into exactly one of these labels:\n' +
  'BUGFIX REFACTOR EXPLAIN ADD_FEATURE WRITE_TEST UNKNOWN\n' +
  'Output only the label, nothing else.';

// ── Anchor token IDs (from active profile) ────────────────────────────────────

const TOKEN_ID_TO_INTENT = new Map<number, Intent>();
for (const [intent, ids] of Object.entries(activeProfile.intentAnchors) as [Intent, readonly number[]][]) {
  for (const id of ids) TOKEN_ID_TO_INTENT.set(id, intent);
}

// ── Regex patterns for CPU-off fallback ───────────────────────────────────────

const REGEX_PATTERNS: { pattern: RegExp; intent: Intent }[] = [
  { pattern: /\b(fix|bug|error|crash|broken|wrong|correg|ripar|sistem)\b/i, intent: 'BUGFIX' },
  { pattern: /\b(refactor|clean|rewrite|simplif|restructur|riorgan|pulisc)\b/i, intent: 'REFACTOR' },
  { pattern: /\b(explain|what|how|why|describe|document|spiega|descrivi|come|perché)\b/i, intent: 'EXPLAIN' },
  { pattern: /\b(add|implement|create|build|new|feature|aggiung|implementa|crea|nuov)\b/i, intent: 'ADD_FEATURE' },
  { pattern: /\b(test|spec|coverage|assert|unit test|integration)\b/i, intent: 'WRITE_TEST' },
];

// ── Prompt builder ────────────────────────────────────────────────────────────

function buildPrompt(userPrompt: string): string {
  // The classifier runs on the SMALL model — use its template when the profile
  // declares one (MacProfile: E2B has no empty-thought-channel insertion, unlike the 12B).
  const build = activeProfile.buildSmallPrompt?.bind(activeProfile) ?? activeProfile.buildPrompt;
  return build({
    systemPrompt: INTENT_SYSTEM_PROMPT,
    userPrompt,
    prefill: activeProfile.intentPrefill,
  });
}

// ── Exported result type ──────────────────────────────────────────────────────

export interface IntentResult {
  intent: Intent;
  anchorLogit: number;
  latencyMs: number;
  promptTokens: number;
}

// ── CPU model-based classifier ────────────────────────────────────────────────

export function routeIntent(userPrompt: string): IntentResult {
  return classifyIntentWith(getSmallModel(), userPrompt);
}

/**
 * Core of routeIntent() with the model as a parameter — shared by the in-process singleton
 * path above and the worker-thread classifier (intentWorker.ts), which owns its OWN model
 * instance and cannot touch the main thread's singleton.
 */
export function classifyIntentWith(model: import('@subvocal/synapse').BaseModel, userPrompt: string): IntentResult {
  const t0 = performance.now();

  const promptTokens = model.tokenize(buildPrompt(userPrompt), true, true);

  const status = model.forward(promptTokens);
  if (status !== 0) {
    throw new Error(`forward() returned non-zero status: ${status}`);
  }

  const logits = model.getLogitsFast();

  let bestIntent: Intent = 'UNKNOWN';
  let bestLogit = -Infinity;

  for (const [tokenId, intent] of TOKEN_ID_TO_INTENT) {
    const v = logits[tokenId];
    if (v > bestLogit) {
      bestLogit = v;
      bestIntent = intent;
    }
  }

  const latencyMs = performance.now() - t0;

  return {
    intent: bestIntent,
    anchorLogit: bestLogit,
    latencyMs,
    promptTokens: promptTokens.length,
  };
}

/**
 * M11.3 variant (b): classify on a model that may already hold a LIVE conversation on seq 0
 * (the shared E2B instance conversation.ts's getE2B() serves to the drafter/generator/
 * distiller), without disturbing it. Decodes the classify prompt on an attention-ISOLATED
 * sequence (seq_id=1, position 0 every call — nothing to persist) via the native
 * decodeAppendSeq(), then wipes it. Requires the model constructed with
 * ModelOptions.auxSeq: true — otherwise decodeAppendSeq rejects seq_id=1 as out of range.
 *
 * Measured 2026-07-06 (doc/substories/M11.3-intent-classifier-shared-e2b.md,
 * encode/src/intentBenchSharedContext.ts): accuracy is provably independent of seq 0's
 * content — 75.0% at prefix lengths 0/2048/8192, byte-identical to the standalone
 * classifyIntentWith() baseline at every length (0.0pp delta). This is the variant that
 * replaced the naive prefix-attending approach (classify by decodeAppendAsync onto seq 0
 * itself), which measured a real accuracy regression (-12.5pp at 2k tokens, -17.5pp at 8k)
 * from the live conversation's causal-attention prefix.
 */
export async function classifyIntentOnAuxSeq(
  model: import('@subvocal/synapse').BaseModel,
  userPrompt: string,
): Promise<IntentResult> {
  const t0 = performance.now();

  const promptTokens = model.tokenize(buildPrompt(userPrompt), true, true);
  const status = await model.decodeAppendSeq(promptTokens, 1, 0);
  if (status !== 0) {
    model.kvCacheSeqRemove(1, 0, -1);
    throw new Error(`decodeAppendSeq() returned non-zero status: ${status}`);
  }

  const logits = model.getLogitsFast();

  let bestIntent: Intent = 'UNKNOWN';
  let bestLogit = -Infinity;

  for (const [tokenId, intent] of TOKEN_ID_TO_INTENT) {
    const v = logits[tokenId];
    if (v > bestLogit) {
      bestLogit = v;
      bestIntent = intent;
    }
  }

  // Always wipe seq 1, success or failure — it must never accumulate across calls (each
  // classify starts fresh at position 0) or leak into seq 0's sequence-id space.
  model.kvCacheSeqRemove(1, 0, -1);

  const latencyMs = performance.now() - t0;

  return {
    intent: bestIntent,
    anchorLogit: bestLogit,
    latencyMs,
    promptTokens: promptTokens.length,
  };
}

// ── Regex-based fallback classifier (<1ms, no model needed) ───────────────────

export function routeIntentRegex(userPrompt: string): IntentResult {
  const t0 = performance.now();

  for (const { pattern, intent } of REGEX_PATTERNS) {
    if (pattern.test(userPrompt)) {
      return { intent, anchorLogit: 0, latencyMs: performance.now() - t0, promptTokens: 0 };
    }
  }

  return { intent: 'UNKNOWN', anchorLogit: 0, latencyMs: performance.now() - t0, promptTokens: 0 };
}
