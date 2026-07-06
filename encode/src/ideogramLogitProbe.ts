/**
 * ideogramLogitProbe.ts
 *
 * Finds the best ideogram anchors for each concept by directly measuring
 * logit values from the model's actual generation space.
 *
 * For each concept (intent class, tool action, AST node type):
 *   1. Run N representative prompts through the classification forward pass
 *   2. Read getLogitsFast() at the last position
 *   3. Extract logits for all 12k ideogram token IDs from TagRegistry
 *   4. Average across prompts → stable ranking
 *   5. Output top-K ideograms per concept
 *
 * This directly answers "which ideogram will the model naturally emit
 * in this classification context?" — the ground truth anchor.
 *
 * Run: SUBVOCAL_MODEL_PROFILE=gemma4 npx tsx ideogramLogitProbe.ts
 */

import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { initSmallModel, getSmallModel } from './smallModel.js';
import { activeProfile } from './modelProfile.js';
import { BENCH_DATASET } from './intentBenchDataset.js';
import type { Intent } from './intentRouter.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const TOP_K = 10;

// ── Load registry ─────────────────────────────────────────────────────────────

interface TagEntry { char: string; tokenId: number; }

const registry: TagEntry[] = JSON.parse(
  fs.readFileSync(activeProfile.tagRegistryPath, 'utf-8')
);
const ideogramIds = new Int32Array(registry.map(e => e.tokenId));

// ── Concept definitions ───────────────────────────────────────────────────────
// Each concept has a set of representative prompts and the system prompt
// that frames the classification task.

const INTENT_SYSTEM =
  'Classify the user request into exactly one of these labels:\n' +
  'BUGFIX REFACTOR EXPLAIN ADD_FEATURE WRITE_TEST UNKNOWN\n' +
  'Output only the label, nothing else.';

// Use the bench dataset for intent concepts (real-world prompts, labeled).
const INTENT_PROMPTS: Record<Intent, string[]> = {
  BUGFIX:      BENCH_DATASET.filter(c => c.expected === 'BUGFIX').map(c => c.prompt),
  REFACTOR:    BENCH_DATASET.filter(c => c.expected === 'REFACTOR').map(c => c.prompt),
  EXPLAIN:     BENCH_DATASET.filter(c => c.expected === 'EXPLAIN').map(c => c.prompt),
  ADD_FEATURE: BENCH_DATASET.filter(c => c.expected === 'ADD_FEATURE').map(c => c.prompt),
  WRITE_TEST:  BENCH_DATASET.filter(c => c.expected === 'WRITE_TEST').map(c => c.prompt),
  UNKNOWN:     BENCH_DATASET.filter(c => c.expected === 'UNKNOWN').map(c => c.prompt),
};

// For tool/AST concepts use a task-description framing.
const TOOL_SYSTEM =
  'You are a coding agent. Output a single control token indicating your next action:\n' +
  'EDIT_AST CMD_READ CMD_EXEC TASK_COMPLETE PAYLOAD_START PAYLOAD_END AFFECTED ERROR';

const TOOL_PROMPTS: Record<string, string[]> = {
  EDIT_AST: [
    'Modify the function signature to add a new parameter',
    'Rewrite this class method to fix the off-by-one error',
    'Update the variable name from usr to user in the AST',
    'Insert a null check before the array access',
  ],
  CMD_READ: [
    'Read the contents of config.json',
    'Fetch the current state of index.ts',
    'Get the list of files in the src directory',
    'Load the package.json to check dependencies',
  ],
  CMD_EXEC: [
    'Run the TypeScript compiler to check for errors',
    'Execute the test suite for the authentication module',
    'Build the project and check for compilation errors',
    'Launch the linter on the changed files',
  ],
  TASK_COMPLETE: [
    'The fix has been applied successfully',
    'All tests pass, the implementation is done',
    'The refactoring is complete',
    'Task finished, no further actions needed',
  ],
  PAYLOAD_START: [
    'Beginning the data block transfer',
    'Starting the structured payload output',
    'Opening the code patch block',
    'Initiating the file content block',
  ],
  PAYLOAD_END: [
    'End of the data block',
    'Closing the structured payload',
    'Patch block complete',
    'File content block closed',
  ],
  AFFECTED: [
    'The following files are affected by this change',
    'This dependency impacts the authentication module',
    'Related targets that need to be updated',
    'Files that reference the modified function',
  ],
  ERROR_SIGNAL: [
    'Undefined variable error detected on line 42',
    'Type error: expected string, got number',
    'LSP validation failed: missing return type',
    'Syntax error in the modified AST node',
  ],
};

// AST node concepts framed as "what kind of node follows?"
const AST_SYSTEM =
  'Classify the type of code structure that follows:\n' +
  'FUNCTION CLASS LOOP CONDITION IMPORT VARIABLE\n' +
  'Output only the label.';

const AST_PROMPTS: Record<string, string[]> = {
  FUNCTION: [
    'def calculate_total(items):',
    'function processPayment(amount, currency) {',
    'async function fetchUserProfile(userId: string) {',
    'const handleClick = (event) => {',
  ],
  CLASS: [
    'class ShoppingCart:',
    'class AuthenticationService {',
    'interface UserRepository {',
    'abstract class BaseModel {',
  ],
  LOOP: [
    'for item in items:',
    'for (let i = 0; i < n; i++) {',
    'while (queue.length > 0) {',
    'items.forEach(item => {',
  ],
  CONDITION: [
    'if user is None:',
    'if (token === null || token === undefined) {',
    'if (count > MAX_RETRIES) {',
    'switch (intent) {',
  ],
  IMPORT: [
    'import numpy as np',
    'import { useState, useEffect } from "react"',
    'from typing import Optional, List',
    'const path = require("path")',
  ],
  VARIABLE: [
    'const MAX_RETRIES = 3',
    'let currentUser: User | null = null',
    'total = sum(item.price for item in items)',
    'private readonly cache: Map<string, Buffer>',
  ],
};

// ── Probe function ────────────────────────────────────────────────────────────

function probeIdeogramLogits(
  systemPrompt: string,
  prompts: string[],
): Float32Array {
  const model = getSmallModel();
  const accumulated = new Float64Array(ideogramIds.length); // double for stability

  for (const userPrompt of prompts) {
    const fullPrompt = activeProfile.buildPrompt({
      systemPrompt,
      userPrompt,
      prefill: activeProfile.intentPrefill,
    });
    const tokens = model.tokenize(fullPrompt, true, true);
    const status = model.forward(tokens);
    if (status !== 0) throw new Error(`forward() returned ${status}`);

    const logits = model.getLogitsFast();
    for (let i = 0; i < ideogramIds.length; i++) {
      accumulated[i] += logits[ideogramIds[i]];
    }
  }

  // Average across prompts → Float32
  const result = new Float32Array(ideogramIds.length);
  for (let i = 0; i < ideogramIds.length; i++) {
    result[i] = accumulated[i] / prompts.length;
  }
  return result;
}

function topK(scores: Float32Array, k: number): Array<{ char: string; tokenId: number; score: number }> {
  const indexed = Array.from(scores, (s, i) => ({ i, s }));
  indexed.sort((a, b) => b.s - a.s);
  return indexed.slice(0, k).map(({ i, s }) => ({
    char: registry[i].char,
    tokenId: registry[i].tokenId,
    score: s,
  }));
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
  console.log('\n' + '='.repeat(64));
  console.log('  Ideogram Logit Probe — Gemma 4 E2B');
  console.log(`  Registry: ${registry.length} ideograms | Profile: ${activeProfile.name}`);
  console.log('='.repeat(64) + '\n');

  initSmallModel({ modelPath: activeProfile.smallModelPath, ...activeProfile.smallOpts });

  const output: Record<string, Array<{ char: string; tokenId: number; score: number }>> = {};

  // ── Intent concepts ────────────────────────────────────────────────────────
  console.log('── Intent classes ────────────────────────────────────────────');
  for (const [intent, prompts] of Object.entries(INTENT_PROMPTS)) {
    process.stdout.write(`  ${intent.padEnd(14)} (${prompts.length} prompts) … `);
    const scores = probeIdeogramLogits(INTENT_SYSTEM, prompts);
    const top = topK(scores, TOP_K);
    output[intent] = top;
    console.log(top.slice(0, 5).map(h => `${h.char}(${h.score.toFixed(1)})`).join('  '));
  }

  // ── Tool action concepts ───────────────────────────────────────────────────
  console.log('\n── Tool actions ──────────────────────────────────────────────');
  for (const [concept, prompts] of Object.entries(TOOL_PROMPTS)) {
    process.stdout.write(`  ${concept.padEnd(14)} (${prompts.length} prompts) … `);
    const scores = probeIdeogramLogits(TOOL_SYSTEM, prompts);
    const top = topK(scores, TOP_K);
    output[concept] = top;
    console.log(top.slice(0, 5).map(h => `${h.char}(${h.score.toFixed(1)})`).join('  '));
  }

  // ── AST node concepts ──────────────────────────────────────────────────────
  console.log('\n── AST node types ────────────────────────────────────────────');
  for (const [concept, prompts] of Object.entries(AST_PROMPTS)) {
    process.stdout.write(`  ${concept.padEnd(14)} (${prompts.length} prompts) … `);
    const scores = probeIdeogramLogits(AST_SYSTEM, prompts);
    const top = topK(scores, TOP_K);
    output[concept] = top;
    console.log(top.slice(0, 5).map(h => `${h.char}(${h.score.toFixed(1)})`).join('  '));
  }

  // ── Save results ───────────────────────────────────────────────────────────
  const outPath = path.join(__dirname, 'ideogramAnchorMap.gemma4.json');
  fs.writeFileSync(outPath, JSON.stringify(output, null, 2), 'utf-8');

  console.log('\n' + '='.repeat(64));
  console.log(`  Saved → ${outPath}`);
  console.log('='.repeat(64) + '\n');

  process.exit(0);
}

main().catch(err => { console.error('Fatal:', err); process.exit(1); });
