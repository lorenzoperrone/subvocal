/**
 * outputEmitter.ts — Substory 5.1
 *
 * Output phase coordinator: diff computation, compiler telemetry, ghost text
 * hints, and NL report generation via the small CPU model.
 *
 * All pure functions (except generateNLReport) are unit-testable without a model.
 */

import type { VFSFile } from './shadowVFS.js';
import type { SyntaxError } from './syntaxValidator.js';
import { sampleGreedy, type BaseModel } from '@subvocal/synapse';
import { activeProfile } from './modelProfile.js';

// ── Types ─────────────────────────────────────────────────────────────────────

export interface DiffSummary {
  filePath: string;
  diff: string;
  insertions: number;
  deletions: number;
}

export interface CompilerStats {
  turns: number;
  totalTokens: number;
  rollbacks: number;
  compileErrors: number;
  sandboxBlocks: number;
  totalMs: number;
  avgMsPerTurn: number;
}

// Structural type matching UtterLogEntry — avoids circular import from utter.ts
type LogEntry = {
  turn: number;
  type: 'start' | 'tool' | 'rollback' | 'commit' | 'sandbox' | 'done' | 'error';
  message: string;
  ms?: number;
};

// ── Diff computation ──────────────────────────────────────────────────────────

/**
 * Line-level diff using multiset frequency counts.
 * Correct for files of any length, guaranteed O(n) with no infinite loops.
 */
function lineDiff(
  oldText: string,
  newText: string,
  filePath: string,
): DiffSummary {
  if (oldText === newText) return { filePath, diff: '', insertions: 0, deletions: 0 };

  const oldLines = oldText.split('\n');
  const newLines = newText.split('\n');

  // Multiset frequency: +1 for each old line, -1 for each new line
  const freq = new Map<string, number>();
  for (const l of oldLines) freq.set(l, (freq.get(l) ?? 0) + 1);
  for (const l of newLines) freq.set(l, (freq.get(l) ?? 0) - 1);

  let deletions = 0, insertions = 0;
  for (const v of freq.values()) {
    if (v > 0) deletions += v;
    if (v < 0) insertions += -v;
  }

  // Produce a minimal unified diff header + changed lines (max 30 context lines)
  const cap = 30;
  const diffLines: string[] = [
    `--- a/${filePath}`, `+++ b/${filePath}`,
    `@@ -1,${oldLines.length} +1,${newLines.length} @@`,
    ...oldLines.slice(0, cap).map(l => `-${l}`),
    ...newLines.slice(0, cap).map(l => `+${l}`),
  ];
  if (oldLines.length > cap || newLines.length > cap)
    diffLines.push(`\\ ... (truncated)`);

  return { filePath, diff: diffLines.join('\n'), insertions, deletions };
}

/**
 * Compute unified diffs between original and mutated file contents.
 * @param files     VFS export from utter.runTask()
 * @param originals Map<filePath, originalContent> recorded before the task ran
 */
export function computeResultDiffs(
  files: VFSFile[],
  originals: Map<string, string>,
): DiffSummary[] {
  return files.map(f => lineDiff(originals.get(f.path) ?? '', f.content, f.path));
}

// ── Compiler telemetry ────────────────────────────────────────────────────────

/** Aggregate per-turn events from utter logs into compiler telemetry. */
export function buildCompilerStats(logs: LogEntry[]): CompilerStats {
  const doneEntry = logs.find(l => l.type === 'done');
  const totalMs = doneEntry?.ms ?? 0;
  const turns = logs.reduce((max, l) => Math.max(max, l.turn), 0);
  const rollbacks = logs.filter(l => l.type === 'rollback').length;
  const compileErrors = logs.filter(l => l.type === 'error' || l.type === 'rollback').length;
  const sandboxBlocks = logs.filter(l => l.type === 'sandbox').length;

  // Token count from the start entry message: "N tokens, M tool call(s)"
  const startEntry = logs.find(l => l.type === 'start');
  const totalTokens = startEntry ? (parseInt(startEntry.message) || 0) : 0;

  return {
    turns,
    totalTokens,
    rollbacks,
    compileErrors,
    sandboxBlocks,
    totalMs,
    avgMsPerTurn: turns > 0 ? Math.round(totalMs / turns) : 0,
  };
}

/**
 * Format CompilerStats as a single esbuild-style telemetry line.
 * Example: "3 turns  512 tokens  1240ms  1 rollback"
 */
export function formatCompilerStats(stats: CompilerStats): string {
  const parts: string[] = [
    `${stats.turns} turn${stats.turns !== 1 ? 's' : ''}`,
    `${stats.totalTokens} tokens`,
    `${stats.totalMs.toFixed(0)}ms`,
  ];
  if (stats.rollbacks > 0)
    parts.push(`${stats.rollbacks} rollback${stats.rollbacks !== 1 ? 's' : ''}`);
  if (stats.sandboxBlocks > 0)
    parts.push(`${stats.sandboxBlocks} sandbox block${stats.sandboxBlocks !== 1 ? 's' : ''}`);
  return parts.join('  ');
}

// ── Diff formatting ───────────────────────────────────────────────────────────

const C = {
  red:   '\x1b[31m',
  green: '\x1b[32m',
  cyan:  '\x1b[36m',
  dim:   '\x1b[2m',
  reset: '\x1b[0m',
} as const;

/** Render DiffSummary[] as ANSI-coloured unified diff blocks. */
export function formatDiffSummaries(diffs: DiffSummary[]): string {
  if (diffs.length === 0) return '';
  return diffs.map(d => {
    if (!d.diff) {
      return `${C.dim}(no changes — ${d.filePath})${C.reset}\n`;
    }
    const coloured = d.diff.split('\n').map(line => {
      if (line.startsWith('+++') || line.startsWith('---')) return `${C.dim}${line}${C.reset}`;
      if (line.startsWith('+')) return `${C.green}${line}${C.reset}`;
      if (line.startsWith('-')) return `${C.red}${line}${C.reset}`;
      if (line.startsWith('@')) return `${C.cyan}${line}${C.reset}`;
      return line;
    }).join('\n');
    const summary = `+${d.insertions}/-${d.deletions}`;
    return `${C.cyan}=== ${d.filePath} (${summary}) ===${C.reset}\n${coloured}`;
  }).join('\n');
}

// ── Ghost text hints (Reverse LSP Injection) ──────────────────────────────────

/**
 * Format TypeScript / LSP diagnostics as Ghost Text inline annotations.
 *
 * The caller obtains diagnostics via typescriptDiagnostics(projectDir) from
 * lspShim.ts, then passes them here to produce dim inline annotations that
 * can be printed alongside or below the relevant file lines in the terminal.
 *
 * Full in-process LSP server is deferred (Substory 5.1 [~]).
 */
export function ghostTextHints(diagnostics: SyntaxError[]): string {
  if (diagnostics.length === 0) return '';
  return diagnostics
    .sort((a, b) => a.line !== b.line ? a.line - b.line : a.column - b.column)
    .map(d => `${C.dim}L${d.line}:${d.column}  ${d.nodeType}: ${d.kind}${C.reset}`)
    .join('\n');
}

// ── NL report (small model) ───────────────────────────────────────────────────

/**
 * Build the prompt string fed to the small model for NL summarization.
 * Pure function — unit-testable without a model.
 */
export function formatTaskSummaryPrompt(
  stats: CompilerStats,
  diffs: DiffSummary[],
): string {
  const changed = diffs
    .filter(d => d.insertions + d.deletions > 0)
    .map(d => `  ${d.filePath}: +${d.insertions}/-${d.deletions}`)
    .join('\n');

  const systemPrompt = 'Summarize what was accomplished in one concise sentence.';
  const userPrompt =
    `Task completed in ${stats.turns} turn(s), ${stats.totalMs.toFixed(0)}ms.\n` +
    `Tokens: ${stats.totalTokens}. Rollbacks: ${stats.rollbacks}.\n` +
    (changed ? `Modified files:\n${changed}\n` : 'No files modified.\n');

  return activeProfile.buildPrompt({ systemPrompt, userPrompt });
}

/**
 * Generate a natural-language task summary using the small CPU model.
 * @param prompt     Built with formatTaskSummaryPrompt().
 * @param model      Small CPU model from getSmallModel().
 * @param maxTokens  Cap on generated tokens (default: 80).
 */
export function generateNLReport(
  prompt: string,
  model: BaseModel,
  maxTokens = 80,
): string {
  const promptTokens = model.tokenize(prompt, true, true);
  model.forward(promptTokens);

  const generated: number[] = [];
  const stopIds = new Set(activeProfile.eotTokenIds);

  for (let i = 0; i < maxTokens; i++) {
    if (i > 0) {
      model.decodeAppend(Int32Array.of(generated[generated.length - 1]));
    }
    const next = sampleGreedy(model.getLogits());
    if (stopIds.has(next)) break;
    generated.push(next);
  }

  return generated.length > 0
    ? model.detokenize(Int32Array.from(generated)).trim()
    : '';
}
