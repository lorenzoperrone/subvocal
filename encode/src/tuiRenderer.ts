/**
 * tuiRenderer.ts — Substory 5.2
 *
 * Brutalista monospazio terminal renderer with double-buffered ANSI differential
 * output. All rendering functions produce strings (render-to-string pattern),
 * making them unit-testable without a real TTY.
 *
 * Interactive features (live TUI loop, animated AST Time-Lapse, binaural
 * acoustic telemetry, Zero-Keystroke Loop, F1/F2 Action Chips) are deferred
 * to Substory 5.2 [~] — the testable rendering core is implemented here.
 */

import type { DiffSummary, CompilerStats } from './outputEmitter.js';

// ── Types ─────────────────────────────────────────────────────────────────────

export interface TuiFrame {
  /** ANSI-escaped lines ready for terminal output. */
  lines: string[];
  /** Render width (characters). */
  width: number;
}

export interface TuiBuffer {
  current: TuiFrame;
  previous: TuiFrame | null;
}

/** Compact representation of a file mutation for TUI display. */
export interface FileMutation {
  filePath: string;
  insertions: number;
  deletions: number;
  /** Abbreviated diff preview (first changed hunk, max 5 lines). */
  preview: string;
}

// ── ANSI constants ────────────────────────────────────────────────────────────

const A = {
  bold:    '\x1b[1m',
  dim:     '\x1b[2m',
  red:     '\x1b[31m',
  green:   '\x1b[32m',
  yellow:  '\x1b[33m',
  blue:    '\x1b[34m',
  magenta: '\x1b[35m',
  cyan:    '\x1b[36m',
  white:   '\x1b[37m',
  reset:   '\x1b[0m',
} as const;

// ── Layout helpers ────────────────────────────────────────────────────────────

function pad(s: string, width: number, char = ' '): string {
  const visible = s.replace(/\x1b\[[^m]*m/g, '').length;
  return s + char.repeat(Math.max(0, width - visible));
}

function ruler(width: number, char = '─'): string {
  return char.repeat(width);
}

function statusBar(label: string, value: string, width: number): string {
  const content = `  ${A.dim}${label}${A.reset}  ${A.bold}${value}${A.reset}`;
  return pad(content, width);
}

// ── Mutation summary ──────────────────────────────────────────────────────────

/**
 * Extract a compact mutation summary from a DiffSummary.
 * Returns the first changed hunk (up to 5 lines) as a preview.
 */
export function extractMutation(diff: DiffSummary): FileMutation {
  const lines = diff.diff.split('\n');
  const hunkLines: string[] = [];
  let inHunk = false;
  for (const line of lines) {
    if (line.startsWith('@@')) { inHunk = true; hunkLines.length = 0; }
    if (inHunk && hunkLines.length < 5) hunkLines.push(line);
    if (inHunk && hunkLines.length >= 5) break;
  }
  return {
    filePath: diff.filePath,
    insertions: diff.insertions,
    deletions: diff.deletions,
    preview: hunkLines.join('\n'),
  };
}

// ── Structural diff ───────────────────────────────────────────────────────────

/**
 * Compute a line-level structural diff summary between two source strings.
 *
 * Returns a human-readable string listing added/removed lines, suitable for
 * display in the TUI mutation log. This is the unit-testable core of the
 * AST Time-Lapse feature — the animated 0.5s replay of syntactic mutations
 * is deferred to the interactive TUI integration ([~]).
 */
export function computeStructuralDiff(before: string, after: string): string {
  if (before === after) return '(no changes)';

  const bLines = before.split('\n');
  const aLines = after.split('\n');
  const bSet = new Set(bLines);
  const aSet = new Set(aLines);

  const removed = bLines.filter(l => l.trim() && !aSet.has(l));
  const added   = aLines.filter(l => l.trim() && !bSet.has(l));

  const parts: string[] = [];
  if (removed.length) {
    parts.push(`${A.red}─ removed (${removed.length})${A.reset}`);
    removed.slice(0, 3).forEach(l => parts.push(`${A.red}  ${l.trim().slice(0, 60)}${A.reset}`));
    if (removed.length > 3) parts.push(`${A.dim}  … +${removed.length - 3} more${A.reset}`);
  }
  if (added.length) {
    parts.push(`${A.green}+ added (${added.length})${A.reset}`);
    added.slice(0, 3).forEach(l => parts.push(`${A.green}  ${l.trim().slice(0, 60)}${A.reset}`));
    if (added.length > 3) parts.push(`${A.dim}  … +${added.length - 3} more${A.reset}`);
  }
  return parts.join('\n') || '(whitespace-only changes)';
}

// ── Frame renderer ────────────────────────────────────────────────────────────

/**
 * Render a complete TUI frame for a finished UtterResult.
 *
 * The frame is a double-buffered ANSI string. Callers write the return value
 * to stdout. For live differential updates, call renderDiff() on the TuiBuffer
 * to emit only changed lines.
 *
 * @param stats      Compiler telemetry (from buildCompilerStats)
 * @param mutations  File mutations (from extractMutation per DiffSummary)
 * @param width      Terminal width in columns (default: 80)
 */
export function renderUtterFrame(
  stats: CompilerStats,
  mutations: FileMutation[],
  width = 80,
): TuiFrame {
  const lines: string[] = [];
  const hr = ruler(width);

  // ── Header ────────────────────────────────────────────────────────────────
  lines.push(`${A.bold}${A.cyan}${hr}${A.reset}`);
  lines.push(pad(`  ${A.bold}SUBVOCAL${A.reset}  output phase`, width));
  lines.push(`${A.dim}${hr}${A.reset}`);

  // ── Compiler stats bar ────────────────────────────────────────────────────
  lines.push(statusBar('turns',    String(stats.turns),                   width));
  lines.push(statusBar('tokens',   String(stats.totalTokens),             width));
  lines.push(statusBar('time',     `${stats.totalMs.toFixed(0)}ms`,       width));
  lines.push(statusBar('avg/turn', `${stats.avgMsPerTurn}ms`,            width));
  if (stats.rollbacks > 0)
    lines.push(statusBar('rollbacks', String(stats.rollbacks),             width));
  if (stats.sandboxBlocks > 0)
    lines.push(statusBar('sandbox',   String(stats.sandboxBlocks) + ' blocked', width));

  // ── Mutations ─────────────────────────────────────────────────────────────
  if (mutations.length > 0) {
    lines.push(`${A.dim}${hr}${A.reset}`);
    lines.push(pad(`  ${A.bold}mutations${A.reset}`, width));
    for (const m of mutations) {
      const badge = `${A.green}+${m.insertions}${A.reset} ${A.red}-${m.deletions}${A.reset}`;
      lines.push(pad(`  ${A.cyan}${m.filePath}${A.reset}  ${badge}`, width));
      if (m.preview) {
        m.preview.split('\n').slice(0, 3).forEach(l => {
          const coloured = l.startsWith('+') ? `${A.green}${l}${A.reset}`
                         : l.startsWith('-') ? `${A.red}${l}${A.reset}`
                         : `${A.dim}${l}${A.reset}`;
          lines.push('    ' + coloured);
        });
      }
    }
  } else {
    lines.push(`${A.dim}${hr}${A.reset}`);
    lines.push(pad(`  ${A.dim}(no mutations)${A.reset}`, width));
  }

  lines.push(`${A.bold}${A.cyan}${hr}${A.reset}`);

  return { lines, width };
}

// ── Double-buffer differential update ────────────────────────────────────────

/**
 * Create a new TuiBuffer with the given frame as the current state.
 */
export function createBuffer(frame: TuiFrame): TuiBuffer {
  return { current: frame, previous: null };
}

/**
 * Advance the buffer to a new frame and return only the changed lines
 * as a partial ANSI update string (differential rendering).
 *
 * Lines are compared using their visible text (ANSI codes stripped) so
 * cosmetic-only changes (colour tweaks) do not trigger re-renders.
 */
export function renderDiff(buffer: TuiBuffer, next: TuiFrame): string {
  buffer.previous = buffer.current;
  buffer.current = next;

  const prev = buffer.previous;
  const out: string[] = [];

  const len = Math.max(prev.lines.length, next.lines.length);
  for (let i = 0; i < len; i++) {
    const prevLine = prev.lines[i] ?? '';
    const nextLine = next.lines[i] ?? '';
    const prevVis = prevLine.replace(/\x1b\[[^m]*m/g, '');
    const nextVis = nextLine.replace(/\x1b\[[^m]*m/g, '');
    if (prevVis !== nextVis) {
      // Move cursor to line i+1, column 1; clear line; write new content
      out.push(`\x1b[${i + 1};1H\x1b[2K${nextLine}`);
    }
  }

  return out.join('');
}

/**
 * Render the full current frame to a string for initial draw or full repaint.
 */
export function renderFull(frame: TuiFrame): string {
  return frame.lines.join('\n') + '\n';
}
