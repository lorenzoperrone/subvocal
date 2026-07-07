/**
 * dualBrainRouter.ts
 *
 * Epic M12.2 — the routing gate: decide per-task whether the fast small generator (E2B,
 * ~5x decode) or the large one (12B) handles the turn.
 *
 * Philosophy (see doc/epics/EPIC-M12-dual-brain-routing.md):
 *   - Conservative default: the LARGE brain generates. E2B gets a task only when it sits in
 *     the high-confidence easy band the re-spike validated (M12.1: mechanical, localized
 *     edits on existing code — rename, docstring/comment, small formatting-grade changes).
 *   - Misroutes are cheap IF caught: an E2B attempt costs ~1/5 of a 12B turn, so the caller
 *     escalates to the large brain when the small one's first step produces nothing usable
 *     (see shouldEscalate()) instead of trusting the gate to be perfect.
 *
 * This gate is regex/shape-based on purpose — it must cost ~0 and work with cpuOff:true.
 * The model-backed intent (M11.2) can feed it later; today intent classes alone are too
 * coarse for "easy vs hard" (ADD_FEATURE spans both a JSDoc line and a new subsystem).
 */

import type { AgentStep } from './agentLoop.js';
import { stripRegistryTags } from './ideogramAllocator.js';

export type Brain = 'small' | 'large';

export interface RouteDecision {
  brain: Brain;
  /** Human-readable reason, for logs and the bench harness. */
  reason: string;
}

// The validated easy band: mechanical/localized edit verbs. Deliberately narrow — growing
// this list is cheap once the escalation path proves itself in real sessions.
const EASY_EDIT_RE =
  /\b(rename|renam\w*|jsdoc|docstring|doc comment|add (a )?comment|comment(a|are)?|typo|riformatt\w*|reformat|format(ting)?|rinomina|commento)\b/i;

// Hard markers that veto the small brain even when an easy verb matches ("rename X and
// restructure the module" is not an easy edit).
const HARD_RE =
  /\b(implement|architect|refactor|redesign|rewrite|restructure|migrate|integrate|from scratch|new (feature|module|class|endpoint|component)|implementa|riscrivi|ristruttura|migra)\b/i;

/** Max request length for the easy band — long prompts describe non-mechanical work. */
const MAX_EASY_PROMPT_CHARS = 200;

/**
 * Route a task to a brain. Pure and cheap (<1ms) — call it per task, log the reason.
 */
export function routeTask(input: { prompt: string; fileContent: string }): RouteDecision {
  if (input.fileContent.trim().length === 0) {
    // Empty file = from-scratch generation. M12.1 showed E2B handles simple cases with the
    // corrected template, but quality risk scales with what "the whole file" turns out to be
    // — stay conservative, large brain.
    return { brain: 'large', reason: 'empty file (from-scratch)' };
  }
  if (input.prompt.length > MAX_EASY_PROMPT_CHARS) {
    return { brain: 'large', reason: `prompt too long for easy band (${input.prompt.length} chars)` };
  }
  if (HARD_RE.test(input.prompt)) {
    return { brain: 'large', reason: 'hard-work marker in prompt' };
  }
  if (EASY_EDIT_RE.test(input.prompt)) {
    return { brain: 'small', reason: 'mechanical-edit verb, short prompt, existing file' };
  }
  return { brain: 'large', reason: 'no easy-band signal (conservative default)' };
}

// The model sees TAGGED file content (astTagger ideograms + M15.2 CRC block anchors + M15.4
// path tokens — spanning many unicode blocks now, not just Greek/Math) and often echoes the
// tags back in its edit's `old`/`new`. Comparing the echoed text against the RAW file must
// therefore ignore those characters, or every tagged echo reads as "target not found" — which
// both escalated spuriously AND made the edit executor reject the edit outright (a real
// pre-existing bug on any brain, surfaced by the M12.2 integration test; widened for M15.2).

/**
 * Candidate de-tagged renderings of model-echoed edit text, in match-priority order:
 * tags were injected as "∀ " (tag + space), so stripping tag+space usually restores the raw
 * text byte-exactly; stripping the bare tag covers echoes that already dropped the space.
 * Callers try each against the raw file and use the first that matches. Uses the FULL registry
 * char set (stripRegistryTags), not a unicode-range guess — see M15.2.
 */
export function detagCandidates(s: string): string[] {
  return [stripRegistryTags(s, true), stripRegistryTags(s, false)];
}

function normalizeTagged(s: string): string {
  return stripRegistryTags(s, false).replace(/[ \t]+/g, ' ');
}

/**
 * Escalation check: after the small brain's FIRST step, decide whether to redo the turn on
 * the large brain. Kept to the first step on purpose — once E2B produces a usable tool call,
 * the turn is committed to it (mid-turn brain switches would need KV surgery for no measured
 * benefit).
 */
export function shouldEscalate(step: AgentStep, fileContent: string): string | null {
  if (fileContent.trim().length > 0 && step.toolCalls.length === 0 && !step.ideogramEdit) {
    return 'small brain produced no tool call on a non-empty file';
  }
  for (const call of step.toolCalls) {
    if (call.name === 'edit') {
      // 2026-07 audit: accept BOTH edit-tool arg shapes. The REPL's built-in tool uses
      // `old`/`new` (AGENT_TOOLS); the TUI declares pi's flat form `oldText`/`newText`
      // (wire.ts FLAT_EDIT_TOOL). This check only ever read `old`, so in the TUI (where the
      // model emits `oldText`) it was always undefined — the "targets missing text" escalation
      // never fired there, and a small-brain edit hallucinating a nonexistent target silently
      // reached the executor instead of triggering a 12B redo.
      const oldText = ((call.arguments.old ?? call.arguments.oldText) as string | undefined) ?? '';
      if (
        oldText &&
        !fileContent.includes(oldText) &&
        !normalizeTagged(fileContent).includes(normalizeTagged(oldText))
      ) {
        return 'small brain edit targets text not present in the file';
      }
    }
  }
  return null;
}
