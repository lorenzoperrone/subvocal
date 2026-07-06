/**
 * intentBenchDataset.ts
 *
 * Labeled dataset for intent router benchmarking.
 * Each entry: [prompt, expectedIntent]
 *
 * Designed to cover:
 *  - Clear-cut cases (easy for both FFI and keyword baseline)
 *  - Ambiguous cases (where FFI semantic understanding matters)
 *  - Italian/mixed language prompts (realistic for this project)
 *  - Adversarial cases (keywords that mislead a regex but have clear semantic intent)
 */

import type { Intent } from './intentRouter.js';

export interface BenchCase {
  prompt: string;
  expected: Intent;
  /** 'easy' = unambiguous, 'hard' = misleading keywords or ambiguous phrasing */
  difficulty: 'easy' | 'hard';
}

export const BENCH_DATASET: BenchCase[] = [
  // ── BUGFIX (easy) ────────────────────────────────────────────────────────────
  { prompt: 'Fix the null pointer exception in the login controller', expected: 'BUGFIX', difficulty: 'easy' },
  { prompt: 'The payment form crashes on mobile — debug it', expected: 'BUGFIX', difficulty: 'easy' },
  { prompt: 'calcola_totale returns wrong value when list is empty', expected: 'BUGFIX', difficulty: 'easy' },
  { prompt: 'Correggi il bug nel parsing della data', expected: 'BUGFIX', difficulty: 'easy' },
  { prompt: "The API endpoint returns 500 when the user doesn't have a profile", expected: 'BUGFIX', difficulty: 'easy' },
  { prompt: 'Session token is not being cleared on logout', expected: 'BUGFIX', difficulty: 'easy' },

  // ── BUGFIX (hard — no "bug/fix" keywords) ───────────────────────────────────
  { prompt: 'The list sometimes shows duplicates after refreshing', expected: 'BUGFIX', difficulty: 'hard' },
  { prompt: 'After adding an item the total does not update correctly', expected: 'BUGFIX', difficulty: 'hard' },
  { prompt: 'IVA viene applicata due volte quando si ha un coupon attivo', expected: 'BUGFIX', difficulty: 'hard' },

  // ── REFACTOR (easy) ──────────────────────────────────────────────────────────
  { prompt: 'Refactor the authentication module to use dependency injection', expected: 'REFACTOR', difficulty: 'easy' },
  { prompt: 'Extract the email validation logic into a separate utility function', expected: 'REFACTOR', difficulty: 'easy' },
  { prompt: 'Rename all occurrences of `usr` to `user` throughout the codebase', expected: 'REFACTOR', difficulty: 'easy' },
  { prompt: 'Semplifica questa funzione, è troppo lunga', expected: 'REFACTOR', difficulty: 'easy' },
  { prompt: 'Split this 200-line class into smaller components', expected: 'REFACTOR', difficulty: 'easy' },

  // ── REFACTOR (hard — sounds like ADD_FEATURE or BUGFIX) ─────────────────────
  { prompt: 'Make the error handling consistent across all API calls', expected: 'REFACTOR', difficulty: 'hard' },
  { prompt: 'Sostituisci i magic numbers con costanti ben nominate', expected: 'REFACTOR', difficulty: 'hard' },
  { prompt: 'Move the database queries from the controller to a repository layer', expected: 'REFACTOR', difficulty: 'hard' },

  // ── EXPLAIN (easy) ───────────────────────────────────────────────────────────
  { prompt: 'Explain how the KV cache eviction works in this decoder', expected: 'EXPLAIN', difficulty: 'easy' },
  { prompt: "What does the `resetNPast` function do exactly?", expected: 'EXPLAIN', difficulty: 'easy' },
  { prompt: 'Help me understand the speculative decoding algorithm', expected: 'EXPLAIN', difficulty: 'easy' },
  { prompt: 'Cosa fa questo metodo `aggiungi_item`?', expected: 'EXPLAIN', difficulty: 'easy' },
  { prompt: 'Why does the model use attention sinks?', expected: 'EXPLAIN', difficulty: 'easy' },

  // ── EXPLAIN (hard — question framing but really add/fix) ────────────────────
  { prompt: 'How would you add rate limiting to this API?', expected: 'EXPLAIN', difficulty: 'hard' },
  { prompt: "What's wrong with this loop?", expected: 'EXPLAIN', difficulty: 'hard' },

  // ── ADD_FEATURE (easy) ───────────────────────────────────────────────────────
  { prompt: 'Add IVA calculation to the cart module', expected: 'ADD_FEATURE', difficulty: 'easy' },
  { prompt: 'Implement dark mode support in the settings page', expected: 'ADD_FEATURE', difficulty: 'easy' },
  { prompt: 'Create a new REST endpoint for bulk user import', expected: 'ADD_FEATURE', difficulty: 'easy' },
  { prompt: 'Aggiungi la logica di sconto per clienti premium', expected: 'ADD_FEATURE', difficulty: 'easy' },
  { prompt: 'Build a rate limiter middleware for the Express app', expected: 'ADD_FEATURE', difficulty: 'easy' },

  // ── ADD_FEATURE (hard — looks like BUGFIX because of "not working") ──────────
  { prompt: 'The export button is missing — add it to the toolbar', expected: 'ADD_FEATURE', difficulty: 'hard' },
  { prompt: 'Users need to be able to filter results by date range', expected: 'ADD_FEATURE', difficulty: 'hard' },

  // ── WRITE_TEST (easy) ────────────────────────────────────────────────────────
  { prompt: 'Write unit tests for the carrello module', expected: 'WRITE_TEST', difficulty: 'easy' },
  { prompt: 'Add test coverage for the edge cases in calcola_totale', expected: 'WRITE_TEST', difficulty: 'easy' },
  { prompt: 'Create integration tests for the login flow', expected: 'WRITE_TEST', difficulty: 'easy' },
  { prompt: 'Scrivi i test per il modulo di autenticazione', expected: 'WRITE_TEST', difficulty: 'easy' },

  // ── WRITE_TEST (hard — "test" used ambiguously) ──────────────────────────────
  { prompt: 'Verify that the IVA calculation handles discount codes correctly', expected: 'WRITE_TEST', difficulty: 'hard' },
  { prompt: 'Make sure the empty cart edge case is properly handled in tests', expected: 'WRITE_TEST', difficulty: 'hard' },

  // ── UNKNOWN ──────────────────────────────────────────────────────────────────
  { prompt: 'Hello, how are you?', expected: 'UNKNOWN', difficulty: 'easy' },
  { prompt: 'What time is it in Tokyo?', expected: 'UNKNOWN', difficulty: 'easy' },
  { prompt: 'Dammi una ricetta per la pasta al pomodoro', expected: 'UNKNOWN', difficulty: 'easy' },
];
