/**
 * gen-stats.ts — tiny shared main-thread readout of the last decode step's throughput.
 *
 * The subvocal-local provider writes it after each turn; the footer component reads it to show a
 * tok/s figure. Kept in its own module (no imports) so the footer doesn't pull in the provider's
 * worker/native machinery just to read a number. Both writer and reader run on the main thread.
 */
export const lastGenStats: { tokPerSec: number; tokens: number } = { tokPerSec: 0, tokens: 0 };
