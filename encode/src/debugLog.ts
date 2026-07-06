/**
 * debugLog.ts — opt-in diagnostic logging.
 *
 * The pipeline's progress chatter (intent, AST tags, cache hits, routing) is useful when
 * debugging but pure noise in a real frontend (it interleaves with the TUI's own rendering).
 * Gate it behind SUBVOCAL_DEBUG so a normal run is quiet; set SUBVOCAL_DEBUG=1 to see it all.
 * Pairs with the native ggml/llama log silencer in the binding, which reads the same env.
 */
export const DEBUG_ENABLED = !!process.env.SUBVOCAL_DEBUG;

/** console.log only when SUBVOCAL_DEBUG is set. */
export function dlog(...args: unknown[]): void {
  if (DEBUG_ENABLED) console.log(...args);
}
