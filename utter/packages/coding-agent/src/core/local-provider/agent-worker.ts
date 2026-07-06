/**
 * agent-worker.ts — worker-thread entry for the subvocal-local provider.
 *
 * Owns the model and the LocalConversationEngine so the synchronous decode loop
 * never touches the TUI's event loop (same rationale — and same client shape —
 * as encode's M11.2 intent worker). Protocol:
 *
 *   main → worker:  { id, type: 'turn', req: TurnRequest }
 *                   { type: 'reset' }        — drop the conversation (e.g. after an abort)
 *   worker → main:  { type: 'ready' }        — model loaded
 *                   { id, type: 'token', text }
 *                   { id, type: 'result', result: TurnResult }
 *                   { id, type: 'error', error }
 *
 * Mid-turn abort can't arrive as a message (the decode loop blocks this thread's
 * event loop too) — the client sets workerData.abortSab[0] = 1 and AgentLoop's
 * shouldStop hook reads it with Atomics between decode iterations.
 */

import { parentPort, workerData } from "node:worker_threads";
import { LocalConversationEngine } from "./conversation.ts";
import type { TurnRequest } from "./wire.ts";

if (!parentPort) throw new Error("agent-worker must run as a worker thread");
const port = parentPort;

const abortFlag = new Int32Array(workerData.abortSab as SharedArrayBuffer);
const engine = new LocalConversationEngine();

port.on("message", async (msg: { id?: number; type: string; req?: TurnRequest }) => {
	if (msg.type === "reset") {
		engine.resetConversation();
		return;
	}
	if (msg.type !== "turn" || msg.id === undefined || !msg.req) return;
	const id = msg.id;
	try {
		const result = await engine.runTurn(msg.req, {
			onToken: (text) => port.postMessage({ id, type: "token", text }),
			shouldStop: () => Atomics.load(abortFlag, 0) === 1,
		});
		port.postMessage({ id, type: "result", result });
	} catch (error) {
		port.postMessage({ id, type: "error", error: error instanceof Error ? error.message : String(error) });
	}
});

// Eager model load: happens off the TUI thread by construction, so unlike the removed
// main-thread prewarm it can't deadlock stdin — and the first turn doesn't pay the ~1min
// 12B load. 'ready' is informational; turns queue on the port either way.
try {
	engine.warm();
	port.postMessage({ type: "ready" });
} catch (error) {
	port.postMessage({ type: "ready", error: error instanceof Error ? error.message : String(error) });
}
