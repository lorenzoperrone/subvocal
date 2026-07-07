/**
 * provider.ts
 *
 * The `subvocal-local` inference provider: runs a GGUF model fully in-process
 * via the @subvocal/ffi-native N-API binder, with no HTTP and no cloud.
 *
 * v4 (M13 P1 fix): the model + AgentLoop live in a WORKER THREAD
 * (agent-worker.ts / conversation.ts). The decode loop is synchronous JS over
 * sync N-API calls — on the main thread it froze the TUI for the whole
 * generation (no rendering, no streaming, Esc ignored). Now:
 *   - tokens stream back per-decode-round (`token` messages → `text_delta`s),
 *   - the TUI stays responsive during generation,
 *   - Esc aborts MID-TURN: the abort flag lives in a SharedArrayBuffer that
 *     AgentLoop's shouldStop hook reads with Atomics between decode iterations
 *     (a postMessage could never interrupt a blocked worker event loop).
 *
 * Turn mapping (v3, unchanged, now inside the worker): first user message →
 * start(), toolResult(s) → continue(), later user messages → followUp() — the
 * KV cache persists across the conversation and the model only ever sees its
 * own real chat template. The harness's tools are declared via the M8 native
 * protocol (see wire.ts for the conversion).
 */

import {
	type AssistantMessage,
	type Context,
	createAssistantMessageEventStream,
	type Message,
	type Model,
	type SimpleStreamOptions,
	type StreamFunction,
	type TextContent,
	type ThinkingContent,
	type ToolCall,
	type Usage,
} from "@earendil-works/pi-ai";
import { registerApiProvider } from "@earendil-works/pi-ai/compat";
import { Worker } from "node:worker_threads";
import { logRawDelta, logRawTurnStart, logToolActivity } from "./activity-log.ts";
import { lastGenStats } from "./gen-stats.ts";
import { StreamSanitizer } from "./stream-sanitize.ts";
import { toToolDefinitions, type TurnRequest, type TurnResult } from "./wire.ts";

export const SUBVOCAL_LOCAL_API = "subvocal-local";
export const SUBVOCAL_LOCAL_PROVIDER = "subvocal-local";
export const SUBVOCAL_LOCAL_MODEL_ID = "subvocal-local-gemma";
export const SUBVOCAL_LOCAL_MODEL_NAME = "Subvocal Local Gemma 4 (FFI)";

// ── Worker client ─────────────────────────────────────────────────────────────

interface TurnCallbacks {
	onToken: (text: string) => void;
	resolve: (result: TurnResult) => void;
	reject: (error: Error) => void;
}

/**
 * Main-thread client for the agent worker. Same lifecycle shape as encode's
 * M11.2 WorkerIntentClassifier: lazy spawn, pending-map dispatch, any worker
 * failure rejects in-flight turns and the next call respawns (model reload is
 * the acceptable price of a crashed worker).
 */
class AgentWorkerClient {
	private worker: Worker | null = null;
	private nextId = 1;
	private pending = new Map<number, TurnCallbacks>();
	private abortFlag = new Int32Array(new SharedArrayBuffer(4));
	/** Serialize turns — the engine is single-conversation, single-decode. */
	private queue: Promise<unknown> = Promise.resolve();

	private getWorker(): Worker {
		if (this.worker) return this.worker;
		// Under tsx the source .ts is the real module; a compiled build ships .js next to this
		// file instead. Resolve whichever variant this module itself is running as.
		const ext = import.meta.url.endsWith(".ts") ? ".ts" : ".js";
		const worker = new Worker(new URL(`./agent-worker${ext}`, import.meta.url), {
			workerData: { abortSab: this.abortFlag.buffer },
		});
		worker.on("message", (msg: { id?: number; type: string; text?: string; result?: TurnResult; error?: string }) => {
			if (msg.id === undefined) return; // 'ready' — informational
			const entry = this.pending.get(msg.id);
			if (!entry) return;
			if (msg.type === "token" && typeof msg.text === "string") {
				entry.onToken(msg.text);
			} else if (msg.type === "result" && msg.result) {
				this.pending.delete(msg.id);
				if (this.pending.size === 0) this.worker?.unref();
				entry.resolve(msg.result);
			} else if (msg.type === "error") {
				this.pending.delete(msg.id);
				if (this.pending.size === 0) this.worker?.unref();
				entry.reject(new Error(msg.error ?? "worker turn failed"));
			}
		});
		const die = (why: string) => {
			for (const { reject } of this.pending.values()) reject(new Error(why));
			this.pending.clear();
			this.worker = null; // next turn respawns
		};
		worker.on("error", (e) => die(`agent worker error: ${e.message}`));
		worker.on("exit", (code) => {
			if (code !== 0) die(`agent worker exited (${code})`);
		});
		// Idle workers must not hold the process open on quit — but an IN-FLIGHT turn must:
		// runTurn() ref()s the worker while anything is pending and unref()s when the pending
		// map drains (an unref'd worker lets Node exit mid-decode, silently dropping the turn).
		worker.unref();
		this.worker = worker;
		return worker;
	}

	/** Request a cooperative mid-turn abort (read by AgentLoop between decode iterations). */
	requestAbort(): void {
		Atomics.store(this.abortFlag, 0, 1);
	}

	/** Drop the worker-side conversation so the next turn re-prefills from scratch. */
	resetConversation(): void {
		this.worker?.postMessage({ type: "reset" });
	}

	runTurn(req: TurnRequest, onToken: (text: string) => void): Promise<TurnResult> {
		const run = () =>
			new Promise<TurnResult>((resolve, reject) => {
				const worker = this.getWorker();
				const id = this.nextId++;
				Atomics.store(this.abortFlag, 0, 0); // fresh turn, clear any stale abort
				this.pending.set(id, { onToken, resolve, reject });
				worker.ref(); // keep the process alive while a turn is in flight
				worker.postMessage({ id, type: "turn", req });
			});
		const turn = this.queue.then(run, run);
		this.queue = turn.catch(() => undefined);
		return turn;
	}
}

const client = new AgentWorkerClient();

// ── Message plumbing helpers ──────────────────────────────────────────────────

function emptyUsage(input: number, output: number): Usage {
	return {
		input,
		output,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: input + output,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	} as Usage;
}

function baseMessage(model: Model<string>): AssistantMessage {
	return {
		role: "assistant",
		content: [],
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: emptyUsage(0, 0),
		stopReason: "stop",
		timestamp: Date.now(),
	} as AssistantMessage;
}

// ── The stream function ───────────────────────────────────────────────────────

export const streamSimple: StreamFunction<string, SimpleStreamOptions> = (model, context, options) => {
	const stream = createAssistantMessageEventStream();

	queueMicrotask(async () => {
		const signal = options?.signal;
		const onAbort = () => client.requestAbort();
		try {
			const partial: AssistantMessage = { ...baseMessage(model), content: [] };
			stream.push({ type: "start", partial: { ...partial } });

			if (signal?.aborted) onAbort();
			signal?.addEventListener("abort", onAbort, { once: true });

			const req: TurnRequest = {
				...(context.systemPrompt ? { systemPrompt: context.systemPrompt } : {}),
				messages: context.messages as Message[],
				toolDefs: toToolDefinitions(context.tools ?? []),
				options: {
					...(options?.maxTokens !== undefined || model.maxTokens ? { maxTokens: options?.maxTokens ?? model.maxTokens } : {}),
					...(options?.temperature !== undefined ? { temperature: options.temperature } : {}),
				},
				cwd: process.cwd(),
			};

			// Live streaming: the RAW decode contains protocol tokens (native tool-call syntax,
			// <|channel> markers, ⊂ideogram⊃ edits, bare AST/CRC anchors) the user should never
			// see scroll by. StreamSanitizer emits only the human-readable prose; the final
			// `done` message still carries the authoritative PARSED content. Markers are whole
			// single tokens (<|tool_call>, ⊂), so the clean text only ever grows — no resets in
			// practice, but the branch is handled for safety.
			// Monitor window (raw mode): mark the turn and mirror the literal decode stream.
			const lastUser = [...(context.messages ?? [])].reverse().find((m) => m.role === "user");
			logRawTurnStart(typeof lastUser?.content === "string" ? lastUser.content : "");

			const sanitizer = new StreamSanitizer();
			let textStarted = false;
			const result = await client.runTurn(req, (tokenText) => {
				logRawDelta(tokenText);
				const r = sanitizer.feed(tokenText);
				const clean = sanitizer.clean;
				if (clean.length === 0) return; // still inside a suppressed region
				if (!textStarted) {
					textStarted = true;
					partial.content = [{ type: "text", text: "" } as TextContent];
					stream.push({ type: "text_start", contentIndex: 0, partial: { ...partial } });
				}
				(partial.content[0] as TextContent).text = clean;
				const delta = "append" in r ? r.append : ""; // reset: partial carries full state
				stream.push({ type: "text_delta", contentIndex: 0, delta, partial: { ...partial } });
			});
			if (textStarted) {
				stream.push({ type: "text_end", contentIndex: 0, content: sanitizer.clean, partial: { ...partial } });
			}

			const { step, inputTokens } = result;

			// Record this decode step's throughput for the footer (tool-only turns count too — the
			// worker times the decode, not the visible stream). Skip trivial/degenerate timings.
			if (step.genMs > 20 && step.tokenCount > 1) {
				lastGenStats.tokens = step.tokenCount;
				lastGenStats.tokPerSec = step.tokenCount / (step.genMs / 1000);
			}

			if (signal?.aborted) {
				// The partial generation is committed in the worker's KV but the harness won't
				// append our (aborted) reply — drop the conversation so the next turn re-prefills
				// a consistent history instead of continuing a half-finished model turn.
				client.resetConversation();
				const aborted = { ...partial, stopReason: "aborted", errorMessage: "aborted" } as AssistantMessage;
				stream.push({ type: "error", reason: "aborted", error: aborted });
				stream.end(aborted);
				return;
			}

			// ── Canonical content blocks (authoritative, parsed) ────────────────
			const content: AssistantMessage["content"] = [];
			if (step.thinking.length > 0) content.push({ type: "thinking", thinking: step.thinking } as ThinkingContent);
			if (step.text.length > 0) content.push({ type: "text", text: step.text } as TextContent);

			let contentIndex = content.length;
			partial.content = [...content];
			for (const tc of step.toolCalls) {
				logToolActivity(tc.name, tc.arguments); // monitor window (structured mode)
				const toolCall: ToolCall = { type: "toolCall", id: tc.id, name: tc.name, arguments: tc.arguments };
				partial.content = [...partial.content, { ...toolCall, arguments: {} } as ToolCall];
				stream.push({ type: "toolcall_start", contentIndex, partial: { ...partial } });
				stream.push({ type: "toolcall_delta", contentIndex, delta: JSON.stringify(tc.arguments), partial: { ...partial } });
				(partial.content[contentIndex] as ToolCall).arguments = tc.arguments;
				stream.push({ type: "toolcall_end", contentIndex, toolCall, partial: { ...partial } });
				content.push(toolCall);
				contentIndex++;
			}

			const finalReason = step.toolCalls.length > 0 ? "toolUse" : step.stoppedNaturally ? "stop" : "length";
			const finalMessage: AssistantMessage = {
				...partial,
				content,
				usage: emptyUsage(inputTokens, step.tokenCount),
				stopReason: finalReason,
				timestamp: Date.now(),
			};
			stream.push({ type: "done", reason: finalReason, message: finalMessage });
			stream.end(finalMessage);
		} catch (error) {
			const message = {
				...baseMessage(model),
				stopReason: "error",
				errorMessage: error instanceof Error ? error.message : String(error),
			} as AssistantMessage;
			stream.push({ type: "error", reason: "error", error: message });
			stream.end(message);
		} finally {
			signal?.removeEventListener("abort", onAbort);
		}
	});

	return stream;
};

/** Register the subvocal-local api handler on the global registry. Idempotent per process. */
export function registerLocalProvider(): void {
	registerApiProvider({ api: SUBVOCAL_LOCAL_API, stream: streamSimple, streamSimple }, SUBVOCAL_LOCAL_API);
}

interface ProviderRegistrar {
	registerProvider(
		providerName: string,
		config: {
			api: string;
			streamSimple: StreamFunction<string, SimpleStreamOptions>;
			baseUrl?: string;
			apiKey?: string;
			models?: Array<{
				id: string;
				name: string;
				api?: string;
				reasoning: boolean;
				input: ("text" | "image")[];
				cost: { input: number; output: number; cacheRead: number; cacheWrite: number };
				contextWindow: number;
				maxTokens: number;
			}>;
		},
	): void;
}

/**
 * Register the local FFI brain as a first-class provider + model on a
 * ModelRegistry so the agent loop can select and stream from it. The dummy
 * `baseUrl`/`apiKey` only satisfy registry validation — the provider runs the
 * GGUF in-process and ignores both.
 */
export function registerLocalProviderModels(registry: ProviderRegistrar): void {
	// Mirrors the worker's LocalConversationEngine.resolveContextSize() (single source of
	// truth over there — this file can't import @subvocal/encode: main thread, no native
	// addons): 32k single-model, 16k when a second E2B instance is resident (drafter or
	// dual-brain generator), SUBVOCAL_LOCAL_CTX overrides. Was hardcoded 8192, which made
	// the statusbar % and the harness's auto-compaction budget disagree with the real window.
	const contextWindow = process.env.SUBVOCAL_LOCAL_CTX
		? Number(process.env.SUBVOCAL_LOCAL_CTX)
		: process.env.SUBVOCAL_LOCAL_DRAFT !== "0" || process.env.SUBVOCAL_LOCAL_DUAL_BRAIN !== "0"
			? 16384
			: 32768;
	registry.registerProvider(SUBVOCAL_LOCAL_PROVIDER, {
		api: SUBVOCAL_LOCAL_API,
		streamSimple,
		baseUrl: "local://subvocal-ffi",
		apiKey: "local",
		models: [
			{
				id: SUBVOCAL_LOCAL_MODEL_ID,
				name: SUBVOCAL_LOCAL_MODEL_NAME,
				api: SUBVOCAL_LOCAL_API,
				reasoning: true,
				input: ["text"],
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
				contextWindow,
				maxTokens: 4096,
			},
		],
	});
}
