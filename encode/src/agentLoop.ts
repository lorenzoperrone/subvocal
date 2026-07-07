/**
 * agentLoop.ts
 *
 * Epico 4.1 + 4.2 — Loop Cibernetico con Ideogram Steering
 *
 * Orchestrates a multi-turn coding session using the GPU model with
 * incremental decode (decodeAppend). Replaces the O(n²) stateless forward()
 * loop with O(n) KV-cache-preserving generation.
 *
 * Supports two edit modes:
 *   - Text-based: model outputs tool calls, preferring this checkpoint's own native
 *     <|tool_call>call:NAME{...}<tool_call|> protocol (see toolParse.ts and Epic M8) when the
 *     active profile declares tools natively, falling back to Hermes/Qwen JSON otherwise
 *   - Ideogram steering: model outputs ⊂ → tag → code → ⊃ sequences
 *
 * Flow:
 *   1. preprocess(input) → intent + AST-tagged code (via small CPU model)
 *   2. Build prompt with tagged source context
 *   3. forward() prefills the KV cache once
 *   4. decodeAppend([token]) autoregressively generates the response
 *   5. Parse edits (text tool calls or ideogram sequences)
 *   6. Caller executes tool, feeds observation back via continue()
 *   7. decodeAppend(observation) appends to KV cache, loop continues
 */

import { ModelGPU, sample, sampleGreedy, SpeculativeDecoder, type DecoderConfig } from '@subvocal/synapse';
import { activeProfile, type ModelProfile } from './modelProfile.js';
import { preprocess, type PreprocessInput, type PreprocessResult } from './index.js';
import { parseAssistantOutput, type ParsedToolCall, type ToolDefinition } from './toolParse.js';
import { IdeogramSteering, TURN_POISON, type IdeogramEdit, type SteeringConfig } from './ideogramSteering.js';
import { intentChar, intentLegend } from './ideogramAllocator.js';
import { getSmallModel } from './smallModel.js';
import { KVColdStore } from './kvColdStore.js';
import { homedir } from 'node:os';
import { basename, join } from 'node:path';

export type { ParsedToolCall, IdeogramEdit };

/** One replayed turn for AgentLoop.replay() — a restored conversation's building block. */
export interface ReplayTurn {
	kind: 'assistant' | 'tool' | 'user';
	/** Assistant raw text (tool-call blocks included, in the profile's surface syntax), tool observation, or user prompt. */
	text: string;
	/** For kind 'tool': the tool that produced the observation. */
	toolName?: string;
}

/** Default disk budget for the auto-constructed cold store (see AgentLoopConfig.coldStore). */
const DEFAULT_COLD_STORE_BUDGET_BYTES = 4 * 1024 * 1024 * 1024; // 4 GiB

/** M3.7: checkpoint ladder granularity — a rung roughly every this many tokens. */
const COLD_BLOCK_TOKENS = 4096;
/** M3.7: max intermediate rungs written during one initial prefill (the full-prompt
 *  checkpoint is always written on top of these). Bounds the transient buffer memory
 *  the fire-and-forget writes hold before compression shrinks them. */
const COLD_MAX_RUNGS = 3;

/**
 * Epic M8 spike (doc/epics/EPIC-M8-grammar-constrained-tool-calls.md): measured a 27%
 * tool-call malformation rate under non-greedy sampling, and EVERY malformed case had a
 * perfectly valid JSON body -- the corruption was always this one literal character, the
 * closing '>' of the Hermes/Qwen "<tool_call>" opening tag (a dropped '>' or merged tokens).
 * forceToolCallTagClose() below fixes exactly that, nothing else -- not a grammar engine.
 */
const TOOL_CALL_TAG_PREFIX = '<tool_call';
const TOOL_CALL_TAG_CLOSE = '>';

/**
 * The two tools AgentLoop exposes to the model today. Kept here (not toolParse.ts) since
 * toolParse.ts is intentionally model/task-agnostic parsing machinery, while this is
 * Subvocal's own agent-loop-specific tool surface -- matches where the equivalent free-text
 * instructions already lived before the Epic M8 native-protocol work.
 */
const AGENT_TOOLS: ToolDefinition[] = [
	{
		name: 'edit',
		description: 'Replace an exact block of existing code with new code in a file.',
		parameters: {
			properties: {
				file: { type: 'STRING', description: 'Path to the file to edit.' },
				old: { type: 'STRING', description: 'The exact existing code to find and replace.' },
				new: { type: 'STRING', description: 'The new code to replace it with.' },
			},
			required: ['file', 'old', 'new'],
		},
	},
	{
		name: 'bash',
		description: 'Run a shell command.',
		parameters: {
			properties: {
				command: { type: 'STRING', description: 'The shell command to execute.' },
			},
			required: ['command'],
		},
	},
];

// ── Config ────────────────────────────────────────────────────────────────────

export interface AgentLoopConfig {
	/** The large GPU model (already loaded). */
	model: ModelGPU;
	/** Cap on generated tokens per turn. */
	maxTokens?: number;
	/** Temperature for sampling. Defaults to the active profile's defaultTemperature (0 if unset = greedy). */
	temperature?: number;
	/**
	 * Nucleus-sampling cutoff, defaults to the active profile's defaultTopP. Strongly
	 * recommended whenever temperature > 0 -- without topP/topK, sample() draws from the
	 * full vocabulary's temperature-scaled distribution with no truncation, which on a
	 * large vocab can produce garbage/foreign-script tokens from the long tail.
	 */
	topP?: number;
	/** Top-K cutoff, defaults to the active profile's defaultTopK. */
	topK?: number;
	/** Override the active model profile (default: SUBVOCAL_MODEL_PROFILE). */
	profile?: ModelProfile;
	/**
	 * Tools declared to the model (via the profile's native protocol when it has one — Epic
	 * M8 — free-text instructions otherwise). Default: AgentLoop's own built-in `edit`/`bash`
	 * pair. A frontend harness that executes its OWN tool set (e.g. the utter TUI's pi tools)
	 * MUST pass that set here, or the model gets declarations whose schemas don't match what
	 * the executor accepts.
	 */
	tools?: ToolDefinition[];
	/**
	 * Base system prompt override for tool-calling mode. When set, replaces AgentLoop's
	 * built-in "precise coding agent" preamble (tool declarations are still appended). Lets a
	 * frontend harness keep its own agent instructions authoritative. Ignored when
	 * `useSteering` is on (steering has its own fixed protocol prompt).
	 */
	systemPrompt?: string;
	/** Enable ideogram token steering (Substory 4.2). Default: true. */
	useSteering?: boolean;
	/**
	 * M15.1: which system prompt steering mode uses. 'exclusive' (default, the pre-M15
	 * behavior) teaches ONLY the ⊂tag code⊃ protocol and forbids tool calls — the legacy
	 * REPL's edit-centric mode. 'hybrid' keeps the tool-calling prompt (harness override +
	 * native declarations) and ADDS the ideogram protocol for single-node edits — the model
	 * picks the cheap path for edits (~20% fewer tokens, M4 showdown) and tools for
	 * everything else. Ignored when useSteering is off.
	 */
	steeringPrompt?: 'exclusive' | 'hybrid';
	/**
	 * What to do when the turn has NO file in context (e.g. a from-scratch "create X" request).
	 * 'fill' (default, legacy REPL single-file mode): instruct the model to output ONLY raw file
	 * content, tool calls forbidden — right for a "here is an empty file, fill it" REPL flow.
	 * 'tools' (agentic frontends like the TUI): use the normal tool-calling prompt so the model
	 * CREATES the file with a `write` tool call instead of dumping content/prose. Without this,
	 * every create-from-scratch task hits the fill branch and the model can never call a tool —
	 * measured 2026-07-05: the 12B emits a perfect `write` call WITH the tool prompt, but the fill
	 * branch tells it "No tool calls… Do NOT wrap output in <tool_call>", so it dumps Python/prose.
	 */
	emptyFileMode?: 'fill' | 'tools';
	/**
	 * M16.1: render the active-file block with a 1-based line-number gutter ("  42| code") so the
	 * model can address exact ranges via a line-anchored edit tool. Only applies to the raw
	 * (non-steering) view — the ideogram-tagged view has its own markers. Default: false.
	 */
	lineGutter?: boolean;
	/** Enable speculative decoding (Substory 13). Default: false. */
	useSpeculation?: boolean;
	/**
	 * Enable suffix-tree (prompt-lookup) speculative decoding (see
	 * doc/research/suffix-tree-speculative-decoding.md). Free drafter (no model, no GPU
	 * cost) — net-positive (1.2-1.3x) on edit/refactor turns where the model mostly
	 * reproduces context already seen, negative on from-scratch generation (cold trie).
	 * Different mechanism from `useSpeculation` (which uses a real CPU drafter model with
	 * a probability-threshold accept rule); this one's accept rule is hard greedy equality,
	 * so it engages when the decode is effectively greedy: `temperature === 0`, OR ideogram
	 * steering active (IdeogramSteering.sample() is argmax over unmodified logits — a pure
	 * observer, never a mask). Falls back to plain per-token decoding otherwise. Takes
	 * priority over `useSpeculation` only when the latter's `specDecoder` isn't active for
	 * that turn (e.g. continue()/followUp() — `useSpeculation`'s decoder is constructed
	 * fresh only in start()). Default: false.
	 */
	useSuffixSpeculation?: boolean;
	/**
	 * M12.3: two-model speculative decoding — a small same-tokenizer model (E2B) drafts, this
	 * loop's model verifies in one batched decode per round. Measured 1.90x on edit-shaped
	 * prompts at 99.5% acceptance (doc/research/exclusions-sweep-2026-07.md; the old "CPU-draft
	 * 1.9x SLOWER" negative was about CPU placement, not the mechanism). The drafter keeps a
	 * SHADOW KV synced with this loop's context (prefilled in start()/replay(), advanced in
	 * continue()/followUp() and per round); accept rule is hard greedy equality (mtpVerifyBatch
	 * argmax) — output is bit-identical to plain greedy, same contract as the suffix drafter.
	 * Engages under the same gate as suffix speculation (greedy decode). Composes with the
	 * suffix trie: a trie hit drafts for free, the model drafts otherwise. Any drafter-side
	 * error disables drafting for the session (never correctness — the target loop stands alone).
	 */
	draftModel?: ModelGPU;
	/** Adaptive draft length: starting K. Doubles on full acceptance, halves on rejection. Default 8. */
	draftInitK?: number;
	/** Adaptive draft length bounds. Defaults 4 / 32 (K=32 measured 1.90x at high acceptance). */
	draftMinK?: number;
	draftMaxK?: number;
	/** Max draft tokens per suffix-tree round. Default 8 (validated value). */
	suffixMaxSpec?: number;
	/** Min cumulative probability to keep extending a draft. Default 0.05 (validated value). */
	suffixMinTokenProb?: number;
	/** Min times a suffix must have been seen to draft from it. Default 1. */
	suffixMinMatchCount?: number;
	/** Min matching suffix length before drafting. Default 3 (validated value). */
	suffixMinMatchLen?: number;
	/** Callback for streaming token output. Called per-token during decode. */
	onToken?: (text: string) => void;
	/**
	 * Cooperative mid-turn abort. Checked between decode iterations (per token on the plain
	 * loop, per round on the suffix-speculation loop); when it returns true the turn ends
	 * gracefully with whatever was generated (`stoppedNaturally: false`). The decode loop is
	 * synchronous JS, so a worker-thread host can't receive abort messages mid-turn — pass a
	 * closure over a SharedArrayBuffer flag (`() => Atomics.load(flag, 0) === 1`) instead.
	 */
	shouldStop?: () => boolean;
	/**
	 * Epic M3 cold tier (see encode/src/kvColdStore.ts and
	 * doc/research/ds4-kvstore-findings.md). `start()` tries to resume from a
	 * disk-persisted KV checkpoint instead of re-prefilling the whole prompt from scratch —
	 * validated at ~13.7x standalone / ~10.9x wired into start() for a 2041-token context
	 * (see doc/research/m1-metal-benchmark.md). Default: ON, using a default store under
	 * `~/.cache/subvocal/kv-cold` (4 GiB budget) -- this is disk I/O, not GPU bandwidth, so
	 * even a one-shot session's worst case (a write that's never reused) doesn't compete
	 * with the scarce resource we're optimizing. Pass a `KVColdStore` instance to use your
	 * own path/budget, or `null` to disable entirely.
	 */
	coldStore?: KVColdStore | null;
}

export interface AgentStep {
	/** Visible text output (tool-call blocks stripped). */
	text: string;
	/** Thinking content inside <think> blocks. */
	thinking: string;
	/** Parsed tool calls (native protocol preferred, Hermes/Qwen JSON/XML fallback; empty if steering active). */
	toolCalls: ParsedToolCall[];
	/** Ideogram edit detected by steering (Substory 4.2), or null. */
	ideogramEdit: IdeogramEdit | null;
	/** Number of tokens generated in this turn. */
	tokenCount: number;
	/** True if a stop token was hit (not truncated by maxTokens). */
	stoppedNaturally: boolean;
}

// ── Implementation ────────────────────────────────────────────────────────────

export class AgentLoop {
	private model: ModelGPU;
	private maxTokens: number;
	private temperature: number;
	private topP?: number;
	private topK?: number;
	private profile: ModelProfile;
	private nPast = 0;
	private firstTurn = true;
	private emptyFileMode: 'fill' | 'tools' = 'fill';
	private lineGutter = false;
	private lastFilePath = '';
	// Raw (un-tagged) source of the current file. Shown to the model instead of the AST-tagged
	// version when steering is OFF — otherwise the injected marker chars (∀ Ω λ ∑ …) are dead
	// weight AND collide with any real source that legitimately uses those Unicode chars (Greek /
	// math operators in ML code, comments, string literals), which the downstream detag would
	// then strip out of edits and corrupt them. Tags are only meaningful when steering consumes them.
	private lastRawSource = '';
	private steering: IdeogramSteering | null = null;
	private useSteering: boolean;
	private onToken?: (text: string) => void;
	private shouldStop?: () => boolean;
	private steeringPromptMode: 'exclusive' | 'hybrid' = 'exclusive';
	private specDecoder: SpeculativeDecoder | null = null;
	private useSpeculation: boolean;
	private useSuffixSpeculation: boolean;
	private suffixMaxSpec: number;
	private suffixMinTokenProb: number;
	private suffixMinMatchCount: number;
	private suffixMinMatchLen: number;
	private suffixUncommittedTail = 0;
	/** Sliding window of recent KV-cache tokens, fed to suffixTreeSpeculate() as match context. */
	private recentTokens: number[] = [];
	/**
	 * File contents already extended into the *current* (not-yet-cleared) suffix trie via
	 * prewarmSuffixTree(). Lets start() skip a wasteful clear+reseed when this exact turn's
	 * file was already prewarmed. Reset whenever the trie itself is cleared.
	 */
	private suffixPrewarmedSet = new Set<string>();
	private coldStore: KVColdStore | null = null;
	/** In-flight background cold-store checkpoint writes (see prefillOrResume + flushColdWrites). */
	private pendingColdWrites: Promise<void>[] = [];
	/** Tail of the serialized checkpoint-write chain — see queueColdWrite's RAM-bounding note. */
	private coldWriteChain: Promise<void> = Promise.resolve();
	/** M3.7: exact rendered text mirror of the KV cache content (prompt + generated + turn
	 *  scaffolding), used to key mid-session checkpoints. Best-effort: a BPE roundtrip mismatch
	 *  at restore time is caught by prefillOrResume's sanity check, never trusted silently. */
	private transcript = '';
	/** M3.7: nPast at the last cold checkpoint — mid-session checkpoints fire on +COLD_BLOCK_TOKENS. */
	private lastColdCheckpointTokens = 0;
	/** Tag injection list from last preprocess — needed by astEditor for ideogram resolution. */
	public lastInjections: import('./astTagger.js').TagInjection[] = [];
	/**
	 * Name of the most recently parsed tool call, updated at the end of decodeLoop(). Used as
	 * continue()'s default `toolName` so buildToolResponse() can render the real tool name
	 * (native protocols need it) without every caller having to track and pass it themselves.
	 */
	private lastToolCallName = 'tool';
	private tools: ToolDefinition[];
	private systemPromptOverride?: string;
	// M12.3 drafter state. draftReady flips false on ANY drafter-side problem (context overflow,
	// decode error) — the target loop then simply runs without model drafts for the session.
	private draftModel: ModelGPU | null;
	private draftReady = false;
	private draftNPast = 0;
	private draftK: number;
	private draftInitK: number;
	private draftMinK: number;
	private draftMaxK: number;

	constructor(config: AgentLoopConfig) {
		this.model = config.model;
		this.maxTokens = config.maxTokens ?? 4096;
		this.profile = config.profile ?? activeProfile;
		this.temperature = config.temperature ?? this.profile.defaultTemperature ?? 0;
		this.topP = config.topP ?? this.profile.defaultTopP;
		this.topK = config.topK ?? this.profile.defaultTopK;
		this.tools = config.tools ?? AGENT_TOOLS;
		this.systemPromptOverride = config.systemPrompt;
		this.draftModel = config.draftModel ?? null;
		this.draftInitK = config.draftInitK ?? 8;
		this.draftMinK = config.draftMinK ?? 4;
		this.draftMaxK = config.draftMaxK ?? 32;
		this.draftK = this.draftInitK;
		this.useSteering = config.useSteering ?? false;
		this.steeringPromptMode = config.steeringPrompt ?? 'exclusive';
		this.emptyFileMode = config.emptyFileMode ?? 'fill';
		this.lineGutter = config.lineGutter ?? false;
		this.useSpeculation = config.useSpeculation ?? false;
		this.useSuffixSpeculation = config.useSuffixSpeculation ?? false;
		this.suffixMaxSpec = config.suffixMaxSpec ?? 8;
		this.suffixMinTokenProb = config.suffixMinTokenProb ?? 0.05;
		this.suffixMinMatchCount = config.suffixMinMatchCount ?? 1;
		this.suffixMinMatchLen = config.suffixMinMatchLen ?? 3;
		this.onToken = config.onToken;
		this.shouldStop = config.shouldStop;
		if (config.coldStore === null) {
			this.coldStore = null; // explicit opt-out
		} else if (config.coldStore) {
			this.coldStore = config.coldStore; // caller-supplied store/path/budget
		} else {
			// Auto-construct from the active profile's cold tier (path + large disk budget).
			// Falls back to the built-in default path/budget for profiles with no cold tier.
			//
			// M12.2: the store lives in a PER-MODEL subdirectory (GGUF basename). Checkpoint
			// keys are prompt-text-only, so without this a checkpoint written by one model
			// would happily restore into a different model sharing the same prompt text —
			// dual-brain routing (12B + E2B generators in one session) made that a real
			// hazard, not a theoretical one. Pre-namespace checkpoints in the parent dir are
			// simply never matched again (harmless leftovers; delete them manually if the
			// disk space matters).
			const cold = this.profile.kvTiers.cold;
			const modelNs = basename(this.profile.largeModelPath).replace(/\.gguf$/i, '');
			this.coldStore = new KVColdStore(
				join(cold?.diskPath ?? join(homedir(), '.cache', 'subvocal', 'kv-cold'), modelNs),
				cold?.budgetBytes ?? DEFAULT_COLD_STORE_BUDGET_BYTES,
			);
		}
	}

	/**
	 * Mechanism C (doc/research/predictive-prefill-while-typing.md /
	 * suffix-tree-speculative-decoding.md): seed the suffix trie with reference content
	 * (typically the file about to be edited, but any related content works -- dependency
	 * files, docs) ahead of calling start(), instead of only at turn-start. Ready-to-call
	 * utility, not wired to any live "file opened" event -- none exists in this codebase yet.
	 *
	 * No-op when suffix speculation is disabled. Requires a loaded model (this.model) but NOT
	 * a started session -- safe to call as soon as the model exists, well before any prompt is
	 * known. start() will skip its usual clear+reseed for any exact content already prewarmed
	 * here, so calling this ahead of time isn't wasted -- but it's a small saving (suffix-tree
	 * insertion itself is cheap, sub-millisecond to low-millisecond for typical file sizes; the
	 * real point is making related-but-not-in-the-prompt content available to match against).
	 */
	prewarmSuffixTree(content: string): void {
		if (!this.useSuffixSpeculation || content.length === 0) return;
		if (this.suffixPrewarmedSet.has(content)) return; // already in the current trie
		const tokens = this.model.tokenize(content, false, false);
		this.model.suffixTreeExtend(tokens);
		this.suffixPrewarmedSet.add(content);
	}

	/**
	 * Start a new conversation turn. Runs the full preprocess pipeline
	 * on the small CPU model, builds a prompt, prefills the GPU KV cache,
	 * and autoregressively decodes the model's response.
	 */
	async start(input: PreprocessInput): Promise<AgentStep> {
		const result = await preprocess(input);
		this.lastFilePath = input.filePath;
		this.lastRawSource = input.fileContent;
		this.lastInjections = result.injections;

		// Init ideogram steering from preprocess result
		if (this.useSteering) {
			this.steering = this.makeSteering(result.tagMap);
		}

		const prompt = this.buildPrompt(result, input.prompt);
		const promptTokens = this.model.tokenize(prompt, true, true);

		if (this.useSpeculation) {
			// Cold-tier resume isn't combined with speculative decoding (SpeculativeDecoder
			// owns its own forward() call internally) — orthogonal feature, not built here.
			try {
				this.specDecoder = new SpeculativeDecoder(
					getSmallModel(),
					this.model,
					{
						draftLength: 7,
						alpha: 0.85,
						temperature: this.temperature,
						maxSteps: Math.floor(this.maxTokens / 7),
						stopTokens: new Set(this.profile.eotTokenIds),
						useTreeSpeculation: false,
						treeBranches: 3,
						attentionSinks: 4,
					}
				);
			} catch {
				this.specDecoder = null;
				await this.model.forwardAsync(promptTokens);
			}
		} else {
			await this.prefillOrResume(prompt, promptTokens);
		}

		this.nPast = promptTokens.length;
		this.firstTurn = false;
		// M3.7: seed the transcript mirror of the KV content for mid-session checkpoint keying.
		this.transcript = prompt;
		this.lastColdCheckpointTokens = promptTokens.length;
		// M12.3: bring the drafter's shadow context up to the same prompt.
		await this.draftPrefill(promptTokens);

		// New session/file — reset and reseed the suffix trie with the fresh prompt, same
		// methodology as the validated bench (clear + seed, see suffix-tree-speculative-decoding.md).
		// Skip the clear when this exact file was already prewarmed via prewarmSuffixTree()
		// ahead of time (mechanism C) -- clearing would discard that head start for nothing.
		if (this.useSuffixSpeculation) {
			if (!this.suffixPrewarmedSet.has(input.fileContent)) {
				this.model.suffixTreeClear();
				this.suffixPrewarmedSet.clear();
			}
			this.recentTokens = [];
			this.feedSuffixTree(promptTokens);
		}

		const step = this.decodeLoop(promptTokens);
		return step;
	}

	/**
	 * M13.3: rebuild a whole conversation's KV WITHOUT generating — the prefill-only
	 * counterpart of start()+continue()+followUp() for restored sessions. The conversation
	 * text is composed exactly as a live session would have built it (same buildPrompt /
	 * buildToolResponse / buildFollowUp composition, assistant outputs appended verbatim), so:
	 *   - the model sees its own real chat template, never a re-serialized story blob;
	 *   - `prefillOrResume` can resume the longest matching prefix from the cold store. The
	 *     ladder rungs over the static prefix (system prompt + file block, deterministic)
	 *     match byte-for-byte; the full-transcript checkpoint written at the end makes a
	 *     SECOND restore of the same session ~free.
	 * Assistant texts re-tokenized from stored messages may differ from the originally
	 * generated token IDs (BPE roundtrip) — harmless for correctness (same text, KV built from
	 * what we append); it only reduces checkpoint hits, and the sanity check below already
	 * guards that class. After replay() the caller drives the LIVE turn with continue() /
	 * followUp() as usual.
	 */
	async replay(input: PreprocessInput, turns: ReplayTurn[]): Promise<void> {
		const result = await preprocess(input);
		this.lastFilePath = input.filePath;
		this.lastRawSource = input.fileContent;
		this.lastInjections = result.injections;
		if (this.useSteering) {
			this.steering = this.makeSteering(result.tagMap);
		}

		let full = this.buildPrompt(result, input.prompt);
		for (const t of turns) {
			if (t.kind === 'assistant') {
				full += t.text;
				this.lastToolCallName = parseAssistantOutput(t.text).toolCalls.at(-1)?.name ?? this.lastToolCallName;
			} else if (t.kind === 'tool') {
				full += this.profile.buildToolResponse(t.text, t.toolName ?? this.lastToolCallName);
			} else {
				full += this.profile.buildFollowUp(t.text);
			}
		}

		const fullTokens = this.model.tokenize(full, true, true);
		await this.prefillOrResume(full, fullTokens);
		this.nPast = fullTokens.length;
		this.firstTurn = false;
		this.transcript = full;
		this.lastColdCheckpointTokens = fullTokens.length;
		// M12.3: bring the drafter's shadow context up to the same replayed conversation.
		await this.draftPrefill(fullTokens);

		if (this.useSuffixSpeculation) {
			if (!this.suffixPrewarmedSet.has(input.fileContent)) {
				this.model.suffixTreeClear();
				this.suffixPrewarmedSet.clear();
			}
			this.recentTokens = [];
			this.feedSuffixTree(fullTokens);
		}
	}

	/**
	 * Prefill the KV cache for `prompt`, resuming from a cold-tier checkpoint if one matches
	 * a prefix of it (see `KVColdStore`). Always leaves the model's KV cache representing
	 * exactly `promptTokens` (the full prompt) when this returns, whether by restore+partial
	 * decode or by a full prefill — callers don't need to know which path was taken.
	 */
	private async prefillOrResume(prompt: string, promptTokens: Int32Array): Promise<void> {
		if (!this.coldStore) {
			const status = await this.model.forwardAsync(promptTokens);
			if (status !== 0) throw new Error(`prefill forward failed (llama_decode status ${status})`);
			return;
		}

		const { matchedChars, matchedTokens } = this.coldStore.tryLoad(this.model, prompt);
		if (matchedTokens > 0) {
			// Sanity check: the matched text must re-tokenize to EXACTLY the full prompt's own
			// first `matchedTokens` token IDs, element-wise — not just to the same COUNT. The
			// count-only check this replaces (2026-07 KV audit) had a silent-corruption hole: a
			// BPE merge across the checkpoint boundary (classic case: a rung ends at "\n" and
			// the new prompt continues with indentation, which the full prompt tokenizes as one
			// fused "\n    " token) can keep the prefix count identical while the token at the
			// seam differs — splicing decodeAppend(promptTokens.slice(matchedTokens)) on top
			// would then silently drop/garble the seam characters in the KV. Same discipline as
			// coldLadderBoundaries' write-side verification, which was already element-wise.
			// (forward() below does llama_memory_clear() first, so falling through after a bad
			// restore still produces a correct, just non-resumed, prefill.)
			const matchedText = prompt.slice(0, matchedChars);
			const sanityTokens = this.model.tokenize(matchedText, true, true);
			let resumable = sanityTokens.length === matchedTokens;
			for (let i = 0; resumable && i < matchedTokens; i++) {
				if (sanityTokens[i] !== promptTokens[i]) resumable = false;
			}
			if (resumable) {
				const remainder = promptTokens.slice(matchedTokens);
				if (remainder.length > 0) {
					// A non-zero status means the KV holds the restored prefix but NOT the
					// remainder — advancing would desync every later position. Fall through to
					// the full prefill instead (forward() clears the partial state first).
					const status = await this.model.decodeAppendAsync(remainder);
					if (status === 0) return; // resumed — KV now represents the full prompt
				} else {
					// Exact full-prompt match (e.g. an identical retry) — kvRestore() brings
					// back the KV cache's history but NOT the last computed logits (those are
					// an ephemeral per-decode output, not part of persisted session state — see
					// the standalone kvColdStore validation notes in
					// doc/epics/EPIC-M3-kv-tiering-unified-memory.md). Without at least one
					// fresh decode, generation would start with no logits to sample from.
					// Rewind the last position and redecode just that one token — cheap (1
					// token), and correct (same KV slot is reused, not appended at a new one).
					this.model.kvCacheSeqRemove(0, promptTokens.length - 1, -1);
					this.model.resetNPast(promptTokens.length - 1);
					const status = await this.model.decodeAppendAsync(promptTokens.slice(-1));
					if (status === 0) return; // resumed
				}
			}
		}

		// No usable checkpoint (or the sanity check failed) — full prefill, then checkpoint
		// it for next time. This is the expensive path the cold tier exists to avoid paying
		// twice for the same static prefix (system prompt, large file context, etc.).
		//
		// M3.7: for long prompts the prefill is STAGED, snapshotting a checkpoint "rung" at
		// verified block boundaries along the way. The payoff is prefix reuse across DIFFERENT
		// prompts: the next REPL turn's prompt shares the static prefix (system prompt + file
		// context) but diverges at the user prompt, so the full-prompt checkpoint below never
		// matches it — a rung inside the shared region does.
		//
		// 2026-07 KV audit: every decode status is checked (a silently-failed chunk used to
		// leave nPast/transcript advanced past a hole AND let the rung below checkpoint a KV
		// that doesn't match its text — a corruption the restore-side sanity check can never
		// catch, since text and token count would both look right). And the ladder is
		// budget-aware: kvSave() serializes the WHOLE context state, so on the 12B every rung
		// carries the ~480 MiB fixed compact-SWA cache no matter how few tokens it covers — a
		// 3-rung ladder plus the full checkpoint was ~2.4+ GiB against MacProfile's 4 GiB
		// budget, thrashing the store's eviction on every long start(). The first rung's
		// actual snapshot size decides: if the full ladder wouldn't fit in half the budget,
		// skip the intermediate rungs (the full-prompt checkpoint — the highest-value one,
		// exact replay resume — is always written).
		const rungs = this.coldLadderBoundaries(prompt, promptTokens);
		let done = 0;
		let rungsSkipped = false;
		for (const rung of rungs) {
			const chunk = promptTokens.subarray(done, rung.tokenCount);
			const status = done === 0
				? await this.model.forwardAsync(chunk)
				: await this.model.decodeAppendAsync(chunk);
			if (status !== 0) throw new Error(`prefill failed at token ${done} (llama_decode status ${status})`);
			done = rung.tokenCount;
			if (rungsSkipped) continue;
			const snap = this.coldStore.snapshot(this.model, rung.text, rung.tokenCount, 'cold');
			const budget = this.coldStore.budget;
			if (budget > 0 && snap.totalSize * (rungs.length + 1) > budget / 2) {
				rungsSkipped = true; // fixed-cost-dominated checkpoints — rungs would thrash the store
			} else {
				this.queueColdWrite(snap);
			}
		}
		const rest = promptTokens.subarray(done);
		if (done === 0 || rest.length > 0) {
			const status = done === 0
				? await this.model.forwardAsync(rest)
				: await this.model.decodeAppendAsync(rest);
			if (status !== 0) throw new Error(`prefill failed at token ${done} (llama_decode status ${status})`);
		}
		// Snapshot the full-prompt KV *now* (synchronous copy, before decode mutates it), then
		// flush it to SSD in the background so the multi-hundred-MB disk write overlaps the
		// response decode instead of blocking before it. Safe: the snapshot is a JS-owned copy,
		// and the write runs on libuv's threadpool, not the decode thread — no compute race, no
		// GPU-bandwidth contention. flushColdWrites() awaits these before the session frees the
		// model.
		this.queueColdWrite(this.coldStore.snapshot(this.model, prompt, promptTokens.length, 'cold'));
	}

	/**
	 * Fire-and-forget a checkpoint write, tracked for flushColdWrites(). Writes are CHAINED
	 * (one in flight at a time, 2026-07 KV audit): each writeSnapshot holds a shuffle copy +
	 * a zstd output alongside the raw snapshot while it runs, so letting a long start()'s
	 * ladder run 4 of those concurrently spiked multiple extra GiB of transient RAM on a
	 * 16 GB machine already holding the 12B. Serializing bounds the overhead to one write's
	 * working set; the snapshots themselves are already-taken copies, so ordering doesn't
	 * affect correctness — only when each hits the SSD.
	 */
	private queueColdWrite(snap: ReturnType<KVColdStore['snapshot']>): void {
		this.coldWriteChain = this.coldWriteChain.then(() =>
			this.coldStore!.writeSnapshot(snap).catch(err => {
				console.warn(`[coldStore] background checkpoint write failed: ${(err as Error).message}`);
			}),
		);
		this.pendingColdWrites.push(this.coldWriteChain);
	}

	/**
	 * M3.7: pick ladder rung boundaries for a staged prefill. A candidate boundary is the next
	 * newline after every COLD_BLOCK_TOKENS-worth of prompt text (estimated by the prompt's own
	 * chars/token ratio), and it is only accepted if the prefix RE-TOKENIZES EXACTLY to a prefix
	 * of `promptTokens` (element-wise) — BPE gives no general guarantee that a char cut lands on
	 * a token boundary, so each rung is verified, never assumed. Rungs whose candidate fails
	 * verification are simply skipped (this is an optimization layer, not a correctness one).
	 * Returns [] for prompts shorter than two blocks — no rung would ever be shared.
	 */
	private coldLadderBoundaries(
		prompt: string,
		promptTokens: Int32Array,
	): { text: string; tokenCount: number }[] {
		if (promptTokens.length < 2 * COLD_BLOCK_TOKENS) return [];
		const charsPerToken = prompt.length / promptTokens.length;
		const out: { text: string; tokenCount: number }[] = [];
		for (let block = 1; block <= COLD_MAX_RUNGS; block++) {
			const targetTokens = block * COLD_BLOCK_TOKENS;
			// Leave at least one block between the last rung and the full prompt — the
			// full-prompt checkpoint already covers the tail.
			if (targetTokens > promptTokens.length - COLD_BLOCK_TOKENS) break;
			const nl = prompt.indexOf('\n', Math.floor(targetTokens * charsPerToken));
			if (nl < 0) break;
			const text = prompt.slice(0, nl + 1);
			const tokens = this.model.tokenize(text, true, true);
			if (tokens.length > promptTokens.length - COLD_BLOCK_TOKENS) break;
			let isPrefix = true;
			for (let i = 0; i < tokens.length; i++) {
				if (tokens[i] !== promptTokens[i]) { isPrefix = false; break; }
			}
			if (!isPrefix) continue; // BPE seam at this cut — skip the rung, try the next block
			// Skip duplicates when two candidates resolve to the same newline.
			if (out.length > 0 && out[out.length - 1].tokenCount >= tokens.length) continue;
			out.push({ text, tokenCount: tokens.length });
		}
		return out;
	}

	/**
	 * M3.7: mid-session checkpoint at a turn boundary (continue()/followUp() entry — the KV is
	 * quiescent there). Fires only when the context grew ≥ COLD_BLOCK_TOKENS since the last
	 * checkpoint. Keyed by the running transcript; restore correctness is protected by
	 * prefillOrResume's re-tokenization sanity check, so a BPE roundtrip mismatch wastes a
	 * little disk, never corrupts a session.
	 */
	private maybeMidSessionCheckpoint(): void {
		if (!this.coldStore || this.transcript.length === 0) return;
		if (this.nPast - this.lastColdCheckpointTokens < COLD_BLOCK_TOKENS) return;
		this.queueColdWrite(this.coldStore.snapshot(this.model, this.transcript, this.nPast, 'continued'));
		this.lastColdCheckpointTokens = this.nPast;
	}

	/**
	 * Await all in-flight background cold-store checkpoint writes. Call before freeing the model
	 * / exiting the process so a checkpoint kicked off during the last turn isn't lost or left
	 * half-written. Safe to call any time; resolves immediately when nothing is pending.
	 */
	async flushColdWrites(): Promise<void> {
		if (this.pendingColdWrites.length === 0) return;
		const pending = this.pendingColdWrites;
		this.pendingColdWrites = [];
		await Promise.all(pending);
	}

	/**
	 * Continue the session with a tool-result observation. Appends the
	 * observation to the KV cache (no re-prefill) and decodes the next
	 * model response.
	 * @param toolName  Name of the tool that produced `observation`. Defaults to the most
	 *                  recently parsed tool call's name (see `lastToolCallName`) -- pass this
	 *                  explicitly when the caller already tracks it, or when running multiple
	 *                  tool calls per turn and responding to a specific one out of order.
	 */
	async continue(observation: string, toolName?: string): Promise<AgentStep> {
		if (this.firstTurn) {
			throw new Error('AgentLoop.continue() called before start() — no KV cache to append to.');
		}
		// M3.7: turn boundary — the KV is quiescent here, checkpoint if a block's worth of
		// context accumulated since the last one.
		this.maybeMidSessionCheckpoint();

		const obsText = this.profile.buildToolResponse(observation, toolName ?? this.lastToolCallName);
		const obsTokens = this.model.tokenize(obsText, false, true);

		// 2026-07 KV audit: a non-zero llama_decode status used to be silently ignored here,
		// advancing nPast/transcript past tokens that never reached the KV — a desync that
		// corrupts every later turn AND every later checkpoint. Throw instead (context
		// overflow already throws from the binding; this covers the residual failure modes).
		const status = await this.model.decodeAppendAsync(obsTokens);
		if (status !== 0) throw new Error(`continue() decode failed (llama_decode status ${status})`);
		this.nPast += obsTokens.length;
		this.transcript += obsText;
		this.feedSuffixTree(obsTokens);
		await this.draftAppend(obsTokens);

		return await this.decodeLoop(obsTokens);
	}

	/**
	 * Follow-up with a new user prompt on the same file.
	 * Appends a new user turn to the KV cache without re-prefilling.
	 * The model continues the conversation from where it left off.
	 */
	async followUp(userPrompt: string): Promise<AgentStep> {
		if (this.firstTurn) {
			throw new Error('AgentLoop.followUp() called before start() — no KV cache to append to.');
		}
		// Reset steering for the new decode (poison tokens re-checked)
		if (this.steering) this.steering.reset();

		// M3.7: turn boundary checkpoint (same rationale as continue()).
		this.maybeMidSessionCheckpoint();

		const followUpText = this.profile.buildFollowUp(userPrompt);
		const followUpTokens = this.model.tokenize(followUpText, false, true);

		// Same status discipline as continue() — see the comment there (2026-07 KV audit).
		const status = await this.model.decodeAppendAsync(followUpTokens);
		if (status !== 0) throw new Error(`followUp() decode failed (llama_decode status ${status})`);
		this.nPast += followUpTokens.length;
		this.transcript += followUpText;
		this.feedSuffixTree(followUpTokens);
		await this.draftAppend(followUpTokens);

		return await this.decodeLoop();
	}

	/** Current KV cache position — needed by StateSnapshotter before continue(). */
	get currentNPast(): number { return this.nPast; }

	/** Free the underlying model and release GPU resources. */
	free(): void {
		this.model.free();
		this.nPast = 0;
		this.firstTurn = true;
	}

	// ── Private helpers ──────────────────────────────────────────────────────

	/**
	 * Construct the ideogram steering machine with the poison set that matches the prompt mode.
	 * In 'hybrid' mode tool calls are LEGAL, so only the always-degenerate turn/channel markers
	 * are poisoned — poisoning the tool-call patterns there would corrupt the native protocol's
	 * closing `<tool_call|>` token (its text starts with `<tool_call`, matching the exclusive
	 * poison), forcing a malformed call the parser rejects. 'exclusive' keeps the full set.
	 */
	private makeSteering(tagMap: Map<number, string>): IdeogramSteering {
		const cfg: SteeringConfig = { model: this.model, tagMap, onToken: this.onToken };
		if (this.steeringPromptMode === 'hybrid') cfg.poisonPatterns = TURN_POISON;
		return new IdeogramSteering(cfg);
	}

	private buildPrompt(result: PreprocessResult, userRequest: string): string {
		const requestText = (result.compressedLabel && result.compressedLabel.length > 0)
			? result.compressedLabel
			: userRequest;
		const taggedTrimmed = result.taggedCode.trim();
		const fileEmpty = taggedTrimmed.length === 0 || taggedTrimmed === '[No file content]';

		const deps = result.multiFileBlocks;
		const depsLine = deps && deps.length > 0
			? `\n[Dependencies: ${deps.map(d => d.filePath.split('/').pop() ?? d.filePath).join(', ')}]`
			: '';
		const depsCode = deps && deps.length > 0
			? '\n\n' + deps.map(d => `[File: ${d.filePath}]\n${d.content}`).join('\n\n')
			: '';

		// Agentic frontends (emptyFileMode 'tools') must NOT enter the single-file-fill mode below:
		// a from-scratch "create X" request has no file in context, and the fill prompt forbids tool
		// calls — so the model would dump raw content/prose instead of calling `write`. Fall through
		// to the normal tool-calling prompt, which renders no file block when the file is empty.
		if (fileEmpty && this.emptyFileMode === 'fill') {
			const systemPrompt =
				'You are a coding agent. Below is an empty file to fill.\n' +
				'Output ONLY the complete file content. No tool calls, no JSON, no markdown fences.\n' +
				'Do NOT wrap output in ``` or <tool_call>. Start with the first character of the file.\n' +
				'Stop after the last character. No explanations before or after.';
			// M3.7 follow-up: the intent line sits NEXT TO the request, not at the head of the
			// prompt — the head is the shared static prefix the checkpoint ladder keys on, and
			// an intent-dependent first line made rungs reusable only between same-intent
			// prompts. Semantically it belongs with the request it classifies anyway.
			// NOT compressed (M15.6 item 1 applies to the tool-calling prompt below, whose system
			// prompt explains the `[∴ <char>]` convention) — this branch's systemPrompt is a
			// minimal, unrelated "output raw content" prompt with no room to explain a new symbol.
			const intentLine = `[Intent: ${result.intent}]`;
			const fileLine = `[File: ${this.lastFilePath}]`;
			const userPrompt = `${fileLine}${depsLine}\n\n${intentLine}\n${requestText}`;
			return this.profile.buildPrompt({ systemPrompt, userPrompt });
		}

		const exclusiveSteeringPrompt =
			'You are a coding agent. Edit only the function/class marked by an ideogram.\n' +
			'Output format: ⊂<ideogram>  <new code>  ⊃\n' +
			'The ideogram is a single special character (like ∀ Ω λ) placed BEFORE the code block.\n' +
			'Find the ideogram that matches your target, then output only:\n' +
			'  ⊂  (fold start marker)\n' +
			'  The ideogram character of the target function/class\n' +
			'  The new code you want to replace it with\n' +
			'  ⊃  (fold end marker)\n' +
			'No explanations. No markdown. No <tool_call>. Just the 4-part sequence.\n' +
			'Example: ⊂∀\ndef calculate_total(items: list[int]) -> int:\n    total = 0\n    for item in items:\n        total += item\n    return total\n⊃';

		// M15.1 hybrid mode: the tool-calling prompt (harness instructions + native
		// declarations) PLUS the ideogram edit protocol for single-node edits — the cheap
		// path for the most common operation, tools for everything else.
		const hybridSteeringSection =
			'\n\nEDIT SHORTCUT — for modifying ONE existing marked code block, do NOT use the edit tool.\n' +
			'Each code block in the file is preceded by a single marker character (like ∀ Ω λ ∑).\n' +
			'To replace that block, output exactly: ⊂ then the marker character, then the complete\n' +
			'replacement code, then ⊃. Nothing else. The replacement must be the whole block, with\n' +
			'no marker characters inside it. For anything that is not a single-block edit (new\n' +
			'files, shell commands, reading files, multi-block changes), use the declared tools.';

		// M15.5: rename/delete ops — a marked node followed by an op character switches the
		// payload from "whole replacement code" to a short verb-specific payload (or none).
		// Validated live on the real 12B (2026-07-06): both ops chosen correctly on the FIRST
		// shot, including in hybrid mode WITH a competing native tool call declared and offered
		// (unlike the whole-block EDIT SHORTCUT above, which measured 0/3 against a competing
		// tool call — see the 2026-07-05 note where this section is consumed). The difference:
		// rename/delete need NO body re-emission, so they don't invite the verbose-tool-edit
		// default that whole-node replacement does.
		const microOpSection =
			'\n\nRENAME/DELETE SHORTCUTS — for these two specific operations on a marked node, do NOT ' +
			'use the edit tool and do NOT re-emit the node\'s body:\n' +
			'  Rename: ⊂ <marker> ⊆ <new name> ⊃\n' +
			'  Delete: ⊂ <marker> ⊇ ⊃\n' +
			'Example: renaming the function marked \'∀\' to \'chargeTax\': ⊂∀⊆chargeTax⊃\n' +
			'Example: deleting the function marked \'∀\' entirely: ⊂∀⊇⊃\n' +
			'For any other kind of edit, use the declared tools or the EDIT SHORTCUT above.';

		const systemPrompt = this.useSteering
			? ((this.steeringPromptMode === 'hybrid')
				? this.buildToolCallingSystemPrompt() + hybridSteeringSection + microOpSection
				: exclusiveSteeringPrompt)
			: this.buildToolCallingSystemPrompt();

		// Same M3.7 rationale as the empty-file branch above: everything before the user's
		// request (file context, dependencies) is the static prefix shared across turns on the
		// same file — keep it intent-independent so ladder rungs match regardless of what the
		// next prompt classifies to. [Intent:] moves down next to the request itself.
		// M15.6 item 1: one registry token instead of the spelled-out class name (~3-4 tokens
		// for e.g. ADD_FEATURE down to 1); explained once in the tool-calling system prompt.
		const intentLine = `[∴ ${intentChar(result.intent)}]`;
		const fileLine = `[File: ${this.lastFilePath}]`;
		// Show AST-tagged code only when steering will consume the markers; otherwise show the raw
		// source, so the injected marker chars neither bloat the prompt nor collide with real code
		// that uses Greek/math Unicode (see lastRawSource). Fall back to tagged if raw is somehow
		// unavailable (never expected — start()/replay() always set it before buildPrompt()).
		const rawView = this.lastRawSource || result.taggedCode;
		// M16.1: gutter the raw view so line-anchored edits (edit_lines) can address exact ranges.
		const gutteredView = this.lineGutter
			? rawView.split('\n').map((l, i) => `${String(i + 1).padStart(4)}| ${l}`).join('\n')
			: rawView;
		const shownCode = this.useSteering ? result.taggedCode : gutteredView;
		// No file in context (a from-scratch create in 'tools' mode): omit the empty [File:] block —
		// there is nothing to show, and an empty block just adds noise before the request.
		const codeBlock = fileEmpty ? '' : `\n[File: ${this.lastFilePath}]\n${shownCode}\n\n`;
		const fileHeader = fileEmpty ? '' : fileLine;
		const userPrompt = `${fileHeader}${depsLine}${codeBlock}${depsCode}\n${intentLine}\n${requestText}`;

		return this.profile.buildPrompt({ systemPrompt, userPrompt });
	}

	/**
	 * Epic M8: declare tools via the active profile's own native protocol when it has one,
	 * instead of instructing a different model family's JSON convention in free text. This is
	 * what actually fixed the malformation the epic measured -- letting the model use its
	 * trained-in tool-calling behavior rather than asking it to imitate Hermes/Qwen's format.
	 * Falls back to the old free-text instructions for profiles without a native mechanism
	 * (Gemma4Profile/Qwen3Profile). See doc/epics/EPIC-M8-grammar-constrained-tool-calls.md.
	 */
	private buildToolCallingSystemPrompt(): string {
		const base = this.systemPromptOverride ??
			'You are a precise coding agent. Your task is classified with an intent.\n' +
			'Below is the target file with structural markers (single uppercase letters before code blocks).\n' +
			'These markers are stable across turns — reference them when proposing edits.\n' +
			// The [∴ X] char is opaque by construction (nothing pairs it with its meaning inline,
			// unlike a node tag or path token) — the legend must be spelled out literally, not
			// described in the abstract, or the model has no way to resolve which class X means.
			`A line like \`[∴ X]\` appears once, right before your task, where X is one of: ` +
			`${intentLegend()} — it marks the task's category and is not part of any file content.\n` +
			'Be concise. Output ONLY code or tool calls, no explanations unless asked.';

		if (this.profile.buildToolDeclarations) {
			return `${base}\n${this.profile.buildToolDeclarations(this.tools)}`;
		}
		return `${base}\n` +
			this.tools.map(t => {
				const args = Object.keys(t.parameters.properties).map(k => `"${k}":"..."`).join(',');
				return `Use <tool_call>{"name":"${t.name}","arguments":{${args}}}</tool_call> for ${t.description}`;
			}).join('\n');
	}

	private async decodeLoop(contextTokens?: Int32Array): Promise<AgentStep> {
		const generated: number[] = [];
		const stopTokenIds = new Set(this.profile.eotTokenIds);
		let stoppedNaturally = false;
		// Tokens pushed to `generated` (so they appear in the output text) but never
		// decodeAppend'ed into the KV: a steering-path stop token, or the last token of a
		// maxTokens-exhausted turn (its decode would have run at the next iteration's start).
		// nPast must count only what's really in the KV — M3.7 checkpoint metadata and the
		// snapshotter both trust it to mirror the native position counter.
		let uncommittedTail = 0;

		if (this.steering) this.steering.reset();

		if (this.specDecoder && contextTokens) {
			const result = this.specDecoder.generate(contextTokens);
			this.specDecoder = null;
			for (const token of result.tokens) {
				if (this.steering) {
					const steerResult = this.steering.feedToken(token);
					generated.push(token);
					if (steerResult.stop) { stoppedNaturally = true; break; }
				} else {
					generated.push(token);
				}
				if (stopTokenIds.has(token)) { stoppedNaturally = true; break; }
			}
		} else if ((this.useSuffixSpeculation || (this.draftModel && this.draftReady)) && (this.temperature === 0 || this.steering)) {
			// Both drafters' accept rule is hard greedy equality (see mtpVerifyBatch in
			// synapse). That matches plain greedy decoding (temperature 0) and ALSO the
			// ideogram-steering path: IdeogramSteering.sample() is argmax over unmodified
			// logits (a pure observer state machine — it never masks or biases), so with
			// steering active the production decode is greedy regardless of temperature.
			stoppedNaturally = await this.decodeLoopSuffixSpec(generated, stopTokenIds);
			uncommittedTail = this.suffixUncommittedTail;
		} else {
			if (this.useSuffixSpeculation || this.draftModel) {
				console.log('⚠️  speculative decoding skipped this turn (needs temperature=0 or ideogram steering)');
			}
			let genText = ''; // only grown/read by the tool-call tag-close fix below
			for (let step = 0; step < this.maxTokens; step++) {
				if (this.shouldStop?.()) break; // cooperative abort — keep what's generated
				if (step === 0) {
				} else {
					const status = await this.model.decodeAppendAsync(Int32Array.of(generated[generated.length - 1]));
					if (status !== 0) return emptyStep(generated.length);
				}

				// Zero-alloc shadow-buffer read (see feedback_getlogitstopk_binding_gap):
				// getLogits() mallocs+copies the full 262k-float vocab EVERY token; the logits
				// are consumed synchronously before the next decode, so the shadow buffer is safe.
				const logits = this.model.getLogitsFast();

				if (this.steering) {
					const { token, stop } = this.steering.sample(logits);
					generated.push(token);
					if (stop || stopTokenIds.has(token)) {
						// Pushed for the output text, but the loop breaks before the next
						// iteration's decodeAppend — it never reaches the KV.
						uncommittedTail = 1;
						stoppedNaturally = true;
						break;
					}
				} else {
					const sampled = this.pickToken(logits);
					if (stopTokenIds.has(sampled)) { stoppedNaturally = true; break; }
					const sampledText = this.model.detokenize(Int32Array.of(sampled));
					const next = this.forceToolCallTagClose(genText, sampled, sampledText);
					const emittedText = next === sampled ? sampledText : TOOL_CALL_TAG_CLOSE;
					genText += emittedText;
					// Stream like the steering path does (steering.sample calls onToken
					// internally) — without this, non-steering loops (e.g. the dual-brain E2B
					// generator) produced no live output at all.
					this.onToken?.(emittedText);
					generated.push(next);
				}
			}
			// maxTokens exhaustion: the last pushed token's decodeAppend never ran.
			if (!stoppedNaturally && generated.length > 0) uncommittedTail = 1;
		}

		this.nPast += generated.length - uncommittedTail;

		const edits = this.steering?.edits ?? [];
		const rawText = generated.length > 0 ? this.model.detokenize(Int32Array.from(generated)) : '';
		// M3.7: keep the transcript in sync with the KV content — the COMMITTED content only
		// (2026-07 KV audit). `generated` can end with an uncommitted tail (a steering stop
		// token, or the last token of a maxTokens-exhausted turn) that is part of the OUTPUT
		// text but never entered the KV; appending it here made tokenize(transcript) disagree
		// with nPast from that turn on, so every later mid-session checkpoint failed the
		// restore-side sanity check forever — hundreds of MiB written per session, none of it
		// ever restorable.
		const committed = generated.length - uncommittedTail;
		this.transcript += committed > 0
			? (committed === generated.length ? rawText : this.model.detokenize(Int32Array.from(generated.slice(0, committed))))
			: '';
		const parsed = parseAssistantOutput(rawText);
		if (parsed.toolCalls.length > 0) {
			// Track the most recent call's name so continue()'s buildToolResponse() default
			// gets the real tool name instead of a placeholder (native protocols need it).
			this.lastToolCallName = parsed.toolCalls[parsed.toolCalls.length - 1].name;
		}

		return {
			text: parsed.text,
			thinking: parsed.thinking,
			toolCalls: parsed.toolCalls,
			ideogramEdit: edits[0] ?? null,
			tokenCount: generated.length,
			stoppedNaturally,
		};
	}

	/** Whether a drafter is currently attached (its shadow KV may or may not be live — see
	 *  draftReady). Lets a host that manages the drafter's lifecycle (the TUI's E2B
	 *  idle-unload) know when to re-attach. */
	get hasDraftModel(): boolean {
		return this.draftModel !== null;
	}

	/**
	 * 2026-07 KV audit: detach the drafter WITHOUT freeing it — for hosts that unload the
	 * shared E2B to reclaim RAM (conversation.ts's idle-unload). Called BEFORE the host frees
	 * the model, so this loop never touches a freed native context. Drafting turns off; the
	 * loop keeps working via trie/plain decode.
	 */
	detachDraftModel(): void {
		this.draftModel = null;
		this.draftReady = false;
		this.draftNPast = 0;
	}

	/**
	 * 2026-07 KV audit: (re)attach a drafter mid-session and rebuild its shadow KV from the
	 * running transcript. The transcript re-tokenization may differ from the originally
	 * committed token IDs (BPE roundtrip — same acknowledged property as replay()); that can
	 * only lower draft ACCEPTANCE, never correctness (mtpVerifyBatch verifies on the target).
	 * Prefill failure (context too small, decode error) just leaves drafting off — same
	 * contract as draftPrefill. No-op cost when called with drafting already live is zero
	 * because callers gate on !hasDraftModel.
	 */
	async attachDraftModel(model: ModelGPU): Promise<void> {
		this.draftModel = model;
		this.draftK = this.draftInitK;
		this.draftReady = false;
		if (this.firstTurn || this.transcript.length === 0) return; // start()/replay() prefill it
		await this.draftPrefill(this.model.tokenize(this.transcript, true, true));
	}

	/** M12.3: (re)prefill the drafter's shadow KV with the full context. Failure ⇒ drafting off. */
	private async draftPrefill(tokens: Int32Array): Promise<void> {
		if (!this.draftModel) return;
		this.draftReady = false;
		try {
			if (tokens.length + this.draftMaxK >= this.draftModel.contextSize) return; // won't fit
			if ((await this.draftModel.forwardAsync(tokens)) !== 0) return;
			this.draftNPast = tokens.length;
			this.draftK = this.draftInitK;
			this.draftReady = true;
		} catch {
			/* drafting stays off for the session; the target loop stands alone */
		}
	}

	/** M12.3: advance the drafter's shadow KV with committed tokens. Failure ⇒ drafting off. */
	private async draftAppend(tokens: Int32Array | readonly number[]): Promise<void> {
		if (!this.draftReady || !this.draftModel) return;
		const arr = tokens instanceof Int32Array ? tokens : Int32Array.from(tokens);
		if (arr.length === 0) return;
		try {
			if (this.draftNPast + arr.length + this.draftMaxK >= this.draftModel.contextSize) {
				this.draftReady = false;
				return;
			}
			if ((await this.draftModel.decodeAppendAsync(arr)) !== 0) {
				this.draftReady = false;
				return;
			}
			this.draftNPast += arr.length;
		} catch {
			this.draftReady = false;
		}
	}

	/** M12.3: roll the drafter's shadow KV back to position `pos` (rejected draft tokens). */
	private draftRollback(pos: number): void {
		if (!this.draftReady || !this.draftModel) return;
		try {
			this.draftModel.kvCacheSeqRemove(0, pos, -1);
			this.draftModel.resetNPast(pos);
			this.draftNPast = pos;
		} catch {
			this.draftReady = false;
		}
	}

	/**
	 * Feed newly-KV'd tokens into the suffix trie and the sliding match-context window.
	 * No-op when suffix speculation is disabled. `tokens` must be NEW tokens only — the
	 * native trie's extend() appends, it does not accept cumulative re-passes of history.
	 */
	private feedSuffixTree(tokens: Int32Array | readonly number[]): void {
		if (!this.useSuffixSpeculation) return;
		const arr = tokens instanceof Int32Array ? tokens : Int32Array.from(tokens);
		if (arr.length === 0) return;
		this.model.suffixTreeExtend(arr);
		for (const t of arr) this.recentTokens.push(t);
		// Keep a generous margin above the trie's max_depth so the match window is never the
		// limiting factor; bound growth so a very long session doesn't grow this unbounded.
		const cap = 256;
		if (this.recentTokens.length > cap) {
			this.recentTokens.splice(0, this.recentTokens.length - cap);
		}
	}

	/**
	 * Suffix-tree speculative decoding round loop: each round either drafts a candidate
	 * continuation from the recent-context window and verifies it in one batched decode
	 * (mtpVerifyBatch — drafter-agnostic, works with any draft source), or, on a cold/no-match
	 * trie, falls back to a single plain decode for that round. See
	 * doc/research/suffix-tree-speculative-decoding.md for the validated bench methodology
	 * this mirrors exactly (same default params, same clear+seed-at-session-start pattern).
	 *
	 * Note on `maxTokens`: a hit round can emit up to `suffixMaxSpec` tokens at once, so the
	 * generated count may overshoot `maxTokens` by up to that many tokens in the worst case.
	 * This is intentional — a verified batch's tokens are already committed to the model's KV
	 * cache and position counter, so truncating them after the fact here (to land exactly on
	 * budget) would desync `this.nPast` from the model's real internal position. Overshooting
	 * by a handful of tokens is harmless; a desynced position counter corrupts every later turn.
	 */
	private async decodeLoopSuffixSpec(generated: number[], stopTokenIds: Set<number>): Promise<boolean> {
		const maxDepth = this.useSuffixSpeculation ? this.model.suffixTreeMaxDepth() : 0;
		this.suffixUncommittedTail = 0;

		while (generated.length < this.maxTokens) {
			if (this.shouldStop?.()) return false; // cooperative abort between rounds
			// Target-side position at round start; the drafter's shadow KV is aligned here
			// (round reconciliation below restores that invariant every round).
			const roundBase = this.nPast + generated.length;

			let emitted: number[];
			let viaVerify = false;
			let acceptedCount = 0;
			let draftedOnDrafter = 0; // tokens the drafter decoded into its own KV this round
			try {
				let draft: Int32Array = new Int32Array(0);

				// Draft source 1 — the small model (M12.3): greedy-decode up to draftK tokens
				// on the drafter's shadow context. getLogitsTopK(1) is the native argmax (no
				// 262k-float JS scan, no copy). Stops break BEFORE pushing, so the draft is
				// stop-free like the clamped trie draft below. The model drafts FIRST, not the
				// trie: measured on an edit prompt, trie-first HALVED the combined win (1.43x
				// vs 1.98x draft-only) — 8-token trie rounds with imperfect acceptance preempt
				// 32-token 99.5%-accepted model rounds while still paying the per-round shadow
				// sync. The trie is the fallback for rounds/sessions without a live drafter.
				//
				// 2026-07 KV audit: the drafter block has its OWN catch. It used to share the
				// round's outer catch (below), which ends the whole turn — so a drafter-side
				// throw (most concretely: the TUI's E2B idle-unload freeing the shadow model
				// between this loop's awaits) at round 0 returned an EMPTY turn instead of the
				// answer. A drafter is an accelerator: on any error, drop it for the session
				// and keep decoding via trie/plain in this same round. The shadow KV may hold
				// un-reconciled draft tokens after a mid-draft throw — irrelevant once
				// draftReady is false (attachDraftModel/draftPrefill re-prefill from scratch).
				if (this.draftModel && this.draftReady) {
					try {
						const cap = Math.min(this.draftK, this.maxTokens - generated.length + 1);
						const buf: number[] = [];
						for (let i = 0; i < cap; i++) {
							const id = this.draftModel.getLogitsTopK(1)[0];
							if (stopTokenIds.has(id)) break;
							if ((await this.draftModel.decodeAppendAsync(Int32Array.of(id))) !== 0) {
								this.draftReady = false;
								break;
							}
							buf.push(id);
						}
						draftedOnDrafter = buf.length;
						this.draftNPast += draftedOnDrafter;
						draft = Int32Array.from(buf);
					} catch {
						this.draftReady = false;
						draftedOnDrafter = 0;
						draft = new Int32Array(0);
					}
				}

				// Draft source 2 — suffix trie (free: pure lookup, no model cost).
				if (draft.length === 0 && this.useSuffixSpeculation) {
					const window = this.recentTokens.length > maxDepth
						? this.recentTokens.slice(-maxDepth)
						: this.recentTokens;
					draft = window.length > 0
						? this.model.suffixTreeSpeculate(
							Int32Array.from(window),
							this.suffixMaxSpec,
							this.suffixMinTokenProb,
							this.suffixMinMatchCount,
							this.suffixMinMatchLen,
						)
						: new Int32Array(0);

					// Clamp the draft at the first stop token: mtpVerifyBatch commits every
					// accepted token to the KV before JS sees the result, and the trie can
					// legitimately draft ACROSS a <turn|> (it was seeded with multi-turn
					// template text). Tokens at/past a stop must never reach the verify batch.
					for (let i = 0; i < draft.length; i++) {
						if (stopTokenIds.has(draft[i])) { draft = draft.slice(0, i); break; }
					}
				}

				if (draft.length === 0) {
					// No draft this round — one plain decode, mirroring the ordinary
					// per-token loop exactly (logits already valid from the prior decode).
					const logits = this.model.getLogitsFast();
					if (this.steering) {
						// steering.sample = greedy + feeds its state machine + streams onToken.
						const { token, stop } = this.steering.sample(logits);
						if (stop || stopTokenIds.has(token)) {
							// Same convention as the plain steering branch: pushed for the
							// output text, never decoded into the KV.
							generated.push(token);
							this.suffixUncommittedTail = 1;
							return true;
						}
						const status = await this.model.decodeAppendAsync(Int32Array.of(token));
						if (status !== 0) return false;
						await this.draftAppend([token]); // keep the shadow context in step
						emitted = [token];
					} else {
						const next = this.pickToken(logits);
						if (stopTokenIds.has(next)) return true;
						const status = await this.model.decodeAppendAsync(Int32Array.of(next));
						if (status !== 0) return false;
						await this.draftAppend([next]); // keep the shadow context in step
						this.onToken?.(this.model.detokenize(Int32Array.of(next)));
						emitted = [next];
					}
				} else {
					const result = this.model.mtpVerifyBatch(draft);
					acceptedCount = result[0];
					emitted = Array.from(result.slice(1));
					viaVerify = true;
				}
			} catch {
				// Native verify/decode error — stop gracefully, keep what's generated so far.
				// The drafter's shadow KV may hold un-reconciled draft tokens (a decodeAppend
				// mid-draft THROWS rather than returning a status, skipping the counters) —
				// disable drafting for the session rather than risk a desynced shadow context.
				if (this.draftModel) this.draftReady = false;
				return false;
			}

			if (viaVerify) {
				// Scan the verified batch. A stop token can only be the trailing correction
				// (the draft was clamped above). Steering — a pure observer — replays the
				// accepted tokens through its state machine exactly as the plain loop would.
				// `keep` = tokens that go into `generated` (the output text); `commit` =
				// tokens that stay in the KV (≤ keep: a steering fold-end is pushed for the
				// text but kept OUT of the KV, matching the plain steering branch).
				let keep = emitted.length;
				let commit = emitted.length;
				let stoppedThisRound = false;
				for (let i = 0; i < emitted.length; i++) {
					const tok = emitted[i];
					if (stopTokenIds.has(tok)) {
						keep = i; commit = i; stoppedThisRound = true;
						break;
					}
					if (this.steering) {
						// 2026-07 audit: poison-token defense for the batch path. The plain loop's
						// sample() masks a poison token (turn/channel markers the observer treats
						// as degeneration) and re-samples; a token that arrived pre-decoded from
						// mtpVerifyBatch skipped that gate. We can't re-sample a committed token
						// here without a fresh decode for valid logits, so the safe, provably-
						// correct action is to truncate BEFORE the poison and stop the round: the
						// clean prefix is kept, poison never reaches the KV (the commit<len rollback
						// below evicts it), and checkPoison() also keeps the steering machine's
						// running text in sync for multi-token patterns. Slightly more conservative
						// than the plain path's mask-and-continue (it ends the turn instead of
						// skipping the one token) — an acceptable trade on the fast path; full parity
						// would need a re-decode from roundBase+i.
						if (this.steering.checkPoison(tok)) {
							keep = i; commit = i; stoppedThisRound = true;
							break;
						}
						const sr = this.steering.feedToken(tok);
						if (sr.stop) {
							keep = i + 1; commit = i; stoppedThisRound = true;
							this.suffixUncommittedTail = 1;
							break;
						}
					}
				}
				if (commit < emitted.length) {
					// mtpVerifyBatch already committed the whole batch (accepted + correction)
					// to the KV and the native position counter. Roll back everything past the
					// stop point so KV and nPast end up exactly what the plain loop would have
					// produced — stop tokens stay OUT of the KV; continue()/followUp() supply
					// the turn structure themselves.
					const target = roundBase + commit;
					this.model.kvCacheSeqRemove(0, target, -1);
					this.model.resetNPast(target);
				}

				// M12.3: reconcile the drafter's shadow KV with what the target committed. The
				// drafter holds its own draft tokens — identical ids up to the ACCEPTED prefix
				// only (the correction token differs from the drafted one by definition), so
				// the match is capped at acceptedCount, never just commit.
				if (this.draftModel && this.draftReady) {
					const matched = Math.min(draftedOnDrafter, commit, acceptedCount);
					if (this.draftNPast > roundBase + matched) this.draftRollback(roundBase + matched);
					if (commit > matched) await this.draftAppend(emitted.slice(matched, commit));
				}
				// M12.3: adaptive draft length — grow on full acceptance, shrink on rejection.
				if (draftedOnDrafter > 0) {
					this.draftK = acceptedCount >= draftedOnDrafter
						? Math.min(this.draftMaxK, this.draftK * 2)
						: Math.max(this.draftMinK, Math.floor(this.draftK / 2));
				}

				const kept = emitted.slice(0, keep);
				generated.push(...kept);
				this.feedSuffixTree(kept);
				if (kept.length > 0) this.onToken?.(this.model.detokenize(Int32Array.from(kept)));
				if (stoppedThisRound) return true;
			} else {
				generated.push(...emitted);
				this.feedSuffixTree(emitted);
			}
		}

		return false;
	}

	private pickToken(logits: Float32Array): number {
		if (this.temperature > 0) {
			return sample(logits, { temperature: this.temperature, topP: this.topP, topK: this.topK });
		}
		return sampleGreedy(logits);
	}

	/**
	 * Epic M8 fix: if the generated text so far ends with the bare "<tool_call" prefix (the
	 * model has committed to opening a tool call but hasn't closed the tag yet) and the
	 * sampled token doesn't continue with '>', override it with the tokenized '>' instead.
	 * No-op everywhere else -- this never touches sampling for the JSON body or normal text.
	 */
	private forceToolCallTagClose(genTextSoFar: string, sampled: number, sampledText: string): number {
		if (!genTextSoFar.endsWith(TOOL_CALL_TAG_PREFIX)) return sampled;
		if (sampledText.startsWith(TOOL_CALL_TAG_CLOSE)) return sampled; // already correct
		const forced = this.model.tokenize(TOOL_CALL_TAG_CLOSE, false, false);
		return forced.length === 1 ? forced[0] : sampled; // only override if it's a single token
	}
}

function emptyStep(tokens: number): AgentStep {
	return { text: '', thinking: '', toolCalls: [], ideogramEdit: null, tokenCount: tokens, stoppedNaturally: false };
}
