/**
 * conversation.ts
 *
 * The subvocal-local conversation engine: owns the GGUF model, the AgentLoop
 * instance and the pi-Context → AgentLoop turn mapping (first user message →
 * start(), toolResult(s) → continue(), later user messages → followUp()) so the
 * KV cache persists across the whole conversation.
 *
 * Runs inside the agent worker thread (agent-worker.ts) — the decode loop is
 * synchronous JS over sync N-API calls, so it must NOT live on the TUI's event
 * loop. Everything in TurnRequest/TurnResult is structured-clone-safe.
 */

import type { Message, ToolResultMessage } from "@earendil-works/pi-ai";
import {
	AgentLoop,
	type AgentStep,
	type Brain,
	type ReplayTurn,
	activeProfile,
	classifyIntentOnAuxSeq,
	computeBlockAnchors,
	detagCandidates,
	detectLanguage,
	deleteASTNode,
	distillContext,
	editASTNode,
	FilePrewarmCache,
	insertAfterNode,
	renameASTNode,
	type IntentResult,
	MacE2BProfile,
	BASH_FAIL_CHAR,
	BASH_PASS_CHAR,
	intentLegend,
	PathRegistry,
	routeTask,
	shouldEscalate,
	UNCHANGED_FILE_CHAR,
} from "@subvocal/encode";
import { ModelGPU } from "@subvocal/synapse";
import { existsSync, readFileSync } from "fs";
import { resolve } from "path";
import { logDualBrainEscalate, logDualBrainRoute, logIntentSource } from "./activity-log.ts";
import type { TurnRequest, TurnResult } from "./wire.ts";

// M15.1 ideogram edit shortcut — OFF by default (2026-07-05, see the useSteering comment below and
// doc/research/m15.7-lora-parked.md). Single source of truth for BOTH the steering toggle and the
// edit-argument detag: when the feature is off the model sees raw (un-tagged) source, so there are
// no injected markers to strip — and detagging anyway would corrupt real code that legitimately
// uses registry chars (Greek/math Unicode). Re-enable the whole path with SUBVOCAL_IDEOGRAM=1.
const IDEOGRAM_ENABLED = process.env.SUBVOCAL_IDEOGRAM === "1";

// M13.4: M12.2's dual-brain generation routing, wired into the TUI (was REPL-only). OFF by
// default. When on, a FRESH conversation is routed to the E2B generator (routeTask, the same
// conservative mechanical-edit gate the REPL uses) and escalated back to the 12B if its first
// step produces nothing usable (shouldEscalate). Orthogonal to SUBVOCAL_LOCAL_DRAFT (M12.3):
// that mechanism speculatively drafts FOR the 12B and never changes which brain answers; this
// one swaps which model generates the whole turn.
const DUAL_BRAIN_ENABLED = process.env.SUBVOCAL_LOCAL_DUAL_BRAIN === "1";

// 2026-07-07: idle-unload the shared E2B (weights + KV, ~2.7-2.9 GiB) after this many ms of no
// use — measured RAM/swap pressure on a 16 GB machine running both models resident for a whole
// session (doc/epics/EPIC-M3's asymmetric-context follow-up). 3 min survives normal think-time
// between turns without reloading on every message; SUBVOCAL_LOCAL_E2B_IDLE_MS overrides,
// SUBVOCAL_LOCAL_E2B_IDLE_MS=0 disables (keep the old always-resident behavior).
const E2B_IDLE_UNLOAD_MS = process.env.SUBVOCAL_LOCAL_E2B_IDLE_MS
	? Number(process.env.SUBVOCAL_LOCAL_E2B_IDLE_MS)
	: 180_000;

// M11.3: model-backed intent classification on the shared E2B's isolated aux sequence
// (classifyIntentOnAuxSeq — variant (b), measured GO: 0.0pp accuracy delta vs standalone at
// every tested conversation length, see doc/substories/M11.3-intent-classifier-shared-e2b.md).
// OFF by default pending a live dual-brain+drafter TUI validation (the story's own acceptance
// step 4, not yet run) — regex intent (cpuOff:true's internal fallback) is unaffected either
// way. Orthogonal to DUAL_BRAIN_ENABLED/SUBVOCAL_LOCAL_DRAFT: intent only steers preprocess
// directives, it does NOT gate dual-brain routing (that gate is regex/shape-based on purpose,
// see dualBrainRouter.ts).
const INTENT_MODEL_ENABLED = process.env.SUBVOCAL_LOCAL_INTENT_MODEL === "1";

// M15.2: content-addressed block anchors are OPT-IN (SUBVOCAL_BLOCK_ANCHORS=1), OFF by default.
// The 12B quality gate showed the prompt-taught model targets whole methods (not the finer
// block anchors), and a live end-to-end run showed WORSE: the model echoes the anchors into its
// edit-tool oldText and CORRUPTS the rare CJK anchor chars (e.g. 楹 → "jv"), so detag can't
// recover them and the edit fails to match the raw file. The node-level tags (few, reproducible
// Greek/Math chars) are safe and stay on. Block anchors need a model trained on the dialect
// (M15.7) before they help rather than hurt — see doc/substories/M15.2-crc-block-anchors.md.

// Appended to the harness's system prompt so the agent explicitly signals task completion —
// the loop ends when the model stops calling tools, and without this the turn just goes quiet.
const COMPLETION_INSTRUCTION =
	"\n\nWhen the task is fully complete and you do not need any more tool calls, end your " +
	"final message with a single line starting with 'DONE — ' that briefly summarizes what you " +
	"changed. Do not write that line while you still intend to call another tool.";

const SUBVOCAL_SYSTEM_PROMPT =
	"You are Subvocal, a local-first coding agent (Gemma-4, in-process, no cloud) operating inside a " +
	"coding-agent harness. You help by reading files, running commands, editing code, and creating new " +
	"files — you ACT directly on the user's real files in the current project through your tools. You " +
	"are NOT a chatbot that returns code for the user to copy: to create a file call `write`, to change " +
	"one call `edit` (exact text) or `edit_lines` (line range), to read a file call `read`, to list a " +
	"directory call `ls`, to search call `grep` or `find`, and to run any other shell command (tests, " +
	"git, npm) call `bash`. NEVER paste file contents into your reply for the user to save by hand, and " +
	"never claim you cannot access the filesystem — you can, so do it with a tool.\n\n" +
	"File listings are shown with a line-number gutter like `  42| code`. The numbers are NOT part of " +
	"the file: never include them in content you write. To replace a known range of lines, prefer " +
	"`edit_lines{path, startLine, endLine, newText}` (numbers from the gutter, 1-based inclusive); " +
	"after an edit_lines call the numbers below the edit shift — re-read the file before another " +
	"line-based edit. For small unique-text replacements `edit` also works.\n\n" +
	// M15.6 item 1: `[∴ X]` replaces the old spelled-out `[Intent: ADD_FEATURE]` line. The char
	// is opaque by construction (nothing pairs it with its meaning inline) so the legend must be
	// spelled out literally here, not described in the abstract.
	`A line like \`[∴ X]\` appears once, right before the task, where X is one of: ${intentLegend()} ` +
	"— it marks the task's category and is not file content, never include it in anything you write.\n\n" +
	// M15.6 item 4: was the sentence `[read <path>: content unchanged, already in context]`. No
	// literal-char legend needed here — unlike pass/fail, there's only one meaning, inferable
	// from the structural pattern itself (a marker + path with nothing following it).
	"If a `read` result looks like `[<marker> <path>]` with no content below it, that marker means " +
	"the file is UNCHANGED since you last saw it — the content is already above in this " +
	"conversation, nothing new was returned.\n\n" +
	// M15.6 item 3: a fixed one-character first line on every bash result. Literal chars named
	// (same reasoning as the intent legend above — pass vs fail isn't inferable from structure).
	`A \`bash\` result starts with a one-character line in brackets before the actual output: ` +
	`\`[${BASH_PASS_CHAR}]\` means the command passed (exit 0), \`[${BASH_FAIL_CHAR}]\` means it ` +
	"failed — read the output below it either way to see what happened.\n\n" +
	"CRITICAL: your reply text does nothing — ONLY your tool calls take effect. Do NOT write shell " +
	"scripts, `cat << EOF` / `touch` / `mkdir` lines, or file contents as text in your message. To " +
	"scaffold a project, emit real tool calls, one per step: a `bash` call for `mkdir <dir>`, then a " +
	"separate `write` call for each file (path + full contents). One tool call at a time; wait for its " +
	"result, then continue.\n\n" +
	"Workflow for every task: 1) UNDERSTAND — read the files and context you need before acting; " +
	"2) PLAN — for non-trivial work, briefly decide the steps; 3) ACT — make the changes with your " +
	"tools; 4) VERIFY — re-read what you wrote and run the relevant test/build/command to confirm it " +
	"works, fixing any problem before continuing; 5) REPORT — end with a short summary of what you " +
	"did. Never stop at describing what should be done — do it, then verify it.\n\n" +
	"Read before you edit; prefer small targeted edits; keep prose between actions brief.\n\n" +
	"Available tools: read, write, edit, bash, ls, grep, find.\n\n" +
	// macOS-specific: warn about BSD-vs-GNU so the model doesn't emit GNU-isms that fail on Mac.
	// This prevents a real class of silent command failures (the single highest-value Mac tuning).
	"Environment: macOS (Apple Silicon), zsh + BSD coreutils (NOT GNU). Use portable flags: " +
	"`sed -i ''` needs an empty backup arg, `grep -E` (not `-P`), `date -v` (not `-d`), `stat -f` (not `-c`), " +
	"`realpath`/`mdfind` (not `readlink -f`/GNU `find -printf`). macOS tools pbcopy/open/mdfind are available; " +
	"mdfind (Spotlight) is often faster than grep -r over large trees.\n\n" +
	"Project context (AGENTS.md, CLAUDE.md, doc/epics/) is available on disk; read them if relevant.";

export interface TurnHooks {
	onToken?: (text: string) => void;
	shouldStop?: () => boolean;
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/** Flatten a message's content blocks to plain text (text blocks only). */
function flattenText(content: string | ReadonlyArray<{ type: string }>): string {
	if (typeof content === "string") return content;
	const parts: string[] = [];
	for (const raw of content) {
		const block = raw as { type: string; [k: string]: unknown };
		if (block.type === "text" && typeof block.text === "string") parts.push(block.text);
		else if (block.type === "image") parts.push("[image omitted]");
	}
	return parts.join("\n");
}

function messageSig(m: Message): string {
	// Assistant entries are our own output; the harness may re-shape their blocks (split
	// text, strip thinking), so match them by role only — user/toolResult content is what
	// actually identifies a conversation.
	if (m.role === "assistant") return "assistant";
	const body = m.role === "toolResult" ? flattenText((m as ToolResultMessage).content) : flattenText(m.content as never);
	return `${m.role}#${body.length}#${body.slice(0, 32)}`;
}

/** True when `messages` extends the already-consumed prefix (same conversation, new tail). */
function extendsPrefix(messages: readonly Message[], prefixSig: readonly string[]): boolean {
	if (messages.length < prefixSig.length) return false;
	for (let i = 0; i < prefixSig.length; i++) {
		if (messageSig(messages[i]) !== prefixSig[i]) return false;
	}
	return true;
}

/**
 * Render a stored assistant message back into the surface text a live turn would have
 * produced: text blocks verbatim, toolCall blocks in the native protocol's syntax (the format
 * toolParse.ts parses — `<|tool_call>call:NAME{key:<|"|>value<|"|>,...}<tool_call|>`). Used by
 * the M13.3 replay path; byte-exactness vs the original generation is best-effort (it only
 * affects cold-store checkpoint hits, never correctness).
 */
function renderAssistantText(msg: Message): string {
	if (typeof msg.content === "string") return msg.content;
	const parts: string[] = [];
	for (const raw of msg.content as unknown as ReadonlyArray<{ type: string; [k: string]: unknown }>) {
		if (raw.type === "text" && typeof raw.text === "string") {
			parts.push(raw.text);
		} else if (raw.type === "toolCall") {
			const name = String(raw.name ?? "");
			// 2026-07 audit: quote-wrap ONLY string values (the native protocol's `<|"|>` marker is
			// for strings). Numbers/booleans go bare — parseNativeArgs reads them from the structural
			// segments and coerces the type back, so a replayed `startLine:5` round-trips as the
			// number 5, not the string "5" a quoted render would have produced (type drift that only
			// costs cold-store checkpoint hits, but free to get right).
			const args = Object.entries((raw.arguments as Record<string, unknown>) ?? {})
				.map(([k, v]) =>
					typeof v === "string" ? `${k}:<|"|>${v}<|"|>`
					: (typeof v === "number" || typeof v === "boolean") ? `${k}:${v}`
					: `${k}:<|"|>${JSON.stringify(v)}<|"|>`)
				.join(",");
			parts.push(`<|tool_call>call:${name}{${args}}<tool_call|>`);
		}
	}
	return parts.join("");
}

/**
 * LAST-RESORT path for a history that can't even be mapped onto replay turns: re-serialize it
 * as a labeled plain-text transcript inside the first user turn. No fake chat-template tokens —
 * just labeled text; the model's real template wraps it once. Restorable histories go through
 * AgentLoop.replay() instead (M13.3).
 *
 * P2-12: tool results go through the caller's renderResult (the M13.1/M13.2 shrinkers) — with
 * fedFiles pre-seeded, a read of the active file collapses to a breadcrumb instead of putting a
 * raw copy in the same prompt as the tagged copy preprocess is about to inject.
 */
function transcriptFallback(messages: readonly Message[], renderResult: (tr: ToolResultMessage) => string): string {
	const lines: string[] = ["[Conversation so far:]"];
	for (const m of messages) {
		if (m.role === "toolResult") {
			const tr = m as ToolResultMessage;
			lines.push(`[${tr.toolName} result${tr.isError ? " (error)" : ""}]:\n${renderResult(tr)}`);
		} else {
			lines.push(`[${m.role}]:\n${flattenText(m.content as never)}`);
		}
	}
	lines.push("[Continue from here.]");
	return lines.join("\n\n");
}

/**
 * The model sees AST-TAGGED file content (astTagger ideograms) and often echoes the tags back
 * in its edit's oldText/newText; the file on disk has no tags, so the executor's exact-match
 * fails. Same fix as encode/src/utter.ts: try detagged candidates against the real file and
 * take the first that matches (transforming newText with the SAME rule so the pair stays
 * consistent).
 */
function detagEditArguments(args: Record<string, unknown>, fileContent: string): Record<string, unknown> {
	const oldText = args.oldText;
	if (typeof oldText !== "string" || oldText.length === 0 || fileContent.includes(oldText)) return args;
	const oldCands = detagCandidates(oldText);
	const newCands = typeof args.newText === "string" ? detagCandidates(args.newText) : null;
	for (let i = 0; i < oldCands.length; i++) {
		if (oldCands[i] !== oldText && fileContent.includes(oldCands[i])) {
			return { ...args, oldText: oldCands[i], ...(newCands ? { newText: newCands[i] } : {}) };
		}
	}
	return args;
}

// ── M13.2: verbose tool output → compact the MODEL copy ──────────────────────
//
// pi already truncates tool output at 2000 lines / 50 KB, but that is still ~12k tokens per
// bash result entering the KV. The user-facing copy in the TUI is a real I/O boundary and
// stays complete; the copy the MODEL conditions on is an intermediate layer where most lines
// carry no signal (tensorProxy.ts's principle, applied to harness tool output). Heuristic
// signal extraction: head + tail + error-ish lines with context, explicit elision breadcrumbs.

const OBS_BUDGET_CHARS = Number(process.env.SUBVOCAL_TOOL_OBS_BUDGET ?? 4000);
const OBS_SIGNAL_RE = /error|fail|warn|exception|traceback|assert|fatal|not found|denied|missing|invalid/i;
const OBS_HEAD_LINES = 15;
const OBS_TAIL_LINES = 15;

function compactObservation(text: string, isError: boolean): string {
	if (text.length <= OBS_BUDGET_CHARS) return text;
	// Failed commands deserve detail — keep them whole within reason.
	if (isError && text.length <= OBS_BUDGET_CHARS * 2) return text;

	const lines = text.split("\n");
	const keep = new Set<number>();
	for (let i = 0; i < Math.min(OBS_HEAD_LINES, lines.length); i++) keep.add(i);
	for (let i = Math.max(0, lines.length - OBS_TAIL_LINES); i < lines.length; i++) keep.add(i);
	for (let i = 0; i < lines.length; i++) {
		if (OBS_SIGNAL_RE.test(lines[i])) {
			keep.add(i);
			if (i > 0) keep.add(i - 1);
			if (i + 1 < lines.length) keep.add(i + 1);
		}
	}

	const out: string[] = [];
	let elided = 0;
	const flushElision = () => {
		if (elided > 0) {
			out.push(`[... ${elided} lines elided — full output shown to the user ...]`);
			elided = 0;
		}
	};
	for (let i = 0; i < lines.length; i++) {
		if (keep.has(i)) {
			flushElision();
			out.push(lines[i]);
		} else {
			elided++;
		}
	}
	flushElision();

	let compact = out.join("\n");
	// Pathological case (every line matches the signal regex): hard-fall back to head+tail.
	if (compact.length > OBS_BUDGET_CHARS * 2) {
		compact =
			lines.slice(0, OBS_HEAD_LINES).join("\n") +
			`\n[... ${lines.length - OBS_HEAD_LINES - OBS_TAIL_LINES} lines elided — full output shown to the user ...]\n` +
			lines.slice(-OBS_TAIL_LINES).join("\n");
	}
	return compact;
}

// ── The engine ────────────────────────────────────────────────────────────────

export class LocalConversationEngine {
	private model: ModelGPU | null = null;
	// ONE resident E2B instance for every small-model role: M12.3 drafter (shadow KV),
	// M13.4 dual-brain generator (own AgentLoop answers routed turns), M16.3 distiller.
	// The roles are never live in the same conversation (small conversations don't draft,
	// large ones don't generate on the E2B) and every role switch re-prefills — the binding's
	// forward() clears the KV, and compaction always drops the loop — so one context serves
	// all. The weights were always mmap-shared; what sharing saves is the per-instance Metal
	// context (KV + compute buffers + SWA cache), which is what OOM'd the 16 GB machine when
	// three instances sat next to the 12B (see warm()).
	private e2b: ModelGPU | null = null;
	/** 2026-07-07: idle-unload timer for the E2B (see scheduleE2BIdleUnload()) — reclaims its
	 *  ~2.7-2.9 GiB (weights + KV) during genuinely idle stretches instead of holding it resident
	 *  for the whole session. Reset on every getE2B() call; fires free() + null after
	 *  E2B_IDLE_UNLOAD_MS of no further use. getE2B()'s existing lazy-create handles the reload. */
	private e2bIdleTimer: ReturnType<typeof setTimeout> | null = null;
	/** 2026-07 KV audit: >0 while runTurn() is executing (counter — the reactive-compaction
	 *  retry re-enters runTurn recursively). The idle-unload timer MUST NOT free the E2B
	 *  mid-turn: turns await async decodes, so the timer callback interleaves with them, and
	 *  llama_free on a context whose AsyncWorker is still running on libuv's threadpool is a
	 *  native use-after-free (Model::Free has no in-flight guard). Even between drafter ops it
	 *  would kill the drafter mid-turn for nothing. */
	private turnActive = 0;
	/** Which brain generated the CURRENT conversation's turns — set when a fresh loop is built. */
	private currentBrain: Brain = "large";
	/** M13.4: buffers the small brain's FIRST step stream until the escalation decision — an
	 *  unusable step must never reach the UI, and without buffering the 12B redo's stream
	 *  would render concatenated after it in the same turn. */
	private smallTokenBuffer: string[] | null = null;
	private loop: AgentLoop | null = null;
	private prefixSig: string[] = [];
	/** M16.3: guards the one-shot compaction retry in the overflow catch (no recursion loops). */
	private compactRetrying = false;
	/** 2026-07 audit: a resetConversation() that arrived mid-turn, applied in runTurn's finally. */
	private pendingReset = false;
	private activeFilePath = "";
	private filePrewarmCache = new FilePrewarmCache();
	// M13.1: file content already fed into THIS conversation's KV (absPath → content).
	// Recorded at start() (the active file preprocess ingested) and on full read results;
	// invalidated when an edit/write touches the path. Lets read results collapse to a
	// breadcrumb instead of re-feeding tokens the model already has (foldTokens pattern).
	private fedFiles = new Map<string, string>();
	// M15.4: per-session path tokens — a path renders as `char (path)` once, bare `char`
	// afterwards; model-emitted tokens in path arguments resolve back BEFORE detagging.
	private paths = new PathRegistry();
	private syntheticCallN = 0;
	// Per-turn hooks, reached from the per-conversation AgentLoop via stable closures
	// (AgentLoop takes onToken/shouldStop at construction, not per call).
	private hooks: TurnHooks = {};

	private emitToken(text: string): void {
		if (this.smallTokenBuffer) {
			this.smallTokenBuffer.push(text);
			return;
		}
		this.hooks.onToken?.(text);
	}

	/** Load the model eagerly (worker startup) so the first turn doesn't pay it.
	 *
	 * NOTE (measured live, 2026-07-06): dual-brain must NOT also load the E2B intent
	 * classifier here — 12B + classifier + generator = THREE resident Metal instances,
	 * which OOM'd the command buffer on the 16 GB machine mid-turn (the REPL-validated
	 * dualBrainMaxCtx=16384 budget assumes TWO). Intent stays regex (cpuOff:true) in the
	 * TUI; the worker-thread classifier remains a future integration with its own budget. */
	warm(): void {
		this.getModel();
	}

	/**
	 * Context size for the 12B. With compact-SWA (patch 102, enabled by
	 * LLAMA_KV_SWA_OFFLOAD=1 in bin/subvocal) + noKvOffload, the KV is CHEAP and nearly
	 * context-independent: SWA is fixed at 480 MiB (GPU), the global KV is ~96 MiB @ 6k /
	 * ~512 MiB @ 32k and lives in RAM (off the Metal working set). So the OOM was the
	 * full-size SWA (~5 GiB at 8k WITHOUT the split), not the context — once the split is on,
	 * we can afford a real coding-agent window. The weights are the budget, not the KV:
	 *   - single model: 32768 (KV ~512 MiB; the REPL runs 128k, we leave more headroom for
	 *                   the heavier pi-harness process).
	 *   - dual model (drafter on, OR M13.4 dual-brain generator on): 8192 (dualBrainMaxCtx) —
	 *                   2026-07-07 (owner call): asymmetric now, the 12B gets a TIGHT window
	 *                   (it's the expensive, rarely-invoked escalation path) while the E2B gets
	 *                   its own wide one (resolveE2BContextSize() below) — they no longer share
	 *                   a single cap. Was 16384 shared.
	 * SUBVOCAL_LOCAL_CTX overrides either.
	 */
	private resolveContextSize(): number {
		if (process.env.SUBVOCAL_LOCAL_CTX) return Number(process.env.SUBVOCAL_LOCAL_CTX);
		return process.env.SUBVOCAL_LOCAL_DRAFT !== "0" || DUAL_BRAIN_ENABLED
			? (activeProfile.dualBrainMaxCtx ?? 8192)
			: 32768;
	}

	/**
	 * Context size for the E2B (drafter or dual-brain generator role), independent of the
	 * 12B's — 2026-07-07 owner call: the cheap/fast model gets a generous window (32768) so it
	 * can hold a long effective conversation on its own, while the expensive 12B stays tight.
	 * As a drafter this is safe even though it's LARGER than the 12B's window: the shadow KV
	 * only ever needs to hold as much of the conversation as the target actually has, so extra
	 * allocated headroom is unused but harmless, not a correctness issue. SUBVOCAL_LOCAL_E2B_CTX
	 * overrides.
	 */
	private resolveE2BContextSize(): number {
		if (process.env.SUBVOCAL_LOCAL_E2B_CTX) return Number(process.env.SUBVOCAL_LOCAL_E2B_CTX);
		return activeProfile.e2bMaxCtx ?? 32768;
	}

	private getModel(): ModelGPU {
		if (this.model) return this.model;
		// Model path and load options come from the active profile (modelProfile.ts);
		// SUBVOCAL_LOCAL_* env vars override.
		const path = process.env.SUBVOCAL_LOCAL_MODEL ?? activeProfile.largeModelPath;
		const gpuLayers = Number(process.env.SUBVOCAL_LOCAL_GPU_LAYERS ?? activeProfile.largeOpts?.gpuLayers ?? 999);
		// noKvOffload keeps the global KV in RAM (the ISWA 2-tier split, enabled by
		// LLAMA_KV_SWA_OFFLOAD=1 in bin/subvocal) instead of competing for the Metal working
		// set — without it a 16 GB M2 Pro OOMs on the GPU command buffer with the harness's
		// large system prompt.
		const noKvOffload = activeProfile.largeOpts?.noKvOffload ?? true;
		this.model = new ModelGPU(path, { contextSize: this.resolveContextSize(), gpuLayers, noKvOffload });
		return this.model;
	}

	/**
	 * The shared E2B instance (see the field comment). Lazy: loaded on the first turn that
	 * needs any small-model role; also freed early after E2B_IDLE_UNLOAD_MS of inactivity
	 * (scheduleE2BIdleUnload()) to reclaim its RAM between bursts of use, not just at teardown.
	 * Context = resolveE2BContextSize() (own window, independent of the 12B's — see that
	 * method's comment). As a drafter, a shadow KV wider than the target it's shadowing is
	 * fine: it only ever holds as much conversation as the 12B actually has.
	 * MacE2BProfile.largeModelPath IS activeProfile.smallModelPath (same gguf).
	 *
	 * M11.3: `auxSeq: true` reserves a second, attention-isolated KV sequence (seq_id=1) —
	 * needed by classifyIntentOnAuxSeq() to classify without disturbing whichever role (drafter
	 * shadow KV, generator conversation, distiller) is live on seq 0. Always on, not gated
	 * behind SUBVOCAL_LOCAL_INTENT_MODEL: it costs ~nothing when unused (kv_unified means the
	 * context window isn't split by reserving the slot) and constructing it conditionally would
	 * risk the flag being flipped after this lazy singleton already loaded without it.
	 */
	private getE2B(): ModelGPU {
		if (!this.e2b) {
			this.e2b = new ModelGPU(MacE2BProfile.largeModelPath, { ...MacE2BProfile.largeOpts, contextSize: this.resolveE2BContextSize(), auxSeq: true });
		}
		this.scheduleE2BIdleUnload();
		return this.e2b;
	}

	/**
	 * Reset (or start) the idle-unload countdown — called on every getE2B(), so any actual use
	 * pushes the deadline back. E2B_IDLE_UNLOAD_MS<=0 disables it (old always-resident behavior).
	 * A stale timer racing a manual free() elsewhere (M13.4's error-recovery path) is harmless:
	 * it just finds `this.e2b` already null and no-ops; the next getE2B() reschedules fresh.
	 * .unref() so the pending timer never keeps the worker thread's event loop alive on its own.
	 */
	private scheduleE2BIdleUnload(): void {
		if (this.e2bIdleTimer) clearTimeout(this.e2bIdleTimer);
		if (E2B_IDLE_UNLOAD_MS <= 0) return;
		this.e2bIdleTimer = setTimeout(() => {
			this.e2bIdleTimer = null;
			if (!this.e2b) return;
			// 2026-07 KV audit — three guards this callback was missing:
			//   1. Mid-turn: never free while runTurn() is live (see turnActive) — the timer
			//      interleaves with the turn's awaits, and freeing a context with an AsyncWorker
			//      in flight is a native use-after-free. This WAS reachable: the deadline only
			//      refreshes on getE2B(), which incremental turns never call, so any turn (or
			//      incremental stretch) longer than the idle window armed it.
			//   2. Small-brain conversations: the E2B *is* the live loop's generator there —
			//      freeing it fails every subsequent turn of the session ("Model has been
			//      freed"). Reschedule; the countdown restarts and unloads once the
			//      conversation moves on.
			//   3. Drafter reference: detach it from the loop BEFORE freeing, so the loop
			//      degrades to trie/plain cleanly; the next turn re-attaches via
			//      attachDraftModel() (see runTurn) instead of losing drafting for the session.
			if (this.turnActive > 0 || (this.currentBrain === "small" && this.loop !== null)) {
				this.scheduleE2BIdleUnload();
				return;
			}
			this.loop?.detachDraftModel();
			try {
				this.e2b.free();
			} catch { /* best-effort */ }
			this.e2b = null;
		}, E2B_IDLE_UNLOAD_MS);
		this.e2bIdleTimer.unref();
	}

	/**
	 * M11.3: classify `userPrompt`'s intent via the shared E2B's isolated aux sequence when
	 * enabled, for the caller to pass as PreprocessInput.precomputedIntent — cpuOff stays true
	 * either way, so the OTHER cpuOff-gated tensor machinery (tensorPayload, logit masking,
	 * compressPrompt) stays off; only the intent SOURCE changes. Returns undefined when the
	 * flag is off, letting preprocess() fall through to its internal regex fallback unchanged.
	 * Failures are non-fatal (regex is always a safe fallback) — logged, not thrown, since a
	 * classify failure must never abort a real turn.
	 */
	private async resolveIntent(userPrompt: string): Promise<IntentResult | undefined> {
		if (!INTENT_MODEL_ENABLED || !userPrompt) return undefined;
		try {
			const result = await classifyIntentOnAuxSeq(this.getE2B(), userPrompt);
			logIntentSource("model-shared-seq", result.intent);
			return result;
		} catch (err) {
			logIntentSource("regex", `model classify failed: ${err instanceof Error ? err.message : String(err)}`);
			return undefined;
		}
	}

	/**
	 * M12.3: E2B drafter for two-model speculative decoding, on by default (SUBVOCAL_LOCAL_DRAFT=0
	 * to disable; 1.90x measured on edit-shaped prompts, output identical to plain greedy — see
	 * doc/research/exclusions-sweep-2026-07.md). Same shared instance as the generator:
	 * AgentLoop's draftPrefill() re-prefills the shadow from scratch (forward() clears KV),
	 * so whatever a previous role left in the context is overwritten before use.
	 */
	private getDraftModel(): ModelGPU | null {
		if (process.env.SUBVOCAL_LOCAL_DRAFT === "0") return null;
		return this.getE2B();
	}

	/** Drop the conversation so the next turn re-prefills from scratch (e.g. after an abort).
	 *
	 * 2026-07 audit: DEFER if a turn is running. The worker's message handler is `async`, so a
	 * `reset` message can be dispatched at any await point of an in-flight runTurn — clearing
	 * loop/fedFiles/paths mid-turn would corrupt that turn's own state. Today the only caller
	 * (provider.ts) posts reset AFTER awaiting the turn, so this can't happen — but that's the
	 * client's discipline, not an invariant this engine can rely on. When turnActive, stash the
	 * request; runTurn's finally applies it once the turn is fully done. */
	resetConversation(): void {
		if (this.turnActive > 0) {
			this.pendingReset = true;
			return;
		}
		this.loop = null;
		this.prefixSig = [];
		this.fedFiles.clear();
		this.paths = new PathRegistry();
	}

	// ── M16.3: asymmetric-context compaction (E2B long-window distiller) ─────────
	// Bench-validated (training/bench-distill.mjs): the 12B on an E2B task-directed distillate
	// matches full-context quality on planted-fact scenarios while naive tail-truncation loses
	// the facts. On context pressure we distill the OLD history via the E2B and keep the recent
	// turns verbatim, instead of overflowing (reactive) or silently truncating.
	// Disable with SUBVOCAL_DISTILL=0. Threshold: SUBVOCAL_COMPACT_AT (default 0.8 of the window).

	/** Distiller model: reuse the shared E2B when loaded (whatever KV a previous role left there
	 *  is rebuilt from scratch by the next fresh AgentLoop / draftPrefill, so clobbering it here
	 *  is safe because compaction always drops the loop); otherwise lazy-load a small dedicated
	 *  E2B and free it after (compaction is rare, and freeing releases its KV while the mmap'd
	 *  weights stay warm in the page cache). */
	private withDistiller<T>(fn: (m: ModelGPU) => T): T {
		if (this.e2b) return fn(this.e2b);
		// "Long-window" distiller: same E2B context budget as the shared instance
		// (resolveE2BContextSize()), not the 12B's tighter one — it needs to see the OLD
		// history being distilled, which can exceed the 12B's own window.
		const m = new ModelGPU(activeProfile.smallModelPath, { contextSize: this.resolveE2BContextSize(), threads: 4, gpuLayers: 999, noKvOffload: true });
		try {
			return fn(m);
		} finally {
			m.free();
		}
	}

	/** Render old messages to plain text and distill them (chunked to the distiller's window). */
	private distillHistory(old: readonly Message[], allMessages: readonly Message[], task: string, cwd: string): string {
		const lines: string[] = [];
		for (const m of old) {
			if (m.role === "toolResult") {
				const tr = m as ToolResultMessage;
				lines.push(`[${tr.toolName} result]:\n${this.renderToolResult(tr, allMessages, cwd)}`);
			} else if (m.role === "assistant") {
				lines.push(`[assistant]:\n${renderAssistantText(m)}`);
			} else {
				lines.push(`[${m.role}]:\n${flattenText(m.content as never)}`);
			}
		}
		const text = lines.join("\n\n");
		return this.withDistiller((model) => {
			// Chunk to fit the distiller window: reserve ~1.5k tokens for system+task+output,
			// ~3.5 chars/token is a conservative estimate for prose+code.
			const budgetChars = Math.max(4000, (model.contextSize - 1500) * 3.5);
			const briefings: string[] = [];
			for (let off = 0; off < text.length; off += budgetChars) {
				const chunk = text.slice(off, off + budgetChars);
				briefings.push(distillContext(model, { context: chunk, task, maxOutputTokens: 350 }).distillate);
			}
			return briefings.join("\n");
		});
	}

	/** Replace everything but the last few messages with one distilled briefing message. */
	private compactMessages(messages: readonly Message[], cwd: string): Message[] {
		const KEEP_RECENT = 5;
		if (messages.length <= KEEP_RECENT + 1) return [...messages];
		const old = messages.slice(0, messages.length - KEEP_RECENT);
		const recent = messages.slice(messages.length - KEEP_RECENT);
		let task = "continue the session";
		for (let i = messages.length - 1; i >= 0; i--) {
			if (messages[i].role === "user") { task = flattenText(messages[i].content as never); break; }
		}
		const distillate = this.distillHistory(old, messages, task, cwd);
		const synthetic = {
			role: "user",
			content:
				`[Distilled summary of the earlier session — produced automatically because the ` +
				`context window filled up. Decisions and constraints below are authoritative.]\n${distillate}`,
			timestamp: Date.now(),
		} as unknown as Message;
		return [synthetic, ...recent];
	}

	/**
	 * M16.2 (fuller): cheap pre-decode estimate of how many tokens THIS turn's new content will
	 * cost, so the proactive-compaction check below can catch "the incoming turn alone pushes
	 * past the window" (a large file write/read in ONE turn) — not just "the accumulated history
	 * already did". Deliberately approximate in one respect only: it mirrors the M13.1 breadcrumb
	 * condition for `read` (an already-fed, unchanged file collapses to ~breadcrumb-sized) so a
	 * routine re-read of a file already in context doesn't trigger a needless compaction — but it
	 * does NOT invoke renderToolResult()'s M15.2 changed-blocks diff or M13.2 compaction, both of
	 * which could shrink a genuinely new/changed read further; this over-estimates in that case,
	 * which is the safe direction. Uses the real tokenizer (cheap, CPU-only, already used
	 * synchronously elsewhere in this file for text this size) rather than a chars-per-token
	 * guess, since the model is already resident.
	 */
	private estimateIncomingTokens(messages: readonly Message[], cwd: string): number {
		const newMessages = messages.slice(this.prefixSig.length);
		// 2026-07 audit: tokenize with a model that's ALREADY resident. The Gemma vocab is shared
		// across the 12B and the E2B, so the count is identical either way — but calling
		// getModel() in a small-brain conversation (where only the E2B is loaded, on purpose, to
		// keep RAM down) would force-load the 12B just to measure. Prefer the live small model.
		const model = (this.currentBrain === "small" && this.e2b) ? this.e2b : this.getModel();
		let tokens = 0;
		for (const m of newMessages) {
			if (m.role === "user") {
				tokens += model.tokenize(flattenText(m.content as never), false, false).length;
			} else if (m.role === "toolResult") {
				const tr = m as ToolResultMessage;
				const text = flattenText(tr.content);
				const call = this.findToolCall(messages, tr.toolCallId);
				const pathArg = typeof call?.args.path === "string" ? resolve(cwd, call.args.path) : null;
				const fed = pathArg ? this.fedFiles.get(pathArg) : undefined;
				if (tr.toolName === "read" && fed && fed.includes(text.trim())) continue; // breadcrumb-sized
				tokens += model.tokenize(text, false, false).length;
			}
		}
		return tokens;
	}

	/** Find the toolCall a toolResult answers (by id) to recover its name/arguments. */
	private findToolCall(messages: readonly Message[], toolCallId: string): { name: string; args: Record<string, unknown> } | null {
		for (let i = messages.length - 1; i >= 0; i--) {
			const msg = messages[i];
			if (msg.role !== "assistant" || !Array.isArray(msg.content)) continue;
			for (const raw of msg.content) {
				const block = raw as { type: string; id?: string; name?: string; arguments?: Record<string, unknown> };
				if (block.type === "toolCall" && block.id === toolCallId) {
					return { name: block.name ?? "", args: block.arguments ?? {} };
				}
			}
		}
		return null;
	}

	/**
	 * Render one toolResult into the MODEL's observation text. The TUI copy is untouched —
	 * these transforms only shrink what enters the KV:
	 *   - M13.1: a `read` whose result is already in the KV collapses to a breadcrumb; an
	 *     `edit`/`write` invalidates the record so the next read re-feeds post-edit truth.
	 *   - M13.2: everything else goes through compactObservation() (signal extraction).
	 */
	private renderToolResult(tr: ToolResultMessage, messages: readonly Message[], cwd: string): string {
		const text = flattenText(tr.content);
		const call = this.findToolCall(messages, tr.toolCallId);
		const pathArg = typeof call?.args.path === "string" ? resolve(cwd, call.args.path) : null;

		if (tr.toolName === "read" && pathArg && !tr.isError) {
			const fed = this.fedFiles.get(pathArg);
			const pathRef = this.paths.render(String(call?.args.path ?? pathArg)); // M15.4
			if (fed && fed.includes(text.trim())) {
				// M15.6 item 4: was `[read <pathRef>: content unchanged, already in context]`
				// (~10 tokens) — UNCHANGED_FILE_CHAR is a static registry marker explained once
				// in SUBVOCAL_SYSTEM_PROMPT, so the meaning is carried by ONE reused token instead
				// of respelling the sentence every re-read.
				return `[${UNCHANGED_FILE_CHAR} ${pathRef}]`;
			}
			const fullRead = call?.args.offset === undefined && call?.args.limit === undefined;
			// M15.2: the file WAS in context and changed — feed only the blocks whose
			// content-addressed anchor is new. Set membership IS the diff.
			if (fed && fullRead) {
				try {
					const lang = detectLanguage(pathArg);
					const before = computeBlockAnchors(fed, lang);
					const after = computeBlockAnchors(text, lang);
					let changed = [...after.entries()].filter(([char]) => !before.has(char));
					// Containment noise: an edit inside a nested block also flips its enclosing
					// blocks' hashes (if/try bodies etc.) — keep only the INNERMOST changed
					// blocks, the outer text would just re-feed them redundantly.
					changed = changed.filter(([, t]) => !changed.some(([, other]) => other !== t && t.includes(other)));
					// Only worth it when a minority of blocks changed; otherwise full feed.
					if (changed.length > 0 && changed.length <= Math.max(2, after.size / 2)) {
						this.fedFiles.set(pathArg, text);
						return (
							`[read ${pathRef}: ${changed.length} changed block(s) since last read — unchanged blocks already in context]\n` +
							changed.map(([, blockText]) => blockText).join("\n\n")
						);
					}
				} catch {
					/* unparseable content — fall through to the full feed */
				}
			}
			// Record only full reads — offset/limit partials can't stand in for the whole file.
			if (fullRead) {
				this.fedFiles.set(pathArg, text);
			}
			return text;
		}
		if ((tr.toolName === "edit" || tr.toolName === "write") && pathArg && !tr.isError) {
			this.fedFiles.delete(pathArg);
		}
		const compacted = compactObservation(text, tr.isError);
		// M15.6 item 3: a fixed pass/fail marker ahead of a bash result — legible exit status
		// without re-reading the extracted lines for error-ish clues. isError is the only signal
		// ToolResultMessage actually carries at this layer (no exit code, no timeout flag), so
		// this stays a plain pass/fail, not the 3-way pass/fail/timeout the story first sketched.
		if (tr.toolName === "bash") {
			return `[${tr.isError ? BASH_FAIL_CHAR : BASH_PASS_CHAR}]\n${compacted}`;
		}
		return compacted;
	}

	/** M13.4: `brain` picks which model generates this conversation's turns — 'large' (default,
	 *  the 12B) or 'small' (the E2B generator, only ever passed when DUAL_BRAIN_ENABLED routed
	 *  here). Config shared between both; only the model/profile/steering/speculation options
	 *  that don't apply to the E2B differ. */
	private newLoop(req: TurnRequest, brain: Brain = "large"): AgentLoop {
		const resolvedSystemPrompt = process.env.SUBVOCAL_KEEP_PI_PROMPT === "1"
			? (req.systemPrompt ?? '')
			: SUBVOCAL_SYSTEM_PROMPT;
		const common = {
			maxTokens: req.options?.maxTokens ?? 1024,
			// This is an agentic frontend: a from-scratch "create X" request has no file in context,
			// and AgentLoop's default 'fill' mode would forbid tool calls and make the model dump raw
			// content/prose. 'tools' keeps the tool-calling prompt so it CREATES files via `write`.
			emptyFileMode: 'tools' as const,
			// M16.1: show the active file with a line-number gutter so edit_lines can address exact
			// ranges (read results are guttered at the source; this covers the injected active file).
			lineGutter: process.env.SUBVOCAL_LINE_GUTTER !== "0",
			// The harness's agent instructions stay authoritative; AgentLoop appends the native
			// tool declarations (Epic M8) for the harness's OWN tool set. We append a short
			// completion-signal instruction so the user gets an explicit "done" line (the
			// agentic loop ends when the model stops calling tools — this makes that boundary
			// legible instead of the turn just going quiet).
			systemPrompt: resolvedSystemPrompt + COMPLETION_INSTRUCTION,
			tools: req.toolDefs,
			onToken: (text: string) => this.emitToken(text),
			shouldStop: () => this.hooks.shouldStop?.() ?? false,
		};

		if (brain === "small") {
			// M12.2's E2B generator, ported from the REPL's smallLoop construction: its own model,
			// its own profile (bare model-turn template, no thought-channel), no steering (never
			// measured on the E2B) and no suffix/draft speculation (nothing to draft FOR here —
			// this loop generates directly).
			return new AgentLoop({
				...common,
				// No edit_lines for the small brain: it is line-number-addressed, so its misuse is
				// invisible to shouldEscalate (which validates `edit`'s oldText against the file).
				// Live run 2026-07-06: the E2B picked edit_lines for a rename and silently
				// duplicated lines instead. The flat `edit` is validated AND detaggable.
				tools: (req.toolDefs ?? []).filter((t) => t.name !== "edit_lines"),
				model: this.getE2B(),
				profile: MacE2BProfile,
				temperature: req.options?.temperature ?? MacE2BProfile.defaultTemperature ?? 0,
				useSteering: false,
			});
		}

		const draftModel = this.getDraftModel();
		return new AgentLoop({
			...common,
			model: this.getModel(),
			...(draftModel ? { draftModel } : {}),
			temperature: req.options?.temperature ?? 0,
			// M15.1 hybrid ideogram steering — DISABLED by default (2026-07-05). Measured: in
			// hybrid mode the 12B chooses native tool calls every time (ideogram-edit 0/3,
			// protocol 3/3), so the ideogram shortcut never fires and only adds prompt bloat +
			// steering logit-bias + the ideogramEdit→edit translation path below for zero benefit.
			// It works 5/5 only in 'exclusive' mode (M4 showdown), which forbids tool calls and so
			// can't be the general agent mode. The intended fluency vehicle (a dialect LoRA) is
			// PARKED (doc/research/m15.7-lora-parked.md) — without it hybrid yields nothing. All the
			// machinery (astTagger/astEditor/ideogramSteering/exclusive mode + the translation
			// below) is kept for a future reimplementation. Re-enable with SUBVOCAL_IDEOGRAM=1.
			useSteering: IDEOGRAM_ENABLED,
			steeringPrompt: 'hybrid',
			// Suffix-tree drafter (doc/research/suffix-tree-speculative-decoding.md): free trie
			// lookup, net-positive on edit/refactor turns — exactly this provider's workload.
			// Engages only when the decode is greedy (the temperature default above is 0).
			useSuffixSpeculation: true,
		});
	}

	/**
	 * Find the file the conversation is currently working on (last read/edit target) and read
	 * its CURRENT content from disk. Content is re-read every turn on purpose: the harness
	 * executes edits between our turns, and preprocessing (AST tags, prewarm cache, suffix
	 * trie) against pre-edit content poisons every downstream consumer.
	 */
	private getActiveFile(messages: readonly Message[], cwd: string): { path: string; content: string } {
		outer: for (let i = messages.length - 1; i >= 0; i--) {
			const msg = messages[i];
			if (msg.role === "assistant" && Array.isArray(msg.content)) {
				for (let j = msg.content.length - 1; j >= 0; j--) {
					const block = msg.content[j] as { type: string; name?: string; arguments?: { path?: unknown } };
					if (block.type === "toolCall" && (block.name === "read" || block.name === "edit")) {
						if (typeof block.arguments?.path === "string") {
							this.activeFilePath = resolve(cwd, block.arguments.path);
							break outer;
						}
					}
				}
			}
		}
		// No tool call has established a file yet (typically turn 1): if the user's prompt names
		// a file that exists, load it so the model sees the REAL file instead of flying blind
		// and hallucinating one. Without this, "fix the bug in src/cart.ts" on turn 1 gives the
		// model an empty file context and it invents plausible-but-wrong code.
		if (!this.activeFilePath) {
			const mentioned = this.findMentionedFile(messages, cwd);
			if (mentioned) this.activeFilePath = mentioned;
		}
		if (!this.activeFilePath) return { path: "", content: "" };
		try {
			return {
				path: this.activeFilePath,
				content: existsSync(this.activeFilePath) ? readFileSync(this.activeFilePath, "utf-8") : "",
			};
		} catch {
			return { path: this.activeFilePath, content: "" };
		}
	}

	/**
	 * Scan the latest user message for a token that looks like a file path and resolves to an
	 * existing file under cwd. Returns the absolute path, or null. Deliberately narrow: matches
	 * tokens with a slash or a known code extension, and REQUIRES existsSync — a false positive
	 * would preprocess the wrong file, so err toward loading nothing (the model can still read()).
	 */
	private findMentionedFile(messages: readonly Message[], cwd: string): string | null {
		let text = "";
		for (let i = messages.length - 1; i >= 0; i--) {
			if (messages[i].role === "user") { text = flattenText(messages[i].content as never); break; }
		}
		if (!text) return null;
		// path-like tokens: optional dirs, a filename, a code extension (strip surrounding quotes/backticks).
		const re = /(?:^|[\s`'"(])([\w./-]+\.(?:ts|tsx|js|jsx|py|json|md|css|html|go|rs|c|cpp|h|java))\b/g;
		let m: RegExpExecArray | null;
		while ((m = re.exec(text)) !== null) {
			const candidate = resolve(cwd, m[1]);
			try {
				if (existsSync(candidate) && readFileSync(candidate, "utf-8").length > 0) return candidate;
			} catch {
				/* not readable — keep scanning */
			}
		}
		return null;
	}

	/**
	 * M15.1: ideogramEdit → standard pi edit call. Resolves the target node via astEditor
	 * against the CURRENT file, detagging newCode on failure (the M4 showdown fix — the model
	 * echoes tags inside replacement code). Returns null when no candidate applies cleanly;
	 * the model's text then stands as-is (best effort, never a corrupt edit).
	 *
	 * M15.5 adds 'rename'/'delete': dispatched before the edit/insert candidate loop below
	 * because their executors work in WHOLE-FILE terms, not a single node's byte range —
	 * rename can touch scattered call sites across the file (renameASTNode's whole-word
	 * substitution), so the resulting {oldText, newText} pair is the FULL file before/after,
	 * not a node-local slice. This costs nothing extra in MODEL tokens (the model only ever
	 * emitted the ~9-token op sequence; the full-file diff is assembled locally in JS for the
	 * executor) — it's just a different shape of `edit` tool-call argument.
	 */
	private translateIdeogramEdit(
		ide: NonNullable<AgentStep["ideogramEdit"]>,
		activeFile: { path: string; content: string },
		loop: AgentLoop,
	): { id: string; name: string; arguments: Record<string, unknown> } | null {
		const language = detectLanguage(activeFile.path);
		const clean = ide.newCode.trim();

		if (ide.mode === "delete") {
			const res = deleteASTNode({
				tokenId: ide.tokenId,
				source: activeFile.content,
				language: language === "python" ? "python" : "typescript",
				injections: loop.lastInjections,
			});
			if (!res.found || res.errors.length > 0 || !res.replaced) return null;
			const oldText = activeFile.content.slice(res.replaced.startIndex, res.replaced.endIndex);
			if (oldText.length === 0) return null;
			return {
				id: `ide_${++this.syntheticCallN}`,
				name: "edit",
				arguments: { path: activeFile.path, oldText, newText: "" },
			};
		}

		if (ide.mode === "rename") {
			// The payload IS the new name, not a code body — no detag candidates needed (there's
			// no code-echo risk for a bare identifier the way there is for a generated code block).
			const res = renameASTNode({
				tokenId: ide.tokenId,
				newName: clean,
				source: activeFile.content,
				language: language === "python" ? "python" : "typescript",
				injections: loop.lastInjections,
			});
			if (!res.found || res.errors.length > 0 || res.newSource === activeFile.content) return null;
			return {
				id: `ide_${++this.syntheticCallN}`,
				name: "edit",
				arguments: { path: activeFile.path, oldText: activeFile.content, newText: res.newSource },
			};
		}

		for (const candidate of [clean, ...detagCandidates(clean)]) {
			const input = {
				tokenId: ide.tokenId,
				newCode: candidate,
				source: activeFile.content,
				language: language === "python" ? ("python" as const) : ("typescript" as const),
				injections: loop.lastInjections,
			};
			const res = ide.mode === "insert" ? insertAfterNode(input) : editASTNode(input);
			if (!res.found || res.errors.length > 0 || !res.replaced) continue;
			const oldText = activeFile.content.slice(res.replaced.startIndex, res.replaced.endIndex);
			const newText = ide.mode === "insert" ? `${oldText}\n\n${candidate}` : candidate;
			if (oldText.length === 0 || oldText === newText) return null;
			return {
				id: `ide_${++this.syntheticCallN}`,
				name: "edit",
				arguments: { path: activeFile.path, oldText, newText },
			};
		}
		return null;
	}

	async runTurn(req: TurnRequest, hooks: TurnHooks = {}): Promise<TurnResult> {
		this.hooks = hooks;
		this.turnActive++; // keep the E2B idle-unload timer out of live turns (see its guards)
		try {
			let messages = req.messages;

			// M16.3/M16.2 PROACTIVE compaction: distill the old history via the E2B and restart the
			// loop on the compacted version BEFORE the next decode overflows. Checks currentNPast
			// (the accumulated history) PLUS this turn's own
			// estimated cost (M16.2 fuller): a single turn that re-reads/writes a large file can
			// push past the window on its own even when the pre-turn KV was comfortably under
			// threshold, and compactMessages() only shrinks the OLD prefix (KEEP_RECENT keeps the
			// last 5 messages — including this turn's new content — verbatim), so what actually
			// helps here is freeing the history's share of the budget for the new content to land
			// in, not shrinking the new content itself. A single-turn payload that alone exceeds
			// the whole window isn't fixable by history compaction either way — that's the
			// REACTIVE catch below (already handled, not this check's job). (The compacted history
			// flows through the normal replay/fresh machinery below; the drafter's shadow KV is
			// rebuilt from scratch by the new loop.)
			if (
				process.env.SUBVOCAL_DISTILL !== "0" &&
				this.loop !== null &&
				extendsPrefix(messages, this.prefixSig)
			) {
				const threshold = Number(process.env.SUBVOCAL_COMPACT_AT || "0.8");
				// 2026-07 KV audit: size the budget on the window of the model actually holding
				// this loop's KV. A small-brain loop lives on the E2B (32k in dual-brain), but
				// this check used resolveContextSize() — the 12B's window (8k in dual-brain) —
				// compacting E2B conversations ~4x too early.
				const window = this.currentBrain === "small" ? this.resolveE2BContextSize() : this.resolveContextSize();
				const budget = window * threshold;
				const incoming = this.estimateIncomingTokens(messages, req.cwd);
				if (this.loop.currentNPast + incoming > budget) {
					messages = this.compactMessages(messages, req.cwd);
					this.loop = null;
					this.prefixSig = [];
				}
			}

			// M13.4: a followUp can leave the easy band ("ora ristruttura il modulo") — the REPL
			// re-routes every task, so re-check each new user turn of a small-brain conversation
			// and rebuild on the 12B when the gate no longer says small. Dropping the loop sends
			// this turn through the non-incremental path below: the M13.3 replay rebuilds the KV
			// on the large brain and the tail dispatches incrementally. Never small→smaller: a
			// large-brain conversation stays large (same commitment rule as the REPL).
			if (this.loop !== null && this.currentBrain === "small" && extendsPrefix(messages, this.prefixSig)) {
				const newUserText = messages
					.slice(this.prefixSig.length)
					.filter((m) => m.role === "user")
					.map((m) => flattenText(m.content as never))
					.join("\n\n");
				if (newUserText.length > 0) {
					const decision = routeTask({ prompt: newUserText, fileContent: this.getActiveFile(messages, req.cwd).content });
					if (decision.brain === "large") {
						logDualBrainRoute("large", `followUp left the easy band: ${decision.reason}`);
						this.currentBrain = "large";
						this.loop = null;
						this.prefixSig = [];
					}
				}
			}

			// Same conversation ⇒ feed only the new tail; anything else ⇒ fresh loop.
			let incremental = this.loop !== null && extendsPrefix(messages, this.prefixSig);
			if (!incremental) {
				// M13.4: route a FRESH conversation to a brain before building its loop — the same
				// conservative gate the REPL uses (mechanical/localized edits → E2B, everything
				// else → 12B). Off by default; when off this.currentBrain stays "large" forever.
				// FRESH means no assistant turn yet: restored sessions and rewritten histories
				// replay 12B-generated turns and dispatch their tail incrementally, where
				// shouldEscalate() can never run — routing them small would leave a misroute with
				// no safety net (and the gate would judge a long conversation by its last message).
				this.currentBrain = "large";
				if (DUAL_BRAIN_ENABLED && !messages.some((m) => m.role === "assistant")) {
					const routeFile = this.getActiveFile(messages, req.cwd);
					let routePrompt = "";
					for (let i = messages.length - 1; i >= 0; i--) {
						if (messages[i].role === "user") { routePrompt = flattenText(messages[i].content as never); break; }
					}
					const decision = routeTask({ prompt: routePrompt, fileContent: routeFile.content });
					this.currentBrain = decision.brain;
					logDualBrainRoute(decision.brain, decision.reason);
				}
				this.loop = this.newLoop(req, this.currentBrain);
				this.prefixSig = [];
				this.fedFiles.clear();
			}
			let loop = this.loop!;

			// M13.3: a history we didn't produce (restored session, compaction) but that maps
			// onto turns gets its KV REBUILT via prefill-only replay — the model sees its own
			// real template and the cold store resumes matching prefixes — instead of being
			// re-serialized as a text blob. After replay the live tail dispatches incrementally.
			if (!incremental && messages.length > 1 && messages[0].role === "user") {
				let lastAssistantIdx = -1;
				for (let i = messages.length - 1; i >= 0; i--) {
					if (messages[i].role === "assistant") { lastAssistantIdx = i; break; }
				}
				// Need at least one assistant turn to replay, and a live tail after it.
				if (lastAssistantIdx > 0 && lastAssistantIdx < messages.length - 1) {
					const activeFile = this.getActiveFile(messages, req.cwd);
					// Record the active file BEFORE mapping the history: replay()'s prompt build
					// feeds its (tagged) content, so reads of it inside the replayed history
					// already collapse to breadcrumbs (M13.1), same as they would have live.
					if (activeFile.path && activeFile.content.length > 0) {
						this.fedFiles.set(activeFile.path, activeFile.content);
					}
					const replayTurns: ReplayTurn[] = messages.slice(1, lastAssistantIdx + 1).map((m): ReplayTurn => {
						if (m.role === "assistant") return { kind: "assistant", text: renderAssistantText(m) };
						if (m.role === "toolResult") {
							const tr = m as ToolResultMessage;
							return { kind: "tool", text: this.renderToolResult(tr, messages, req.cwd), toolName: tr.toolName };
						}
						return { kind: "user", text: flattenText(m.content as never) };
					});
					const replayPrompt = flattenText(messages[0].content as never);
					await loop.replay(
						{
							prompt: replayPrompt,
							filePath: activeFile.path,
							fileContent: activeFile.content,
							cpuOff: true, // M11.3: intent source is precomputedIntent below when enabled, regex
							// otherwise — the rest of cpuOff's tensor machinery (payload, logit mask) stays off.
							precomputedIntent: await this.resolveIntent(replayPrompt),
							// (2026-07 KV audit: a per-turn `new KVCacheManager(...)` was passed here for a
							// while — inert by construction, a fresh instance has no baseline to diff against.
							// See kvCacheManager.ts for what wiring it for real would require.)
							filePrewarmCache: this.filePrewarmCache,
						},
						replayTurns,
					);
					this.prefixSig = messages.slice(0, lastAssistantIdx + 1).map(messageSig);
					incremental = true; // the tail below dispatches on the rebuilt KV
				}
			}

			// 2026-07 KV audit: if the idle-unload detached the drafter between turns, re-attach
			// a fresh E2B now — its shadow KV is rebuilt from the loop's transcript (a few
			// seconds of E2B prefill vs losing the ~2x draft speedup for the whole rest of the
			// session). Large-brain only: a small-brain loop's generator IS the E2B — it never
			// has a drafter, and the unload guard above keeps it alive while its conversation
			// lives. getDraftModel() already honors SUBVOCAL_LOCAL_DRAFT=0 by returning null.
			if (incremental && this.currentBrain === "large" && !loop.hasDraftModel) {
				const dm = this.getDraftModel();
				if (dm) await loop.attachDraftModel(dm);
			}

			const newMessages = messages.slice(this.prefixSig.length);
			const newToolResults = newMessages.filter((m): m is ToolResultMessage => m.role === "toolResult");
			const newUserTexts = newMessages
				.filter((m) => m.role === "user")
				.map((m) => flattenText(m.content as never))
				.filter((t) => t.length > 0);

			const activeFile = this.getActiveFile(messages, req.cwd);
			const nPastBefore = loop.currentNPast;

			let step: AgentStep;
			const genStart = performance.now();
			if (incremental && newToolResults.length > 0 && newUserTexts.length === 0) {
				// Tool observation(s) → continue() on the live KV. Multiple results in one batch
				// are merged into one observation (buildToolResponse renders a single block).
				// Each result goes through the M13.1/M13.2 shrinkers (breadcrumb / compaction).
				const observation = newToolResults
					.map((tr) => (newToolResults.length > 1 ? `[${tr.toolName}${tr.isError ? " error" : ""}]\n` : "") + this.renderToolResult(tr, messages, req.cwd))
					.join("\n\n");
				step = await loop.continue(observation, newToolResults[newToolResults.length - 1].toolName);
			} else if (incremental && newUserTexts.length > 0 && newToolResults.length === 0) {
				// New user turn on the same conversation → followUp() on the live KV.
				step = await loop.followUp(newUserTexts.join("\n\n"));
			} else {
				// Fresh conversation — or a shape we can't map incrementally (mixed tool results
				// + user message in one batch, rewritten history): full start(), with the prior
				// history replayed as plain text when there is any.
				let lastUserIdx = -1;
				for (let i = messages.length - 1; i >= 0; i--) {
					if (messages[i].role === "user") { lastUserIdx = i; break; }
				}
				const promptText = lastUserIdx >= 0 ? flattenText(messages[lastUserIdx].content as never) : "";
				// P2-12: preprocess is about to inject the active file's (tagged) content — seed
				// fedFiles BEFORE rendering the transcript so read results of that same file
				// collapse to breadcrumbs instead of duplicating it raw in the same prompt
				// (mirrors the replay path above).
				if (lastUserIdx > 0 && activeFile.path && activeFile.content.length > 0) {
					this.fedFiles.set(activeFile.path, activeFile.content);
				}
				const prompt = lastUserIdx > 0
					? `${transcriptFallback(messages.slice(0, lastUserIdx), (tr) => this.renderToolResult(tr, messages, req.cwd))}\n\n${promptText}`
					: promptText;
				const startInput = {
					prompt,
					filePath: activeFile.path,
					fileContent: activeFile.content,
					cpuOff: true, // M11.3: intent source is precomputedIntent below when enabled, regex
					// otherwise — the rest of cpuOff's tensor machinery (payload, logit mask) stays off.
					precomputedIntent: await this.resolveIntent(prompt),
					// (2026-07 KV audit: per-turn KVCacheManager removed — see the replay-path note.)
					filePrewarmCache: this.filePrewarmCache,
				};
				// M13.4: buffer the small brain's first-step stream — if it escalates, the unusable
				// step must never reach the UI (the 12B redo would otherwise render right after it
				// in the same turn). The E2B is ~5x decode, so the held-back stream is brief.
				if (this.currentBrain === "small") this.smallTokenBuffer = [];
				try {
					step = await loop.start(startInput);
				} catch (err) {
					// M13.4: a small-brain FAILURE (e.g. Metal OOM bringing up the second instance
					// under memory pressure — seen live 2026-07-06) is just another misroute: redo
					// on the 12B instead of failing the turn. Free the broken E2B instance so a
					// later small route reloads it fresh instead of reusing a poisoned context.
					if (this.currentBrain !== "small") throw err;
					this.smallTokenBuffer = null;
					logDualBrainEscalate(`small brain failed: ${err instanceof Error ? err.message : String(err)}`);
					try {
						this.e2b?.free();
					} catch { /* best-effort */ }
					this.e2b = null;
					this.currentBrain = "large";
					loop = this.newLoop(req, "large");
					this.loop = loop;
					step = await loop.start(startInput);
				}
				// M13.1: preprocess just fed this file's (tagged) content into the KV — reads of
				// the same unchanged file can now collapse to a breadcrumb.
				if (activeFile.path && activeFile.content.length > 0) {
					this.fedFiles.set(activeFile.path, activeFile.content);
				}

				// M13.4: escalation — a misroute to the small brain is cheap ONLY if caught here.
				// If its first step produced nothing usable, redo the turn on the 12B and stay
				// there for the rest of the conversation (this.currentBrain persists via this.loop).
				if (this.currentBrain === "small") {
					const buffered = this.smallTokenBuffer ?? [];
					this.smallTokenBuffer = null;
					const escalateReason = shouldEscalate(step, activeFile.content);
					if (escalateReason) {
						// The buffered stream is discarded — the redo below streams live.
						logDualBrainEscalate(escalateReason);
						this.currentBrain = "large";
						loop = this.newLoop(req, "large");
						this.loop = loop;
						step = await loop.start(startInput);
						if (activeFile.path && activeFile.content.length > 0) {
							this.fedFiles.set(activeFile.path, activeFile.content);
						}
					} else {
						for (const text of buffered) this.hooks.onToken?.(text);
					}
				}
			}

			// The loop consumed everything up to this turn's input; our own reply becomes the
			// next context's trailing assistant message — pre-record its role-only signature.
			this.prefixSig = messages.map(messageSig);
			this.prefixSig.push("assistant");

			const toolCalls = step.toolCalls.map((tc) => {
				// M15.4: resolve path tokens BEFORE detagging — detag would strip the token.
				let args = tc.arguments;
				if (typeof args.path === "string") {
					args = { ...args, path: this.paths.resolve(args.path) };
				}
				return {
					id: tc.id,
					name: tc.name,
					// Only detag edits when the ideogram path is active — otherwise the model saw raw
					// source (no markers to strip) and detagging would corrupt real registry chars.
					arguments: (IDEOGRAM_ENABLED && tc.name === "edit") ? detagEditArguments(args, activeFile.content) : args,
				};
			});

			// M15.1: translate an ideogram edit into a standard pi edit call — the model
			// spoke the cheap protocol, the executor sees a normal tool call.
			let stepText = step.text;
			if (step.ideogramEdit && activeFile.path && activeFile.content.length > 0) {
				const synthetic = this.translateIdeogramEdit(step.ideogramEdit, activeFile, loop);
				if (synthetic) {
					toolCalls.push(synthetic);
					// Strip the raw ⊂...⊃ sequence from the visible text (the diff renders
					// through the tool call now).
					stepText = stepText.replace(/⊂[\s\S]*?⊃/g, "").trim();
				}
			}

			return {
				step: {
					text: stepText,
					thinking: step.thinking,
					toolCalls,
					tokenCount: step.tokenCount,
					stoppedNaturally: step.stoppedNaturally,
					genMs: performance.now() - genStart,
				},
				inputTokens: Math.max(0, loop.currentNPast - nPastBefore - step.tokenCount),
			};
		} catch (err) {
			// Context overflow: the KV filled up (a big file write/read can do it in ONE turn, which
			// cross-turn compaction can't help). Recover gracefully instead of crashing the session:
			// clear the KV, drop the loop so the next turn re-prefills a fresh (compacted) history,
			// and return a readable message. Other errors propagate.
			const msg = err instanceof Error ? err.message : String(err);
			if (!/exceed context size/i.test(msg)) throw err;
			try {
				const m = this.getModel();
				m.kvCacheSeqRemove(0, 0, -1);
				m.resetNPast(0);
			} catch { /* best-effort */ }
			this.loop = null;
			this.prefixSig = [];
			// M16.3 REACTIVE: retry ONCE on a distilled history before giving up — the overflow may
			// be the accumulated session (distillable) rather than one atomic oversized turn.
			if (process.env.SUBVOCAL_DISTILL !== "0" && !this.compactRetrying && req.messages.length > 6) {
				this.compactRetrying = true;
				try {
					return await this.runTurn({ ...req, messages: this.compactMessages(req.messages, req.cwd) }, hooks);
				} catch { /* fall through to the plain message below */ } finally {
					this.compactRetrying = false;
				}
			}
			const ctx = this.resolveContextSize();
			return {
				step: {
					text:
						`⚠️ Contesto pieno: questo turno ha superato la finestra di ${(ctx / 1024).toFixed(0)}k token ` +
						`(scrivere/rileggere un file grande può farlo in un colpo solo). Ho ripulito il contesto — ` +
						`riprova con una modifica più piccola, oppure rilancia con più contesto ` +
						`(single-model = 32k, o SUBVOCAL_LOCAL_CTX=32768).`,
					thinking: "",
					toolCalls: [],
					tokenCount: 0,
					stoppedNaturally: true,
					genMs: 0,
				},
				inputTokens: 0,
			};
		} finally {
			this.turnActive--;
			this.hooks = {};
			this.smallTokenBuffer = null;
			// 2026-07 audit: apply a reset that arrived mid-turn, now that the turn is fully done
			// and turnActive is back to 0 (guard against a nested/reactive-retry frame still live).
			if (this.pendingReset && this.turnActive === 0) {
				this.pendingReset = false;
				this.resetConversation();
			}
		}
	}
}
