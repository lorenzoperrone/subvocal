/**
 * utter.ts
 *
 * Substory 4.1.10 + 18 — The Subvocal Agent Harness.
 *
 * Interactive REPL agent: start it, type prompts, model edits files, keep going.
 * Like claude code / opencode / earendil-pi, but local via GGUF.
 *
 * CLI:
 *   npx tsx encode/src/utter.ts [--file path/to/file.ts]
 *
 *   > add type hints to calculateDiscount
 *   [edits file, shows diff]
 *   > also add JSDoc
 *   [more edits]
 *   > :quit
 */

import { execSync } from 'child_process';
import { readFileSync, writeFileSync } from 'fs';
import { createInterface } from 'readline';

import { FilePrewarmCache } from './filePrewarm.js';
import { IncrementalIntentClassifier } from './incrementalIntent.js';
import type { IntentResult } from './intentRouter.js';

import { initSmallModel, freeSmallModel, getSmallModel } from './smallModel.js';
import { initLargeModel, freeLargeModel, getLargeModel } from './largeModel.js';
import { AgentLoop } from './agentLoop.js';
import { StateSnapshotter } from './stateSnapshotter.js';
import { ShadowVFS, type VFSFile } from './shadowVFS.js';
import { buildSandbox, checkCommand, type SandboxRule } from './tokenSandbox.js';
import { computeDiff } from './osSandbox.js';
import { heuristicDistill, distillObservation } from './observationDistiller.js';
import { validateSyntaxForPath, formatSyntaxErrors } from './syntaxValidator.js';
import { compileCheck } from './compileValidator.js';
import { activeProfile } from './modelProfile.js';
import { gitInit, gitStageAll, gitCommit } from '@subvocal/synapse';
import { editASTNode, insertAfterNode, type ASTEditInput, type ASTEditResult } from './astEditor.js';
import { detectLanguage } from './astTagger.js';
import { PipelineController } from './dualBrainPipeline.js';
import { ModelGPU, type BaseModel } from '@subvocal/synapse';
import { MacE2BProfile } from './modelProfile.js';
import { detagCandidates, routeTask, shouldEscalate } from './dualBrainRouter.js';
import { WorkerIntentClassifier } from './workerIntent.js';
import {
  computeResultDiffs,
  buildCompilerStats,
  formatCompilerStats,
  formatDiffSummaries,
  type DiffSummary,
  type CompilerStats,
} from './outputEmitter.js';

// ── Public types ───────────────────────────────────────────────────────────────

export interface UtterConfig {
	smallModelPath: string;
	largeModelPath: string;
	/** Project directory for sandbox scanning and command execution. */
	projectDir?: string;
	/** Maximum agentic turns before forcing exit (default: 10). */
	maxTurns?: number;
	/** Sampling temperature for the large model (default: 0 = greedy). */
	temperature?: number;
	/**
	 * Substory 5.1: Auto-commit all VFS changes via libgit2 after task completion.
	 * Requires projectDir to be an initialized git repository.
	 */
	autoCommit?: boolean;
	/** Substory 5.1: Suppress all output on success (UNIX silent mode). */
	quiet?: boolean;
	/**
	 * Substory 17: Disable CPU small model entirely.
	 * Intent classification falls back to regex. Observation distillation
	 * falls back to truncation. Reduces RAM usage by ~2.4 GB.
	 */
	cpuOff?: boolean;
	/**
	 * Substory 18: Enable dual-brain pipeline — overlap CPU pre-processing
	 * of the next turn with GPU decode of the current turn.
	 * Default: true.
	 */
	pipeline?: boolean;
	/**
	 * Epic M12.2: dual-brain GENERATION routing. Loads a second E2B instance (MacE2BProfile,
	 * own KV context — deliberately NOT shared with the intent-classifier instance, whose
	 * forward() calls would clobber a generator's live KV mid-turn) and routes easy-band
	 * tasks to it (~5x decode), escalating back to the large brain when its first step
	 * produces nothing usable. Default: false.
	 */
	dualBrain?: boolean;
}

export interface UtterLogEntry {
	turn: number;
	type: 'start' | 'tool' | 'rollback' | 'commit' | 'sandbox' | 'done' | 'error';
	message: string;
	ms?: number;
}

export interface UtterSession {
	config: UtterConfig;
	smallModel: BaseModel | null;
	largeModel: ModelGPU;
	loop: AgentLoop;
	snapshotter: StateSnapshotter;
	vfs: ShadowVFS;
	sandbox: SandboxRule[];
	log: UtterLogEntry[];
	pipeline?: PipelineController;
	/**
	 * Mechanism A (doc/research/predictive-prefill-while-typing.md): per-session cache of the
	 * file-dependent (prompt-independent) preprocess phases. Warmed at file-open time (the
	 * REPL `:file` command / `--file` flag) so the work is already done by submit time.
	 */
	filePrewarmCache: FilePrewarmCache;
	/** M12.2: E2B generator model + loop, present only when config.dualBrain is set. */
	smallGenModel?: ModelGPU | null;
	smallLoop?: AgentLoop | null;
	/** M12.3: E2B drafter instance (SUBVOCAL_DRAFT=on) — shadow context for two-model specdec. */
	draftModel?: ModelGPU | null;
	/** M12.2: which brain ran the LAST runTask() turn — REPL follow-ups must go to the loop
	 *  whose KV actually holds the conversation, not unconditionally to the large one. */
	lastBrain?: 'small' | 'large';
}

export interface UtterTask {
	prompt: string;
	filePath: string;
	fileContent: string;
}

export interface UtterResult {
	/** Files modified during the task (VFS export). */
	files: VFSFile[];
	turns: number;
	totalTokens: number;
	rollbacks: number;
	errors: number;
	logs: UtterLogEntry[];
	/** Substory 5.1: Per-file unified diffs (original → mutated). */
	diffs: DiffSummary[];
	/** Substory 5.1: Aggregated compiler telemetry. */
	compilerStats: CompilerStats;
	/** Substory 5.1: libgit2 commit SHA if autoCommit was requested, else null. */
	commitSha: string | null;
}

// ── Session lifecycle ──────────────────────────────────────────────────────────

/**
 * Resolve the large model's context size for a session. Whenever a SECOND model shares the
 * unified-memory pool — the E2B classifier (--cpu-model on) or the E2B generator
 * (--dual-brain on, M12.2) — the profile may cap it (ModelProfile.dualBrainMaxCtx) so both
 * fit without swapping; an explicit SUBVOCAL_CONTEXT_SIZE always wins. Returns undefined when
 * no cap applies (initLargeModel then uses the profile's warm-tier default).
 */
export function resolveLargeContextSize(cpuOff: boolean, dualBrain = false): number | undefined {
	if (cpuOff && !dualBrain) return undefined;
	if (process.env.SUBVOCAL_CONTEXT_SIZE) return undefined; // explicit override wins
	const cap = activeProfile.dualBrainMaxCtx;
	if (!cap) return undefined;
	return Math.min(cap, activeProfile.kvTiers.warm.contextSize);
}

/**
 * Initialize models and allocate session resources.
 * Call once at startup; reuse for multiple runTask() calls.
 */
export async function initSession(config: UtterConfig): Promise<UtterSession> {
	let smallModel: BaseModel | null = null;

	if (!config.cpuOff) {
		// Backend + layers come from the profile (Mac: ModelGPU/Metal, gpuLayers 999).
		initSmallModel({
			modelPath: config.smallModelPath,
			...activeProfile.smallOpts,
		});
		smallModel = getSmallModel();
	}

	const largeCtx = resolveLargeContextSize(config.cpuOff ?? false, config.dualBrain ?? false);
	initLargeModel({
		modelPath: config.largeModelPath,
		gpuLayers: 999,
		...(largeCtx !== undefined ? { contextSize: largeCtx } : {}),
	});
	const largeModel = getLargeModel();

	// M12.3: optional E2B drafter for two-model speculative decoding (1.90x measured,
	// doc/research/exclusions-sweep-2026-07.md). Opt-in via SUBVOCAL_DRAFT=on: it costs the
	// E2B weights (2.4 GiB mmap, shared with any other E2B instance) plus a shadow KV sized
	// to the large model's live context. Deliberately its OWN instance — the M11.2 classifier
	// and the M12.2 smallGenModel have their own KV lifecycles a drafter would clobber.
	let draftModel: ModelGPU | null = null;
	if ((process.env.SUBVOCAL_DRAFT ?? 'off') === 'on') {
		draftModel = new ModelGPU(config.smallModelPath, {
			contextSize: largeCtx ?? activeProfile.largeOpts.contextSize,
			threads: 4,
			gpuLayers: 999,
		});
		console.log('[draft] E2B drafter loaded (SUBVOCAL_DRAFT=on): two-model specdec active');
	}

	const loop = new AgentLoop({
		model: largeModel,
		temperature: config.temperature ?? activeProfile.defaultTemperature ?? 0,
		maxTokens: 2048,
		useSteering: true,
		// Suffix-tree drafter (doc/research/suffix-tree-speculative-decoding.md): free trie
		// lookup, validated net-positive on edit-shaped turns. Composes with steering because
		// IdeogramSteering samples greedily (pure observer) — and it makes the file-open
		// prewarmSuffixTree() call below actually live (it no-ops when this flag is off).
		useSuffixSpeculation: true,
		...(draftModel ? { draftModel } : {}),
		onToken: config.quiet ? undefined : (text: string) => process.stdout.write(text),
	});

	const pipeline = config.pipeline !== false
		? new PipelineController()
		: undefined;

	// M12.2: dual-brain generation — a SECOND E2B instance dedicated to generating (own KV
	// context; the classifier instance from --cpu-model on stays untouched). Its AgentLoop
	// runs MacE2BProfile (bare model-turn openers — E2B's real template) and gets its own
	// cold-store namespace automatically (per-model subdir, see AgentLoop's constructor).
	let smallGenModel: ModelGPU | null = null;
	let smallLoop: AgentLoop | null = null;
	if (config.dualBrain) {
		smallGenModel = new ModelGPU(MacE2BProfile.largeModelPath, {
			...MacE2BProfile.largeOpts,
		});
		smallLoop = new AgentLoop({
			model: smallGenModel,
			profile: MacE2BProfile,
			temperature: config.temperature ?? MacE2BProfile.defaultTemperature ?? 0,
			maxTokens: 2048,
			useSteering: false,
			onToken: config.quiet ? undefined : (text: string) => process.stdout.write(text),
		});
		console.log(`[dualBrain] E2B generator loaded: ${MacE2BProfile.largeModelPath}`);
	}

	return {
		config,
		smallModel,
		largeModel,
		loop,
		snapshotter: new StateSnapshotter(largeModel),
		vfs: new ShadowVFS(),
		sandbox: buildSandbox(config.projectDir ?? process.cwd()),
		log: [],
		pipeline,
		filePrewarmCache: new FilePrewarmCache(),
		smallGenModel,
		smallLoop,
		draftModel,
	};
}

/** Release GPU/RAM resources acquired by initSession(). */
export async function freeSession(session: UtterSession): Promise<void> {
	// Let any background cold-store checkpoint write (kicked off during the last turn to overlap
	// decode) finish before we tear down — otherwise the checkpoint would be lost/half-written.
	await session.loop.flushColdWrites();
	if (session.smallLoop) await session.smallLoop.flushColdWrites();
	session.draftModel?.free();
	session.smallGenModel?.free();
	freeSmallModel();
	freeLargeModel();
	session.vfs.reset();
}

// ── Task runner ────────────────────────────────────────────────────────────────

/** Substory 17: distillation helper — uses the CPU model if available, otherwise truncates. */
function makeDistill(smallModel: BaseModel | null | undefined): (raw: string) => { label: string } {
	return (raw: string) => {
		if (smallModel) {
			return heuristicDistill(raw, smallModel) ?? distillObservation(raw, smallModel);
		}
		return { label: raw.slice(0, 80).replace(/\n/g, ' ') };
	};
}

/** Mutable turn counters shared between runTask() and the tool loop. */
interface ToolLoopCounters {
	turns: number;
	totalTokens: number;
	rollbacks: number;
	errors: number;
}

/**
 * The agentic tool loop: execute one tool call per iteration (edit with syntax+compiler
 * gates and VFS/KV rollback, sandboxed bash, unknown-tool recovery) until the model stops
 * calling tools or maxTurns is reached. Extracted from runTask() (2026-07 audit item) so REPL
 * follow-ups run the SAME loop instead of silently dropping tool calls.
 * Mutates `counters` and appends to session.log; returns the final step.
 */
async function executeToolLoop(
	session: UtterSession,
	loop: AgentLoop,
	snapshotter: StateSnapshotter,
	task: UtterTask,
	firstStep: import('./agentLoop.js').AgentStep,
	counters: ToolLoopCounters,
	opts?: { keepAlive?: boolean },
): Promise<import('./agentLoop.js').AgentStep> {
	const { vfs, sandbox, config, log } = session;
	const distill = makeDistill(session.smallModel);
	const maxTurns = config.maxTurns ?? 10;
	const projectDir = config.projectDir ?? process.cwd();
	let step = firstStep;

	while (counters.turns < maxTurns && step.toolCalls.length > 0) {
		const toolCall = step.toolCalls[0];
		const tTool = performance.now();

		if (toolCall.name === 'edit') {
			const file = (toolCall.arguments.file as string | undefined) ?? task.filePath;
			let oldCode = (toolCall.arguments.old as string | undefined) ?? '';
			let newCode = (toolCall.arguments.new as string | undefined) ?? '';
			const currentContent = vfs.read(file) ?? '';

			// The model saw TAGGED content (astTagger ideograms) and echoes the tags back in
			// old/new, but the VFS holds the RAW file — de-tag before matching or every
			// tag-echoing edit fails "target text not found" (pre-existing bug surfaced by the
			// M12.2 integration test; it bit every brain, not just the small one).
			if (oldCode && !currentContent.includes(oldCode)) {
				const oldCandidates = detagCandidates(oldCode);
				const newCandidates = detagCandidates(newCode);
				for (let c = 0; c < oldCandidates.length; c++) {
					if (currentContent.includes(oldCandidates[c])) {
						oldCode = oldCandidates[c];
						newCode = newCandidates[c];
						break;
					}
				}
			}

			// Snapshot state before applying the edit
			snapshotter.snapshot(loop.currentNPast, { source: currentContent, tagMap: new Map(), injections: [] }, null);
			vfs.snapshot();

			// Guard: old text must be present
			if (oldCode && !currentContent.includes(oldCode)) {
				vfs.rollback();
				snapshotter.rollback(0);
				counters.errors++;
				const msg = `Edit failed: target text not found in ${file}.`;
				log.push({ turn: counters.turns, type: 'error', message: msg, ms: performance.now() - tTool });
				step = await loop.continue(`${msg} Verify the exact text to replace and retry.`);
				counters.turns++;
				counters.totalTokens += step.tokenCount;
				continue;
			}

			const editedContent = oldCode ? currentContent.replace(oldCode, newCode) : newCode;
			const validation = validateSyntaxForPath(file, editedContent);

			if (!validation.ok && !validation.skipped) {
				// Syntax error: rollback VFS + KV, distill error, retry
				vfs.rollback();
				snapshotter.rollback(0);
				counters.rollbacks++;
				counters.errors++;

				const rawErr = formatSyntaxErrors(validation);
				const distilled = distill(rawErr);
				log.push({ turn: counters.turns, type: 'rollback', message: `syntax error → ${distilled.label}`, ms: performance.now() - tTool });

				step = await loop.continue(`Syntax error — edit rolled back. ${distilled.label}. Fix and retry.`);
			} else {
				// ── Substory 4.3: Compiler check before VFS commit ────────────────
				const compile = compileCheck(file, editedContent);
				if (!compile.ok) {
					// Language compiler rejected the code — rollback and retry
					vfs.rollback();
					snapshotter.rollback(0);
					counters.rollbacks++;
					counters.errors++;

					const distilled = distill(compile.message);
					log.push({ turn: counters.turns, type: 'rollback', message: `${compile.tool} → ${distilled.label}`, ms: performance.now() - tTool });

					step = await loop.continue(`Compiler error — edit rolled back. ${distilled.label}. Fix and retry.`);
				} else {
					vfs.write(file, editedContent);
					vfs.commit();
					snapshotter.commit();
					const compileNote = compile.tool !== 'skipped' ? ` (${compile.tool} ✓)` : '';
					log.push({ turn: counters.turns, type: 'commit', message: `edited ${file}${compileNote}`, ms: performance.now() - tTool });

					step = await loop.continue('Edit applied successfully, compiler clean.');
				}
			}

		} else if (toolCall.name === 'bash') {
			const command = (toolCall.arguments.command as string | undefined) ?? '';
			const sandboxResult = checkCommand(command, sandbox);

			if (!sandboxResult.allowed) {
				log.push({ turn: counters.turns, type: 'sandbox', message: `blocked: ${command}`, ms: performance.now() - tTool });
				step = await loop.continue(`Command rejected by sandbox: ${sandboxResult.reason}`);
			} else {
				try {
					const output = execSync(command, {
						cwd: projectDir,
						timeout: 30_000,
						encoding: 'utf-8',
					});
					log.push({ turn: counters.turns, type: 'tool', message: `ran: ${command}`, ms: performance.now() - tTool });
					step = await loop.continue(`Command output:\n${String(output).slice(0, 500)}`);
				} catch (e) {
					const stderr = (e as { stderr?: Buffer }).stderr?.toString('utf-8') ?? String(e);
					counters.errors++;
					const distilled = distill(stderr);
					log.push({ turn: counters.turns, type: 'error', message: `bash failed: ${command}`, ms: performance.now() - tTool });
					step = await loop.continue(`Command failed: ${distilled.label}`);
				}
			}

		} else {
			// Unknown tool: tell the model and let it recover
			const msg = `Unknown tool '${toolCall.name}'. Use 'edit' or 'bash'.`;
			log.push({ turn: counters.turns, type: 'error', message: msg, ms: performance.now() - tTool });
			step = await loop.continue(msg);
		}

		// Substory 18: take precomputed, schedule next for REPL mode
		if (opts?.keepAlive) {
			await session.pipeline?.take();
			session.pipeline?.schedule({
				prompt: task.prompt,
				filePath: task.filePath,
				fileContent: vfs.read(task.filePath) ?? task.fileContent,
				cpuOff: config.cpuOff,
				multiFile: true,
				filePrewarmCache: session.filePrewarmCache,
			});
		}

		counters.turns++;
		counters.totalTokens += step.tokenCount;
	}

	return step;
}

/**
 * Apply an ideogram edit via astEditor, validate, commit or rollback.
 * Shared between runTask and REPL follow-up.
 */
async function applyIdeogramEdit(
	session: UtterSession,
	ideogram: import('./agentLoop.js').IdeogramEdit,
	taskFilePath: string,
	originalPrompt?: string,
): Promise<boolean> {
	const { loop, snapshotter, vfs, config, smallModel, log } = session;
	const lang = detectLanguage(taskFilePath);
	const currentContent = vfs.read(taskFilePath) ?? readFileSync(taskFilePath, 'utf-8');
	const MAX_RETRIES = 2;

	for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
		const editInput: ASTEditInput = {
			tokenId: ideogram.tokenId,
			newCode: ideogram.newCode.trim(),
			source: currentContent,
			language: lang === 'python' ? 'python' : 'typescript',
			injections: loop.lastInjections,
		};

		const editResult = ideogram.mode === 'insert'
			? insertAfterNode(editInput)
			: editASTNode(editInput);

		if (editResult.found && editResult.errors.length === 0) {
			const compile = compileCheck(taskFilePath, editResult.newSource);
			if (compile.ok) {
				vfs.write(taskFilePath, editResult.newSource);
				vfs.commit();
				log.push({ turn: 1, type: 'commit', message: `ideogram edit → ${ideogram.nodeLabel}`, ms: 0 });
				return true;
			}
			if (attempt < MAX_RETRIES) {
				// Auto-retry: feed compiler error back to model
				const retryPrompt = `Edit failed: ${compile.message.slice(0, 100)}. Retry with correct syntax.`;
				const step = await loop.continue(retryPrompt);
				if (step.ideogramEdit) {
					ideogram = step.ideogramEdit;
					continue;
				}
			}
			vfs.rollback();
			snapshotter.rollback(0);
			return false;
		}

		if (editResult.found && attempt < MAX_RETRIES) {
			const retryPrompt = `Edit failed: syntax error. Retry with correct syntax.`;
			const step = await loop.continue(retryPrompt);
			if (step.ideogramEdit) {
				ideogram = step.ideogramEdit;
				continue;
			}
		}

		vfs.rollback();
		snapshotter.rollback(0);
		log.push({ turn: 1, type: 'rollback', message: `edit failed (attempt ${attempt + 1})`, ms: 0 });
		return false;
	}

	return false;
}

/**
 * Run a single coding task end-to-end and return the result.
 * Can be called multiple times on the same session with different tasks.
 */
export async function runTask(session: UtterSession, task: UtterTask, opts?: { keepAlive?: boolean; precomputedIntent?: IntentResult }): Promise<UtterResult> {
  const { vfs, sandbox, config, smallModel, log } = session;

  // M12.2: dual-brain routing — pick the generating brain for THIS task. The snapshotter is
  // model-bound (KV rollback), so it must follow the routed model, not stay on the large one.
  let loop = session.loop;
  let snapshotter = session.snapshotter;
  let routedSmall = false;
  if (session.smallLoop && session.smallGenModel) {
    const decision = routeTask({ prompt: task.prompt, fileContent: task.fileContent });
    if (decision.brain === 'small') {
      loop = session.smallLoop;
      snapshotter = new StateSnapshotter(session.smallGenModel);
      routedSmall = true;
    }
    if (!config.quiet) console.log(`[dualBrain] route → ${decision.brain} (${decision.reason})`);
  }

  const distill = makeDistill(smallModel);
	const maxTurns = config.maxTurns ?? 10;
	const projectDir = config.projectDir ?? process.cwd();

	let turns = 0;
	let totalTokens = 0;
	let rollbacks = 0;
	let errors = 0;
	const t0 = performance.now();

	// Substory 5.1: Record original contents before any mutations (for diff computation)
	const originalContents = new Map<string, string>();
	originalContents.set(task.filePath, task.fileContent);

	// Reset per-task state
	vfs.reset();
	vfs.write(task.filePath, task.fileContent);
	log.length = 0;

	// Turn 0: preprocess + prefill + first generation
	const startInput = {
		prompt: task.prompt,
		filePath: task.filePath,
		fileContent: task.fileContent,
		cpuOff: config.cpuOff,
		multiFile: true,
		// Mechanisms A + B: reuse the file-open prewarm and the while-typing intent, if present.
		filePrewarmCache: session.filePrewarmCache,
		precomputedIntent: opts?.precomputedIntent,
	};
	let step0 = await loop.start(startInput);
	turns++;
	totalTokens += step0.tokenCount;

	// M12.2: escalation — a misroute to the small brain is cheap ONLY if caught here. If its
	// first step produced nothing usable, redo the turn on the large brain and stay there.
	if (routedSmall) {
		const escalateReason = shouldEscalate(step0, task.fileContent);
		if (escalateReason) {
			if (!config.quiet) console.log(`[dualBrain] escalate → large (${escalateReason})`);
			log.push({ turn: turns, type: 'error', message: `dual-brain escalation: ${escalateReason}`, ms: performance.now() - t0 });
			loop = session.loop;
			snapshotter = session.snapshotter;
			routedSmall = false;
			step0 = await loop.start(startInput);
			turns++;
			totalTokens += step0.tokenCount;
		}
	}

	log.push({
		turn: turns,
		type: 'start',
		message: `${step0.tokenCount} tokens, ${step0.toolCalls.length} tool call(s)${routedSmall ? ' [small brain]' : ''}`,
		ms: performance.now() - t0,
	});

	// Substory 18: schedule precompute for REPL mode
	if (opts?.keepAlive) {
		session.pipeline?.schedule({
			prompt: task.prompt,
			filePath: task.filePath,
			fileContent: vfs.read(task.filePath) ?? task.fileContent,
			cpuOff: config.cpuOff,
			multiFile: true,
			filePrewarmCache: session.filePrewarmCache,
		});
	}

	let step = step0;

	// Fallback: when no tool calls parsed and the file is empty, try to extract
	// generated content from model output (handles models that spill code inline
	// without closing </tool_call>, or generate raw code directly).
	if (step.toolCalls.length === 0 && task.fileContent.length === 0 && step.text.length > 20) {
		// Strip Gemma4-specific token garbage from start (may be nested).
		// Known prefixes: <|channel>, <channel|>, _think, _thought
		const garbageRE = /^(<\|?channel\|?>|_think|_thought)\s*/i;
		let content = step.text;
		let prev = '';
		while (prev !== content) {
			prev = content;
			content = content.replace(garbageRE, '');
		}
		content = content.trim();
		// If the first line still looks like a tool call or channel header, skip it
		const nlIdx = content.indexOf('\n');
		if (nlIdx !== -1) {
			const firstLine = content.slice(0, nlIdx);
			if (firstLine.includes('tool_call') || firstLine.startsWith('{')) {
				content = content.slice(nlIdx + 1).trim();
			}
		}
		if (content.length > 50) {
			vfs.write(task.filePath, content);
			vfs.commit();
			log.push({ turn: turns, type: 'commit', message: `wrote ${task.filePath} (direct: ${content.length} chars)`, ms: performance.now() - t0 });
		}
	}

	// Substory 4.2: Ideogram steering edit — apply via astEditor
	if (step.ideogramEdit && task.fileContent.length > 0) {
		const ideogram = step.ideogramEdit;
		const lang = detectLanguage(task.filePath);
		const currentContent = vfs.read(task.filePath) ?? task.fileContent;

		// Trim leading/trailing whitespace from generated code (model may add spaces/newlines)
		const cleanCode = ideogram.newCode.trim();

		snapshotter.snapshot(loop.currentNPast, { source: currentContent, tagMap: new Map(), injections: [] }, null);
		vfs.snapshot();

		// M4 showdown finding (2026-07-04, doc/epics/EPIC-M4): the model echoes AST tags (∅ …)
		// INSIDE its replacement code too, not just in tool-call arguments — the echoed tag is a
		// syntax error in the result. Same detag treatment as the M12 edit path: try the raw
		// code first, fall back to detagged candidates on failure. This single fix took the
		// ideogram edit suite from 4/5 to 5/5.
		const attemptEdit = (candidate: string): ASTEditResult => {
			const editInput: ASTEditInput = {
				tokenId: ideogram.tokenId,
				newCode: candidate,
				source: currentContent,
				language: lang === 'python' ? 'python' : 'typescript',
				injections: loop.lastInjections,
			};
			return ideogram.mode === 'insert' ? insertAfterNode(editInput) : editASTNode(editInput);
		};
		const candidates = [cleanCode, ...detagCandidates(cleanCode)];
		let editResult: ASTEditResult = attemptEdit(candidates[0]);
		for (let ci = 1; ci < candidates.length && !(editResult.found && editResult.errors.length === 0); ci++) {
			const attempt = attemptEdit(candidates[ci]);
			// On success switch to the detagged result; on failure keep the FIRST attempt so
			// error reporting reflects what the model actually emitted.
			if (attempt.found && attempt.errors.length === 0) editResult = attempt;
		}
		if (editResult.found && editResult.errors.length === 0) {
			const compile = compileCheck(task.filePath, editResult.newSource);
			if (compile.ok) {
				vfs.write(task.filePath, editResult.newSource);
				vfs.commit();
				log.push({ turn: turns, type: 'commit', message: `ideogram edit → ${ideogram.nodeLabel}`, ms: performance.now() - t0 });
				step = await loop.continue(
					`Edit applied to ${ideogram.nodeLabel} via ideogram steering. Continue or stop.`,
				);
			} else {
				vfs.rollback();
				snapshotter.rollback(0);
				rollbacks++;
				errors++;
				const distilled = distill(compile.message);
				log.push({ turn: turns, type: 'rollback', message: `ideogram compile → ${distilled.label}`, ms: performance.now() - t0 });
				step = await loop.continue(`Compiler error on ideogram edit — rolled back. ${distilled.label}. Fix and retry.`);
			}
		} else if (editResult.found) {
			vfs.rollback();
			snapshotter.rollback(0);
			rollbacks++;
			errors++;
			const rawErr = formatSyntaxErrors({ ok: false, skipped: false, errors: editResult.errors, language: null });
			const distilled = distill(rawErr);
			log.push({ turn: turns, type: 'rollback', message: `ideogram syntax → ${distilled.label}`, ms: performance.now() - t0 });
			step = await loop.continue(`Syntax error on ideogram edit — rolled back. ${distilled.label}. Fix and retry.`);
		} else {
			vfs.rollback();
			snapshotter.rollback(0);
			errors++;
			log.push({ turn: turns, type: 'error', message: `ideogram tag not found: ${ideogram.tokenId} (${ideogram.nodeLabel})`, ms: performance.now() - t0 });
			step = await loop.continue(`Ideogram tag ${ideogram.nodeLabel} not found in file. Try a different tag.`);
		}
		turns++;
		totalTokens += step.tokenCount;
	}

	// Agentic loop: one tool call handled per iteration (shared with REPL follow-ups).
	const counters: ToolLoopCounters = { turns, totalTokens, rollbacks, errors };
	step = await executeToolLoop(session, loop, snapshotter, task, step, counters, opts);
	({ turns, totalTokens, rollbacks, errors } = counters);

	// M12.2: remember which brain owns the live conversation KV for REPL follow-ups.
	session.lastBrain = routedSmall ? 'small' : 'large';

	log.push({ turn: turns, type: 'done', message: `${turns} turn(s), ${totalTokens} token(s), ${rollbacks} rollback(s)`, ms: performance.now() - t0 });

	// Substory 5.1: Compute diffs and telemetry while GPU is still alive
	const exportedFiles = vfs.export();
	const diffs = computeResultDiffs(exportedFiles, originalContents);
	const compilerStats = buildCompilerStats(log);

	// Substory 5.1: GPU Muting — free VRAM immediately after generation ends,
	// before post-processing (diff, report, git). Skip in REPL mode (keepAlive).
	if (!opts?.keepAlive) freeLargeModel();

	// Substory 5.1: libgit2 soft commit (auto-commit mode)
	let commitSha: string | null = null;
	if (config.autoCommit && exportedFiles.length > 0) {
		try {
			gitInit(projectDir);
			gitStageAll();
			commitSha = gitCommit(
				`subvocal: ${task.prompt.slice(0, 72)}`,
				'subvocal',
				'subvocal@local',
			);
		} catch {
			// Non-fatal — commit failure does not abort the result
		}
	}

	return {
		files: exportedFiles,
		turns,
		totalTokens,
		rollbacks,
		errors,
		logs: [...log],
		diffs,
		compilerStats,
		commitSha,
	};
}

// ── CLI ────────────────────────────────────────────────────────────────────────

async function cli(): Promise<void> {
	const args = process.argv.slice(2);
	let prompt = '';
	let filePath = '';
	let quiet = false;
	let autoCommit = false;
	let cpuOff = true;
	let dualBrain = false; // M12.2: E2B generation routing, opt-in via --dual-brain on

	for (let i = 0; i < args.length; i++) {
		if (args[i] === '--prompt' && args[i + 1]) { prompt = args[++i]; }
		else if (args[i] === '--file' && args[i + 1]) { filePath = args[++i]; }
		else if (args[i] === '--quiet' || args[i] === '--unix') { quiet = true; }
		else if (args[i] === '--commit') { autoCommit = true; }
		else if (args[i] === '--cpu-model' && args[i + 1] === 'off') { cpuOff = true; i++; }
		else if (args[i] === '--cpu-model' && args[i + 1] === 'on') { cpuOff = false; i++; }
		else if (args[i] === '--dual-brain' && args[i + 1] === 'on') { dualBrain = true; i++; }
		else if (args[i] === '--dual-brain' && args[i + 1] === 'off') { dualBrain = false; i++; }
	}

	// One-shot mode: --prompt provided
	if (prompt) {
		const fileContent = filePath ? readFileSync(filePath, 'utf-8') : '';
		const session = await initSession({
			smallModelPath: process.env.SUBVOCAL_SMALL_MODEL ?? activeProfile.smallModelPath,
			largeModelPath: process.env.SUBVOCAL_LARGE_MODEL ?? activeProfile.largeModelPath,
			projectDir: process.cwd(),
			autoCommit,
			quiet,
			cpuOff,
			dualBrain,
		});
		try {
			const result = await runTask(session, {
				prompt,
				filePath: filePath || '/tmp/utter-scratch.txt',
				fileContent,
			});
			printResult(result, quiet);
		} finally {
			await freeSession(session);
		}
		return;
	}

	// REPL mode: interactive agent harness
	console.log('🧠 Subvocal — local agent harness');
	console.log('   Type a prompt. :file <path> to switch files. :quit to exit.\n');

	const session = await initSession({
		smallModelPath: process.env.SUBVOCAL_SMALL_MODEL ?? activeProfile.smallModelPath,
		largeModelPath: process.env.SUBVOCAL_LARGE_MODEL ?? activeProfile.largeModelPath,
		projectDir: process.cwd(),
		autoCommit,
		quiet,
		cpuOff,
		dualBrain,
	});

	const rl = createInterface({ input: process.stdin, output: process.stdout, prompt: '> ' });

	let isFirstPrompt = true;
	let currentFile = '';
	let turnActive = false;

	// Mechanism B: classify intent incrementally while the user types (idle CPU, no GPU
	// decode loop to pace against — a superseded/stale classification just gets discarded).
	// With the small model on, classification runs in a worker thread (own model context on
	// mmap-shared weights, see intentWorker.ts) so the ~99ms forward never blocks keystroke
	// echo; if the worker fails, IncrementalIntentClassifier falls back in-thread.
	const workerClassifier = !cpuOff
		? new WorkerIntentClassifier(process.env.SUBVOCAL_SMALL_MODEL ?? activeProfile.smallModelPath)
		: null;
	const intentClassifier = new IncrementalIntentClassifier({
		cpuOff,
		classifyAsync: workerClassifier ? (text) => workerClassifier.classify(text) : undefined,
	});

	// Mechanisms A + C: file-open is the "dead time" before the user starts typing a prompt.
	// Do the prompt-independent work now (AST tagging + multi-file context, and pre-seed the
	// suffix-tree drafter) so it's already a cache hit / warm trie by submit time.
	const openFile = (path: string): void => {
		if (!path) return;
		try {
			const content = readFileSync(path, 'utf-8');
			session.filePrewarmCache.warm(path, content, { multiFile: true });
			session.loop.prewarmSuffixTree(content);
			if (!quiet) console.log(`   ⚡ prewarmed ${path}`);
		} catch (e) {
			if (!quiet) console.log(`   (prewarm skipped: ${(e as Error).message})`);
		}
	};

	// Feed the current input buffer to the incremental classifier on each keystroke. readline
	// keeps rl.line updated in cooked/terminal mode; on non-TTY input no keypress fires and
	// this stays dormant (B just doesn't kick in, which is fine).
	process.stdin.on('keypress', () => {
		if (turnActive) return;
		const buf = rl.line?.trim();
		if (buf && !buf.startsWith(':')) intentClassifier.feed(buf);
	});

	// Prewarm the file passed via --file before the first prompt is even typed.
	if (filePath) openFile(filePath);

	try {
		rl.prompt();

		for await (const line of rl) {
			const trimmed = line.trim();
			if (!trimmed) { rl.prompt(); continue; }

			// REPL commands
			if (trimmed.startsWith(':')) {
				const parts = trimmed.slice(1).split(/\s+/);
				const cmd = parts[0];
				if (cmd === 'quit' || cmd === 'q' || cmd === 'exit') break;
				if (cmd === 'file' || cmd === 'f') {
					filePath = parts[1] ?? filePath;
					isFirstPrompt = true; // file switch = fresh start
					console.log(`   File: ${filePath}`);
					openFile(filePath); // Mechanisms A + C: warm on file-open, before any prompt
				} else if (cmd === 'diff' || cmd === 'd') {
					// A real diff (2026-07 audit item): pending session changes = VFS vs disk.
					// The old behavior printed the whole file, which is what :file already shows.
					if (filePath) {
						const diskContent = readFileSync(filePath, 'utf-8');
						const vfsContent = session.vfs.read(filePath);
						if (vfsContent === undefined || vfsContent === diskContent) {
							console.log('   (no pending changes)');
						} else {
							console.log(computeDiff(diskContent, vfsContent, filePath));
						}
					}
				} else {
					console.log(`   Unknown command: :${cmd}`);
				}
				rl.prompt();
				continue;
			}

			// Mechanism B: reuse the while-typing classification only if it was computed from
			// exactly this prompt (a pause >= debounce before Enter). Otherwise fall back to
			// classifying at submit time inside preprocess(), as before.
			const precomputedIntent = intentClassifier.getLatestText() === trimmed
				? intentClassifier.getLatest() ?? undefined
				: undefined;
			intentClassifier.cancel();

			// Agent turn
			if (isFirstPrompt || filePath !== currentFile) {
				// Fresh start: full preprocess + forward
				currentFile = filePath;
				const fileContent = filePath ? readFileSync(filePath, 'utf-8') : '';
				turnActive = true;
				try {
					const result = await runTask(session, {
						prompt: trimmed,
						filePath: filePath || '/tmp/utter-scratch.txt',
						fileContent,
					}, { keepAlive: true, precomputedIntent });
					printResult(result, quiet);
				} finally {
					turnActive = false;
				}
				isFirstPrompt = false;
			} else {
				// Follow-up: append to existing KV cache — on the loop whose KV actually holds
				// the conversation (with --dual-brain on, that may be the small brain).
				turnActive = true;
				try {
					const followLoop = session.lastBrain === 'small' && session.smallLoop
						? session.smallLoop
						: session.loop;
					let step = await followLoop.followUp(trimmed);
					if (step.toolCalls.length > 0) {
						// 2026-07 audit item: follow-ups run the SAME tool loop as runTask now,
						// instead of silently dropping tool calls. Seed the VFS from disk when the
						// session hasn't touched this file yet, so the edit branch sees real content.
						const followPath = filePath || '/tmp/utter-scratch.txt';
						const diskNow = filePath ? readFileSync(filePath, 'utf-8') : '';
						if (session.vfs.read(followPath) === undefined) session.vfs.write(followPath, diskNow);
						const followSnapshotter = session.lastBrain === 'small' && session.smallGenModel
							? new StateSnapshotter(session.smallGenModel)
							: session.snapshotter;
						const counters: ToolLoopCounters = { turns: 1, totalTokens: step.tokenCount, rollbacks: 0, errors: 0 };
						step = await executeToolLoop(
							session, followLoop, followSnapshotter,
							{ prompt: trimmed, filePath: followPath, fileContent: session.vfs.read(followPath) ?? diskNow },
							step, counters, { keepAlive: true },
						);
						console.log(`  [follow-up] ${counters.turns} turn(s), ${counters.totalTokens} tokens, ${counters.rollbacks} rollback(s)`);
					}
					if (step.ideogramEdit) {
						await applyIdeogramEdit(session, step.ideogramEdit, filePath || '/tmp/utter-scratch.txt');
					}
					// Show diffs if file was modified
					const currentContent = filePath ? readFileSync(filePath, 'utf-8') : '';
					const vfsContent = session.vfs.read(filePath || '') ?? currentContent;
					if (vfsContent !== currentContent) {
						writeFileSync(filePath || '/tmp/utter-scratch.txt', vfsContent, 'utf-8');
						console.log(`  [follow-up] ${step.tokenCount} tokens`);
					}
				} finally {
					turnActive = false;
				}
			}

			rl.prompt();
		}
	} finally {
		rl.close();
		await workerClassifier?.terminate();
		await freeSession(session);
	}
}

function printResult(result: UtterResult, quiet: boolean): void {
	if (quiet && result.errors === 0) return;
	const stats = formatCompilerStats(result.compilerStats);
	process.stderr.write(`\n  ${stats}\n`);
	if (result.diffs.length > 0) {
		const diffText = formatDiffSummaries(result.diffs);
		if (diffText) console.log(diffText);
	}
	// Always write modified files to disk
	for (const f of result.files) {
		if (f.content) writeFileSync(f.path, f.content, 'utf-8');
	}
}

// Run as CLI when invoked directly (tsx / node)
if (process.argv[1]?.endsWith('utter.ts') || process.argv[1]?.endsWith('utter.js')) {
	cli().catch(e => { console.error(e); process.exit(1); });
}
