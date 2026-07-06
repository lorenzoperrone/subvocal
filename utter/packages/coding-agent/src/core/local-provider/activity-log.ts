/**
 * activity-log.ts — optional live activity feed for a second, read-only "monitor" window.
 *
 * When SUBVOCAL_MONITOR_LOG points at a file (set by bin/subvocal when SUBVOCAL_MONITOR is on),
 * the provider appends a human-readable line per model action here. bin/subvocal opens a second
 * Terminal window that `tail -f`s this file, so you can watch WHAT the agent writes and WHERE in
 * real time, separately from the conversation. Two modes (SUBVOCAL_MONITOR):
 *   - "1" / "activity" (default): structured tool activity (WRITE path + preview, EDIT, BASH, …).
 *   - "raw": the literal decode stream (protocol tokens included) as it is generated.
 * All writes are best-effort and never throw into the turn path.
 */
import { appendFileSync } from "node:fs";

const LOG = process.env.SUBVOCAL_MONITOR_LOG;
const MODE = process.env.SUBVOCAL_MONITOR ?? "";
const RAW = MODE === "raw";

function stamp(): string {
	return new Date().toISOString().slice(11, 19); // HH:MM:SS
}

function append(text: string): void {
	if (!LOG) return;
	try {
		appendFileSync(LOG, text);
	} catch {
		/* monitor is best-effort — never disturb the turn */
	}
}

const indent = (s: string, prefix = "    ") =>
	s.split("\n").map((l) => prefix + l).join("\n");

/** Structured activity mode: one entry per tool call, with a content preview for write/edit. */
export function logToolActivity(name: string, args: Record<string, unknown>): void {
	if (!LOG || RAW) return;
	const path = typeof args.path === "string" ? args.path : "";
	const s = (v: unknown) => (typeof v === "string" ? v : "");
	let entry: string;
	switch (name) {
		case "write": {
			const content = s(args.content);
			const lines = content.split("\n");
			const preview = lines.slice(0, 14).join("\n");
			entry =
				`[${stamp()}] WRITE  ${path}  (${content.length} chars, ${lines.length} lines)\n` +
				indent(preview) +
				(lines.length > 14 ? "\n    … (truncated)" : "") +
				"\n";
			break;
		}
		case "edit": {
			entry =
				`[${stamp()}] EDIT   ${path}\n` +
				indent(s(args.oldText), "  - ") +
				"\n" +
				indent(s(args.newText), "  + ") +
				"\n";
			break;
		}
		case "bash":
			entry = `[${stamp()}] BASH   $ ${s(args.command)}\n`;
			break;
		case "read":
			entry = `[${stamp()}] READ   ${path}\n`;
			break;
		case "ls":
			entry = `[${stamp()}] LS     ${path || "."}\n`;
			break;
		case "grep":
			entry = `[${stamp()}] GREP   ${s(args.pattern)}\n`;
			break;
		case "find":
			entry = `[${stamp()}] FIND   ${s(args.pattern) || s(args.name)}\n`;
			break;
		default:
			entry = `[${stamp()}] ${name.toUpperCase()}  ${JSON.stringify(args).slice(0, 160)}\n`;
	}
	append(entry + "\n");
}

/** Raw mode: mark a turn boundary. */
export function logRawTurnStart(userText: string): void {
	if (!LOG || !RAW) return;
	append(`\n──────── [${stamp()}] user: ${userText.slice(0, 120).replace(/\n/g, " ")}\n`);
}

/** Raw mode: append the literal decode text as it streams. */
export function logRawDelta(text: string): void {
	if (!LOG || !RAW) return;
	append(text);
}

/** M13.4: dual-brain routing decision for a fresh conversation/task (visible in both modes). */
export function logDualBrainRoute(brain: string, reason: string): void {
	if (!LOG) return;
	append(`[${stamp()}] DUAL-BRAIN route → ${brain}  (${reason})\n`);
}

/** M13.4: the small brain's first step was unusable — the turn was redone on the large brain. */
export function logDualBrainEscalate(reason: string): void {
	if (!LOG) return;
	append(`[${stamp()}] DUAL-BRAIN escalate → large  (${reason})\n`);
}

/** M11.3: which intent source classified this turn — model-shared-seq (isolated aux
 *  sequence on the shared E2B) or regex (the internal preprocess() fallback). */
export function logIntentSource(source: "model-shared-seq" | "regex", intent: string): void {
	if (!LOG) return;
	append(`[${stamp()}] INTENT [${source}] → ${intent}\n`);
}
