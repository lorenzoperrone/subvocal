/**
 * edit-lines.ts — M16.1: anchor-based editing by 1-based line range.
 *
 * Companion to the exact-text `edit` tool: instead of matching fragile oldText (which fails when
 * the model reconstructs indentation imperfectly), the model addresses the range it saw in the
 * line-number gutter that `read` (and the active-file view) now render. Probe-verified on the
 * 12B: it picks exact ranges and keeps the gutter out of the replacement text; the executor
 * still strips a leaked gutter defensively (all-or-nothing rule, see gutter.ts).
 *
 * Line numbers SHIFT after an edit — the result tells the model to re-read before another
 * line-based edit on the same file, and the executor rejects out-of-range requests loudly.
 */
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { readFile as fsReadFile, writeFile as fsWriteFile } from "fs/promises";
import { Container, Text } from "@earendil-works/pi-tui";
import { type Static, Type } from "typebox";
import type { Theme } from "../../modes/interactive/theme/theme.ts";
import type { ToolDefinition } from "../extensions/types.ts";
import { detectLineEnding, generateDiffString, generateUnifiedPatch, normalizeToLF, restoreLineEndings, stripBom } from "./edit-diff.ts";
import { withFileMutationQueue } from "./file-mutation-queue.ts";
import { stripLineGutter } from "./gutter.ts";
import { resolveToCwd } from "./path-utils.ts";
import { renderToolPath, str } from "./render-utils.ts";
import { wrapToolDefinition } from "./tool-definition-wrapper.ts";

const editLinesSchema = Type.Object({
	path: Type.String({ description: "Path to the file to edit (relative or absolute)" }),
	startLine: Type.Number({ description: "First line to replace (1-based, inclusive, as shown in the read gutter)" }),
	endLine: Type.Number({ description: "Last line to replace (1-based, inclusive)" }),
	newText: Type.String({ description: "Replacement for that line range. Do NOT include the line-number gutter." }),
});

export type EditLinesToolInput = Static<typeof editLinesSchema>;

export interface EditLinesToolDetails {
	diff: string;
	firstChangedLine?: number;
	patch?: string;
}

export interface EditLinesOperations {
	readFile: (absolutePath: string) => Promise<Buffer>;
	writeFile: (absolutePath: string, content: string) => Promise<void>;
}

const defaultOps: EditLinesOperations = {
	readFile: (p) => fsReadFile(p),
	writeFile: (p, c) => fsWriteFile(p, c, "utf-8"),
};

export interface EditLinesToolOptions {
	operations?: EditLinesOperations;
}

export function createEditLinesToolDefinition(
	cwd: string,
	options?: EditLinesToolOptions,
): ToolDefinition<typeof editLinesSchema, EditLinesToolDetails | undefined> {
	const ops = options?.operations ?? defaultOps;
	return {
		name: "edit_lines",
		label: "edit_lines",
		description:
			"Replace a range of lines in a file with new text. Line numbers are 1-based and refer to the " +
			"numbered gutter shown by read. Use this instead of edit when you know the exact line range; " +
			"after the call, line numbers below the edit shift — re-read the file before another line edit.",
		promptSnippet: "Replace a line range (numbers from the read gutter)",
		promptGuidelines: [
			"Use edit_lines with the gutter line numbers for block replacements; never include the gutter in newText.",
		],
		parameters: editLinesSchema,
		async execute(_toolCallId, { path, startLine, endLine, newText }: EditLinesToolInput, signal?: AbortSignal) {
			const absolutePath = resolveToCwd(path, cwd);
			return withFileMutationQueue(absolutePath, async () => {
				const throwIfAborted = (): void => {
					if (signal?.aborted) throw new Error("Operation aborted");
				};
				throwIfAborted();

				const raw = (await ops.readFile(absolutePath)).toString("utf-8");
				throwIfAborted();
				const { bom, text } = stripBom(raw);
				const ending = detectLineEnding(text);
				const normalized = normalizeToLF(text);
				const lines = normalized.split("\n");

				const s = Math.trunc(startLine);
				const e = Math.trunc(endLine);
				if (!(s >= 1 && e >= s && e <= lines.length)) {
					throw new Error(
						`Invalid line range ${startLine}-${endLine} for ${path} (${lines.length} lines). ` +
							`Line numbers may have shifted since your last read — re-read the file and retry.`,
					);
				}

				// Defensive: strip a leaked gutter (all-or-nothing rule) and normalize endings.
				const replacement = normalizeToLF(stripLineGutter(newText));
				const newLines = replacement.length > 0 ? replacement.split("\n") : [];
				const before = lines.slice(0, s - 1);
				const after = lines.slice(e);
				const newContent = [...before, ...newLines, ...after].join("\n");
				if (newContent === normalized) {
					throw new Error(`No changes made to ${path}: the replacement is identical to lines ${s}-${e}.`);
				}
				throwIfAborted();

				await ops.writeFile(absolutePath, bom + restoreLineEndings(newContent, ending));
				throwIfAborted();

				const removed = e - s + 1;
				const added = newLines.length;
				const shift = added - removed;
				const diffResult = generateDiffString(normalized, newContent);
				const patch = generateUnifiedPatch(path, normalized, newContent);
				return {
					content: [
						{
							type: "text",
							text:
								`Replaced lines ${s}-${e} in ${path} (-${removed}/+${added}).` +
								(shift !== 0
									? ` Line numbers after ${s - 1 + added} shifted by ${shift > 0 ? "+" : ""}${shift} — re-read before another line edit.`
									: ""),
						},
					],
					details: { diff: diffResult.diff, patch, firstChangedLine: diffResult.firstChangedLine },
				};
			});
		},
		renderCall(args, theme, context) {
			const a = args as Partial<EditLinesToolInput> | undefined;
			const pathDisplay = renderToolPath(str(a?.path), theme, context.cwd);
			const range =
				a?.startLine !== undefined && a?.endLine !== undefined ? `  ${a.startLine}-${a.endLine}` : "";
			const component = (context.lastComponent as Text | undefined) ?? new Text("", 0, 0);
			component.setText(`${theme.fg("toolTitle", theme.bold("Edit-lines"))} ${pathDisplay}${theme.fg("muted", range)}`);
			return component;
		},
		renderResult(result, options, theme, context) {
			const details = (result as { details?: EditLinesToolDetails }).details;
			if (context.isError) {
				const msg = result.content
					.filter((c) => c.type === "text")
					.map((c) => (c as { text?: string }).text || "")
					.join("\n");
				const t = (context.lastComponent as Text | undefined) ?? new Text("", 0, 0);
				t.setText(`\n${theme.fg("error", msg)}`);
				return t;
			}
			if (!details?.diff) {
				const c = (context.lastComponent as Container | undefined) ?? new Container();
				c.clear();
				return c;
			}
			const lines = details.diff.split("\n");
			const max = options.expanded ? lines.length : 14;
			const shown = lines.slice(0, max).join("\n");
			const t = (context.lastComponent as Text | undefined) ?? new Text("", 0, 0);
			t.setText(
				`\n${shown}${lines.length > max ? theme.fg("muted", `\n… (${lines.length - max} more diff lines)`) : ""}`,
			);
			return t;
		},
	};
}

export function createEditLinesTool(cwd: string, options?: EditLinesToolOptions): AgentTool<typeof editLinesSchema> {
	return wrapToolDefinition(createEditLinesToolDefinition(cwd, options));
}
