/**
 * gutter.ts — M16.1: line-number gutter helpers for anchor-based (line-range) edits.
 *
 * File content shown to the model carries a `  42| ` gutter so `edit_lines` can address exact
 * 1-based ranges without fragile exact-text matching (probe-verified: the 12B picks the right
 * range and keeps the gutter out of its replacement text). The strip helper is the defensive
 * counterpart used by the write/edit/edit_lines executors in case a model ever echoes the
 * gutter back into content it writes.
 */

const GUTTER_RE = /^\s*\d+\| ?/;

/** Prefix each line with a right-aligned 1-based line number: "  42| code". */
export function addLineGutter(text: string, startLine = 1): string {
	const lines = text.split("\n");
	const width = String(startLine + lines.length - 1).length;
	return lines.map((l, i) => `${String(startLine + i).padStart(Math.max(4, width))}| ${l}`).join("\n");
}

/**
 * If EVERY non-empty line carries a line-number gutter, strip it; otherwise return the text
 * unchanged. The all-or-nothing rule keeps this safe for code that legitimately contains
 * `123| something` on some line (e.g. a table in a string) — a partial match is never stripped.
 */
export function stripLineGutter(text: string): string {
	const lines = text.split("\n");
	const nonEmpty = lines.filter((l) => l.trim().length > 0);
	if (nonEmpty.length === 0 || !nonEmpty.every((l) => GUTTER_RE.test(l))) return text;
	return lines.map((l) => l.replace(GUTTER_RE, "")).join("\n");
}
