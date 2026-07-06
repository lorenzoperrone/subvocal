/**
 * stream-sanitize.ts — turn the model's RAW decode stream into human-readable prose for live
 * display. The raw stream contains protocol tokens the user should never see scrolling by:
 *   - native tool calls:  <|tool_call>call:edit{...}<tool_call|>
 *   - thought channel:    <|channel>thought\n<channel|>
 *   - ideogram edits:     ⊂ <anchor> <code> ⊃   (M15.1)
 *   - bare AST tags / CRC anchors / path tokens echoed into text
 * The final `done` message already carries the clean PARSED content; this only affects what the
 * user sees WHILE generating. Conservative by construction: it suppresses clearly-delimited
 * protocol regions and strips registry characters, never ordinary prose.
 */

import { stripRegistryTags } from "@subvocal/encode";

const TOOL_OPEN = "<|tool_call>";
const TOOL_CLOSE = "<tool_call|>";
const CHANNEL_OPEN = "<|channel>";
const CHANNEL_CLOSE = "<channel|>";
const IDEO_OPEN = "⊂";
const IDEO_CLOSE = "⊃";

// M15.1 ideogram path is OFF by default (SUBVOCAL_IDEOGRAM=1 re-enables). When off, the model
// emits no ideogram/registry markers, so we must NOT strip ⊂…⊃ / bare Greek-math Unicode from the
// display — real prose or code legitimately using those characters would be silently mangled. The
// native tool-call and thought-channel stripping stays unconditional (that protocol is always on).
const IDEOGRAM_ENABLED = process.env.SUBVOCAL_IDEOGRAM === "1";

/**
 * Remove protocol regions from a COMPLETE accumulated string. Idempotent and total — callers
 * pass the whole raw text so far and get the whole clean text so far, then diff for the delta.
 */
export function sanitizeStreamText(raw: string): string {
	let out = raw;
	// Drop delimited regions (inclusive). Non-greedy; tolerate an unclosed trailing region
	// (still generating) by dropping from the opener to end-of-string.
	out = out.replace(new RegExp(`${escape(TOOL_OPEN)}[\\s\\S]*?${escape(TOOL_CLOSE)}`, "g"), "");
	out = out.replace(new RegExp(`${escape(TOOL_OPEN)}[\\s\\S]*$`), "");
	out = out.replace(new RegExp(`${escape(CHANNEL_OPEN)}[\\s\\S]*?${escape(CHANNEL_CLOSE)}`, "g"), "");
	if (IDEOGRAM_ENABLED) {
		out = out.replace(new RegExp(`${escape(IDEO_OPEN)}[\\s\\S]*?${escape(IDEO_CLOSE)}`, "g"), "");
		out = out.replace(new RegExp(`${IDEO_OPEN}[\\s\\S]*$`), ""); // unclosed ideogram still streaming
		// Strip any bare registry ideograms (anchors/paths/tags) the model echoed into prose.
		out = stripRegistryTags(out, true);
	}
	return out;
}

function escape(s: string): string {
	return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Incremental sanitizer for a live stream. Feed each raw piece; get back the clean text to
 * APPEND to the display (may be empty while inside a suppressed region). When a region closes
 * and the clean text would shrink, returns { reset: cleanSoFar } so the caller can replace the
 * block instead of appending.
 */
export class StreamSanitizer {
	private raw = "";
	private emitted = "";

	feed(piece: string): { append: string } | { reset: string } {
		this.raw += piece;
		const clean = sanitizeStreamText(this.raw);
		if (clean.startsWith(this.emitted)) {
			const append = clean.slice(this.emitted.length);
			this.emitted = clean;
			return { append };
		}
		// Clean text diverged (a partial marker completed and got removed) — reset the block.
		this.emitted = clean;
		return { reset: clean };
	}

	get clean(): string {
		return this.emitted;
	}
}
