/**
 * toolParse.ts
 *
 * Extract tool calls from generated assistant text. Supports three conventions:
 *
 * 1. Native (preferred): `<|tool_call>call:edit{file:<|"|>...<|"|>,...}<tool_call|>` -- this
 *    checkpoint's OWN special-token tool-calling protocol, confirmed byte-for-byte from the
 *    GGUF's own tokenizer.chat_template (format_tool_call / format_argument macros). See
 *    doc/epics/EPIC-M8-grammar-constrained-tool-calls.md. `<|tool_call>`/`<tool_call|>` and the
 *    `<|"|>` string-quote marker are real single special tokens (verified ids 48/49/52 on the
 *    12B checkpoint) -- there is no multi-token opening/closing sequence to corrupt the way the
 *    Hermes convention below has, which is the whole reason this is now the preferred path.
 * 2. Hermes/Qwen JSON (fallback): `<tool_call>{"name":"edit","arguments":{...}}</tool_call>`
 * 3. Qwen2.5-Coder XML (fallback): `<edit file="..." old="..." new="..."/>` or `<bash command="..."/>`
 *
 * Pure and model-independent for unit testing.
 */

export interface ParsedToolCall {
	id: string;
	name: string;
	arguments: Record<string, unknown>;
}

export interface ParsedAssistantOutput {
	/** Visible text with all tool-call and think blocks stripped. */
	text: string;
	/** Concatenated <think> reasoning content, if the model emitted any. */
	thinking: string;
	toolCalls: ParsedToolCall[];
	/** True when at least one tool-call block was found (even if malformed). */
	hadToolCallSyntax: boolean;
}

/** JSON-Schema-ish parameter type, matching the native protocol's `type:<|"|>...<|"|>` values. */
export interface ToolParameterSchema {
	type: 'STRING' | 'NUMBER' | 'BOOLEAN' | 'OBJECT' | 'ARRAY';
	description?: string;
}

/** A tool's declared shape, rendered into the native protocol by ModelProfile.buildToolDeclarations(). */
export interface ToolDefinition {
	name: string;
	description: string;
	parameters: {
		properties: Record<string, ToolParameterSchema>;
		required: string[];
	};
}

const TOOL_CALL_RE = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g;
const THINK_RE     = /<think>\s*([\s\S]*?)\s*<\/think>/g;

// Qwen2.5-Coder XML attribute format: <edit file="p" old="o" new="n"/>
const XML_TOOL_RE  = /<(edit|bash|write)\s+([^>]*?)\s*\/?>/g;

// Native protocol: <|tool_call>call:NAME{key:value,...}<tool_call|>. Deliberately matches the
// literal special-token surface text (not `<tool_call>`, which is the different Hermes/Qwen
// convention above) -- `<|tool_call>`/`<tool_call|>` always detokenize to exactly this text.
const NATIVE_TOOL_CALL_RE = /<\|tool_call>([\s\S]*?)<tool_call\|>/g;
const NATIVE_QUOTE = '<|"|>';

let counter = 0;
function nextId(): string {
	counter += 1;
	return `call_${Date.now().toString(36)}_${counter}`;
}

// ── JSON payload parser (Hermes/Qwen convention) ──────────────────────────

function stripFences(body: string): string {
	let s = body.trim();
	if (s.startsWith("```")) {
		s = s.replace(/^```[a-zA-Z]*\n?/, "").replace(/```\s*$/, "").trim();
	}
	return s;
}

function parseOne(payload: string): ParsedToolCall | null {
	let body = stripFences(payload);
	let obj: unknown;
	try {
		obj = JSON.parse(body);
	} catch {
		return null;
	}
	if (typeof obj !== "object" || obj === null) return null;
	const record = obj as Record<string, unknown>;
	const name = record.name;
	if (typeof name !== "string" || name.length === 0) return null;

	let args: Record<string, unknown> = {};
	const rawArgs = record.arguments;
	if (typeof rawArgs === "string") {
		try {
			const parsed = JSON.parse(rawArgs);
			if (parsed && typeof parsed === "object") args = parsed as Record<string, unknown>;
		} catch { /* leave args empty */ }
	} else if (rawArgs && typeof rawArgs === "object") {
		args = rawArgs as Record<string, unknown>;
	}

	return { id: nextId(), name, arguments: args };
}

// ── Native protocol parser (this checkpoint's own special-token convention) ────

/**
 * Parse `key:<|"|>value<|"|>,key2:<|"|>value2<|"|>` (a native tool-call's argument body).
 * Splits on the `<|"|>` quote marker FIRST, before any structural parsing -- this is why it's
 * safe against arbitrary code content in string values (braces, commas, colons in an `old`/`new`
 * code snippet can never be misread as protocol structure, unlike naive brace-counting).
 * Odd-indexed segments after the split are the quoted string values; the key for each is the
 * `\w+:` immediately preceding it in the prior (even-indexed) structural segment.
 */
function parseNativeArgs(body: string): Record<string, unknown> {
	const args: Record<string, unknown> = {};
	const parts = body.split(NATIVE_QUOTE);
	for (let i = 1; i < parts.length; i += 2) {
		const keyMatch = /(\w+):\s*$/.exec(parts[i - 1]);
		if (keyMatch) args[keyMatch[1]] = parts[i];
	}
	// Non-string values (number/boolean): scan the STRUCTURAL (even-index) segments — string
	// contents live in the odd segments, so numbers inside quoted code can never be misread as
	// arguments. Needed for mixed-type tools (M16.1 edit_lines carries quoted path/newText AND
	// numeric startLine/endLine; the old parts.length===1 gate silently dropped the numbers).
	for (let i = 0; i < parts.length; i += 2) {
		const re = /(\w+):\s*(true|false|-?\d+(?:\.\d+)?)\s*(?=[,}]|$)/g;
		let m: RegExpExecArray | null;
		while ((m = re.exec(parts[i])) !== null) {
			if (!(m[1] in args)) {
				args[m[1]] = m[2] === 'true' ? true : m[2] === 'false' ? false : Number(m[2]);
			}
		}
	}
	return args;
}

function parseNativeToolCall(body: string): ParsedToolCall | null {
	const nameMatch = /^call:([A-Za-z_]\w*)\{/.exec(body);
	if (!nameMatch) return null;
	const closeIdx = body.lastIndexOf('}');
	if (closeIdx < nameMatch[0].length) return null;
	const argsBody = body.slice(nameMatch[0].length, closeIdx);
	return { id: nextId(), name: nameMatch[1], arguments: parseNativeArgs(argsBody) };
}

// ── XML attribute parser (Qwen2.5-Coder convention) ───────────────────────

function parseXmlAttrs(attrStr: string): Record<string, string> {
	const attrs: Record<string, string> = {};
	const re = /(\w+)\s*=\s*"([^"]*)"|(\w+)\s*=\s*'([^']*)'/g;
	let m: RegExpExecArray | null;
	while ((m = re.exec(attrStr)) !== null) {
		const key = m[1] ?? m[3] ?? '';
		const val = m[2] ?? m[4] ?? '';
		if (key) attrs[key] = val;
	}
	return attrs;
}

function parseXmlTool(tagName: string, attrStr: string): ParsedToolCall | null {
	const attrs = parseXmlAttrs(attrStr);
	if (Object.keys(attrs).length === 0) return null;
	return { id: nextId(), name: tagName, arguments: attrs as Record<string, unknown> };
}

// ── Main export ───────────────────────────────────────────────────────────

export function parseAssistantOutput(raw: string): ParsedAssistantOutput {
	const toolCalls: ParsedToolCall[] = [];
	let hadToolCallSyntax = false;

	// Native protocol (preferred): <|tool_call>call:NAME{...}<tool_call|> — parsed on the RAW
	// text, BEFORE any markdown-fence stripping: a native string value may legitimately
	// contain ``` fences (e.g. an edit writing a README), and the fence-strip below used to
	// corrupt them (2026-07 audit finding).
	NATIVE_TOOL_CALL_RE.lastIndex = 0;
	let match: RegExpExecArray | null;
	while ((match = NATIVE_TOOL_CALL_RE.exec(raw)) !== null) {
		hadToolCallSyntax = true;
		const parsed = parseNativeToolCall(match[1] ?? "");
		if (parsed) toolCalls.push(parsed);
	}

	// Strip native blocks first, THEN markdown fences — the fence-strip only exists for the
	// fallback conventions below (Qwen often wraps tool calls in ```xml).
	const nativeStripped = raw.replace(NATIVE_TOOL_CALL_RE, "");
	const clean = nativeStripped.replace(/```(?:xml|json|tool)?\s*/gi, '').replace(/```\s*/g, '');

	// Hermes/Qwen JSON format (fallback): <tool_call>{...}</tool_call>
	TOOL_CALL_RE.lastIndex = 0;
	while ((match = TOOL_CALL_RE.exec(clean)) !== null) {
		hadToolCallSyntax = true;
		const parsed = parseOne(match[1] ?? "");
		if (parsed) toolCalls.push(parsed);
	}

	// Qwen2.5-Coder XML attribute format: <edit file="p" old="o" new="n"/>
	XML_TOOL_RE.lastIndex = 0;
	while ((match = XML_TOOL_RE.exec(clean)) !== null) {
		hadToolCallSyntax = true;
		const parsed = parseXmlTool(match[1] ?? '', match[2] ?? '');
		if (parsed) toolCalls.push(parsed);
	}

	// <think> reasoning
	const thinkParts: string[] = [];
	THINK_RE.lastIndex = 0;
	let thinkMatch: RegExpExecArray | null;
	while ((thinkMatch = THINK_RE.exec(clean)) !== null) {
		const body = (thinkMatch[1] ?? "").trim();
		if (body.length > 0) thinkParts.push(body);
	}

	let text = clean.replace(TOOL_CALL_RE, "").replace(XML_TOOL_RE, "").replace(THINK_RE, ""); // native blocks already stripped above

	// Unterminated <think> handling
	const danglingThink = text.indexOf("<think>");
	if (danglingThink !== -1) {
		const tail = text.slice(danglingThink + "<think>".length).trim();
		if (tail.length > 0) thinkParts.push(tail);
		text = text.slice(0, danglingThink);
	}
	return { text: text.trim(), thinking: thinkParts.join("\n\n"), toolCalls, hadToolCallSyntax };
}
