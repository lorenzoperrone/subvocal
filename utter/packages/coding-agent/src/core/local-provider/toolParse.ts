/**
 * toolParse.ts
 *
 * Extract Hermes/Qwen `<tool_call>{json}</tool_call>` blocks from generated
 * assistant text. Returns the visible text (tool-call blocks removed) plus the
 * parsed tool calls. Pure and model-independent for unit testing.
 */

export interface ParsedToolCall {
	id: string;
	name: string;
	arguments: Record<string, unknown>;
}

export interface ParsedAssistantOutput {
	/** Visible text with all <tool_call> and <think> blocks stripped. */
	text: string;
	/** Concatenated <think> reasoning content, if the model emitted any. */
	thinking: string;
	toolCalls: ParsedToolCall[];
	/** True when at least one <tool_call> block was found (even if malformed). */
	hadToolCallSyntax: boolean;
}

const TOOL_CALL_RE = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g;
const THINK_RE = /<think>\s*([\s\S]*?)\s*<\/think>/g;

let counter = 0;
function nextId(): string {
	counter += 1;
	return `call_${Date.now().toString(36)}_${counter}`;
}

/**
 * Parse one tool-call payload. Tolerates the model wrapping the JSON in code
 * fences or emitting `arguments` as a JSON string instead of an object.
 */
function parseOne(payload: string): ParsedToolCall | null {
	let body = payload.trim();
	if (body.startsWith("```")) {
		body = body.replace(/^```[a-zA-Z]*\n?/, "").replace(/```$/, "").trim();
	}
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
		} catch {
			// leave args empty on malformed nested JSON
		}
	} else if (rawArgs && typeof rawArgs === "object") {
		args = rawArgs as Record<string, unknown>;
	}

	return { id: nextId(), name, arguments: args };
}

export function parseAssistantOutput(raw: string): ParsedAssistantOutput {
	const toolCalls: ParsedToolCall[] = [];
	let hadToolCallSyntax = false;

	TOOL_CALL_RE.lastIndex = 0;
	let match: RegExpExecArray | null;
	// biome-ignore lint/suspicious/noAssignInExpressions: standard regex exec loop
	while ((match = TOOL_CALL_RE.exec(raw)) !== null) {
		hadToolCallSyntax = true;
		const parsed = parseOne(match[1] ?? "");
		if (parsed) toolCalls.push(parsed);
	}

	const thinkParts: string[] = [];
	THINK_RE.lastIndex = 0;
	let thinkMatch: RegExpExecArray | null;
	// biome-ignore lint/suspicious/noAssignInExpressions: standard regex exec loop
	while ((thinkMatch = THINK_RE.exec(raw)) !== null) {
		const body = (thinkMatch[1] ?? "").trim();
		if (body.length > 0) thinkParts.push(body);
	}

	let text = raw.replace(TOOL_CALL_RE, "").replace(THINK_RE, "");
	// Handle an unterminated <think> (model hit the token limit mid-reasoning):
	// everything after the dangling tag is thinking, not visible text.
	const danglingThink = text.indexOf("<think>");
	if (danglingThink !== -1) {
		const tail = text.slice(danglingThink + "<think>".length).trim();
		if (tail.length > 0) thinkParts.push(tail);
		text = text.slice(0, danglingThink);
	}
	return { text: text.trim(), thinking: thinkParts.join("\n\n"), toolCalls, hadToolCallSyntax };
}
