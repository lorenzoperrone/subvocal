/**
 * chatml.ts
 *
 * Render an @subvocal/ai `Context` (system prompt + messages + tool defs) into a
 * single Qwen3 ChatML prompt string for the local FFI brain. Tool calling uses
 * the Hermes/Qwen `<tool_call>{json}</tool_call>` convention, which the decode
 * loop parses back out of the generated text.
 *
 * This is a pure function with no model dependency so it can be unit-tested in
 * isolation.
 */

import type { Context, Message, Tool } from "@earendil-works/pi-ai";

const IM_START = "<|im_start|>";
const IM_END = "<|im_end|>";

/** Flatten a message content array (text/image/toolCall blocks) to plain text. */
function renderContentBlocks(content: string | ReadonlyArray<{ type: string }>): string {
	if (typeof content === "string") return content;
	const parts: string[] = [];
	for (const raw of content) {
		const block = raw as { type: string; [k: string]: unknown };
		if (block.type === "text" && typeof block.text === "string") {
			parts.push(block.text);
		} else if (block.type === "thinking" && typeof block.thinking === "string") {
			// Thinking is internal; don't replay it back into the prompt.
			continue;
		} else if (block.type === "toolCall") {
			const name = String(block.name ?? "");
			const args = JSON.stringify(block.arguments ?? {});
			parts.push(`<tool_call>\n{"name": ${JSON.stringify(name)}, "arguments": ${args}}\n</tool_call>`);
		} else if (block.type === "image") {
			// The local text model can't see images; note the omission.
			parts.push("[image omitted]");
		}
	}
	return parts.join("\n");
}

/** Build the `# Tools` section appended to the system prompt. */
export function renderToolsSection(tools: ReadonlyArray<Tool>): string {
	if (tools.length === 0) return "";
	const specs = tools.map((t) =>
		JSON.stringify({
			name: t.name,
			description: t.description,
			// `parameters` is a TypeBox/JSON-schema object; serialize as-is.
			parameters: t.parameters,
		}),
	);
	return [
		"",
		"# Tools",
		"",
		"You may call one or more functions to assist with the user query.",
		"You are provided with function signatures within <tools></tools>:",
		"<tools>",
		...specs,
		"</tools>",
		"",
		"For each function call, return a json object with function name and arguments within <tool_call></tool_call> tags:",
		"<tool_call>",
		'{"name": <function-name>, "arguments": <args-json-object>}',
		"</tool_call>",
	].join("\n");
}

export interface ChatMLOptions {
	/** Append the assistant generation primer (default true). */
	addGenerationPrompt?: boolean;
}

/** Render a full Context into a ChatML prompt string. */
export function formatChatML(context: Context, options: ChatMLOptions = {}): string {
	const addGenerationPrompt = options.addGenerationPrompt ?? true;
	const tools = context.tools ?? [];
	const segments: string[] = [];

	const systemBody = (context.systemPrompt ?? "").trimEnd();
	const toolsSection = renderToolsSection(tools);
	if (systemBody || toolsSection) {
		segments.push(`${IM_START}system\n${systemBody}${toolsSection}${IM_END}`);
	}

	for (const message of context.messages as Message[]) {
		if (message.role === "user") {
			segments.push(`${IM_START}user\n${renderContentBlocks(message.content)}${IM_END}`);
		} else if (message.role === "assistant") {
			segments.push(`${IM_START}assistant\n${renderContentBlocks(message.content)}${IM_END}`);
		} else if (message.role === "toolResult") {
			// Qwen expects tool output under the `tool` role, wrapped in
			// <tool_response> so multiple results in a turn stay delimited.
			const body = renderContentBlocks(message.content);
			segments.push(`${IM_START}tool\n<tool_response>\n${body}\n</tool_response>${IM_END}`);
		}
	}

	let prompt = segments.join("\n");
	if (addGenerationPrompt) {
		prompt += `\n${IM_START}assistant\n`;
	}
	return prompt;
}

export const CHATML_STOP_STRINGS = [IM_END, "<|endoftext|>"];
