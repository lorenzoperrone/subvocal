/**
 * wire.ts — postMessage-safe types and pure helpers shared between the
 * subvocal-local provider (main thread) and the agent worker.
 *
 * Deliberately NO runtime imports of @subvocal/encode or @subvocal/synapse:
 * the main thread must stay free of native addons (model FFI, tree-sitter) —
 * those load only inside the worker. Type-only imports are erased.
 */

import type { Message, Tool } from "@earendil-works/pi-ai";
import type { ToolDefinition, ToolParameterSchema } from "@subvocal/encode";

// ── Wire types ────────────────────────────────────────────────────────────────

export interface TurnRequest {
	systemPrompt?: string;
	messages: Message[];
	/** Already converted on the main thread (TypeBox schemas don't survive cloning intact). */
	toolDefs: ToolDefinition[];
	options?: { maxTokens?: number; temperature?: number };
	cwd: string;
}

export interface SerializableStep {
	text: string;
	thinking: string;
	toolCalls: Array<{ id: string; name: string; arguments: Record<string, unknown> }>;
	tokenCount: number;
	stoppedNaturally: boolean;
	/** Wall-clock ms spent decoding this step (tool-exec excluded) — for the tok/s footer. */
	genMs: number;
}

export interface TurnResult {
	step: SerializableStep;
	inputTokens: number;
}

// ── pi Tool → native ToolDefinition conversion ────────────────────────────────

/** Map a JSON-schema/TypeBox `type` to the native protocol's type tags. */
function nativeType(t: unknown): ToolParameterSchema["type"] {
	switch (t) {
		case "number":
		case "integer":
			return "NUMBER";
		case "boolean":
			return "BOOLEAN";
		case "array":
			return "ARRAY";
		case "object":
			return "OBJECT";
		default:
			return "STRING";
	}
}

/**
 * pi's `edit` tool natively takes `{path, edits: [{oldText, newText}, ...]}` — a nested array
 * schema the M8 declaration renderer can't fully express (it renders `type:ARRAY` without item
 * shapes). The executor also accepts the LEGACY flat form `{path, oldText, newText}`
 * (single targeted replacement), which IS fully expressible — declare that instead.
 */
const FLAT_EDIT_TOOL: ToolDefinition = {
	name: "edit",
	description: "Replace one exact block of existing text with new text in a file.",
	parameters: {
		properties: {
			path: { type: "STRING", description: "Path to the file to edit (relative or absolute)." },
			oldText: { type: "STRING", description: "Exact existing text to find and replace. Must be unique in the file." },
			newText: { type: "STRING", description: "The replacement text." },
		},
		required: ["path", "oldText", "newText"],
	},
};

export function toToolDefinitions(tools: ReadonlyArray<Tool>): ToolDefinition[] {
	return tools.map((t) => {
		if (t.name === "edit") return FLAT_EDIT_TOOL;
		const schema = t.parameters as unknown as {
			properties?: Record<string, { type?: unknown; description?: unknown }>;
			required?: string[];
		};
		const properties: Record<string, ToolParameterSchema> = {};
		for (const [key, p] of Object.entries(schema.properties ?? {})) {
			properties[key] = {
				type: nativeType(p?.type),
				...(typeof p?.description === "string" ? { description: p.description } : {}),
			};
		}
		return {
			name: t.name,
			description: t.description,
			parameters: { properties, required: schema.required ?? [] },
		};
	});
}
