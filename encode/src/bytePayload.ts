export const PAYLOAD_START = '[PAYLOAD_START]';
export const PAYLOAD_END = '[PAYLOAD_END]';

const PAYLOAD_REGEX = /\[PAYLOAD_START\]\s*([\s\S]*?)\s*\[PAYLOAD_END\]/g;
// Matches ALL_CAPS or UPPER_SNAKE_CASE identifiers (min 2 chars or contains underscore).
const ACTION_REGEX = /(?:\[)?([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)(?:\])?|(?:\[)?([A-Z]{2,})(?:\])?/;

export interface PayloadBlock {
  /** The action/command token that preceded this payload (e.g. "GIT_COMMIT", "API_CALL"). */
  action: string;
  /** Raw text content between PAYLOAD_START and PAYLOAD_END. */
  payload: string;
  /** Start and end indices in the original text. */
  startIndex: number;
  endIndex: number;
}

export interface ParsedPayloadText {
  /** Text with all payload blocks removed. */
  cleanText: string;
  /** Extracted payload blocks. */
  blocks: PayloadBlock[];
}

/**
 * Parse text containing [PAYLOAD_START]...[PAYLOAD_END] blocks.
 * The regex looks for: <action_token> [PAYLOAD_START] <content> [PAYLOAD_END]
 * where action_token is any uppercase identifier preceding the block.
 */
export function parsePayloadBlocks(text: string): ParsedPayloadText {
  const blocks: PayloadBlock[] = [];
  let cleanText = '';
  let lastEnd = 0;

  let match: RegExpExecArray | null;
  const regex = new RegExp(PAYLOAD_REGEX.source, 'g');

  while ((match = regex.exec(text)) !== null) {
    const payload = match[1];
    const payloadStart = match.index;
    const payloadEnd = regex.lastIndex;

    // Find the action token preceding this block and its start position.
    const actionInfo = findLastActionWithPos(text, payloadStart);

    // Append text to cleanText, excluding the action token.
    if (actionInfo) {
      cleanText += text.slice(lastEnd, actionInfo.startIndex);
    } else {
      cleanText += text.slice(lastEnd, payloadStart);
    }

    blocks.push({
      action: actionInfo?.name ?? '',
      payload,
      startIndex: payloadStart,
      endIndex: payloadEnd,
    });

    lastEnd = payloadEnd;
  }

  cleanText += text.slice(lastEnd);

  return { cleanText: cleanText.trim(), blocks };
}

/**
 * Build a payload-wrapped message for the model.
 * Returns: ACTION_NAME [PAYLOAD_START] payload [PAYLOAD_END]
 */
export function wrapPayload(action: string, payload: string): string {
  return `${action} ${PAYLOAD_START} ${payload} ${PAYLOAD_END}`;
}

function findLastActionWithPos(text: string, upToIndex: number): { name: string; startIndex: number } | null {
  const preceding = text.slice(0, upToIndex);
  const regex = new RegExp(ACTION_REGEX.source, 'g');
  let lastMatch: { name: string; startIndex: number } | null = null;
  let m: RegExpExecArray | null;
  while ((m = regex.exec(preceding)) !== null) {
    lastMatch = { name: m[1] || m[2], startIndex: m.index };
  }
  return lastMatch;
}
