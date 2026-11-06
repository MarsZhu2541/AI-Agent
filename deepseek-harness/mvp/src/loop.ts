import type { Completer, Message, Tool, ToolCall } from "./types.js";
import { toolByName, toolSchemas } from "./tools.js";

const DEFAULT_MAX_ROUNDS = 8;

export class ToolRoundLimitError extends Error {
  constructor(maxRounds: number) {
    super(`工具轮次超过 ${maxRounds}`);
    this.name = "ToolRoundLimitError";
  }
}

export async function runLoop(input: {
  messages: Message[];
  tools: Tool[];
  workspace: string;
  complete: Completer;
  maxRounds?: number;
  onMessage?: (message: Message) => void;
}): Promise<string | null> {
  const maxRounds = input.maxRounds ?? DEFAULT_MAX_ROUNDS;
  const schemas = toolSchemas(input.tools);

  for (let round = 0; round < maxRounds; round += 1) {
    const assistant = await input.complete(input.messages, schemas);
    pushMessage(input, assistant);
    const calls = assistant.tool_calls ?? [];
    if (calls.length === 0) {
      return assistant.content;
    }
    for (const call of calls) {
      const content = await executeCall(input.tools, input.workspace, call);
      pushMessage(input, {
        role: "tool",
        tool_call_id: call.id,
        content,
      });
    }
  }

  throw new ToolRoundLimitError(maxRounds);
}

async function executeCall(tools: Tool[], workspace: string, call: ToolCall): Promise<string> {
  const tool = toolByName(tools, call.function.name);
  if (!tool) {
    return `未知工具: ${call.function.name}`;
  }
  let args: unknown = {};
  const raw = call.function.arguments?.trim() ?? "";
  if (raw) {
    try {
      args = JSON.parse(raw) as unknown;
    } catch {
      return "工具参数不是合法 JSON";
    }
  }
  try {
    const content = await tool.execute(args, { workspace });
    return content.trim() ? content : "(空)";
  } catch (error) {
    return error instanceof Error ? error.message : String(error);
  }
}

function pushMessage(input: { messages: Message[]; onMessage?: (message: Message) => void }, message: Message) {
  input.messages.push(message);
  input.onMessage?.(message);
}
