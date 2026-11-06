import type { Completer, Message, ToolCall } from "./types.js";

const DEFAULT_BASE_URL = "https://api.deepseek.com/v1";
const DEFAULT_MODEL = "deepseek-v4.1-flash";


export function createComplete1(url: string, apiKey: string, model: string): Completer {

  return async (messages, tools) => {
    const response = await fetch(`${url}/chat/completions`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ model, messages, tools }),
      signal: AbortSignal.timeout(60_000),
    });
    const text = await response.text();
    if (!response.ok) {
      throw new Error(`模型请求失败 ${response.status}: ${text.slice(0, 500)}`);
    }
    return assistantFrom(parseJson(text));
  };
}


export function createComplete(env: NodeJS.ProcessEnv = process.env): Completer {
  const apiKey = env.DEEPSEEK_API_KEY?.trim() || env.OPENAI_API_KEY?.trim();
  if (!apiKey) {
    throw new Error("缺少 DEEPSEEK_API_KEY");
  }
  const baseUrl = (env.DEEPSEEK_BASE_URL?.trim() || DEFAULT_BASE_URL).replace(/\/$/, "");
  const model = env.DEEPSEEK_MODEL?.trim() || DEFAULT_MODEL;

  return async (messages, tools) => {
    const response = await fetch(`${baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ model, messages, tools }),
      signal: AbortSignal.timeout(60_000),
    });
    const text = await response.text();
    if (!response.ok) {
      throw new Error(`模型请求失败 ${response.status}: ${text.slice(0, 500)}`);
    }
    return assistantFrom(parseJson(text));
  };
}

function parseJson(text: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    throw new Error("模型返回的不是 JSON");
  }
}

function assistantFrom(payload: unknown): Extract<Message, { role: "assistant" }> {
  if (!payload || typeof payload !== "object") {
    throw new Error("模型返回缺少 choices");
  }
  const choices = (payload as { choices?: unknown }).choices;
  const first = Array.isArray(choices) ? choices[0] : undefined;
  const message = first && typeof first === "object" ? (first as { message?: unknown }).message : undefined;
  if (!message || typeof message !== "object") {
    throw new Error("模型返回缺少 message");
  }
  const record = message as { content?: unknown; tool_calls?: unknown };
  const content = typeof record.content === "string" ? record.content : null;
  const toolCalls = parseToolCalls(record.tool_calls);
  return toolCalls.length > 0 ? { role: "assistant", content, tool_calls: toolCalls } : { role: "assistant", content };
}

function parseToolCalls(value: unknown): ToolCall[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item, index) => {
    if (!item || typeof item !== "object") return [];
    const record = item as { id?: unknown; function?: unknown };
    const fn = record.function;
    if (!fn || typeof fn !== "object") return [];
    const name = (fn as { name?: unknown }).name;
    if (typeof name !== "string" || name.trim() === "") return [];
    const id = typeof record.id === "string" && record.id ? record.id : `call_${index}`;
    const args = (fn as { arguments?: unknown }).arguments;
    const argumentsText = typeof args === "string" ? args : JSON.stringify(args ?? {});
    return [{ id, type: "function" as const, function: { name, arguments: argumentsText } }];
  });
}
