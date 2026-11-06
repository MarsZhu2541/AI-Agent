import { appendFile, mkdir, readFile } from "node:fs/promises";
import path from "node:path";

import type { Message, ToolCall } from "./types.js";

export async function loadMessages(file: string): Promise<Message[]> {
  let text: string;
  try {
    text = await readFile(file, "utf8");
  } catch (error) {
    const code = error && typeof error === "object" && "code" in error ? error.code : "";
    if (code === "ENOENT") return [];
    throw error;
  }
  return text
    .split("\n")
    .flatMap((line, index) => {
      const trimmed = line.trim();
      if (!trimmed) return [];
      let value: unknown;
      try {
        value = JSON.parse(trimmed) as unknown;
      } catch {
        throw new Error(`${file}:${index + 1} 不是合法 JSON`);
      }
      return [parseMessage(value, `${file}:${index + 1}`)];
    });
}

export async function appendMessages(file: string, messages: Message[]): Promise<void> {
  if (messages.length === 0) return;
  await mkdir(path.dirname(file), { recursive: true });
  const body = messages.map((message) => JSON.stringify(message)).join("\n") + "\n";
  await appendFile(file, body, "utf8");
}

function parseMessage(value: unknown, where: string): Message {
  if (!value || typeof value !== "object") {
    throw new Error(`${where} 不是消息`);
  }
  const record = value as { role?: unknown; content?: unknown; tool_calls?: unknown; tool_call_id?: unknown };
  if (record.role === "system" || record.role === "user") {
    if (typeof record.content !== "string") throw new Error(`${where} 缺少 content`);
    return { role: record.role, content: record.content };
  }
  if (record.role === "assistant") {
    const content = typeof record.content === "string" ? record.content : null;
    const toolCalls = parseToolCalls(record.tool_calls, where);
    return toolCalls.length > 0 ? { role: "assistant", content, tool_calls: toolCalls } : { role: "assistant", content };
  }
  if (record.role === "tool") {
    if (typeof record.tool_call_id !== "string" || typeof record.content !== "string") {
      throw new Error(`${where} 的 tool 消息不完整`);
    }
    return { role: "tool", tool_call_id: record.tool_call_id, content: record.content };
  }
  throw new Error(`${where} 的 role 无法识别`);
}

function parseToolCalls(value: unknown, where: string): ToolCall[] {
  if (value === undefined) return [];
  if (!Array.isArray(value)) throw new Error(`${where} 的 tool_calls 不是数组`);
  return value.map((item, index) => {
    if (!item || typeof item !== "object") throw new Error(`${where} 的 tool_calls[${index}] 无法识别`);
    const record = item as { id?: unknown; function?: unknown };
    const fn = record.function;
    if (!fn || typeof fn !== "object") throw new Error(`${where} 的 tool_calls[${index}] 缺少 function`);
    const name = (fn as { name?: unknown }).name;
    const args = (fn as { arguments?: unknown }).arguments;
    if (typeof record.id !== "string" || typeof name !== "string" || typeof args !== "string") {
      throw new Error(`${where} 的 tool_calls[${index}] 字段不完整`);
    }
    return { id: record.id, type: "function", function: { name, arguments: args } };
  });
}
