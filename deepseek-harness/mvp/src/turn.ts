import { runLoop } from "./loop.js";
import { appendMessages, loadMessages } from "./session.js";
import type { Completer, Message, Tool } from "./types.js";
import { workspaceTools } from "./tools.js";

export const SYSTEM_PROMPT =
  "你是一个编程助手。查看工作区时使用工具，不要编造没有读到的文件内容。回复使用中文。";

export async function runTurn(input: {
  sessionPath: string;
  workspace: string;
  prompt: string;
  complete: Completer;
  tools?: Tool[];
  maxRounds?: number;
  onMessage?: (message: Message) => void;
}): Promise<string | null> {
  const messages = await loadMessages(input.sessionPath);
  const start = messages.length;
  if (!messages.some((message) => message.role === "system")) {
    messages.push({ role: "system", content: SYSTEM_PROMPT });
  }
  messages.push({ role: "user", content: input.prompt });
  const answer = await runLoop({
    messages,
    tools: input.tools ?? workspaceTools(),
    workspace: input.workspace,
    complete: input.complete,
    maxRounds: input.maxRounds,
    onMessage: input.onMessage,
  });
  await appendMessages(input.sessionPath, messages.slice(start));
  return answer;
}
