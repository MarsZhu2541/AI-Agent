import path from "node:path";

import { createComplete } from "./client.js";
import { runTurn } from "./turn.js";
import type { Message } from "./types.js";

const sessionDefault = path.resolve(import.meta.dirname, "../.sessions/default.jsonl");

const help = `用法: npm start -- [--workspace 目录] [--session 日志] 你的问题

DEEPSEEK_API_KEY   必填，没有时改读 OPENAI_API_KEY
DEEPSEEK_BASE_URL  默认 https://api.deepseek.com/v1
DEEPSEEK_MODEL     默认 deepseek-v4.1-flash

工具只有 list_dir 和 read_file，路径锁在工作区里。一轮失败不会写入会话日志。
`;

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help || !args.prompt) {
    process.stderr.write(help);
    process.exit(args.help ? 0 : 1);
  }
  const complete = createComplete();
  const answer = await runTurn({
    sessionPath: args.session,
    workspace: path.resolve(args.workspace),
    prompt: args.prompt,
    complete,
    onMessage: logMessage,
  });
  if (answer) process.stdout.write(`${answer}\n`);
}

function logMessage(message: Message) {
  if (message.role === "assistant" && message.tool_calls?.length) {
    for (const call of message.tool_calls) {
      process.stderr.write(`[tool] ${call.function.name} ${call.function.arguments}\n`);
    }
  }
}

export function parseArgs(argv: string[]) {
  let workspace = process.cwd();
  let session = sessionDefault;
  let helpFlag = false;
  const prompt: string[] = [];
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--help" || arg === "-h") {
      helpFlag = true;
    } else if (arg === "--workspace") {
      workspace = requireValue(argv, (i += 1), arg);
    } else if (arg === "--session") {
      session = requireValue(argv, (i += 1), arg);
    } else {
      prompt.push(arg);
    }
  }
  return { workspace, session, help: helpFlag, prompt: prompt.join(" ").trim() };
}

function requireValue(argv: string[], index: number, flag: string): string {
  const value = argv[index];
  if (!value || value.startsWith("--")) {
    throw new Error(`${flag} 缺少参数`);
  }
  return value;
}

main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  process.stderr.write(`${message}\n`);
  process.exit(1);
});
