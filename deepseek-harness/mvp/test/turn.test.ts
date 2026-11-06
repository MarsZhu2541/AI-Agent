import assert from "node:assert/strict";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import { loadMessages } from "../src/session.js";
import { runTurn } from "../src/turn.js";
import type { Completer, Tool } from "../src/types.js";

const echo: Tool = {
  name: "echo",
  description: "原样返回",
  parameters: {
    type: "object",
    properties: {},
    additionalProperties: false,
  },
  async execute() {
    return "pong";
  },
};

async function sessionFile() {
  const dir = await mkdtemp(path.join(tmpdir(), "harness-turn-"));
  return path.join(dir, "session.jsonl");
}

test("成功的一轮写入系统提示、用户、工具和回答", async () => {
  const file = await sessionFile();
  let calls = 0;
  const complete: Completer = async () => {
    calls += 1;
    if (calls === 1) {
      return {
        role: "assistant",
        content: null,
        tool_calls: [{ id: "call_1", type: "function", function: { name: "echo", arguments: "{}" } }],
      };
    }
    return { role: "assistant", content: "完成" };
  };

  const answer = await runTurn({
    sessionPath: file,
    workspace: process.cwd(),
    prompt: "调用 echo",
    complete,
    tools: [echo],
  });

  assert.equal(answer, "完成");
  const saved = await loadMessages(file);
  assert.deepEqual(
    saved.map((message) => message.role),
    ["system", "user", "assistant", "tool", "assistant"],
  );
});

test("第二轮沿用已有的系统提示", async () => {
  const file = await sessionFile();
  const complete: Completer = async () => ({ role: "assistant", content: "好" });
  await runTurn({ sessionPath: file, workspace: process.cwd(), prompt: "第一", complete, tools: [] });
  await runTurn({ sessionPath: file, workspace: process.cwd(), prompt: "第二", complete, tools: [] });
  const saved = await loadMessages(file);
  assert.equal(saved.filter((message) => message.role === "system").length, 1);
  assert.equal(saved.filter((message) => message.role === "user").length, 2);
});

test("模型请求失败时不写入这一轮", async () => {
  const file = await sessionFile();
  const complete: Completer = async () => {
    throw new Error("上游不可用");
  };
  await assert.rejects(
    () =>
      runTurn({
        sessionPath: file,
        workspace: process.cwd(),
        prompt: "你好",
        complete,
        tools: [],
      }),
    /上游不可用/,
  );
  assert.deepEqual(await loadMessages(file), []);
});
