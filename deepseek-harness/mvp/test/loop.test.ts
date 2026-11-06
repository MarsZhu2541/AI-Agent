import assert from "node:assert/strict";
import test from "node:test";

import { runLoop, ToolRoundLimitError } from "../src/loop.js";
import type { Completer, Message, Tool } from "../src/types.js";

const echo: Tool = {
  name: "echo",
  description: "原样返回 text",
  parameters: {
    type: "object",
    properties: { text: { type: "string" } },
    required: ["text"],
    additionalProperties: false,
  },
  async execute(args) {
    return (args as { text: string }).text;
  },
};

test("模型要工具时执行并把结果交回去", async () => {
  let calls = 0;
  const complete: Completer = async (messages) => {
    calls += 1;
    if (calls === 1) {
      assert.equal(messages.at(-1)?.role, "user");
      return {
        role: "assistant",
        content: null,
        tool_calls: [
          { id: "call_1", type: "function", function: { name: "echo", arguments: "{\"text\":\"pong\"}" } },
        ],
      };
    }
    const tool = messages.at(-1);
    assert.equal(tool?.role, "tool");
    if (tool?.role === "tool") assert.equal(tool.content, "pong");
    return { role: "assistant", content: "收到 pong" };
  };

  const messages: Message[] = [{ role: "user", content: "ping" }];
  const answer = await runLoop({
    messages,
    tools: [echo],
    workspace: process.cwd(),
    complete,
  });

  assert.equal(answer, "收到 pong");
  assert.equal(messages.length, 4);
});

test("未知工具和坏参数会写回工具消息", async () => {
  let calls = 0;
  const complete: Completer = async () => {
    calls += 1;
    if (calls === 1) {
      return {
        role: "assistant",
        content: null,
        tool_calls: [
          { id: "bad", type: "function", function: { name: "echo", arguments: "{" } },
          { id: "missing", type: "function", function: { name: "nope", arguments: "{}" } },
        ],
      };
    }
    return { role: "assistant", content: "结束" };
  };

  const messages: Message[] = [{ role: "user", content: "试一下" }];
  await runLoop({ messages, tools: [echo], workspace: process.cwd(), complete });
  const toolMessages = messages.filter((message) => message.role === "tool");
  assert.deepEqual(
    toolMessages.map((message) => (message.role === "tool" ? message.content : "")),
    ["工具参数不是合法 JSON", "未知工具: nope"],
  );
});

test("超过轮次上限时抛出并且不再请求模型", async () => {
  let calls = 0;
  const complete: Completer = async () => {
    calls += 1;
    return {
      role: "assistant",
      content: null,
      tool_calls: [{ id: "call_1", type: "function", function: { name: "echo", arguments: "{\"text\":\"x\"}" } }],
    };
  };
  await assert.rejects(
    () =>
      runLoop({
        messages: [{ role: "user", content: "循环" }],
        tools: [echo],
        workspace: process.cwd(),
        complete,
        maxRounds: 2,
      }),
    ToolRoundLimitError,
  );
  assert.equal(calls, 2);
});
