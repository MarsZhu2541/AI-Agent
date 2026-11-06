import assert from "node:assert/strict";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import { appendMessages, loadMessages } from "../src/session.js";
import type { Message } from "../src/types.js";

test("日志按追加顺序重放", async () => {
  const dir = await mkdtemp(path.join(tmpdir(), "harness-session-"));
  const file = path.join(dir, "nested", "session.jsonl");
  const messages: Message[] = [
    { role: "user", content: "你好" },
    {
      role: "assistant",
      content: null,
      tool_calls: [{ id: "call_1", type: "function", function: { name: "list_dir", arguments: "{}" } }],
    },
    { role: "tool", tool_call_id: "call_1", content: "a.txt" },
  ];

  assert.deepEqual(await loadMessages(file), []);
  await appendMessages(file, messages);
  await appendMessages(file, [{ role: "assistant", content: "看到 a.txt" }]);
  assert.deepEqual(await loadMessages(file), [...messages, { role: "assistant", content: "看到 a.txt" }]);
});
