import assert from "node:assert/strict";
import { mkdtemp, mkdir, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import { resolveInWorkspace, workspaceTools } from "../src/tools.js";

async function workspace() {
  return mkdtemp(path.join(tmpdir(), "harness-mvp-"));
}

test("list_dir 和 read_file 只访问工作区内的文件", async () => {
  const root = await workspace();
  await mkdir(path.join(root, "notes"));
  await writeFile(path.join(root, "notes", "a.txt"), "hello");
  const [listDir, readFileTool] = workspaceTools();

  const listed = await listDir.execute({ path: "notes" }, { workspace: root });
  assert.equal(listed, "a.txt");
  const text = await readFileTool.execute({ path: "notes/a.txt" }, { workspace: root });
  assert.equal(text, "hello");
});

test("符号链接不能逃出工作区", async () => {
  const root = await workspace();
  const outside = await workspace();
  await writeFile(path.join(outside, "secret.txt"), "nope");
  await symlink(outside, path.join(root, "escape"));

  await assert.rejects(() => resolveInWorkspace(root, "escape/secret.txt"), /路径超出工作区/);
});

test("拒绝读取 .env", async () => {
  const root = await workspace();
  await writeFile(path.join(root, ".env"), "KEY=secret");
  const [, readFileTool] = workspaceTools();
  await assert.rejects(() => readFileTool.execute({ path: ".env" }, { workspace: root }), /敏感文件/);
});
