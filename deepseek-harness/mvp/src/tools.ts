import { open, readdir, realpath, stat } from "node:fs/promises";
import path from "node:path";

import type { Tool, ToolSchema } from "./types.js";

const MAX_ENTRIES = 200;
const MAX_READ_BYTES = 16 * 1024;

export function toolSchemas(tools: Tool[]): ToolSchema[] {
  return tools.map((tool) => ({
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters,
    },
  }));
}

export function workspaceTools(): Tool[] {
  return [listDir, readFileTool];
}

export async function resolveInWorkspace(workspace: string, requested = "."): Promise<string> {
  const root = await realpath(path.resolve(workspace));
  const target = path.resolve(root, requested);
  let realTarget: string;
  try {
    realTarget = await realpath(target);
  } catch (error) {
    const code = error && typeof error === "object" && "code" in error ? error.code : "";
    if (code === "ENOENT") {
      throw new Error(`路径不存在: ${requested}`);
    }
    throw error;
  }
  const relative = path.relative(root, realTarget);
  if (relative === ".." || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
    throw new Error(`路径超出工作区: ${requested}`);
  }
  if (isSecret(realTarget)) {
    throw new Error(`拒绝读取敏感文件: ${requested}`);
  }
  return realTarget;
}

const listDir: Tool = {
  name: "list_dir",
  description: "列出工作区内某个目录的文件名。path 省略时列出工作区根目录。",
  parameters: {
    type: "object",
    properties: {
      path: { type: "string", description: "相对工作区的目录，默认 ." },
    },
    additionalProperties: false,
  },
  async execute(args, context) {
    const requested = optionalPath(args);
    const target = await resolveInWorkspace(context.workspace, requested);
    const info = await stat(target);
    if (!info.isDirectory()) {
      throw new Error(`不是目录: ${requested}`);
    }
    const names = (await readdir(target))
      .filter((name) => name !== "node_modules" && name !== ".git" && !isSecretName(name))
      .sort((a, b) => a.localeCompare(b));
    const shown = names.slice(0, MAX_ENTRIES);
    const omitted = names.length - shown.length;
    const body = shown.join("\n");
    return omitted > 0 ? `${body}\n[另有 ${omitted} 项未列出]` : body || "(空目录)";
  },
};

const readFileTool: Tool = {
  name: "read_file",
  description: "读取工作区内的文本文件。超出长度会被截断。",
  parameters: {
    type: "object",
    properties: {
      path: { type: "string", description: "相对工作区的文件路径" },
    },
    required: ["path"],
    additionalProperties: false,
  },
  async execute(args, context) {
    const requested = requiredPath(args);
    const target = await resolveInWorkspace(context.workspace, requested);
    const info = await stat(target);
    if (!info.isFile()) {
      throw new Error(`不是文件: ${requested}`);
    }
    const length = Math.min(info.size, MAX_READ_BYTES);
    const buffer = Buffer.alloc(length);
    const handle = await open(target, "r");
    try {
      if (length > 0) await handle.read(buffer, 0, length, 0);
    } finally {
      await handle.close();
    }
    const text = buffer.toString("utf8");
    return info.size > MAX_READ_BYTES ? `${text}\n[已截断，文件共 ${info.size} 字节]` : text || "(空文件)";
  },
};

function optionalPath(args: unknown): string {
  const value = pathValue(args);
  return value ?? ".";
}

function requiredPath(args: unknown): string {
  const value = pathValue(args);
  if (!value) {
    throw new Error("path 必须是非空字符串");
  }
  return value;
}

function pathValue(args: unknown): string | undefined {
  if (args === undefined || args === null) return undefined;
  if (typeof args !== "object" || Array.isArray(args)) {
    throw new Error("工具参数必须是对象");
  }
  const value = (args as { path?: unknown }).path;
  if (value === undefined) return undefined;
  if (typeof value !== "string" || value.trim() === "") {
    throw new Error("path 必须是非空字符串");
  }
  return value;
}

function isSecret(target: string): boolean {
  return target.split(path.sep).some((part) => isSecretName(part));
}

function isSecretName(name: string): boolean {
  return name === ".env" || name.startsWith(".env.");
}

export function toolByName(tools: Tool[], name: string): Tool | undefined {
  return tools.find((tool) => tool.name === name);
}
