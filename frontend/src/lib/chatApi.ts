import type { ThreadMessage, ThreadMessageLike } from "@assistant-ui/react";

type ChatTextPart = { type: "text"; text: string };
type ChatImagePart = { type: "image"; image: string };
type ChatFilePart = {
  type: "file";
  filename: string;
  content: string;
  mime: string;
};
export type ActivityName = "retrieve_lore" | "search_web" | "get_weather";
export type ActivityStatus = "running" | "complete" | "error";
export type ChatActivityPart = {
  type: "activity";
  id: string;
  name: ActivityName;
  status: ActivityStatus;
  title: string;
  detail: string;
};
type ChatPart = ChatTextPart | ChatImagePart | ChatFilePart | ChatActivityPart;
type ChatContent = string | ChatPart[];

export type DownloadProposal = {
  filename: string;
  content: string;
  mime: string;
};

export type ChatEvent =
  | { type: "text"; text: string }
  | ({ type: "file" } & DownloadProposal)
  | ChatActivityPart;

type ChatSegment =
  | { kind: "text"; text: string }
  | { kind: "file"; file: DownloadProposal }
  | { kind: "activity"; activity: ChatActivityPart };

const DOWNLOAD_TOOL = "prepare_download";

type DownloadArgs = {
  filename: string;
  content: string;
  mime: string;
};

type ActivityArgs = {
  title: string;
  detail: string;
};

type ToolCallPart =
  | {
      type: "tool-call";
      toolCallId: string;
      toolName: typeof DOWNLOAD_TOOL;
      args: DownloadArgs;
      argsText: string;
      result: { ready: true };
    }
  | {
      type: "tool-call";
      toolCallId: string;
      toolName: ActivityName;
      args: ActivityArgs;
      argsText: string;
      result?: { title: string; detail: string; status: ActivityStatus };
      isError?: boolean;
    };

type AssistantPart = { type: "text"; text: string } | ToolCallPart;

const fileTool = (file: DownloadProposal, index: number): ToolCallPart => {
  const args = {
    filename: file.filename,
    content: file.content,
    mime: file.mime,
  };
  return {
    type: "tool-call",
    toolCallId: `${DOWNLOAD_TOOL}-${index}`,
    toolName: DOWNLOAD_TOOL,
    args,
    argsText: file.filename,
    result: { ready: true },
  };
};

const activityTool = (activity: ChatActivityPart): ToolCallPart => {
  const args = { title: activity.title, detail: activity.detail };
  const part: ToolCallPart = {
    type: "tool-call",
    toolCallId: activity.id,
    toolName: activity.name,
    args,
    argsText: activity.title,
  };
  if (activity.status === "running") return part;
  return {
    ...part,
    isError: activity.status === "error",
    result: {
      title: activity.title,
      detail: activity.detail,
      status: activity.status,
    },
  };
};

export function applyChatEvent(segments: ChatSegment[], event: ChatEvent) {
  if (event.type === "text") {
    const last = segments[segments.length - 1];
    if (last?.kind === "text") last.text += event.text;
    else segments.push({ kind: "text", text: event.text });
    return;
  }
  if (event.type === "file") {
    segments.push({
      kind: "file",
      file: {
        filename: event.filename,
        content: event.content,
        mime: event.mime,
      },
    });
    return;
  }
  const index = segments.findIndex(
    (segment) => segment.kind === "activity" && segment.activity.id === event.id,
  );
  const activity: ChatActivityPart = {
    type: "activity",
    id: event.id,
    name: event.name,
    status: event.status,
    title: event.title,
    detail: event.detail,
  };
  if (index === -1) segments.push({ kind: "activity", activity });
  else segments[index] = { kind: "activity", activity };
}

export function threadParts(segments: readonly ChatSegment[]): AssistantPart[] {
  const content: AssistantPart[] = [];
  let files = 0;
  for (const segment of segments) {
    if (segment.kind === "text") {
      if (segment.text) content.push({ type: "text", text: segment.text });
      continue;
    }
    if (segment.kind === "file") {
      content.push(fileTool(segment.file, files));
      files += 1;
      continue;
    }
    content.push(activityTool(segment.activity));
  }
  return content;
}

const IMAGE_URL =
  /^(?:https?:\/\/\S+|\/aiapi\/images\/\d+|data:image\/(?:jpeg|jpg|png|gif|webp);base64,[a-z0-9+/=\s]+)$/i;

const textOf = (message: ThreadMessage) =>
  message.content
    .filter((part) => part.type === "text")
    .map((part) => part.text)
    .join("");

const imagePartsOf = (message: ThreadMessage): ChatImagePart[] => {
  if (message.role !== "user") return [];

  const images: ChatImagePart[] = [];
  const push = (image: string | undefined) => {
    if (!image || !IMAGE_URL.test(image)) return;
    if (images.some((part) => part.image === image)) return;
    images.push({ type: "image", image });
  };

  for (const part of message.content) {
    if (part.type === "image") push(part.image);
  }
  for (const attachment of message.attachments) {
    for (const part of attachment.content) {
      if (part.type === "image") push(part.image);
    }
  }
  return images;
};

const stringOf = (value: unknown) => (typeof value === "string" ? value : "");

const isActivityName = (name: string): name is ActivityName =>
  name === "retrieve_lore" || name === "search_web" || name === "get_weather";

const activityPartOf = (
  part: ThreadMessage["content"][number],
): ChatActivityPart | null => {
  if (part.type !== "tool-call" || !isActivityName(part.toolName)) return null;
  const result =
    part.result && typeof part.result === "object"
      ? (part.result as { title?: unknown; detail?: unknown; status?: unknown })
      : undefined;
  const status: ActivityStatus =
    part.isError || result?.status === "error"
      ? "error"
      : result
        ? "complete"
        : "running";
  return {
    type: "activity",
    id: part.toolCallId,
    name: part.toolName,
    status,
    title: stringOf(result?.title ?? part.args.title),
    detail: stringOf(result?.detail ?? part.args.detail),
  };
};

const filePartOf = (
  part: ThreadMessage["content"][number],
): ChatFilePart | null => {
  if (part.type !== "tool-call" || part.toolName !== DOWNLOAD_TOOL) return null;
  const filename = part.args.filename;
  const content = part.args.content;
  const mime = part.args.mime;
  if (typeof filename !== "string" || typeof content !== "string") return null;
  return {
    type: "file",
    filename,
    content,
    mime: typeof mime === "string" ? mime : "text/plain",
  };
};

const contentOf = (message: ThreadMessage): ChatContent => {
  if (message.role !== "user") {
    const parts: ChatPart[] = [];
    for (const part of message.content) {
      if (part.type === "text") {
        if (part.text) parts.push({ type: "text", text: part.text });
        continue;
      }
      const activity = activityPartOf(part);
      if (activity) {
        parts.push(activity);
        continue;
      }
      const file = filePartOf(part);
      if (file) parts.push(file);
    }
    if (parts.length === 0) return "";
    if (parts.length === 1 && parts[0].type === "text") return parts[0].text;
    return parts;
  }

  const images = imagePartsOf(message);
  const text = textOf(message).trim();
  if (images.length === 0) return text;
  return [{ type: "text", text: text || "请识别这张图片。" }, ...images];
};

type HistoryMessage = {
  id?: string;
  role?: string;
  content?: ChatContent;
  createdAt?: string;
};

const toThreadMessage = (message: HistoryMessage): ThreadMessageLike | null => {
  if (
    message.role !== "user" &&
    message.role !== "assistant" &&
    message.role !== "system"
  ) {
    return null;
  }

  const content = normalizeHistoryContent(message.content, message.role);
  if (content === null) return null;

  return {
    role: message.role,
    content,
    ...(message.id ? { id: message.id } : {}),
    ...(message.createdAt ? { createdAt: new Date(message.createdAt) } : {}),
    ...(message.role === "assistant"
      ? { status: { type: "complete" as const, reason: "stop" as const } }
      : {}),
  };
};

const fileProposalOf = (part: ChatPart): DownloadProposal | null => {
  if (part.type !== "file") return null;
  return {
    filename: part.filename,
    content: part.content,
    mime: part.mime || "text/plain",
  };
};

const storedActivityOf = (part: ChatPart): ChatActivityPart | null => {
  if (part.type !== "activity" || !isActivityName(part.name)) return null;
  if (part.status !== "running" && part.status !== "complete" && part.status !== "error") {
    return null;
  }
  if (!part.id) return null;
  return {
    type: "activity",
    id: part.id,
    name: part.name,
    status: part.status,
    title: part.title || "",
    detail: part.detail || "",
  };
};

const normalizeHistoryContent = (
  content: ChatContent | undefined,
  role: HistoryMessage["role"],
): ThreadMessageLike["content"] | null => {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return null;

  if (role !== "user") {
    const parts: AssistantPart[] = [];
    let files = 0;
    for (const part of content) {
      if (part.type === "text") {
        if (part.text) parts.push({ type: "text", text: part.text });
        continue;
      }
      const activity = storedActivityOf(part);
      if (activity) {
        parts.push(activityTool(activity));
        continue;
      }
      const file = fileProposalOf(part);
      if (file) {
        parts.push(fileTool(file, files));
        files += 1;
      }
    }
    if (parts.length === 0) return "";
    if (parts.length === 1 && parts[0].type === "text") return parts[0].text;
    return parts;
  }

  const parts: Array<
    { type: "text"; text: string } | { type: "image"; image: string }
  > = [];
  for (const part of content) {
    if (part.type === "text" && part.text) {
      parts.push({ type: "text", text: part.text });
    } else if (part.type === "image" && part.image) {
      parts.push({ type: "image", image: part.image });
    }
  }
  if (parts.length === 0) return "";
  if (parts.every((part) => part.type === "text")) {
    return parts.map((part) => part.text).join("");
  }
  return parts;
};

export type ChatSessionSummary = {
  id: string;
  title: string;
  updatedAt: string;
};

export async function loadSessions(): Promise<ChatSessionSummary[]> {
  const response = await fetch("/aiapi/sessions");
  if (!response.ok) {
    throw new Error(`Sessions request failed: ${response.status}`);
  }
  const body = (await response.json()) as { sessions?: ChatSessionSummary[] };
  return body.sessions ?? [];
}

export async function loadMessages(sessionId: string): Promise<ThreadMessageLike[]> {
  const response = await fetch(
    `/aiapi/messages/${encodeURIComponent(sessionId)}`,
  );
  if (!response.ok) {
    throw new Error(`History request failed: ${response.status}`);
  }

  const body = (await response.json()) as { messages?: HistoryMessage[] };
  return (body.messages ?? []).flatMap((message) => {
    const threadMessage = toThreadMessage(message);
    return threadMessage ? [threadMessage] : [];
  });
}

const chatEvent = (value: unknown): ChatEvent | null => {
  if (!value || typeof value !== "object") return null;
  const event = value as Record<string, unknown>;
  if (event.type === "text" && typeof event.text === "string") {
    return { type: "text", text: event.text };
  }
  if (
    event.type === "file" &&
    typeof event.filename === "string" &&
    typeof event.content === "string"
  ) {
    return {
      type: "file",
      filename: event.filename,
      content: event.content,
      mime: typeof event.mime === "string" ? event.mime : "text/plain",
    };
  }
  if (
    event.type === "activity" &&
    typeof event.id === "string" &&
    typeof event.name === "string" &&
    isActivityName(event.name) &&
    (event.status === "running" ||
      event.status === "complete" ||
      event.status === "error") &&
    typeof event.title === "string" &&
    typeof event.detail === "string"
  ) {
    return {
      type: "activity",
      id: event.id,
      name: event.name,
      status: event.status,
      title: event.title,
      detail: event.detail,
    };
  }
  return null;
};

export async function* streamChat(
  sessionId: string,
  userIp: string,
  messages: readonly ThreadMessage[],
  signal: AbortSignal,
  outfitChat: boolean
): AsyncGenerator<ChatEvent> {
  const path = outfitChat ? "/aiapi/outfit/chat" : "/aiapi/chat";
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      user_ip: userIp,
      messages: messages.map((message) => ({
        role: message.role,
        content: contentOf(message),
      })),
    }),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(`Chat request failed: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const lines = buffer.split("\n");
    buffer = done ? "" : (lines.pop() ?? "");
    for (const line of lines) {
      if (!line.trim()) continue;
      const event = chatEvent(JSON.parse(line));
      if (event) yield event;
    }
    if (done) {
      if (buffer.trim()) {
        const event = chatEvent(JSON.parse(buffer));
        if (event) yield event;
      }
      break;
    }
  }
}
