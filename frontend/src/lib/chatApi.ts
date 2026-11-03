import type { ThreadMessage, ThreadMessageLike } from "@assistant-ui/react";

type ChatTextPart = { type: "text"; text: string };
type ChatImagePart = { type: "image"; image: string };
type ChatFilePart = {
  type: "file";
  filename: string;
  content: string;
  mime: string;
};
type ChatPart = ChatTextPart | ChatImagePart | ChatFilePart;
type ChatContent = string | ChatPart[];

export type DownloadProposal = {
  filename: string;
  content: string;
  mime: string;
};

export type ChatEvent =
  | { type: "text"; text: string }
  | ({ type: "file" } & DownloadProposal);

const TOOL_NAME = "prepare_download";

export function assistantParts(text: string, files: readonly DownloadProposal[]) {
  const content: Array<
    | { type: "text"; text: string }
    | {
        type: "tool-call";
        toolCallId: string;
        toolName: typeof TOOL_NAME;
        args: DownloadProposal;
        argsText: string;
        result: { ready: true };
      }
  > = [];
  if (text) content.push({ type: "text", text });
  files.forEach((file, index) => {
    const args = {
      filename: file.filename,
      content: file.content,
      mime: file.mime,
    };
    content.push({
      type: "tool-call",
      toolCallId: `${TOOL_NAME}-${index}`,
      toolName: TOOL_NAME,
      args,
      argsText: JSON.stringify(args),
      result: { ready: true },
    });
  });
  return content;
}

const IMAGE_URL =
  /^(?:https?:\/\/\S+|data:image\/(?:jpeg|jpg|png|gif|webp);base64,[a-z0-9+/=\s]+)$/i;

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

const downloadPartsOf = (message: ThreadMessage): ChatFilePart[] => {
  if (message.role === "user") return [];
  const files: ChatFilePart[] = [];
  for (const part of message.content) {
    if (part.type !== "tool-call" || part.toolName !== TOOL_NAME) continue;
    const filename = part.args.filename;
    const content = part.args.content;
    const mime = part.args.mime;
    if (typeof filename !== "string" || typeof content !== "string") continue;
    files.push({
      type: "file",
      filename,
      content,
      mime: typeof mime === "string" ? mime : "text/plain",
    });
  }
  return files;
};

const contentOf = (message: ThreadMessage): ChatContent => {
  if (message.role !== "user") {
    const text = textOf(message);
    const files = downloadPartsOf(message);
    if (files.length === 0) return text;
    return [...(text ? [{ type: "text" as const, text }] : []), ...files];
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

const normalizeHistoryContent = (
  content: ChatContent | undefined,
  role: HistoryMessage["role"],
): ThreadMessageLike["content"] | null => {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return null;

  if (role !== "user") {
    const text = content
      .filter((part) => part.type === "text")
      .map((part) => part.text)
      .join("");
    const files = content.flatMap((part) => {
      const file = fileProposalOf(part);
      return file ? [file] : [];
    });
    if (files.length === 0) return text;
    return assistantParts(text, files);
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
  const response = await fetch("/api/sessions");
  if (!response.ok) {
    throw new Error(`Sessions request failed: ${response.status}`);
  }
  const body = (await response.json()) as { sessions?: ChatSessionSummary[] };
  return body.sessions ?? [];
}

export async function loadMessages(sessionId: string): Promise<ThreadMessageLike[]> {
  const response = await fetch(
    `/api/messages/${encodeURIComponent(sessionId)}`,
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

export async function* streamChat(
  sessionId: string,
  messages: readonly ThreadMessage[],
  signal: AbortSignal,
): AsyncGenerator<ChatEvent> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
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
      yield JSON.parse(line) as ChatEvent;
    }
    if (done) {
      if (buffer.trim()) yield JSON.parse(buffer) as ChatEvent;
      break;
    }
  }
}
