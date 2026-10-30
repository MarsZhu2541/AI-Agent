import type { ThreadMessage, ThreadMessageLike } from "@assistant-ui/react";

type ChatTextPart = { type: "text"; text: string };
type ChatImagePart = { type: "image"; image: string };
type ChatContent = string | Array<ChatTextPart | ChatImagePart>;

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

const contentOf = (message: ThreadMessage): ChatContent => {
  if (message.role !== "user") return textOf(message);

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

const normalizeHistoryContent = (
  content: ChatContent | undefined,
  role: HistoryMessage["role"],
): ThreadMessageLike["content"] | null => {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return null;

  if (role !== "user") {
    return content
      .filter((part) => part.type === "text")
      .map((part) => part.text)
      .join("");
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

export async function loadMessages(): Promise<ThreadMessageLike[]> {
  const response = await fetch("/api/messages");
  if (!response.ok) {
    throw new Error(`History request failed: ${response.status}`);
  }

  const body = (await response.json()) as { messages?: HistoryMessage[] };
  return (body.messages ?? []).flatMap((message) => {
    const threadMessage = toThreadMessage(message);
    return threadMessage ? [threadMessage] : [];
  });
}

export async function streamChat(
  messages: readonly ThreadMessage[],
  signal: AbortSignal,
): Promise<ReadableStreamDefaultReader<Uint8Array>> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
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

  return response.body.getReader();
}
