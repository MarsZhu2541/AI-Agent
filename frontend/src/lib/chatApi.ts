import type { ThreadMessage } from "@assistant-ui/react";

const textOf = (message: ThreadMessage) =>
  message.content
    .filter((part) => part.type === "text")
    .map((part) => part.text)
    .join("");

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
        content: textOf(message),
      })),
    }),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(`Chat request failed: ${response.status}`);
  }

  return response.body.getReader();
}
