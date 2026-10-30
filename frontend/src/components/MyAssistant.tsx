import { Thread } from "@/components/assistant-ui/elements/thread.aui";
import { loadMessages, streamChat } from "@/lib/chatApi";
import { ImageAttachmentAdapter } from "@/lib/imageAttachment";
import {
  AssistantRuntimeProvider,
  useLocalRuntime,
  type ChatModelAdapter,
  type ThreadMessageLike,
} from "@assistant-ui/react";
import { useEffect, useState } from "react";

const imageAttachment = new ImageAttachmentAdapter();

const chatModel: ChatModelAdapter = {
  async *run({ messages, abortSignal }) {
    const reader = await streamChat(messages, abortSignal);
    const decoder = new TextDecoder();
    let text = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      text += decoder.decode(value, { stream: true });
      yield { content: [{ type: "text", text }] };
    }

    text += decoder.decode();
    if (text) {
      yield { content: [{ type: "text", text }] };
    }
  },
};

function AssistantThread({
  initialMessages,
}: {
  initialMessages: ThreadMessageLike[];
}) {
  const runtime = useLocalRuntime(chatModel, {
    initialMessages,
    adapters: { attachments: imageAttachment },
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="h-dvh">
        <Thread />
      </div>
    </AssistantRuntimeProvider>
  );
}

export function MyAssistant() {
  const [initialMessages, setInitialMessages] = useState<
    ThreadMessageLike[] | null
  >(null);

  useEffect(() => {
    let cancelled = false;
    loadMessages()
      .then((messages) => {
        if (!cancelled) setInitialMessages(messages);
      })
      .catch(() => {
        if (!cancelled) setInitialMessages([]);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (initialMessages === null) {
    return <div className="h-dvh" />;
  }

  return <AssistantThread initialMessages={initialMessages} />;
}
