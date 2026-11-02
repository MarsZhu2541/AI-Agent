import { DownloadFileToolUI } from "@/components/DownloadFile";
import { Thread } from "@/components/assistant-ui/elements/thread.aui";
import {
  assistantParts,
  loadMessages,
  streamChat,
  type DownloadProposal,
} from "@/lib/chatApi";
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
    let text = "";
    const files: DownloadProposal[] = [];

    for await (const event of streamChat(messages, abortSignal)) {
      if (event.type === "text") text += event.text;
      else files.push(event);
      const content = assistantParts(text, files);
      if (content.length > 0) yield { content };
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
      <DownloadFileToolUI />
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
