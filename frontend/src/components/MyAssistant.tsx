import { Thread } from "@/components/assistant-ui/elements/thread.aui";
import { streamChat } from "@/lib/chatApi";
import {
  AssistantRuntimeProvider,
  useLocalRuntime,
  type ChatModelAdapter,
} from "@assistant-ui/react";

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

export function MyAssistant() {
  const runtime = useLocalRuntime(chatModel);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="h-dvh">
        <Thread />
      </div>
    </AssistantRuntimeProvider>
  );
}
