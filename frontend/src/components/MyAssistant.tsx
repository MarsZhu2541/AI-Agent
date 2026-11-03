import { DownloadFileToolUI } from "@/components/DownloadFile";
import { SessionSidebar, type ChatSession } from "@/components/SessionSidebar";
import { Thread } from "@/components/assistant-ui/elements/thread.aui";
import {
  assistantParts,
  loadMessages,
  loadSessions,
  streamChat,
  type DownloadProposal,
} from "@/lib/chatApi";
import { ImageAttachmentAdapter } from "@/lib/imageAttachment";
import { createSessionId } from "@/lib/sessionId";
import {
  AssistantRuntimeProvider,
  useLocalRuntime,
  type ChatModelAdapter,
  type ThreadMessage,
  type ThreadMessageLike,
} from "@assistant-ui/react";
import { useEffect, useRef, useState } from "react";

const imageAttachment = new ImageAttachmentAdapter();

function sessionTitle(messages: readonly ThreadMessage[]): string {
  const user = [...messages].reverse().find((message) => message.role === "user");
  const text = user?.content
    .flatMap((part) => (part.type === "text" ? [part.text] : []))
    .join(" ")
    .replace(/\s+/g, " ")
    .trim();
  if (text) return text.slice(0, 30);
  if (user?.content.some((part) => part.type === "image")) return "图片";
  return "新对话";
}

function AssistantThread({
  sessionId,
  initialMessages,
  onSend,
}: {
  sessionId: string;
  initialMessages: ThreadMessageLike[];
  onSend: (sessionId: string, title: string) => void;
}) {
  const sessionRef = useRef(sessionId);
  const onSendRef = useRef(onSend);
  useEffect(() => {
    sessionRef.current = sessionId;
    onSendRef.current = onSend;
  });

  const [chatModel] = useState<ChatModelAdapter>(() => ({
    async *run({ messages, abortSignal }) {
      onSendRef.current(sessionRef.current, sessionTitle(messages));
      let text = "";
      const files: DownloadProposal[] = [];
      for await (const event of streamChat(
        sessionRef.current,
        messages,
        abortSignal,
      )) {
        if (event.type === "text") text += event.text;
        else files.push(event);
        const content = assistantParts(text, files);
        if (content.length > 0) yield { content };
      }
    },
  }));

  const runtime = useLocalRuntime(chatModel, {
    initialMessages,
    adapters: { attachments: imageAttachment },
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <DownloadFileToolUI />
      <div className="min-w-0 flex-1">
        <Thread />
      </div>
    </AssistantRuntimeProvider>
  );
}

export function MyAssistant() {
  const [initialMessages, setInitialMessages] = useState<
    ThreadMessageLike[] | null
  >(null);
  const [loadedSessionId, setLoadedSessionId] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessions, setSessions] = useState<ChatSession[]>([]);

  const noteSession = (id: string, title: string) => {
    const updatedAt = new Date().toISOString();
    setSessions((current) => {
      const existing = current.find((session) => session.id === id);
      const next = {
        id,
        title: existing?.title || title,
        updatedAt,
      };
      return [next, ...current.filter((session) => session.id !== id)];
    });
  };

  useEffect(() => {
    let cancelled = false;
    loadSessions()
      .then((list) => {
        if (cancelled) return;
        setSessions(list);
        setSessionId(list[0]?.id ?? createSessionId());
      })
      .catch(() => {
        if (cancelled) return;
        setSessions([]);
        setSessionId(createSessionId());
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (sessionId === null) return;
    let cancelled = false;
    loadMessages(sessionId)
      .then((messages) => {
        if (cancelled) return;
        setInitialMessages(messages);
        setLoadedSessionId(sessionId);
      })
      .catch(() => {
        if (cancelled) return;
        setInitialMessages([]);
        setLoadedSessionId(sessionId);
      });
    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  if (sessionId === null) {
    return <div className="h-dvh" />;
  }

  return (
    <div className="flex h-dvh">
      <SessionSidebar
        sessions={sessions}
        sessionId={sessionId}
        onSelect={setSessionId}
        onNew={() => setSessionId(createSessionId())}
      />
      {loadedSessionId === sessionId && initialMessages !== null ? (
        <AssistantThread
          key={sessionId}
          sessionId={sessionId}
          initialMessages={initialMessages}
          onSend={noteSession}
        />
      ) : (
        <div className="min-w-0 flex-1" />
      )}
    </div>
  );
}
