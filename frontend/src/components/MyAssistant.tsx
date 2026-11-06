import { DownloadFileToolUI } from "@/components/DownloadFile";
import { SessionSidebar, type ChatSession } from "@/components/SessionSidebar";
import { Thread } from "@/components/assistant-ui/elements/thread.aui";
import { Button } from "@/components/ui/button";
import {
  assistantParts,
  loadMessages,
  loadSessions,
  streamChat,
  type DownloadProposal,
} from "@/lib/chatApi";
import { ImageAttachmentAdapter } from "@/lib/imageAttachment";
import { createSessionId } from "@/lib/sessionId";
import { cn } from "@/lib/utils";
import {
  AssistantRuntimeProvider,
  useLocalRuntime,
  type ChatModelAdapter,
  type ThreadMessage,
  type ThreadMessageLike,
} from "@assistant-ui/react";
import { PanelLeftIcon } from "lucide-react";
import { useEffect, useRef, useState } from "react";

const NARROW_VIEWPORT = "(max-width: 767px)";

function useNarrowViewport() {
  const [narrow, setNarrow] = useState(
    () => window.matchMedia(NARROW_VIEWPORT).matches,
  );

  useEffect(() => {
    const media = window.matchMedia(NARROW_VIEWPORT);
    const update = () => setNarrow(media.matches);
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);

  return narrow;
}

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
      <div className="h-full min-h-0 min-w-0 flex-1">
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
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const narrow = useNarrowViewport();

  useEffect(() => {
    if (!narrow || !sidebarOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setSidebarOpen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [narrow, sidebarOpen]);

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
    <div className="flex h-dvh overflow-hidden">
      {narrow && sidebarOpen && (
        <button
          type="button"
          aria-label="关闭会话列表"
          className="fixed inset-0 z-30 bg-black/40 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}
      <aside
        id="session-sidebar"
        className={cn(
          "bg-background fixed inset-y-0 left-0 z-40 flex w-64 max-w-[75vw] transition-transform duration-200 ease-out md:static md:z-auto md:max-w-none md:shrink-0 md:translate-x-0",
          sidebarOpen ? "translate-x-0 shadow-xl md:shadow-none" : "-translate-x-full",
        )}
        inert={narrow && !sidebarOpen ? true : undefined}
      >
        <SessionSidebar
          sessions={sessions}
          sessionId={sessionId}
          onSelect={(id) => {
            setSessionId(id);
            setSidebarOpen(false);
          }}
          onNew={() => {
            setSessionId(createSessionId());
            setSidebarOpen(false);
          }}
        />
      </aside>
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <div className="flex h-11 shrink-0 items-center border-b px-1.5 md:hidden">
          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label="打开会话列表"
            aria-expanded={sidebarOpen}
            aria-controls="session-sidebar"
            onClick={() => setSidebarOpen(true)}
          >
            <PanelLeftIcon />
          </Button>
        </div>
        {loadedSessionId === sessionId && initialMessages !== null ? (
          <AssistantThread
            key={sessionId}
            sessionId={sessionId}
            initialMessages={initialMessages}
            onSend={noteSession}
          />
        ) : (
          <div className="min-h-0 min-w-0 flex-1" />
        )}
      </div>
    </div>
  );
}
