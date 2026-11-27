import {
  LoreActivityToolUI,
  SearchActivityToolUI,
  WeatherActivityToolUI,
} from "@/components/ActivityTool";
import { DownloadFileToolUI } from "@/components/DownloadFile";
import { SessionSidebar, type ChatSession } from "@/components/SessionSidebar";
import { Thread } from "@/components/assistant-ui/elements/thread.aui";
import { Button } from "@/components/ui/button";
import {
  applyChatEvent,
  loadMessages,
  loadSessions,
  streamChat,
  threadParts,
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
import { useEffect, useMemo, useRef, useState } from "react";

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
  initialMessages, outfitChat, setOutfitChat,userIp,
  onSend,
}: {
  sessionId: string;
  initialMessages: ThreadMessageLike[];
  outfitChat: boolean;
  setOutfitChat: (value: boolean) => void;
  userIp: string;
  onSend: (sessionId: string, title: string) => void;
}) {
  const sessionRef = useRef(sessionId);
  const onSendRef = useRef(onSend);
  const outfitChatRef = useRef(outfitChat);
  const userIpRef = useRef(userIp);
  useEffect(() => {
    outfitChatRef.current = outfitChat;
    userIpRef.current = userIp
    sessionRef.current = sessionId;
    onSendRef.current = onSend;
  });

  const ComposerExtra = useMemo(() => {
    function OutfitToggle() {
      return (
        <Button
          type="button"
          size="sm"
          aria-label="穿搭推荐"
          aria-pressed={outfitChat}
          variant={outfitChat ? "default" : "ghost"}
          className={cn(
            "h-7 rounded-full px-2.5 text-xs",
            !outfitChat && "border-border text-muted-foreground",
          )}
          onClick={() => setOutfitChat(!outfitChat)}
        >
          穿搭推荐
        </Button>
      );
    }
    return OutfitToggle;
  }, [outfitChat, setOutfitChat]);


  const [chatModel] = useState<ChatModelAdapter>(() => ({
    async *run({ messages, abortSignal }) {
      onSendRef.current(sessionRef.current, sessionTitle(messages), );
      const segments: Parameters<typeof applyChatEvent>[0] = [];
      for await (const event of streamChat(
        sessionRef.current,
        userIpRef.current,
        messages,
        abortSignal,
        outfitChatRef.current,
      )) {
        applyChatEvent(segments, event);
        const content = threadParts(segments);
        if (content.length > 0) yield { content };
      }
      let unfinished = false;
      for (const segment of segments) {
        if (segment.kind !== "activity" || segment.activity.status !== "running") {
          continue;
        }
        unfinished = true;
        segment.activity = {
          ...segment.activity,
          status: "error",
          detail: segment.activity.detail || "未完成",
        };
      }
      if (unfinished) {
        const content = threadParts(segments);
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
      <LoreActivityToolUI />
      <SearchActivityToolUI />
      <WeatherActivityToolUI />
      <div className="h-full min-h-0 min-w-0 flex-1">
        <Thread components={{ ComposerExtra }}  />
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
  const [outfitChat, setOutfitChat] = useState(false);
  const narrow = useNarrowViewport();
  const [userIp, setUserIp] = useState("");

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
    console.log("start query user ip: ")

    async function fetchIp() {
      try {
        const ipRes = await fetch("http://ip-api.com/json/?lang=zh-CN");
        if (!ipRes.ok) throw new Error(`IP 请求失败: ${ipRes.status}`);
        const data = await ipRes.json();
        console.log("user ip: " + data.query)
        if (!cancelled) setUserIp(data.query);
      } catch (err) {
        if (!cancelled) {
          // 这里可以 setError(err.message)，或者至少 console.error
          console.error("获取 IP 失败:", err);
        }
      }
    }

    fetchIp();
    return () => { cancelled = true; };
  }, []);

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
            outfitChat = {outfitChat}
            setOutfitChat={setOutfitChat}
            userIp = {userIp}
            onSend={noteSession}
          />
        ) : (
          <div className="min-h-0 min-w-0 flex-1" />
        )}
      </div>
    </div>
  );
}
