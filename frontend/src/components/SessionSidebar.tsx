"use client";

import { Button } from "@/components/ui/button";
import { ThreadListPrimitive } from "@assistant-ui/react";
import { PlusIcon } from "lucide-react";
import type { FC } from "react";

export type ChatSession = {
  id: string;
  title: string;
  updatedAt: string;
};

type SessionSidebarProps = {
  sessions: ChatSession[];
  sessionId: string;
  onSelect: (sessionId: string) => void;
  onNew?: () => void;
};

export const SessionSidebar: FC<SessionSidebarProps> = ({
  sessions,
  sessionId,
  onSelect,
  onNew,
}) => {
  return (
    <ThreadListPrimitive.Root className="aui-root aui-thread-list-root bg-muted/30 flex h-full w-full min-w-0 flex-col gap-1.5 overflow-x-hidden border-r p-2">
      {onNew && (
        <Button
          variant="ghost"
          className="aui-thread-list-new hover:bg-muted flex items-center justify-start gap-1 rounded-lg px-2.5 py-2 text-start"
          onClick={onNew}
        >
          <PlusIcon />
          新对话
        </Button>
      )}
      <div className="flex min-h-0 flex-1 flex-col gap-1 overflow-y-auto">
        {sessions.length === 0 && (
          <p className="text-muted-foreground px-2.5 py-2 text-sm">还没有对话</p>
        )}
        {sessions.map((session) => {
          const active = session.id === sessionId;
          return (
            <Button
              key={session.id}
              variant="ghost"
              aria-current={active ? "true" : undefined}
              data-active={active ? "" : undefined}
              className="aui-thread-list-item hover:bg-muted data-active:bg-muted h-auto w-full min-w-0 justify-start overflow-hidden rounded-lg px-3 py-2 text-start font-normal"
              onClick={() => onSelect(session.id)}
            >
              <span className="min-w-0 truncate">{session.title || "新对话"}</span>
            </Button>
          );
        })}
      </div>
    </ThreadListPrimitive.Root>
  );
};
