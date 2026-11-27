"use client";

import { makeAssistantToolUI } from "@assistant-ui/react";
import { CheckIcon, LoaderIcon, XCircleIcon } from "lucide-react";

type ActivityArgs = {
  title?: string;
  detail?: string;
};

type ActivityResult = {
  title?: string;
  detail?: string;
  status?: string;
};

const textOf = (value: unknown) => (typeof value === "string" ? value.trim() : "");

const ActivityCard = ({
  args,
  result,
  status,
}: {
  args: ActivityArgs;
  result?: ActivityResult;
  status?: { type?: string };
}) => {
  const failed = status?.type === "incomplete" || result?.status === "error";
  const running = !failed && result == null;
  const title = textOf(result?.title) || textOf(args.title) || "工具";
  const detail = running ? "" : textOf(result?.detail) || textOf(args.detail);
  const icon = running ? (
    <LoaderIcon className="text-muted-foreground size-3.5 shrink-0 animate-spin [animation-duration:0.6s]" />
  ) : failed ? (
    <XCircleIcon className="text-destructive size-3.5 shrink-0" />
  ) : (
    <CheckIcon className="text-muted-foreground size-3.5 shrink-0" />
  );

  if (!detail) {
    return (
      <div className="text-muted-foreground my-1 flex items-center gap-2 text-sm">
        {icon}
        <span className="min-w-0 truncate">{title}</span>
      </div>
    );
  }

  return (
    <details className="border-border/70 bg-muted/30 my-1 w-full max-w-xl rounded-lg border">
      <summary className="flex cursor-pointer list-none items-center gap-2 px-2.5 py-1.5 text-sm [&::-webkit-details-marker]:hidden">
        {icon}
        <span className="min-w-0 flex-1 truncate">{title}</span>
      </summary>
      <pre className="text-foreground/90 max-h-48 overflow-auto px-2.5 pb-2.5 text-xs break-words whitespace-pre-wrap">
        {detail}
      </pre>
    </details>
  );
};

const renderActivity = ({
  args,
  result,
  status,
}: {
  args: ActivityArgs;
  result?: ActivityResult;
  status?: { type?: string };
}) => <ActivityCard args={args} result={result} status={status} />;

export const LoreActivityToolUI = makeAssistantToolUI<ActivityArgs, ActivityResult>({
  toolName: "retrieve_lore",
  display: "standalone",
  render: renderActivity,
});

export const SearchActivityToolUI = makeAssistantToolUI<ActivityArgs, ActivityResult>({
  toolName: "search_web",
  display: "standalone",
  render: renderActivity,
});

export const WeatherActivityToolUI = makeAssistantToolUI<ActivityArgs, ActivityResult>({
  toolName: "get_weather",
  display: "standalone",
  render: renderActivity,
});
