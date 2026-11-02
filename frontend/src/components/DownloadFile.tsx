"use client";

import { Button } from "@/components/ui/button";
import { makeAssistantToolUI } from "@assistant-ui/react";
import { FileTextIcon } from "lucide-react";
import { useState } from "react";

type DownloadArgs = {
  filename?: string;
  content?: string;
  mime?: string;
};

const FILENAME = /^\.?[A-Za-z0-9][A-Za-z0-9._-]{0,80}$/;

const safeFilename = (name: string | undefined) => {
  const trimmed = name?.trim() ?? "";
  if (
    !trimmed ||
    trimmed.includes("/") ||
    trimmed.includes("\\") ||
    trimmed === "." ||
    trimmed === ".." ||
    !FILENAME.test(trimmed) ||
    !trimmed.includes(".")
  ) {
    return null;
  }
  return trimmed;
};

const formatSize = (bytes: number) => {
  if (bytes < 1024) return `${bytes} B`;
  return `${(bytes / 1024).toFixed(1)} KB`;
};

const saveFile = (filename: string, content: string, mime: string) => {
  const blob = new Blob([content], { type: `${mime};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
};

const DownloadFileCard = ({ args }: { args: DownloadArgs }) => {
  const filename = safeFilename(args.filename);
  const content = typeof args.content === "string" ? args.content : "";
  const mime = typeof args.mime === "string" ? args.mime : "text/plain";
  const [choice, setChoice] = useState<"pending" | "saved" | "dismissed">(
    "pending",
  );

  if (!filename || !content) {
    return (
      <p className="text-muted-foreground text-sm">这个文件还不能下载。</p>
    );
  }

  const bytes = new TextEncoder().encode(content).length;
  const preview = content.length > 400 ? `${content.slice(0, 400)}…` : content;

  return (
    <div className="border-border bg-muted/40 my-2 flex w-full max-w-xl flex-col gap-3 rounded-lg border p-3">
      <div className="flex items-center gap-2">
        <FileTextIcon className="text-muted-foreground size-4 shrink-0" />
        <span className="min-w-0 flex-1 truncate text-sm font-medium">
          {filename}
        </span>
        <span className="text-muted-foreground shrink-0 text-xs">
          {formatSize(bytes)}
        </span>
      </div>
      <pre className="bg-background max-h-32 overflow-auto rounded-md p-2 text-xs break-words whitespace-pre-wrap">
        {preview}
      </pre>
      {choice === "pending" ? (
        <div className="flex items-center gap-2">
          <Button
            size="sm"
            type="button"
            onClick={() => {
              saveFile(filename, content, mime);
              setChoice("saved");
            }}
          >
            确认下载
          </Button>
          <Button
            size="sm"
            type="button"
            variant="outline"
            onClick={() => setChoice("dismissed")}
          >
            取消
          </Button>
        </div>
      ) : (
        <div className="flex items-center gap-2">
          <span className="text-muted-foreground text-sm">
            {choice === "saved" ? "已下载到本机" : "已取消"}
          </span>
          {choice === "saved" ? (
            <Button
              size="sm"
              type="button"
              variant="outline"
              onClick={() => saveFile(filename, content, mime)}
            >
              再次下载
            </Button>
          ) : null}
        </div>
      )}
    </div>
  );
};

export const DownloadFileToolUI = makeAssistantToolUI<
  DownloadArgs,
  { ready?: boolean }
>({
  toolName: "prepare_download",
  display: "standalone",
  render: ({ args }) => <DownloadFileCard args={args} />,
});
