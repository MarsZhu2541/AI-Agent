"""Prepare text files for the browser to download after the user confirms."""

from __future__ import annotations

import re
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Annotated

from langchain_core.tools import tool

MAX_FILE_BYTES = 256 * 1024
MAX_FILES = 3

_NAME = re.compile(r"^\.?[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
_EXTENSIONS = frozenset(
    {
        "txt",
        "md",
        "markdown",
        "json",
        "csv",
        "tsv",
        "py",
        "js",
        "ts",
        "tsx",
        "jsx",
        "css",
        "html",
        "xml",
        "yml",
        "yaml",
        "toml",
        "ini",
        "log",
        "sql",
        "sh",
        "go",
        "rs",
        "java",
        "c",
        "cpp",
        "h",
        "rb",
        "php",
        "vue",
        "env",
    }
)
_MIME = {
    "md": "text/markdown",
    "markdown": "text/markdown",
    "json": "application/json",
    "csv": "text/csv",
    "tsv": "text/tab-separated-values",
    "html": "text/html",
    "xml": "application/xml",
    "js": "text/javascript",
    "css": "text/css",
    "yml": "application/yaml",
    "yaml": "application/yaml",
    "sh": "text/x-shellscript",
}

@dataclass(frozen=True)
class PreparedFile:
    """One text file waiting for the user to confirm a browser download."""

    filename: str
    content: str
    mime: str

    def as_part(self) -> dict[str, str]:
        """Return the file as a chat content part."""
        return {
            "type": "file",
            "filename": self.filename,
            "content": self.content,
            "mime": self.mime,
        }


_pending: ContextVar[list[PreparedFile] | None] = ContextVar(
    "prepared_files",
    default=None,
)


def parse_download(filename: str, content: str) -> PreparedFile | str:
    """Validate a text file, or return a message the model can correct."""
    name = str(filename).strip()
    if (
        not name
        or "/" in name
        or "\\" in name
        or name in {".", ".."}
        or not _NAME.fullmatch(name)
    ):
        return "文件名不合法。使用不含路径的名字，例如 notes.md"
    extension = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if extension not in _EXTENSIONS:
        shown = f".{extension}" if extension else "无扩展名"
        return f"不支持的文件类型：{shown}。请改成 .md、.txt、.json、.csv 或 .py"
    if not isinstance(content, str) or "\x00" in content:
        return "文件内容必须是文本"
    size = len(content.encode("utf-8"))
    if size == 0:
        return "文件内容为空"
    if size > MAX_FILE_BYTES:
        return f"文件不能超过 {MAX_FILE_BYTES // 1024}KB"
    return PreparedFile(
        filename=name,
        content=content,
        mime=_MIME.get(extension, "text/plain"),
    )


def bind_prepared_files() -> Token[list[PreparedFile] | None]:
    """Start collecting files for the current chat request."""
    return _pending.set([])


def reset_prepared_files(token: Token[list[PreparedFile] | None]) -> None:
    """Stop collecting files for the current chat request."""
    _pending.reset(token)


def take_prepared_files() -> list[PreparedFile]:
    """Return files prepared since the last take, then clear them."""
    bucket = _pending.get()
    if not bucket:
        return []
    found = list(bucket)
    bucket.clear()
    return found


@tool(
    description=(
        "生成一个供用户下载的文本文件。文件不会自动下载，用户必须在页面上点击确认。"
        "filename 不要带路径，使用常见文本扩展名，例如 notes.md、data.csv、main.py。"
        "content 是完整文件内容。不要在回复里再贴出整份文件。"
    ),
)
async def prepare_download(
    filename: Annotated[str, "文件名，例如 notes.md"],
    content: Annotated[str, "完整的文本内容"],
) -> str:
    """Stage a text file for the user to confirm and download."""
    parsed = parse_download(filename, content)
    if isinstance(parsed, str):
        return parsed
    bucket = _pending.get()
    if bucket is None:
        return "当前会话不能生成文件"
    if len(bucket) >= MAX_FILES:
        return f"一次最多生成 {MAX_FILES} 个文件"
    bucket.append(parsed)
    size = len(parsed.content.encode("utf-8"))
    return (
        f"已生成 {parsed.filename}（{size} 字节）。"
        "请告诉用户在页面上确认后才会下载，不要复述文件全文。"
    )
