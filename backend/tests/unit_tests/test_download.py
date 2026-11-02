"""Download proposals stay text files with safe names."""

import asyncio

from agent.downloads import (
    bind_prepared_files,
    parse_download,
    prepare_download,
    reset_prepared_files,
    take_prepared_files,
)


def test_parse_download_accepts_a_text_file() -> None:
    parsed = parse_download("notes.md", "# hi")
    assert not isinstance(parsed, str)
    assert parsed.filename == "notes.md"
    assert parsed.mime == "text/markdown"
    assert parsed.as_part()["content"] == "# hi"


def test_parse_download_rejects_paths_and_unknown_types() -> None:
    assert "文件名" in parse_download("../secret.txt", "x")
    assert "文件类型" in parse_download("run.exe", "x")
    assert "为空" in parse_download("empty.txt", "")
    assert "超过" in parse_download("big.txt", "x" * (256 * 1024 + 1))


def test_prepare_download_records_only_inside_a_request() -> None:
    async def outside() -> str:
        return await prepare_download.ainvoke(
            {"filename": "a.txt", "content": "hi"}
        )

    assert "不能生成文件" in asyncio.run(outside())

    async def inside() -> tuple[str, list]:
        token = bind_prepared_files()
        try:
            message = await prepare_download.ainvoke(
                {"filename": "a.txt", "content": "hi"}
            )
            return message, take_prepared_files()
        finally:
            reset_prepared_files(token)

    message, files = asyncio.run(inside())
    assert "已生成" in message
    assert files[0].filename == "a.txt"
    assert take_prepared_files() == []
