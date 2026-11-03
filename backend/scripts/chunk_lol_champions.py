"""Split champion biographies into paragraph-aligned chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CHUNK_LIMIT = 700

FACTIONS = {
    "zaun": "祖安",
    "void": "虚空之地",
    "ixtal": "以绪塔尔",
    "piltover": "皮尔特沃夫",
    "shadow-isles": "暗影岛",
    "bandle-city": "班德尔城",
    "shurima": "恕瑞玛",
    "mount-targon": "巨神峰",
    "demacia": "德玛西亚",
    "bilgewater": "比尔吉沃特",
    "noxus": "诺克萨斯",
    "freljord": "弗雷尔卓德",
    "ionia": "艾欧尼亚",
    "unaffiliated": "无所属",
}


def paragraphs_of(biography: str) -> list[str]:
    """Return non-empty biography paragraphs."""
    return [part.strip() for part in biography.split("\n\n") if part.strip()]


def chunk_paragraphs(paragraphs: list[str], limit: int) -> list[list[int]]:
    """Group adjacent paragraphs until the next one would pass the limit.

    The next chunk repeats the previous chunk's last paragraph.
    """
    chunks: list[list[int]] = []
    buf: list[int] = []
    size = 0
    for index, paragraph in enumerate(paragraphs):
        if buf and size + len(paragraph) > limit:
            chunks.append(buf)
            buf = [buf[-1], index]
            size = len(paragraphs[buf[0]]) + len(paragraph)
        else:
            buf.append(index)
            size += len(paragraph)
    if buf:
        chunks.append(buf)
    return chunks


def header_of(champion: dict[str, Any]) -> str:
    """Build the identity line stored at the start of every chunk."""
    slug = champion.get("faction_slug") or ""
    faction = FACTIONS.get(slug) or slug
    parts = [f"{champion['name']}（{champion['title']}）"]
    races = champion.get("races") or []
    if races:
        parts.append("种族：" + "、".join(races))
    if faction:
        parts.append("地区：" + faction)
    return "｜".join(parts)


def chunk_champion(champion: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    """Split one champion into retrieval records."""
    paragraphs = paragraphs_of(champion.get("biography") or "")
    if not paragraphs:
        return []
    header = header_of(champion)
    grouped = chunk_paragraphs(paragraphs, limit)
    total = len(grouped)
    records = []
    for chunk_index, indexes in enumerate(grouped):
        body = "\n\n".join(paragraphs[index] for index in indexes)
        records.append(
            {
                "id": f"{champion['slug']}-{chunk_index}",
                "slug": champion["slug"],
                "name": champion["name"],
                "title": champion["title"],
                "chunk_index": chunk_index,
                "chunk_total": total,
                "paragraph_start": indexes[0],
                "paragraph_end": indexes[-1],
                "header": header,
                "text": f"{header}\n\n{body}",
                "roles": champion.get("roles") or [],
                "races": champion.get("races") or [],
                "faction": FACTIONS.get(champion.get("faction_slug") or "", ""),
                "faction_slug": champion.get("faction_slug") or "",
                "related": champion.get("related") or [],
                "quote": champion.get("quote") or "",
                "quote_author": champion.get("quote_author") or "",
                "biography_short": champion.get("biography_short") or "",
                "source_url": champion.get("source_url") or "",
            }
        )
    return records


def chunk_all(champions: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Split every champion biography."""
    records: list[dict[str, Any]] = []
    for champion in champions:
        records.extend(chunk_champion(champion, limit))
    return records


def main() -> None:
    """Read champion biographies and write chunk records."""
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1] / "data" / "lol"
    parser.add_argument("--input", type=Path, default=root / "champions.json")
    parser.add_argument("--output", type=Path, default=root / "chunks.json")
    parser.add_argument("--limit", type=int, default=CHUNK_LIMIT)
    args = parser.parse_args()
    champions = json.loads(args.input.read_text(encoding="utf-8"))
    records = chunk_all(champions, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved {len(records)} chunks from {len(champions)} champions to {args.output}")


if __name__ == "__main__":
    main()
