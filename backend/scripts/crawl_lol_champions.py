"""Download League of Legends champion biographies from the Universe JSON API."""

from __future__ import annotations

import argparse
import html
import json
import re
import time
from pathlib import Path
from typing import Any

import httpx

INDEX_URL = "https://yz.lol.qq.com/v1/zh_cn/search/index.json"
CHAMPION_URL = "https://yz.lol.qq.com/v1/zh_cn/champions/{slug}/index.json"
STORY_URL = "https://yz.lol.qq.com/zh_CN/story/champion/{slug}/"
USER_AGENT = "ai-agent-lore-crawler/0.1"

TAG_RE = re.compile(r"<[^>]+>")
BLOCK_RE = re.compile(r"</p>|<br\s*/?>", re.IGNORECASE)


def strip_html(value: str) -> str:
    """Turn biography HTML into plain paragraphs."""
    text = BLOCK_RE.sub("\n", value)
    text = TAG_RE.sub("", text)
    text = html.unescape(text)
    lines = [line.strip() for line in text.splitlines()]
    return "\n\n".join(line for line in lines if line)


def names(items: Any) -> list[str]:
    """Collect display names from role or race objects."""
    if not isinstance(items, list):
        return []
    return [item["name"] for item in items if isinstance(item, dict) and item.get("name")]


def champion_record(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep the fields needed for later retrieval."""
    champion = payload["champion"]
    biography = champion.get("biography") or {}
    related = []
    for item in payload.get("related-champions") or []:
        if isinstance(item, dict) and item.get("slug"):
            related.append(
                {
                    "slug": item["slug"],
                    "name": item.get("name") or "",
                    "title": item.get("title") or "",
                }
            )
    slug = champion["slug"]
    full = biography.get("full") or ""
    short = biography.get("short") or ""
    return {
        "slug": slug,
        "name": champion.get("name") or "",
        "title": champion.get("title") or "",
        "release_date": champion.get("release-date") or "",
        "faction": champion.get("associated-faction") or "",
        "faction_slug": champion.get("associated-faction-slug") or "",
        "roles": names(champion.get("roles")),
        "races": names(champion.get("races")),
        "quote": strip_html(biography.get("quote") or ""),
        "quote_author": biography.get("quote-author") or "",
        "biography_short": strip_html(short),
        "biography": strip_html(full),
        "related": related,
        "source_url": STORY_URL.format(slug=slug),
    }


def fetch_json(client: httpx.Client, url: str) -> dict[str, Any]:
    """GET a JSON document, retrying transient failures."""
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = client.get(url)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError(f"expected object from {url}")
            return payload
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc
            time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}") from last_error


def crawl(output: Path, delay: float) -> None:
    """Download every champion biography and write one JSON file."""
    output.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    with httpx.Client(headers=headers, timeout=20.0, follow_redirects=True) as client:
        index = fetch_json(client, INDEX_URL)
        champions = index.get("champions") or []
        slugs = [item["slug"] for item in champions if isinstance(item, dict) and item.get("slug")]
        records: list[dict[str, Any]] = []
        failures: list[str] = []
        for index_no, slug in enumerate(slugs, start=1):
            try:
                payload = fetch_json(client, CHAMPION_URL.format(slug=slug))
                records.append(champion_record(payload))
            except (RuntimeError, KeyError) as exc:
                failures.append(f"{slug}: {exc}")
            if delay:
                time.sleep(delay)
            if index_no % 25 == 0 or index_no == len(slugs):
                print(f"{index_no}/{len(slugs)}")
    records.sort(key=lambda item: item["slug"])
    output.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    empty = [item["slug"] for item in records if not item["biography"]]
    print(f"saved {len(records)} champions to {output}")
    if empty:
        print(f"empty biography: {', '.join(empty)}")
    if failures:
        print("failures:")
        for failure in failures:
            print(failure)
        raise SystemExit(1)


def main() -> None:
    """Parse arguments and crawl."""
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = Path(__file__).resolve().parents[1] / "data" / "lol" / "champions.json"
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--delay", type=float, default=0.05)
    args = parser.parse_args()
    crawl(args.output, args.delay)


if __name__ == "__main__":
    main()
