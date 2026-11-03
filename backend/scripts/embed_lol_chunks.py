"""Embed champion chunks and upsert them into PostgreSQL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.embeddings import embedding_model
from agent.vectors import upsert_embeddings


def main() -> None:
    """Embed chunk text with the configured model, then store the vectors."""
    parser = argparse.ArgumentParser(description=__doc__)
    default_input = Path(__file__).resolve().parents[1] / "data" / "lol" / "chunks.json"
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    records = json.loads(args.input.read_text(encoding="utf-8"))
    total = len(records)
    model = embedding_model.model
    print(f"model {model}", flush=True)
    saved = 0
    for start in range(0, total, args.batch):
        batch = records[start : start + args.batch]
        for offset, record in enumerate(batch, start=1):
            index = start + offset
            print(
                f"embed {index}/{total} {record['id']} {record.get('name', '')}",
                flush=True,
            )
        embedded = embedding_model.embed([record["text"] for record in batch])
        upsert_embeddings(
            [
                {**record, "embedding": vector, "model": model}
                for record, vector in zip(batch, embedded, strict=True)
            ]
        )
        saved += len(batch)
        print(f"saved {saved}/{total}", flush=True)
    print(f"saved {saved} embeddings from {model}", flush=True)


if __name__ == "__main__":
    main()
