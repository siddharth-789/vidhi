"""Load chunks into the database and embed the ones missing an embedding.

Run standalone with:
    python -m ingest.embed [path/to/chunks.jsonl] [--concurrency N] [--skip-load]

Resumable by construction: rows are selected where embedding is null, so an
interrupted run can simply be restarted and will continue from where it left off.
task_type is RETRIEVAL_DOCUMENT for every chunk embedded here, matching the document
side of Gemini's retrieval embedding API (queries use RETRIEVAL_QUERY instead, see
app/llm.py).

One embed_content call embeds exactly one chunk: gemini-embedding-2 does not batch
multiple documents into a single call, so chunks are embedded a handful at a time
concurrently via asyncio.gather (--concurrency) rather than in one batched request.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from pgvector.asyncpg import Vector

from app.config import get_config
from app.db import create_pool, upsert_chunk
from app.llm import build_client, embed_text

logger = logging.getLogger(__name__)

_DEFAULT_CONCURRENCY = 5


def load_chunks(path: Path) -> list[dict]:
    """Read chunks.jsonl into a list of dicts, one per line."""

    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


async def load_chunks_into_db(pool, chunks: list[dict]) -> None:
    """Upsert every chunk's text and metadata, without touching its embedding column."""

    for chunk in chunks:
        await upsert_chunk(pool, chunk)


async def _embed_one(client, row, embed_model: str, embed_dim: int):
    """Embed one chunk row and return its id alongside the resulting vector."""

    vector = await embed_text(
        client, row["content"], embed_model, embed_dim, task_type="RETRIEVAL_DOCUMENT"
    )
    return row["id"], vector


async def embed_missing_chunks(
    pool,
    client,
    embed_model: str,
    embed_dim: int,
    concurrency: int = _DEFAULT_CONCURRENCY,
) -> int:
    """Embed every chunk row still missing an embedding, a few at a time concurrently.

    The read of which rows are missing an embedding and the write of each row's
    embedding both happen on the same held connection, rather than round tripping
    through the pool per call. Supabase's pooler was observed to not reliably make a
    write on one pooled connection visible to an immediately following read on a
    different pooled connection, which caused rows to be reselected and reprocessed.
    """
    total_embedded = 0
    async with pool.acquire() as conn:
        while True:
            rows = await conn.fetch(
                "select id, chunk_uid, content from vidhi.chunks "
                "where embedding is null limit $1",
                concurrency,
            )
            if not rows:
                break

            results = await asyncio.gather(
                *(_embed_one(client, row, embed_model, embed_dim) for row in rows)
            )
            for chunk_id, vector in results:
                await conn.execute(
                    "update vidhi.chunks set embedding = $1 where id = $2",
                    Vector(vector),
                    chunk_id,
                )

            total_embedded += len(rows)
            logger.info("embedded %d chunks so far", total_embedded)

    return total_embedded


async def _main(
    chunks_path_arg: str | None,
    concurrency: int = _DEFAULT_CONCURRENCY,
    skip_load: bool = False,
) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    config = get_config()
    if not config.database_url:
        raise RuntimeError("DATABASE_URL is not set")
    if not config.gemini_api_key_task:
        raise RuntimeError("GEMINI_API_KEY_TASK is not set")

    pool = await create_pool(config.database_url)
    try:
        if skip_load:
            print("skipping chunk load, assuming vidhi.chunks is already populated")
        else:
            chunks_path = Path(chunks_path_arg or "data/chunks.jsonl")
            chunks = load_chunks(chunks_path)
            print(f"loaded {len(chunks)} chunks from {chunks_path}")
            await load_chunks_into_db(pool, chunks)
            print("upserted all chunks into the database")

        client = build_client(config.gemini_api_key_task)
        total_embedded = await embed_missing_chunks(
            pool, client, config.embed_model, config.embed_dim, concurrency=concurrency
        )
        print(f"embedded {total_embedded} chunks this run")

        async with pool.acquire() as conn:
            remaining = await conn.fetchval(
                "select count(*) from vidhi.chunks where embedding is null"
            )
        print(f"{remaining} chunks still missing an embedding")
    finally:
        await pool.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunks_path", nargs="?", default=None)
    parser.add_argument("--concurrency", type=int, default=_DEFAULT_CONCURRENCY)
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Skip upserting chunks.jsonl into the database, use when it was already loaded",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    asyncio.run(_main(args.chunks_path, concurrency=args.concurrency, skip_load=args.skip_load))
