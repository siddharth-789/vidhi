"""Load chunks into the database and embed the ones missing an embedding.

Run standalone with:
    python -m ingest.embed [path/to/chunks.jsonl] [--max-calls N] [--delay-seconds S]
                                                    [--concurrency N] [--api-key KEY]

Resumable by construction: rows are selected where embedding is null, so an
interrupted run, or a run deliberately stopped partway through because of a daily API
call quota, can simply be restarted later, with the same or a different API key, and
will continue from where it left off. task_type is RETRIEVAL_DOCUMENT for every chunk
embedded here, see CLAUDE.md section 3 embedding rule 1.

One embed_content call embeds exactly one chunk. gemini-embedding-2 was confirmed
empirically, including against its own SDK documented usage example, to return only
one embedding no matter how many texts are passed in a single call, so there is no
batching to be had here, each chunk costs one call. The free Gemini tier used on this
project caps requests per day at a low four figure number, so a full ingestion of a
few thousand chunks may need --max-calls to spend exactly the calls left in a day's
quota, or --api-key to resume with a different key, across more than one day.

Note on normalization: gemini-embedding-2 was verified empirically against the live
API to already return unit length vectors at 768 output dimensions, across multiple
sample inputs. Per a deliberate decision on this project, no additional L2
normalization step is implemented here, the API's output is stored as returned.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from pgvector.asyncpg import Vector

from app.config import get_config
from app.db import create_pool, upsert_chunk
from app.llm import build_client, embed_text

logger = logging.getLogger(__name__)

_DEFAULT_DELAY_SECONDS = 0.0
_DEFAULT_CONCURRENCY = 5


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


async def load_chunks_into_db(pool, chunks: list[dict]) -> None:
    for chunk in chunks:
        await upsert_chunk(pool, chunk)


async def _embed_one(client, row, embed_model: str, embed_dim: int):
    vector = await embed_text(
        client, row["content"], embed_model, embed_dim, task_type="RETRIEVAL_DOCUMENT"
    )
    return row["id"], vector


async def embed_missing_chunks(
    pool,
    client,
    embed_model: str,
    embed_dim: int,
    max_calls: int | None = None,
    delay_seconds: float = _DEFAULT_DELAY_SECONDS,
    concurrency: int = _DEFAULT_CONCURRENCY,
) -> int:
    """Embed rows missing an embedding, one chunk per API call, several concurrently.

    The read of which rows are missing an embedding and the write of each row's
    embedding both happen on the same held connection, rather than round tripping
    through the pool per call. Supabase's pooler was observed to not reliably make a
    write on one pooled connection visible to an immediately following read on a
    different pooled connection, which caused rows to be reselected and reprocessed.
    Each concurrent task still performs exactly one embed_text call for exactly one
    row, so there is no reintroduction of the earlier batching mismatch, only the
    fan out and gather is concurrent, the API call to row mapping stays 1 to 1.
    """
    total_embedded = 0
    calls_made = 0
    async with pool.acquire() as conn:
        while True:
            if max_calls is not None and calls_made >= max_calls:
                logger.info("reached max_calls=%d for this run, stopping", max_calls)
                break

            batch_size = min(concurrency, max_calls - calls_made) if max_calls is not None else concurrency
            rows = await conn.fetch(
                "select id, chunk_uid, content from vidhi.chunks "
                "where embedding is null limit $1",
                batch_size,
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

            calls_made += len(rows)
            total_embedded += len(rows)
            logger.info(
                "embedded %d chunks so far (%d calls made this run)",
                total_embedded,
                calls_made,
            )

            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)

    return total_embedded


async def _main(
    chunks_path_arg: str | None,
    max_calls: int | None = None,
    delay_seconds: float = _DEFAULT_DELAY_SECONDS,
    api_key: str | None = None,
    concurrency: int = _DEFAULT_CONCURRENCY,
    skip_load: bool = False,
) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    config = get_config()
    if not config.database_url:
        raise RuntimeError("DATABASE_URL is not set")
    resolved_api_key = api_key or config.gemini_api_key_task
    if not resolved_api_key:
        raise RuntimeError("GEMINI_API_KEY_TASK is not set and no --api-key was given")

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

        client = build_client(resolved_api_key)
        total_embedded = await embed_missing_chunks(
            pool,
            client,
            config.embed_model,
            config.embed_dim,
            max_calls=max_calls,
            delay_seconds=delay_seconds,
            concurrency=concurrency,
        )
        print(f"embedded {total_embedded} chunks this run")

        async with pool.acquire() as conn:
            remaining = await conn.fetchval(
                "select count(*) from vidhi.chunks where embedding is null"
            )
        print(f"{remaining} chunks still missing an embedding, rerun this command to continue")
    finally:
        await pool.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunks_path", nargs="?", default=None)
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument("--delay-seconds", type=float, default=_DEFAULT_DELAY_SECONDS)
    parser.add_argument("--concurrency", type=int, default=_DEFAULT_CONCURRENCY)
    parser.add_argument("--api-key", default=None, help="Override GEMINI_API_KEY_TASK for this run")
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Skip upserting chunks.jsonl into the database, use when it was already loaded",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    asyncio.run(
        _main(
            args.chunks_path,
            max_calls=args.max_calls,
            delay_seconds=args.delay_seconds,
            api_key=args.api_key,
            concurrency=args.concurrency,
            skip_load=args.skip_load,
        )
    )
