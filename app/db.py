"""Thin repository module over asyncpg. Raw SQL only, no ORM.

Every table lives in the vidhi schema and every query qualifies it explicitly, rather
than relying on search_path, because Supabase's connection pooler does not guarantee
session state such as search_path persists across statements on a pooled connection.

Run standalone with: python -m app.db
This issues one trivial query against DATABASE_URL and prints the chunk count, the
same warm up query the API lifespan uses at startup.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import uuid
from typing import Any

import asyncpg
from pgvector.asyncpg import Vector, register_vector

from app.config import get_config
from app.errors import DatabaseError

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 3
_BASE_DELAY_SECONDS = 0.5


async def _with_retry(description: str, fn):
    """Retry a connection-level failure a few times with exponential backoff and jitter."""

    last_exc: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return await fn()
        except (asyncpg.PostgresConnectionError, OSError) as exc:
            last_exc = exc
            if attempt == _MAX_ATTEMPTS:
                break
            delay = _BASE_DELAY_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
            logger.warning(
                "%s failed on attempt %d/%d, retrying in %.2fs: %s",
                description,
                attempt,
                _MAX_ATTEMPTS,
                delay,
                exc,
            )
            await asyncio.sleep(delay)
    logger.error("%s failed after %d attempts: %s", description, _MAX_ATTEMPTS, last_exc)
    raise DatabaseError(f"{description} failed after {_MAX_ATTEMPTS} attempts") from last_exc


async def _init_connection(conn: asyncpg.Connection) -> None:
    await register_vector(conn)
    # asyncpg has no built in codec for jsonb, it round trips Python str only. Encode
    # dict and list values as JSON text on the way in and decode on the way out, so
    # callers can pass and receive native Python objects for jsonb columns such as
    # vidhi.traces.sub_queries, stages, latency_ms, tokens, and citations.
    await conn.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
        format="text",
    )


async def create_pool(database_url: str) -> asyncpg.Pool:
    """Build the process-wide asyncpg connection pool."""

    return await asyncpg.create_pool(
        database_url, min_size=1, max_size=5, init=_init_connection
    )


async def warm_pool(pool: asyncpg.Pool) -> None:
    """Issue one trivial query so the first real request does not pay for connection setup."""

    async def _run():
        async with pool.acquire() as conn:
            await conn.fetchval("select 1")

    await _with_retry("warm pool", _run)


async def get_chunk_count(pool: asyncpg.Pool) -> int:
    """Return how many chunk rows exist, used by /healthz to confirm data is loaded."""

    async def _run():
        async with pool.acquire() as conn:
            return await conn.fetchval("select count(*) from vidhi.chunks")

    return await _with_retry("get chunk count", _run)


async def upsert_chunk(pool: asyncpg.Pool, chunk: dict[str, Any]) -> None:
    """Insert or update a chunk row by chunk_uid. Does not touch embedding.

    Used by ingest/embed.py before the embedding pass, so embeddings can be added in
    a second pass without duplicating rows.
    """

    async def _run():
        async with pool.acquire() as conn:
            await conn.execute(
                """
                insert into vidhi.chunks (chunk_uid, content, heading_path, chapter,
                                           page_start, page_end, token_count)
                values ($1, $2, $3, $4, $5, $6, $7)
                on conflict (chunk_uid) do update set
                    content = excluded.content,
                    heading_path = excluded.heading_path,
                    chapter = excluded.chapter,
                    page_start = excluded.page_start,
                    page_end = excluded.page_end,
                    token_count = excluded.token_count
                """,
                chunk["chunk_uid"],
                chunk["content"],
                chunk["heading_path"],
                chunk["chapter"],
                chunk["page_start"],
                chunk["page_end"],
                chunk["token_count"],
            )

    await _with_retry("upsert chunk", _run)


async def get_chunks_missing_embedding(pool: asyncpg.Pool, limit: int) -> list[asyncpg.Record]:
    """Return up to limit chunk rows that have not yet been embedded."""

    async def _run():
        async with pool.acquire() as conn:
            return await conn.fetch(
                "select id, chunk_uid, content from vidhi.chunks "
                "where embedding is null limit $1",
                limit,
            )

    return await _with_retry("select chunks missing embedding", _run)


async def set_embedding(pool: asyncpg.Pool, chunk_id: int, embedding: list[float]) -> None:
    """Write one chunk's embedding vector."""

    async def _run():
        async with pool.acquire() as conn:
            await conn.execute(
                "update vidhi.chunks set embedding = $1 where id = $2",
                Vector(embedding),
                chunk_id,
            )

    await _with_retry("set embedding", _run)


async def insert_trace(pool: asyncpg.Pool, trace: dict[str, Any]) -> None:
    """Insert one trace row. Callers must catch DatabaseError and continue: a trace
    write failure must never fail the user-facing request."""

    async def _run():
        async with pool.acquire() as conn:
            await conn.execute(
                """
                insert into vidhi.traces (request_id, query, route, sub_queries, stages,
                                           latency_ms, tokens, answer, citations, grounded,
                                           abstained, degraded, error)
                values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                on conflict (request_id) do nothing
                """,
                trace["request_id"],
                trace["query"],
                trace.get("route"),
                trace.get("sub_queries"),
                trace.get("stages"),
                trace.get("latency_ms"),
                trace.get("tokens"),
                trace.get("answer"),
                trace.get("citations"),
                trace.get("grounded"),
                trace.get("abstained"),
                trace.get("degraded", False),
                trace.get("error"),
            )

    await _with_retry("insert trace", _run)


async def get_trace(pool: asyncpg.Pool, request_id: uuid.UUID) -> asyncpg.Record | None:
    """Look up one stored trace row by its request id, or None if not found."""

    async def _run():
        async with pool.acquire() as conn:
            return await conn.fetchrow(
                "select * from vidhi.traces where request_id = $1", request_id
            )

    return await _with_retry("get trace", _run)


async def _main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    config = get_config()
    if not config.database_url:
        raise DatabaseError("DATABASE_URL is not set")

    pool = await create_pool(config.database_url)
    try:
        await warm_pool(pool)
        count = await get_chunk_count(pool)
        print(f"chunks table row count: {count}")
    finally:
        await pool.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_main())
