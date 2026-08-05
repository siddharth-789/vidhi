"""Dense (semantic) and sparse (keyword) retrieval, combined with reciprocal rank
fusion.

Dense and sparse searches run concurrently. Fusion is a pure function so it can be
unit tested against a hand-computed example without touching the database.

Run standalone with: python -m app.retrieve "some question"
This embeds the question, runs both searches against DATABASE_URL, and prints the
fused candidate list.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import asyncpg

from app.errors import RetrievalError

logger = logging.getLogger(__name__)

_RRF_K = 60


@dataclass(frozen=True)
class Candidate:
    """One retrieved chunk, plus where it ranked in each retrieval stage."""

    chunk_uid: str
    content: str
    heading_path: str | None
    chapter: str | None
    page_start: int
    page_end: int
    dense_rank: int | None
    sparse_rank: int | None
    fusion_score: float
    rerank_score: float | None = None


async def dense_search(pool: asyncpg.Pool, embedding: list[float], k: int) -> list[asyncpg.Record]:
    """Nearest-neighbor search by cosine distance over chunk embeddings."""

    from pgvector.asyncpg import Vector

    async with pool.acquire() as conn:
        return await conn.fetch(
            """
            select chunk_uid, content, heading_path, chapter, page_start, page_end
            from vidhi.chunks
            where embedding is not null
            order by embedding <=> $1
            limit $2
            """,
            Vector(embedding),
            k,
        )


async def sparse_search(pool: asyncpg.Pool, query: str, k: int) -> list[asyncpg.Record]:
    """Full-text keyword search over chunk content, ranked by Postgres's ts_rank_cd."""

    async with pool.acquire() as conn:
        return await conn.fetch(
            """
            select chunk_uid, content, heading_path, chapter, page_start, page_end,
                   ts_rank_cd(tsv, websearch_to_tsquery('english', $1)) as rank
            from vidhi.chunks
            where tsv @@ websearch_to_tsquery('english', $1)
            order by rank desc
            limit $2
            """,
            query,
            k,
        )


def fuse(dense_rows: list[asyncpg.Record], sparse_rows: list[asyncpg.Record]) -> list[Candidate]:
    """Reciprocal rank fusion. score = sum(1 / (RRF_K + rank)) across lists.

    A chunk that ranks near the top of either the dense or the sparse list scores
    well, without needing the two lists' raw scores to be on a comparable scale.
    Rank is 1-indexed. A chunk missing from a list contributes nothing from that list
    and keeps a null rank for that list (never a sentinel number).
    """

    by_uid: dict[str, dict] = {}

    for rank, row in enumerate(dense_rows, start=1):
        uid = row["chunk_uid"]
        entry = by_uid.setdefault(
            uid,
            {
                "row": row,
                "dense_rank": None,
                "sparse_rank": None,
                "score": 0.0,
            },
        )
        entry["dense_rank"] = rank
        entry["score"] += 1.0 / (_RRF_K + rank)

    for rank, row in enumerate(sparse_rows, start=1):
        uid = row["chunk_uid"]
        entry = by_uid.setdefault(
            uid,
            {
                "row": row,
                "dense_rank": None,
                "sparse_rank": None,
                "score": 0.0,
            },
        )
        entry["sparse_rank"] = rank
        entry["score"] += 1.0 / (_RRF_K + rank)

    fused = [
        Candidate(
            chunk_uid=entry["row"]["chunk_uid"],
            content=entry["row"]["content"],
            heading_path=entry["row"]["heading_path"],
            chapter=entry["row"]["chapter"],
            page_start=entry["row"]["page_start"],
            page_end=entry["row"]["page_end"],
            dense_rank=entry["dense_rank"],
            sparse_rank=entry["sparse_rank"],
            fusion_score=entry["score"],
        )
        for entry in by_uid.values()
    ]
    fused.sort(key=lambda c: c.fusion_score, reverse=True)
    return fused


@dataclass(frozen=True)
class RetrievalResult:
    """The fused candidate list for one query, plus whether anything degraded."""

    candidates: list[Candidate]
    degraded: bool
    degraded_reason: str | None = None


async def retrieve(
    pool: asyncpg.Pool,
    query: str,
    embedding: list[float] | None,
    use_hybrid: bool,
    dense_k: int,
    sparse_k: int,
    fusion_k: int,
) -> RetrievalResult:
    """Run dense and, when enabled and available, sparse search, then fuse.

    A failed embedding upstream is signalled by embedding being None, which forces
    sparse-only retrieval. A dense or sparse search that itself raises is caught here
    and degrades rather than propagating, unless both fail, which raises
    RetrievalError.
    """

    dense_rows: list[asyncpg.Record] = []
    sparse_rows: list[asyncpg.Record] = []
    degraded = False
    degraded_reason: str | None = None

    dense_task = (
        asyncio.create_task(dense_search(pool, embedding, dense_k))
        if embedding is not None
        else None
    )
    sparse_task = (
        asyncio.create_task(sparse_search(pool, query, sparse_k))
        if use_hybrid or embedding is None
        else None
    )

    dense_exc: Exception | None = None
    sparse_exc: Exception | None = None

    if dense_task is not None:
        try:
            dense_rows = await dense_task
        except Exception as exc:  # noqa: BLE001 - narrowed below by re-raise path
            dense_exc = exc
            logger.warning("dense search failed: %s", exc)

    if sparse_task is not None:
        try:
            sparse_rows = await sparse_task
        except Exception as exc:  # noqa: BLE001 - narrowed below by re-raise path
            sparse_exc = exc
            logger.warning("sparse search failed: %s", exc)

    if embedding is None:
        degraded = True
        degraded_reason = "query embedding failed, sparse only retrieval"

    if dense_task is not None and dense_exc is not None and (
        sparse_task is None or sparse_exc is not None
    ):
        raise RetrievalError("both dense and sparse retrieval failed") from dense_exc

    if dense_exc is not None:
        degraded = True
        degraded_reason = "dense search failed, sparse only retrieval"
    if sparse_exc is not None and use_hybrid:
        degraded = True
        degraded_reason = "sparse search failed, dense only retrieval"

    fused = fuse(dense_rows, sparse_rows)[:fusion_k]
    return RetrievalResult(candidates=fused, degraded=degraded, degraded_reason=degraded_reason)


async def _main() -> None:
    import sys

    from dotenv import load_dotenv

    from app.config import get_config
    from app.db import create_pool
    from app.llm import build_client, embed_query

    load_dotenv()
    question = sys.argv[1] if len(sys.argv) > 1 else "What is the GST rate for textiles?"

    config = get_config()
    if not config.database_url:
        raise RetrievalError("DATABASE_URL is not set")

    pool = await create_pool(config.database_url)
    try:
        client = build_client(config.gemini_api_key_task)
        embedding = await embed_query(client, question, config.embed_model, config.embed_dim)
        result = await retrieve(
            pool, question, embedding, use_hybrid=True, dense_k=30, sparse_k=30, fusion_k=12
        )
        print(f"degraded={result.degraded} reason={result.degraded_reason}")
        for c in result.candidates:
            print(
                f"pages {c.page_start}-{c.page_end} dense={c.dense_rank} "
                f"sparse={c.sparse_rank} score={c.fusion_score:.4f} {c.heading_path!r}"
            )
    finally:
        await pool.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_main())
