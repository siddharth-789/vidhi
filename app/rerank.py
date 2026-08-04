"""Reranking with three interchangeable backends selected by RERANK_PROVIDER.

See CLAUDE.md section 3. Default is llm, a single listwise call to TASK_MODEL, so the
project runs on a Gemini key alone. cohere and jina are hosted API backends behind the
same interface. If reranking fails or times out after 4 seconds, the caller falls back
to fusion order, sets degraded=True, and continues, see the degradation table in
section 11. Reranking must never fail the request.

Run standalone with: python -m app.rerank "some question"
This runs retrieval, then reranks the fused candidates and prints the reordering.
"""

from __future__ import annotations

import asyncio
import json
import logging

import httpx

from app.config import get_config
from app.errors import RerankError
from app.retrieve import Candidate

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 4.0
_COHERE_RERANK_URL = "https://api.cohere.com/v1/rerank"
_COHERE_MODEL = "rerank-v3.5"


async def rerank(
    query: str,
    candidates: list[Candidate],
    top_k: int,
) -> tuple[list[Candidate], bool]:
    """Return up to top_k candidates reordered by relevance, best first.

    Returns (candidates, degraded). degraded is True whenever reranking could not be
    used and the fusion order was kept instead. Never raises, callers do not need to
    catch RerankError, it is caught here.
    """

    if not candidates:
        return [], False

    config = get_config()
    try:
        scored = await asyncio.wait_for(
            _rerank_dispatch(config.rerank_provider, query, candidates, config),
            timeout=_TIMEOUT_SECONDS,
        )
        return scored[:top_k], False
    except (RerankError, asyncio.TimeoutError, TimeoutError) as exc:
        reason = str(exc) or f"{config.rerank_provider} rerank timed out after {_TIMEOUT_SECONDS}s"
        logger.warning("rerank failed or timed out, falling back to fusion order: %s", reason)
        return candidates[:top_k], True


async def _rerank_dispatch(
    provider: str, query: str, candidates: list[Candidate], config
) -> list[Candidate]:
    if provider == "cohere":
        return await _rerank_cohere(query, candidates, config.rerank_api_key)
    if provider == "jina":
        return await _rerank_jina(query, candidates, config.rerank_api_key)
    if provider == "llm":
        return await _rerank_llm(query, candidates, config)
    raise RerankError(f"unknown RERANK_PROVIDER {provider!r}")


async def _rerank_cohere(
    query: str, candidates: list[Candidate], api_key: str | None
) -> list[Candidate]:
    if not api_key:
        raise RerankError("RERANK_API_KEY is not set, cannot call cohere")

    documents = [c.content for c in candidates]
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                _COHERE_RERANK_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": _COHERE_MODEL,
                    "query": query,
                    "documents": documents,
                    "top_n": len(documents),
                },
                timeout=_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        raise RerankError(f"cohere rerank request failed: {exc}") from exc

    try:
        results = payload["results"]
        ordered = []
        for item in results:
            candidate = candidates[item["index"]]
            ordered.append(
                Candidate(
                    chunk_uid=candidate.chunk_uid,
                    content=candidate.content,
                    heading_path=candidate.heading_path,
                    chapter=candidate.chapter,
                    page_start=candidate.page_start,
                    page_end=candidate.page_end,
                    dense_rank=candidate.dense_rank,
                    sparse_rank=candidate.sparse_rank,
                    fusion_score=candidate.fusion_score,
                    rerank_score=float(item["relevance_score"]),
                )
            )
        return ordered
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        logger.warning("malformed cohere rerank response: %s", payload)
        raise RerankError(f"malformed cohere rerank response: {exc}") from exc


async def _rerank_jina(
    query: str, candidates: list[Candidate], api_key: str | None
) -> list[Candidate]:
    if not api_key:
        raise RerankError("RERANK_API_KEY is not set, cannot call jina")

    documents = [c.content for c in candidates]
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.jina.ai/v1/rerank",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "jina-reranker-v2-base-multilingual",
                    "query": query,
                    "documents": documents,
                    "top_n": len(documents),
                },
                timeout=_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        raise RerankError(f"jina rerank request failed: {exc}") from exc

    try:
        results = payload["results"]
        ordered = []
        for item in results:
            candidate = candidates[item["index"]]
            ordered.append(
                Candidate(
                    chunk_uid=candidate.chunk_uid,
                    content=candidate.content,
                    heading_path=candidate.heading_path,
                    chapter=candidate.chapter,
                    page_start=candidate.page_start,
                    page_end=candidate.page_end,
                    dense_rank=candidate.dense_rank,
                    sparse_rank=candidate.sparse_rank,
                    fusion_score=candidate.fusion_score,
                    rerank_score=float(item["relevance_score"]),
                )
            )
        return ordered
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        logger.warning("malformed jina rerank response: %s", payload)
        raise RerankError(f"malformed jina rerank response: {exc}") from exc


_LLM_RERANK_PROMPT = """You are ranking candidate excerpts from a GST manual by relevance \
to a question. Return ONLY a JSON array, no other text, of objects with integer fields \
"index" (the candidate number below, 0 indexed) and "score" (relevance from 0 to 1). \
Order the array best match first. Include every candidate exactly once.

Question: {query}

Candidates:
{candidates_block}
"""


async def _rerank_llm(query: str, candidates: list[Candidate], config) -> list[Candidate]:
    from app.llm import build_client

    if not config.gemini_api_key_task:
        raise RerankError("GEMINI_API_KEY_TASK is not set, cannot call llm reranker")

    candidates_block = "\n".join(
        f"[{i}] {c.content[:500]}" for i, c in enumerate(candidates)
    )
    prompt = _LLM_RERANK_PROMPT.format(query=query, candidates_block=candidates_block)

    client = build_client(config.gemini_api_key_task)
    try:
        response = await client.aio.models.generate_content(
            model=config.task_model,
            contents=prompt,
        )
        raw_text = response.text or ""
    except Exception as exc:  # noqa: BLE001 - any SDK failure degrades rerank, never crashes
        raise RerankError(f"llm rerank generation failed: {exc}") from exc

    try:
        parsed = _parse_llm_rerank_json(raw_text)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("malformed llm rerank JSON, raw text: %s", raw_text)
        raise RerankError(f"malformed llm rerank JSON: {exc}") from exc

    ordered = []
    seen_indices = set()
    for item in parsed:
        idx = item.get("index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(candidates) or idx in seen_indices:
            continue
        seen_indices.add(idx)
        candidate = candidates[idx]
        score = item.get("score")
        ordered.append(
            Candidate(
                chunk_uid=candidate.chunk_uid,
                content=candidate.content,
                heading_path=candidate.heading_path,
                chapter=candidate.chapter,
                page_start=candidate.page_start,
                page_end=candidate.page_end,
                dense_rank=candidate.dense_rank,
                sparse_rank=candidate.sparse_rank,
                fusion_score=candidate.fusion_score,
                rerank_score=float(score) if isinstance(score, (int, float)) else None,
            )
        )

    if not ordered:
        raise RerankError("llm rerank produced no valid indices")

    for i, candidate in enumerate(candidates):
        if i not in seen_indices:
            ordered.append(candidate)

    return ordered


def _parse_llm_rerank_json(raw_text: str) -> list[dict]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[: -3]
        text = text.strip()
        if text.startswith("json"):
            text = text[4:].strip()

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON array found in llm rerank response")

    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, list):
        raise ValueError("llm rerank response was not a JSON array")
    return parsed


async def _main() -> None:
    import sys

    from dotenv import load_dotenv

    from app.db import create_pool
    from app.llm import build_client, embed_query
    from app.retrieve import retrieve

    load_dotenv()
    question = sys.argv[1] if len(sys.argv) > 1 else "What is the GST rate for textiles?"

    config = get_config()
    pool = await create_pool(config.database_url)
    try:
        client = build_client(config.gemini_api_key_task)
        embedding = await embed_query(client, question, config.embed_model, config.embed_dim)
        result = await retrieve(
            pool, question, embedding, use_hybrid=True, dense_k=30, sparse_k=30, fusion_k=12
        )
        reranked, degraded = await rerank(question, result.candidates, top_k=5)
        print(f"degraded={degraded}")
        for c in reranked:
            print(f"pages {c.page_start}-{c.page_end} rerank={c.rerank_score} {c.heading_path!r}")
    finally:
        await pool.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_main())
