"""Reranks retrieved chunks by asking the task model to judge relevance.

After hybrid search returns a shortlist of candidate chunks, this module sends the
question and all candidates in one prompt and asks the model to return a JSON array
ranking them best-first with a relevance score. This is cheaper to operate than a
hosted cross-encoder API and needs no extra dependency, at the cost of one extra
model call per request.

Reranking must never fail the request: if the call errors, times out, or returns
unparseable output, the caller gets back the original fusion order with degraded=True
set, and the pipeline continues without reranking.

Run standalone with: python -m app.rerank "some question"
This runs retrieval, then reranks the fused candidates and prints the reordering.
"""

from __future__ import annotations

import asyncio
import json
import logging

from app.config import get_config
from app.errors import RerankError
from app.retrieve import Candidate

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 15.0

_RERANK_PROMPT = """You are ranking candidate excerpts from a GST manual by relevance \
to a question. Return ONLY a JSON array, no other text, of objects with integer fields \
"index" (the candidate number below, 0 indexed) and "score" (relevance from 0 to 1). \
Order the array best match first. Include every candidate exactly once.

Question: {query}

Candidates:
{candidates_block}
"""


async def rerank(query: str, candidates: list[Candidate] ,top_k: int) -> tuple[list[Candidate], bool]:
    """Return up to top_k candidates reordered by relevance, best first.

    degraded is True whenever reranking could not be used and the incoming fusion
    order was kept instead. Never raises, callers do not need to catch RerankError.
    """

    if not candidates:
        return [], False

    config = get_config()
    try:
        scored = await asyncio.wait_for(
            _rerank_llm(query, candidates, config), timeout=_TIMEOUT_SECONDS
        )
        return scored[:top_k], False
    except (RerankError, asyncio.TimeoutError, TimeoutError) as exc:
        reason = str(exc) or f"rerank timed out after {_TIMEOUT_SECONDS}s"
        logger.warning("rerank failed or timed out, falling back to fusion order: %s", reason)
        return candidates[:top_k], True


async def _rerank_llm(query: str, candidates: list[Candidate], config) -> list[Candidate]:
    """Send one listwise ranking prompt to the task model and reorder candidates."""

    from app.llm import build_client

    if not config.gemini_api_key_task:
        raise RerankError("GEMINI_API_KEY_TASK is not set, cannot call the reranker")

    candidates_block = "\n".join(
        f"[{i}] {c.content[:500]}" for i, c in enumerate(candidates)
    )
    prompt = _RERANK_PROMPT.format(query=query, candidates_block=candidates_block)

    client = build_client(config.gemini_api_key_task)
    try:
        response = await client.aio.models.generate_content(
            model=config.task_model,
            contents=prompt,
        )
        raw_text = response.text or ""
    except Exception as exc:  # noqa: BLE001 - any SDK failure degrades rerank, never crashes
        raise RerankError(f"rerank generation failed: {exc}") from exc

    try:
        parsed = _parse_rerank_json(raw_text)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("malformed rerank JSON, raw text: %s", raw_text)
        raise RerankError(f"malformed rerank JSON: {exc}") from exc

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
        raise RerankError("rerank produced no valid indices")

    for i, candidate in enumerate(candidates):
        if i not in seen_indices:
            ordered.append(candidate)

    return ordered


def _parse_rerank_json(raw_text: str) -> list[dict]:
    """Extract the JSON array from a model response, stripping markdown fences if present."""

    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        if text.startswith("json"):
            text = text[4:].strip()

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON array found in rerank response")

    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, list):
        raise ValueError("rerank response was not a JSON array")
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
