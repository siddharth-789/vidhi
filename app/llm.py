"""Wrapper around the Gemini API for embeddings and generation.

Every call goes through a retry wrapper with exponential backoff and jitter on 429,
and two retries then a typed error on 5xx or timeout, see CLAUDE.md section 2 and the
degradation table in section 11.

Two separate genai.Client instances are constructed, one per API key, because the
task key and the judge key belong to different Google Cloud projects with separate
free tier quotas. Never share a client across the two roles.
"""

from __future__ import annotations

import asyncio
import logging
import random

import google.genai as genai
from google.genai import types as genai_types
from google.genai import errors as genai_errors

from app.errors import EmbeddingError, LLMError

logger = logging.getLogger(__name__)

_MAX_RETRIES = 2
_BASE_DELAY_SECONDS = 1.0

# Vectors returned by gemini-embedding-2 at any requested output_dimensionality have
# been observed to already be unit length (verified empirically against the live API
# at 768 dimensions across multiple inputs, norm approximately 1.0 in every case).
# This contradicts the general Gemini embedding API documentation, which states
# vectors below 3072 dimensions are not normalized. Per an explicit decision on this
# project, we rely on the API's output directly and do not renormalize.


def build_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


async def _with_retry(description: str, fn, error_type: type[Exception]):
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return await fn()
        except genai_errors.ClientError as exc:
            if getattr(exc, "code", None) != 429 or attempt == _MAX_RETRIES:
                logger.error("%s failed, not retrying: %s", description, exc)
                raise error_type(f"{description} failed: {exc}") from exc
            last_exc = exc
            delay = _BASE_DELAY_SECONDS * (2**attempt) + random.uniform(0, 0.5)
            logger.warning(
                "%s hit rate limit, retrying in %.2fs (attempt %d/%d)",
                description,
                delay,
                attempt + 1,
                _MAX_RETRIES,
            )
            await asyncio.sleep(delay)
        except genai_errors.ServerError as exc:
            last_exc = exc
            if attempt == _MAX_RETRIES:
                logger.error("%s failed after retries: %s", description, exc)
                raise error_type(f"{description} failed after retries: {exc}") from exc
            delay = _BASE_DELAY_SECONDS * (2**attempt) + random.uniform(0, 0.5)
            logger.warning(
                "%s hit a server error, retrying in %.2fs (attempt %d/%d)",
                description,
                delay,
                attempt + 1,
                _MAX_RETRIES,
            )
            await asyncio.sleep(delay)
    raise error_type(f"{description} failed: {last_exc}") from last_exc


async def embed_text(
    client: genai.Client,
    text: str,
    model: str,
    output_dimensionality: int,
    task_type: str,
) -> list[float]:
    """Embed a single text. task_type must be RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY.

    See CLAUDE.md section 3 embedding rules 1 and 2.

    gemini-embedding-2 does not batch multiple documents into one embed_content call
    the way its own SDK usage example implies, confirmed empirically: passing a list
    of N texts as contents returns exactly one embedding, not N. One call per text is
    required.
    """

    async def _run():
        response = await client.aio.models.embed_content(
            model=model,
            contents=text,
            config=genai_types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=output_dimensionality,
            ),
        )
        return list(response.embeddings[0].values)

    return await _with_retry("embed 1 text", _run, EmbeddingError)


async def embed_texts(
    client: genai.Client,
    texts: list[str],
    model: str,
    output_dimensionality: int,
    task_type: str,
) -> list[list[float]]:
    """Embed each text in turn, one embed_content call per text. See embed_text."""

    return [
        await embed_text(client, text, model, output_dimensionality, task_type)
        for text in texts
    ]


async def embed_query(
    client: genai.Client, text: str, model: str, output_dimensionality: int
) -> list[float]:
    return await embed_text(
        client, text, model, output_dimensionality, task_type="RETRIEVAL_QUERY"
    )


async def _main() -> None:
    import os

    from dotenv import load_dotenv

    load_dotenv()
    client = build_client(os.environ["GEMINI_API_KEY_TASK"])
    vector = await embed_query(
        client, "What is the GST rate for textiles?", "gemini-embedding-2", 768
    )
    print(f"embedding dimension: {len(vector)}")
    norm = sum(v * v for v in vector) ** 0.5
    print(f"embedding norm: {norm}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_main())
