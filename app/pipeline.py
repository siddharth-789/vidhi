"""The single query path: route, retrieve, rerank, generate, verify, one question in,
one streamed answer out.

answer_question is consumed by both app/api.py and the evaluation harness, so
evaluation always exercises the exact same code the live API runs, not a parallel
reimplementation. PipelineConfig selects which stages are turned on, which is what
lets the four ablation configs (A/B/C/D) share this one function.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, Union

import asyncpg
import dspy
import google.genai as genai

from app.config import Config
from app.db import insert_trace
from app.errors import DatabaseError, EmbeddingError, GuardrailAbstain, LLMError, RetrievalError
from app.guardrails import check_pre_generation, validate_citations
from app.llm import embed_query
from app.programs import AnswerFromManual, CheckGrounded, RAGProgram, RouteQuery
from app.rerank import rerank as rerank_candidates
from app.retrieve import Candidate, retrieve
from app.trace import StageTimer, Trace, new_trace

logger = logging.getLogger(__name__)

_ABSTENTION_MESSAGE = (
    "I could not find a confident answer to this in Bharat's GST Smart Guide. "
    "Please rephrase the question or ask something covered by the manual."
)

_PLAIN_PROMPT_TEMPLATE = """You are answering a question using only the context below, \
excerpted from a GST reference manual. If the context does not contain the answer, say \
plainly that the manual does not cover this rather than guessing or using outside \
knowledge. Cite page numbers that appear in the context using the format (p. N).

Context:
{context}

Question: {question}

Answer:"""


@dataclass(frozen=True)
class PipelineConfig:
    """Selects which pipeline stages run, so the four ablation configs (A/B/C/D) can
    share one code path and differ only in these flags."""

    name: str
    use_hybrid: bool  # False means dense-only retrieval, no keyword search
    use_rerank: bool  # whether the LLM reranker runs on the fused candidates
    use_dspy: bool  # DSPy routing + ChainOfThought generation vs a hand-written prompt
    use_compiled: bool  # load the MIPROv2-compiled program instead of a fresh one
    dense_k: int = 30
    sparse_k: int = 30
    fusion_k: int = 12
    final_k: int = 5


CONFIGS: dict[str, PipelineConfig] = {
    "A_vanilla": PipelineConfig(
        name="A_vanilla", use_hybrid=False, use_rerank=False, use_dspy=False, use_compiled=False
    ),
    "B_hybrid": PipelineConfig(
        name="B_hybrid", use_hybrid=True, use_rerank=True, use_dspy=False, use_compiled=False
    ),
    "C_dspy": PipelineConfig(
        name="C_dspy", use_hybrid=True, use_rerank=True, use_dspy=True, use_compiled=False
    ),
    "D_optimized": PipelineConfig(
        name="D_optimized", use_hybrid=True, use_rerank=True, use_dspy=True, use_compiled=True
    ),
}


@dataclass(frozen=True)
class RouteEvent:
    """Emitted once routing decides which path the question takes."""

    route: str
    sub_queries: list[str]


@dataclass(frozen=True)
class RetrievalEvent:
    """Emitted once retrieval (and rerank, if enabled) has picked the final candidates."""

    candidates: list[dict[str, Any]]
    degraded: bool


@dataclass(frozen=True)
class TokenEvent:
    """One chunk of streamed answer text."""

    text: str


@dataclass(frozen=True)
class DoneEvent:
    """The final event for a request: the full answer plus all its metadata."""

    request_id: str
    answer: str
    citations: list[int]
    grounded: bool | None
    unsupported_claims: list[str]
    abstained: bool
    degraded: bool
    latency_ms: dict[str, float]
    tokens: dict[str, int]


@dataclass(frozen=True)
class ErrorEvent:
    """Emitted instead of DoneEvent when the request could not be completed."""

    request_id: str
    code: str
    message: str


PipelineEvent = Union[RouteEvent, RetrievalEvent, TokenEvent, DoneEvent, ErrorEvent]


def _candidate_to_dict(candidate: Candidate) -> dict[str, Any]:
    """Slim a Candidate down to what's safe to send over SSE (no raw chunk text)."""

    return {
        "chunk_uid": candidate.chunk_uid,
        "page_start": candidate.page_start,
        "page_end": candidate.page_end,
        "heading_path": candidate.heading_path,
        "dense_rank": candidate.dense_rank,
        "sparse_rank": candidate.sparse_rank,
        "fusion_score": candidate.fusion_score,
        "rerank_score": candidate.rerank_score,
    }


def _build_context(candidates: list[Candidate]) -> str:
    """Join retrieved chunks into labeled [Page N] text blocks for the generation prompt."""

    blocks = []
    for c in candidates:
        page_label = (
            f"Page {c.page_start}"
            if c.page_start == c.page_end
            else f"Pages {c.page_start}-{c.page_end}"
        )
        blocks.append(f"[{page_label}] {c.content}")
    return "\n\n".join(blocks)


async def _route_question(question: str, use_dspy: bool, task_lm: dspy.LM | None) -> tuple[str, list[str]]:
    """Classify the question into a route. Defaults to lookup on any router failure,
    so a broken router degrades the experience rather than aborting the request."""

    if not use_dspy:
        return "lookup", []

    try:
        with dspy.context(lm=task_lm):
            router = dspy.Predict(RouteQuery)
            prediction = router(question=question)
        return prediction.route, list(prediction.sub_questions)
    except Exception as exc:  # noqa: BLE001 - router failure must never abort the request
        logger.warning("router failed, defaulting to lookup route: %s", exc)
        return "lookup", []


async def _embed_question(question: str, config: Config, judge_or_task_client: genai.Client) -> list[float] | None:
    """Embed the question for retrieval. Returns None (never raises) if embedding
    fails, so the caller can degrade to sparse-only retrieval instead of aborting."""

    try:
        return await embed_query(
            judge_or_task_client, question, config.embed_model, config.embed_dim
        )
    except EmbeddingError as exc:
        logger.warning("query embedding failed, degrading to sparse only retrieval: %s", exc)
        return None


async def _generate_plain(
    context: str,
    question: str,
    task_client: genai.Client,
    task_model: str,
    usage_holder: dict[str, int],
) -> AsyncIterator[str]:
    """Yield answer token text as it streams. usage_holder is owned by the caller, a
    fresh dict per call, and is filled from the last streamed chunk's usage_metadata,
    which Gemini's streaming API reports cumulatively, so the final chunk carries the
    request's true totals.
    """

    prompt = _PLAIN_PROMPT_TEMPLATE.format(context=context, question=question)
    try:
        stream = await task_client.aio.models.generate_content_stream(
            model=task_model, contents=prompt
        )
        async for chunk in stream:
            if chunk.text:
                yield chunk.text
            if chunk.usage_metadata:
                usage_holder["input"] = chunk.usage_metadata.prompt_token_count or 0
                usage_holder["output"] = chunk.usage_metadata.candidates_token_count or 0
    except Exception as exc:  # noqa: BLE001 - mapped to LLMError for the caller to handle
        raise LLMError(f"plain generation failed: {exc}") from exc


def _extract_plain_citations(answer_text: str) -> list[int]:
    """Extract page numbers from citation groups shaped like the prompt's (p. N)
    format. The model sometimes packs more than one page into one parenthetical, for
    example (p. 131, p. 134), so this matches each p. N inside every (...) group
    rather than requiring exactly one per group.
    """

    import re

    citations: list[int] = []
    for group in re.findall(r"\(([^()]*\bp\.\s*\d+[^()]*)\)", answer_text):
        citations.extend(int(m) for m in re.findall(r"p\.\s*(\d+)", group))
    return citations


async def _generate_dspy(
    context: str,
    question: str,
    rag_program: RAGProgram,
    task_lm: dspy.LM,
    result_holder: dict[str, dspy.Prediction],
    usage_holder: dict[str, int],
) -> AsyncIterator[str]:
    """Yield answer token text as it streams, and stash the final Prediction into
    result_holder, since an async generator's return value cannot itself be
    observed by the caller's async for loop. result_holder is owned by the caller,
    a fresh dict per call, so concurrent requests never share state.

    usage_holder is filled from task_lm.history after the call, dspy.LM (via litellm)
    records a usage dict with prompt_tokens and completion_tokens per call. The router
    call also appends to this same history when use_dspy is set, so only the last
    entry, the answer predictor's call, is read here.
    """

    stream_program = dspy.streamify(
        rag_program,
        stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
    )
    try:
        with dspy.context(lm=task_lm):
            async for chunk in stream_program(context=context, question=question):
                if isinstance(chunk, dspy.streaming.StreamResponse):
                    if chunk.chunk:
                        yield chunk.chunk
                elif isinstance(chunk, dspy.Prediction):
                    result_holder["prediction"] = chunk
    except Exception as exc:  # noqa: BLE001 - mapped to LLMError for the caller to handle
        raise LLMError(f"dspy generation failed: {exc}") from exc

    if "prediction" not in result_holder:
        raise LLMError("dspy generation produced no final prediction")

    if task_lm.history:
        usage = task_lm.history[-1].get("usage") or {}
        usage_holder["input"] = usage.get("prompt_tokens", 0) or 0
        usage_holder["output"] = usage.get("completion_tokens", 0) or 0
        # litellm's usage accounting is unreliable for both the streamify path (has
        # been observed to report completion_tokens correctly but prompt_tokens as 0)
        # and cache hits (has been observed to report an empty usage dict entirely).
        # Known limitation, not fixed here: see README section 9.


async def answer_question(
    question: str,
    pipeline_config: PipelineConfig,
    config: Config,
    pool: asyncpg.Pool,
    task_client: genai.Client,
    task_lm: dspy.LM | None,
    judge_lm: dspy.LM | None = None,
    compiled_rag_program: RAGProgram | None = None,
) -> AsyncIterator[PipelineEvent]:
    """Answer one question: route, embed, retrieve, rerank, guardrail, generate,
    validate, verify, persist, yielding a PipelineEvent at each step.

    task_lm is required when pipeline_config.use_dspy is True. judge_lm, when
    provided, is used for the post-generation groundedness check via CheckGrounded so
    that check can run on JUDGE_MODEL rather than TASK_MODEL. When judge_lm is None,
    task_lm is used for grounding too.

    compiled_rag_program is the module cached once at API startup, never loaded per
    request. It is used only when pipeline_config.use_compiled is True; otherwise a
    fresh uncompiled RAGProgram is constructed per request, matching config C's intent
    of exercising the DSPy program without any optimizer output.
    """

    trace: Trace = new_trace(question)
    stages: dict[str, Any] = {}

    try:
        with StageTimer(trace, "route"):
            route, sub_queries = await _route_question(question, pipeline_config.use_dspy, task_lm)
        trace.route = route
        trace.sub_queries = sub_queries
        yield RouteEvent(route=route, sub_queries=sub_queries)

        if route == "out_of_scope":
            trace.abstained = True
            trace.answer = _ABSTENTION_MESSAGE
            await _persist_trace(pool, trace)
            yield DoneEvent(
                request_id=str(trace.request_id),
                answer=_ABSTENTION_MESSAGE,
                citations=[],
                grounded=None,
                unsupported_claims=[],
                abstained=True,
                degraded=False,
                latency_ms=trace.latency_ms,
                tokens=trace.tokens,
            )
            return

        with StageTimer(trace, "embed_query"):
            embedding = await _embed_question(question, config, task_client)
        if embedding is None:
            trace.degraded = True
            trace.degraded_reason = "query embedding failed, sparse only retrieval"

        with StageTimer(trace, "retrieval"):
            retrieval_result = await retrieve(
                pool,
                question,
                embedding,
                use_hybrid=pipeline_config.use_hybrid,
                dense_k=pipeline_config.dense_k,
                sparse_k=pipeline_config.sparse_k,
                fusion_k=pipeline_config.fusion_k,
            )
        if retrieval_result.degraded:
            trace.degraded = True
            trace.degraded_reason = retrieval_result.degraded_reason

        candidates = retrieval_result.candidates
        rerank_degraded = False
        if pipeline_config.use_rerank:
            with StageTimer(trace, "rerank"):
                candidates, rerank_degraded = await rerank_candidates(
                    question, candidates, top_k=pipeline_config.final_k
                )
            if rerank_degraded:
                trace.degraded = True
                trace.degraded_reason = "rerank failed or timed out, used fusion order"
        else:
            candidates = candidates[: pipeline_config.final_k]

        trace.record_stage(
            "retrieval",
            {
                "candidate_count": len(candidates),
                "degraded": trace.degraded,
                # Full chunk content, unlike the SSE-facing RetrievalEvent below,
                # which is deliberately kept to a slim per-candidate summary with no
                # content field. Trace.stages is never sent over SSE, so it is the
                # right place for eval/run_eval.py to read real context text for
                # RAGAS from, instead of substituting heading_path, which is not the
                # actual retrieved text.
                "context_texts": [c.content for c in candidates],
            },
        )
        yield RetrievalEvent(
            candidates=[_candidate_to_dict(c) for c in candidates[:12]],
            degraded=trace.degraded,
        )

        check_pre_generation(candidates, config.retrieval_floor)

        context = _build_context(candidates)
        answer_text = ""
        citations: list[int] = []
        usage_holder: dict[str, int] = {}

        with StageTimer(trace, "generation"):
            if pipeline_config.use_dspy:
                if task_lm is None:
                    raise LLMError("use_dspy is set but no task_lm was configured")
                if pipeline_config.use_compiled and compiled_rag_program is not None:
                    rag_program = compiled_rag_program
                else:
                    rag_program = RAGProgram()
                result_holder: dict[str, dspy.Prediction] = {}
                async for token_text in _generate_dspy(
                    context, question, rag_program, task_lm, result_holder, usage_holder
                ):
                    answer_text += token_text
                    yield TokenEvent(text=token_text)
                prediction = result_holder.get("prediction")
                if prediction is not None:
                    answer_text = prediction.answer
                    citations = [int(p) for p in prediction.citations]
            else:
                async for token_text in _generate_plain(
                    context, question, task_client, config.task_model, usage_holder
                ):
                    answer_text += token_text
                    yield TokenEvent(text=token_text)
                citations = _extract_plain_citations(answer_text)

        trace.tokens["input"] = usage_holder.get("input", 0)
        trace.tokens["output"] = usage_holder.get("output", 0)

        with StageTimer(trace, "citation_validation"):
            valid_citations, any_dropped = validate_citations(citations, candidates)
        low_confidence = any_dropped

        grounded: bool | None = None
        unsupported_claims: list[str] = []
        with StageTimer(trace, "groundedness_check"):
            try:
                grounding_lm = judge_lm or task_lm
                if grounding_lm is not None:
                    with dspy.context(lm=grounding_lm):
                        checker = dspy.Predict(CheckGrounded)
                        grounding_prediction = checker(context=context, answer=answer_text)
                    grounded = bool(grounding_prediction.grounded)
                    unsupported_claims = list(grounding_prediction.unsupported_claims)
            except Exception as exc:  # noqa: BLE001 - answer still ships, badge reads not verified
                logger.warning("groundedness check failed, answer ships unverified: %s", exc)
                grounded = None

        trace.answer = answer_text
        trace.citations = valid_citations
        trace.grounded = grounded
        trace.unsupported_claims = unsupported_claims
        if low_confidence:
            trace.record_stage("low_confidence", True)

        await _persist_trace(pool, trace)

        yield DoneEvent(
            request_id=str(trace.request_id),
            answer=answer_text,
            citations=valid_citations,
            grounded=grounded,
            unsupported_claims=unsupported_claims,
            abstained=False,
            degraded=trace.degraded,
            latency_ms=trace.latency_ms,
            tokens=trace.tokens,
        )

    except GuardrailAbstain as exc:
        trace.abstained = True
        trace.answer = _ABSTENTION_MESSAGE
        trace.record_stage("abstain_reason", exc.reason)
        await _persist_trace(pool, trace)
        yield DoneEvent(
            request_id=str(trace.request_id),
            answer=_ABSTENTION_MESSAGE,
            citations=[],
            grounded=None,
            unsupported_claims=[],
            abstained=True,
            degraded=trace.degraded,
            latency_ms=trace.latency_ms,
            tokens=trace.tokens,
        )
    except RetrievalError as exc:
        trace.error = str(exc)
        await _persist_trace(pool, trace)
        yield ErrorEvent(
            request_id=str(trace.request_id), code="retrieval_failed", message=str(exc)
        )
    except LLMError as exc:
        trace.error = str(exc)
        await _persist_trace(pool, trace)
        yield ErrorEvent(request_id=str(trace.request_id), code="llm_failed", message=str(exc))


async def _persist_trace(pool: asyncpg.Pool, trace: Trace) -> None:
    try:
        await insert_trace(pool, trace.to_db_row())
    except DatabaseError as exc:
        logger.warning("trace insert failed, continuing without it: %s", exc)
