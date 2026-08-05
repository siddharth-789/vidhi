"""FastAPI app: the HTTP surface of the project, and its one exception boundary.

One service serves the JSON and SSE API and the single static HTML page. The lifespan
context manager handles startup validation, database pool creation and warming, and
compiled DSPy program loading, all before the first request is served.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import asyncpg
import dspy
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from app.config import Config, get_config, require_serving_config
from app.db import create_pool, get_chunk_count, get_trace, warm_pool
from app.errors import ConfigError, DatabaseError
from app.llm import build_client
from app.pipeline import CONFIGS, DoneEvent, ErrorEvent, RetrievalEvent, RouteEvent, TokenEvent, answer_question
from app.programs import build_lm, configure_lm_cache

logger = logging.getLogger(__name__)

_HEARTBEAT_INTERVAL_SECONDS = 10.0

_state: dict[str, Any] = {}


def _git_commit_hash() -> str:
    """Resolve the running commit hash, preferring the build-time GIT_SHA env var."""

    env_sha = os.environ.get("GIT_SHA")
    if env_sha:
        return env_sha
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning("could not resolve git commit hash: %s", exc)
        return "unknown"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Validate config, open and warm the database pool, load DSPy LMs and the
    compiled program if present, and cache the chunk count, all before serving."""

    config = get_config()
    require_serving_config(config)

    pool = await create_pool(config.database_url)
    await warm_pool(pool)

    task_client = build_client(config.gemini_api_key_task)

    task_lm: dspy.LM | None = None
    judge_lm: dspy.LM | None = None
    compiled_rag_program = None
    compiled_loaded = False
    try:
        configure_lm_cache()
        task_lm = build_lm(config.task_model, config.gemini_api_key_task)
        if config.gemini_api_key_judge:
            judge_lm = build_lm(config.judge_model, config.gemini_api_key_judge)
    except Exception as exc:  # noqa: BLE001 - DSPy setup failure must not block plain configs
        logger.warning("failed to configure DSPy LMs, dspy backed configs will fail: %s", exc)

    if os.path.exists(config.compiled_program_path):
        try:
            from app.programs import RAGProgram

            compiled_rag_program = RAGProgram()
            compiled_rag_program.load(config.compiled_program_path)
            compiled_loaded = True
        except Exception as exc:  # noqa: BLE001 - startup must not crash on a bad artifact
            logger.warning("failed to load compiled program, falling back to uncompiled: %s", exc)

    chunk_count = await get_chunk_count(pool)

    _state.update(
        {
            "config": config,
            "pool": pool,
            "task_client": task_client,
            "task_lm": task_lm,
            "judge_lm": judge_lm,
            "compiled_rag_program": compiled_rag_program,
            "compiled_loaded": compiled_loaded,
            "chunk_count": chunk_count,
            "git_commit": _git_commit_hash(),
            "started_at": time.time(),
        }
    )

    logger.info(
        "startup complete: chunk_count=%d compiled_loaded=%s",
        chunk_count,
        compiled_loaded,
    )

    yield

    await pool.close()


app = FastAPI(title="Bharat's GST Smart Guide RAG", lifespan=lifespan)


class AskRequest(BaseModel):
    """Request body for POST /ask: the question plus which ablation config to use."""

    question: str = Field(..., min_length=1, max_length=1000)
    config: str

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value

    @field_validator("config")
    @classmethod
    def _config_known(cls, value: str) -> str:
        if value not in CONFIGS:
            raise ValueError(f"unknown config {value!r}, valid names: {sorted(CONFIGS)}")
        return value


def _sse_frame(event: str, data: dict[str, Any]) -> str:
    """Format one Server-Sent Events frame. JSON must stay single-line: an embedded
    newline inside data: would break the frame."""

    payload = json.dumps(data, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"


async def _event_stream(question: str, config_name: str) -> AsyncIterator[str]:
    """Run the pipeline and translate each PipelineEvent into an SSE frame, inserting
    a heartbeat comment if more than 10 seconds pass with no event (keeps proxies
    from timing out an idle connection while reranking or generation is slow)."""

    config: Config = _state["config"]
    pool: asyncpg.Pool = _state["pool"]
    pipeline_config = CONFIGS[config_name]

    task_lm = _state["task_lm"] if pipeline_config.use_dspy else None
    judge_lm = _state["judge_lm"]
    compiled_rag_program = _state.get("compiled_rag_program") if pipeline_config.use_compiled else None

    last_yield = time.perf_counter()

    try:
        async for event in answer_question(
            question,
            pipeline_config,
            config,
            pool,
            _state["task_client"],
            task_lm,
            judge_lm,
            compiled_rag_program,
        ):
            now = time.perf_counter()
            if now - last_yield > _HEARTBEAT_INTERVAL_SECONDS:
                yield ": heartbeat\n\n"
            last_yield = now

            if isinstance(event, RouteEvent):
                yield _sse_frame(
                    "route", {"route": event.route, "sub_queries": event.sub_queries}
                )
            elif isinstance(event, RetrievalEvent):
                yield _sse_frame(
                    "retrieval",
                    {"candidates": event.candidates, "degraded": event.degraded},
                )
            elif isinstance(event, TokenEvent):
                yield _sse_frame("token", {"text": event.text})
            elif isinstance(event, DoneEvent):
                yield _sse_frame(
                    "done",
                    {
                        "request_id": event.request_id,
                        "answer": event.answer,
                        "citations": event.citations,
                        "grounded": event.grounded,
                        "unsupported_claims": event.unsupported_claims,
                        "abstained": event.abstained,
                        "degraded": event.degraded,
                        "latency_ms": event.latency_ms,
                        "tokens": event.tokens,
                    },
                )
            elif isinstance(event, ErrorEvent):
                yield _sse_frame(
                    "error",
                    {
                        "request_id": event.request_id,
                        "code": event.code,
                        "message": event.message,
                    },
                )
    except Exception as exc:  # noqa: BLE001 - stream must end cleanly, never abandon the client
        logger.error("unhandled error in event stream: %s", exc)
        yield _sse_frame(
            "error",
            {"request_id": str(uuid.uuid4()), "code": "internal_error", "message": str(exc)},
        )


@app.post("/ask")
async def ask(request: AskRequest) -> StreamingResponse:
    """Stream an answer to one question as Server-Sent Events."""

    return StreamingResponse(
        _event_stream(request.question, request.config),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/healthz")
async def healthz() -> JSONResponse:
    """Liveness check: reports model IDs, chunk count, and whether the compiled
    program was found. Fails with a non-200 status if the database is unreachable or
    no chunks are loaded, so this doubles as the post-deploy smoke test target."""

    config: Config = _state.get("config")
    pool: asyncpg.Pool | None = _state.get("pool")

    if config is None or pool is None:
        return JSONResponse(status_code=503, content={"error": "not initialized"})

    try:
        chunk_count = await get_chunk_count(pool)
    except DatabaseError as exc:
        return JSONResponse(status_code=503, content={"error": f"database unreachable: {exc}"})

    body = {
        "git_commit": _state["git_commit"],
        "task_model": config.task_model,
        "judge_model": config.judge_model,
        "embed_model": config.embed_model,
        "embed_dim": config.embed_dim,
        "chunk_count": chunk_count,
        "compiled": _state.get("compiled_loaded", False),
    }

    if chunk_count == 0:
        return JSONResponse(status_code=503, content=body)

    return JSONResponse(status_code=200, content=body)


@app.get("/trace/{request_id}")
async def get_trace_endpoint(request_id: str) -> JSONResponse:
    """Return the full stored trace for one past request, for debugging and demos."""

    try:
        parsed_id = uuid.UUID(request_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="request_id must be a valid UUID")

    pool: asyncpg.Pool = _state["pool"]
    row = await get_trace(pool, parsed_id)
    if row is None:
        raise HTTPException(status_code=404, detail="trace not found")

    return JSONResponse(content=dict(row), default=str)


@app.get("/program")
async def get_program() -> JSONResponse:
    """Return the currently active instruction text and demo count for each DSPy
    predictor, so a reader can see exactly what the optimizer produced (or, if
    nothing is compiled yet, the original hand-written instructions)."""

    compiled_rag_program = _state.get("compiled_rag_program")
    compiled_loaded = _state.get("compiled_loaded", False)

    if compiled_loaded and compiled_rag_program is not None:
        predictors = {
            name: {
                "instructions": predictor.signature.instructions,
                "demo_count": len(predictor.demos),
            }
            for name, predictor in compiled_rag_program.named_predictors()
        }
        return JSONResponse(content={"compiled": True, "predictors": predictors})

    from app.programs import AnswerFromManual, RouteQuery

    return JSONResponse(
        content={
            "compiled": False,
            "predictors": {
                "route": {"instructions": RouteQuery.__doc__ or "", "demo_count": 0},
                "answer": {"instructions": AnswerFromManual.__doc__ or "", "demo_count": 0},
            },
        }
    )


@app.get("/configs")
async def get_configs() -> JSONResponse:
    """List the four ablation configs and their flags, for the UI's config selector."""

    return JSONResponse(
        content={
            name: {
                "use_hybrid": cfg.use_hybrid,
                "use_rerank": cfg.use_rerank,
                "use_dspy": cfg.use_dspy,
                "use_compiled": cfg.use_compiled,
            }
            for name, cfg in CONFIGS.items()
        }
    )


_RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eval", "results")


def _latest_result_file(config_name: str) -> str | None:
    """Find the most recently written eval result file for one config, if any."""

    matches = sorted(glob.glob(os.path.join(_RESULTS_DIR, f"{config_name}_*.json")))
    return matches[-1] if matches else None


def _nan_to_none(value: Any) -> Any:
    """Recursively convert NaN floats to null, since JSON has no NaN literal."""

    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, dict):
        return {k: _nan_to_none(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_nan_to_none(v) for v in value]
    return value


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return _nan_to_none(json.load(f))


@app.get("/metrics")
async def get_metrics() -> JSONResponse:
    """Serve the latest eval, latency, and optimizer result files as one JSON blob,
    for the metrics dashboard page."""

    configs: dict[str, Any] = {}
    for name in CONFIGS:
        config_entry: dict[str, Any] = {}

        result_path = _latest_result_file(name)
        if result_path is not None:
            try:
                data = _load_json(result_path)
                config_entry["group1_retrieval"] = data.get("group1_retrieval")
                config_entry["group2_ragas"] = data.get("group2_ragas")
                config_entry["n"] = data.get("n")
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("failed to read eval result %s: %s", result_path, exc)

        latency_path = os.path.join(_RESULTS_DIR, f"latency_{name}.json")
        if os.path.exists(latency_path):
            try:
                config_entry["latency"] = _load_json(latency_path)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("failed to read latency result %s: %s", latency_path, exc)

        configs[name] = config_entry or None

    optimizer: Any = None
    optimize_log_path = os.path.join(_RESULTS_DIR, "optimize_log.json")
    if os.path.exists(optimize_log_path):
        try:
            optimizer = _load_json(optimize_log_path)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("failed to read optimize log %s: %s", optimize_log_path, exc)

    return JSONResponse(content={"configs": configs, "optimizer": optimizer})


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the single-page UI, with no-cache so a redeploy is never masked by a
    stale cached page."""

    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content, headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")))


@app.exception_handler(Exception)
async def _exception_boundary(request: Request, exc: Exception) -> JSONResponse:
    """The one exception boundary in the project.

    Maps typed exceptions to status codes and always returns a structured JSON body
    carrying request_id, so a client-visible error is never an opaque 500 with no
    correlation handle.
    """

    request_id = str(uuid.uuid4())

    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"request_id": request_id, "error": exc.detail},
        )
    if isinstance(exc, ConfigError):
        logger.error("config error [%s]: %s", request_id, exc)
        return JSONResponse(
            status_code=503, content={"request_id": request_id, "error": str(exc)}
        )
    if isinstance(exc, DatabaseError):
        logger.error("database error [%s]: %s", request_id, exc)
        return JSONResponse(
            status_code=503, content={"request_id": request_id, "error": str(exc)}
        )

    logger.error("unhandled exception [%s]: %s", request_id, exc)
    return JSONResponse(
        status_code=500, content={"request_id": request_id, "error": "internal server error"}
    )
