"""Trace object accumulated across the query path, see CLAUDE.md section 7 and the
traces table in sql/schema.sql.

Every stage of the query path writes timing into this object, see CLAUDE.md section 2.
StageTimer is a small context manager so app/pipeline.py can wrap each stage without
repeating perf_counter bookkeeping at every call site.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Trace:
    request_id: uuid.UUID
    query: str
    route: str | None = None
    sub_queries: list[str] = field(default_factory=list)
    stages: dict[str, Any] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)
    answer: str | None = None
    citations: list[int] = field(default_factory=list)
    grounded: bool | None = None
    unsupported_claims: list[str] = field(default_factory=list)
    abstained: bool = False
    degraded: bool = False
    degraded_reason: str | None = None
    error: str | None = None

    def record_stage(self, name: str, payload: Any) -> None:
        self.stages[name] = payload

    def to_db_row(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "query": self.query,
            "route": self.route,
            "sub_queries": self.sub_queries,
            "stages": self.stages,
            "latency_ms": self.latency_ms,
            "tokens": self.tokens,
            "answer": self.answer,
            "citations": self.citations,
            "grounded": self.grounded,
            "abstained": self.abstained,
            "degraded": self.degraded,
            "error": self.error,
        }

    def to_response_dict(self) -> dict[str, Any]:
        return {
            "request_id": str(self.request_id),
            "answer": self.answer,
            "citations": self.citations,
            "grounded": self.grounded,
            "unsupported_claims": self.unsupported_claims,
            "abstained": self.abstained,
            "degraded": self.degraded,
            "latency_ms": self.latency_ms,
            "tokens": self.tokens,
        }


class StageTimer:
    """Context manager that records elapsed milliseconds for one stage on a Trace.

    Usage:
        async with StageTimer(trace, "dense_search"):
            rows = await dense_search(...)
    """

    def __init__(self, trace: Trace, stage_name: str) -> None:
        self._trace = trace
        self._stage_name = stage_name
        self._start: float = 0.0

    def __enter__(self) -> "StageTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        self._trace.latency_ms[self._stage_name] = elapsed_ms

    async def __aenter__(self) -> "StageTimer":
        self._start = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        self._trace.latency_ms[self._stage_name] = elapsed_ms


def new_trace(query: str) -> Trace:
    return Trace(request_id=uuid.uuid4(), query=query)


if __name__ == "__main__":
    import asyncio

    async def _demo() -> None:
        trace = new_trace("What is the GST rate for cotton textiles?")
        async with StageTimer(trace, "embed_query"):
            await asyncio.sleep(0.01)
        trace.route = "lookup"
        trace.record_stage("retrieval", {"candidate_count": 5})
        print(trace.latency_ms)
        print(trace.to_response_dict())

    asyncio.run(_demo())
