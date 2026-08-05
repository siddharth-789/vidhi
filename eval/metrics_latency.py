"""Production and latency metrics: how fast the service actually responds, end to end.

Warm and cold measurements are always kept in separate result sets, never blended,
because a blended p95 describes nothing real: cold start only happens after the
service has been idle long enough to scale to zero.

Run standalone with:
    python -m eval.metrics_latency warm --url http://localhost:8000 --n 10
    python -m eval.metrics_latency cold --url https://<service-url> --n 5 --wait-seconds 900

Cold mode is a manual measurement tool. It does not force Cloud Run to scale to zero,
it assumes the caller has waited out the idle window (or passes --wait-seconds to
sleep first).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RequestTiming:
    """Timing and token counts for one /ask request."""

    time_to_first_byte_ms: float
    time_to_first_token_ms: float | None
    total_wall_clock_ms: float
    stage_latency_ms: dict[str, float]
    input_tokens: int
    output_tokens: int
    output_token_count_streamed: int


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile of a list of values."""

    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


async def time_one_request(
    client: httpx.AsyncClient, base_url: str, question: str, config_name: str
) -> RequestTiming:
    """Issue one /ask request and measure time to first byte, time to first token,
    and total wall clock by consuming the SSE stream exactly as app/static/index.html
    does.
    """

    start = time.perf_counter()
    first_byte_ms: float | None = None
    first_token_ms: float | None = None
    stage_latency_ms: dict[str, float] = {}
    tokens: dict[str, int] = {}
    streamed_token_count = 0

    buffer = ""
    async with client.stream(
        "POST", f"{base_url}/ask", json={"question": question, "config": config_name}
    ) as response:
        async for chunk in response.aiter_text():
            if first_byte_ms is None:
                first_byte_ms = (time.perf_counter() - start) * 1000
            buffer += chunk
            while "\n\n" in buffer:
                frame, buffer = buffer.split("\n\n", 1)
                if not frame or frame.startswith(":"):
                    continue
                event_name = None
                data_line = None
                for line in frame.split("\n"):
                    if line.startswith("event:"):
                        event_name = line[len("event:") :].strip()
                    elif line.startswith("data:"):
                        data_line = line[len("data:") :].strip()
                if data_line is None:
                    continue
                import json

                try:
                    payload = json.loads(data_line)
                except json.JSONDecodeError:
                    logger.warning("malformed SSE frame, skipping: %s", frame)
                    continue

                if event_name == "token" and first_token_ms is None:
                    first_token_ms = (time.perf_counter() - start) * 1000
                if event_name == "token":
                    streamed_token_count += 1
                if event_name == "done":
                    stage_latency_ms = payload.get("latency_ms", {})
                    tokens = payload.get("tokens", {})

    total_ms = (time.perf_counter() - start) * 1000
    return RequestTiming(
        time_to_first_byte_ms=first_byte_ms or total_ms,
        time_to_first_token_ms=first_token_ms,
        total_wall_clock_ms=total_ms,
        stage_latency_ms=stage_latency_ms,
        input_tokens=tokens.get("input", 0),
        output_tokens=tokens.get("output", 0),
        output_token_count_streamed=streamed_token_count,
    )


@dataclass
class LatencyReport:
    """Aggregated latency stats (percentiles, per-stage breakdown) over N requests."""

    mode: str
    n: int
    ttft_p50_ms: float
    ttft_p95_ms: float
    total_p50_ms: float
    total_p95_ms: float
    per_stage_mean_ms: dict[str, float]
    per_stage_p95_ms: dict[str, float]
    tokens_per_second_mean: float
    input_tokens_mean: float
    output_tokens_mean: float

    def to_dict(self) -> dict:
        """Serialize the report to a plain dict for JSON output."""

        return {
            "mode": self.mode,
            "n": self.n,
            "ttft_p50_ms": self.ttft_p50_ms,
            "ttft_p95_ms": self.ttft_p95_ms,
            "total_p50_ms": self.total_p50_ms,
            "total_p95_ms": self.total_p95_ms,
            "per_stage_mean_ms": self.per_stage_mean_ms,
            "per_stage_p95_ms": self.per_stage_p95_ms,
            "tokens_per_second_mean": self.tokens_per_second_mean,
            "input_tokens_mean": self.input_tokens_mean,
            "output_tokens_mean": self.output_tokens_mean,
        }


def summarize(mode: str, timings: list[RequestTiming]) -> LatencyReport:
    """Aggregate a list of per-request timings into percentiles and per-stage stats."""

    ttfts = [t.time_to_first_token_ms for t in timings if t.time_to_first_token_ms is not None]
    totals = [t.total_wall_clock_ms for t in timings]

    stage_names: set[str] = set()
    for t in timings:
        stage_names.update(t.stage_latency_ms.keys())

    per_stage_mean = {}
    per_stage_p95 = {}
    for name in stage_names:
        values = [t.stage_latency_ms[name] for t in timings if name in t.stage_latency_ms]
        per_stage_mean[name] = sum(values) / len(values) if values else 0.0
        per_stage_p95[name] = percentile(values, 95)

    tokens_per_second = []
    for t in timings:
        gen_ms = t.stage_latency_ms.get("generation")
        if gen_ms and gen_ms > 0 and t.output_token_count_streamed > 0:
            tokens_per_second.append(t.output_token_count_streamed / (gen_ms / 1000))

    return LatencyReport(
        mode=mode,
        n=len(timings),
        ttft_p50_ms=percentile(ttfts, 50),
        ttft_p95_ms=percentile(ttfts, 95),
        total_p50_ms=percentile(totals, 50),
        total_p95_ms=percentile(totals, 95),
        per_stage_mean_ms=per_stage_mean,
        per_stage_p95_ms=per_stage_p95,
        tokens_per_second_mean=(
            sum(tokens_per_second) / len(tokens_per_second) if tokens_per_second else 0.0
        ),
        input_tokens_mean=(
            sum(t.input_tokens for t in timings) / len(timings) if timings else 0.0
        ),
        output_tokens_mean=(
            sum(t.output_tokens for t in timings) / len(timings) if timings else 0.0
        ),
    )


async def run_warm(base_url: str, n: int, question: str, config_name: str) -> LatencyReport:
    """Fire N requests back to back against an already-warm instance."""

    timings = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        for i in range(n):
            timing = await time_one_request(client, base_url, question, config_name)
            timings.append(timing)
            logger.info("warm request %d/%d: total=%.0fms", i + 1, n, timing.total_wall_clock_ms)
    return summarize("warm", timings)


async def run_cold(
    base_url: str, n: int, question: str, config_name: str, wait_seconds: float
) -> LatencyReport:
    """Measure cold start by waiting wait_seconds before each request, so the
    instance has a chance to scale to zero. Reported separately from warm numbers,
    never blended.
    """

    timings = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        for i in range(n):
            logger.info("waiting %.0fs for scale to zero before cold request %d/%d",
                        wait_seconds, i + 1, n)
            await asyncio.sleep(wait_seconds)
            timing = await time_one_request(client, base_url, question, config_name)
            timings.append(timing)
            logger.info("cold request %d/%d: total=%.0fms", i + 1, n, timing.total_wall_clock_ms)
    return summarize("cold", timings)


def estimate_cost_per_thousand(
    input_tokens_mean: float,
    output_tokens_mean: float,
    input_price_per_million: float,
    output_price_per_million: float,
) -> float:
    """Estimated cost per thousand queries at a published paid-tier rate.

    Prices are passed in explicitly rather than hardcoded, since a rate baked into
    code goes stale silently. Callers should pass the rate from the current AI
    Studio pricing page.
    """

    cost_per_query = (
        input_tokens_mean / 1_000_000 * input_price_per_million
        + output_tokens_mean / 1_000_000 * output_price_per_million
    )
    return cost_per_query * 1000


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["warm", "cold"])
    parser.add_argument("--url", required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--wait-seconds", type=float, default=0.0)
    parser.add_argument("--question", default="What is the GST rate for cotton textiles?")
    parser.add_argument("--config", default="D_optimized")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.mode == "warm":
        report = await run_warm(args.url, args.n, args.question, args.config)
    else:
        report = await run_cold(
            args.url, args.n, args.question, args.config, args.wait_seconds
        )

    import json

    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
