"""Command line entry point for the query path, see CLAUDE.md Phase 2 checkpoint.

Run with: python -m scripts.answer_cli "some question" --config D_optimized

Prints the streamed answer as it arrives, then the full trace as formatted JSON, so
the Phase 2 checkpoint can be verified: that citations point at correct pages, that an
out of scope question abstains without calling the generation model, and that the
same chunk appears in both the dense and sparse candidate lists for a lookup question.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging

from dotenv import load_dotenv

import dspy

from app.config import get_config
from app.db import create_pool
from app.llm import build_client
from app.pipeline import CONFIGS, DoneEvent, ErrorEvent, RetrievalEvent, RouteEvent, TokenEvent, answer_question
from app.programs import build_lm, configure_lm_cache

logger = logging.getLogger(__name__)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Answer one question and print the full trace.")
    parser.add_argument("question", help="The question to ask")
    parser.add_argument(
        "--config",
        default="D_optimized",
        choices=sorted(CONFIGS.keys()),
        help="Ablation configuration to use",
    )
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    config = get_config()
    pipeline_config = CONFIGS[args.config]

    pool = await create_pool(config.database_url)
    task_client = build_client(config.gemini_api_key_task)

    task_lm = None
    judge_lm = None
    if pipeline_config.use_dspy:
        configure_lm_cache()
        task_lm = build_lm(config.task_model, config.gemini_api_key_task)
        dspy.configure(lm=task_lm)
        if config.gemini_api_key_judge:
            judge_lm = build_lm(config.judge_model, config.gemini_api_key_judge)

    final_event: DoneEvent | ErrorEvent | None = None
    route_event: RouteEvent | None = None
    retrieval_event: RetrievalEvent | None = None

    try:
        print(f"\n--- Answer ({args.config}) ---\n")
        async for event in answer_question(
            args.question, pipeline_config, config, pool, task_client, task_lm, judge_lm
        ):
            if isinstance(event, RouteEvent):
                route_event = event
                print(f"[route: {event.route}, sub_queries: {event.sub_queries}]\n")
            elif isinstance(event, RetrievalEvent):
                retrieval_event = event
            elif isinstance(event, TokenEvent):
                print(event.text, end="", flush=True)
            elif isinstance(event, (DoneEvent, ErrorEvent)):
                final_event = event
        print("\n")
    finally:
        await pool.close()

    trace_summary = {
        "route_event": route_event.__dict__ if route_event else None,
        "retrieval_event": retrieval_event.__dict__ if retrieval_event else None,
        "final_event": final_event.__dict__ if final_event else None,
    }
    print("--- Trace ---")
    print(json.dumps(trace_summary, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
