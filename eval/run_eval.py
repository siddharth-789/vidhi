"""Runs one or more named pipeline configurations against the dev split and writes a
full results file plus a markdown comparison table, see CLAUDE.md section 10.

This harness calls app.pipeline.answer_question directly, the same production code
path the API uses, per CLAUDE.md section 7: evaluation must exercise production code,
not a parallel reimplementation.

Group 1 (deterministic retrieval and citation metrics) always runs on the full dev
split. Group 2 (RAGAS) defaults to a subset of dev questions, sized by --ragas-n, and
is skipped entirely with --skip-ragas. Context precision and context recall are cached
per retrieval configuration, since B, C, and D share retrieval, per section 10.

Run standalone with:
    python -m eval.run_eval --configs A_vanilla --split dev --n 5
    python -m eval.run_eval --configs A_vanilla,B_hybrid --split dev
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import get_config
from app.db import create_pool
from app.llm import build_client
from app.pipeline import CONFIGS, DoneEvent, ErrorEvent, PipelineConfig, RetrievalEvent, answer_question
from app.programs import build_lm, configure_lm_cache
from eval.dataset import EvalExample, filter_split, load_trainset
from eval.metrics_ragas import RagasRow, compute_ragas_metrics
from eval.metrics_retrieval import QuestionRecord, compute_all

logger = logging.getLogger(__name__)

_RETRIEVAL_SHARING = {
    "A_vanilla": "dense_only",
    "B_hybrid": "hybrid_rerank",
    "C_dspy": "hybrid_rerank",
    "D_optimized": "hybrid_rerank",
}


@dataclass
class PerQuestionRecord:
    example_id: str
    question: str
    category: str
    answerable: bool
    gold_pages: list[int]
    route: str
    abstained: bool
    degraded: bool
    answer: str
    citations: list[int]
    grounded: bool | None
    retrieved_page_spans: list[tuple[int, int]]
    context_texts: list[str]
    latency_ms: dict[str, float]
    tokens: dict[str, int]
    error: str | None = None


def _git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning("could not resolve git commit hash: %s", exc)
        return "unknown"


async def _run_one_question(
    example: EvalExample,
    pipeline_config: PipelineConfig,
    config,
    pool,
    task_client,
    task_lm,
    judge_lm,
    compiled_rag_program,
) -> PerQuestionRecord:
    route = "unknown"
    abstained = False
    degraded = False
    answer_text = ""
    citations: list[int] = []
    grounded: bool | None = None
    retrieved_spans: list[tuple[int, int]] = []
    context_texts: list[str] = []
    latency_ms: dict[str, float] = {}
    tokens: dict[str, int] = {}
    error: str | None = None

    async for event in answer_question(
        example.question, pipeline_config, config, pool, task_client, task_lm, judge_lm,
        compiled_rag_program,
    ):
        if isinstance(event, RetrievalEvent):
            degraded = degraded or event.degraded
            retrieved_spans = [(c["page_start"], c["page_end"]) for c in event.candidates]
            context_texts = [c.get("heading_path") or "" for c in event.candidates]
        elif isinstance(event, DoneEvent):
            abstained = event.abstained
            degraded = degraded or event.degraded
            answer_text = event.answer
            citations = event.citations
            grounded = event.grounded
            latency_ms = event.latency_ms
            tokens = event.tokens
        elif isinstance(event, ErrorEvent):
            error = event.message
        else:
            route = getattr(event, "route", route)

    return PerQuestionRecord(
        example_id=example.id,
        question=example.question,
        category=example.category,
        answerable=example.answerable,
        gold_pages=example.gold_pages,
        route=route,
        abstained=abstained,
        degraded=degraded,
        answer=answer_text,
        citations=citations,
        grounded=grounded,
        retrieved_page_spans=retrieved_spans,
        context_texts=context_texts,
        latency_ms=latency_ms,
        tokens=tokens,
        error=error,
    )


def _checkpoint_path(config_name: str, split: str) -> Path:
    """Stable (non timestamped) per config, per split checkpoint path, so a run
    interrupted by key rotation or a quota exhaustion mid run can resume instead of
    restarting from question 1. Distinct from the timestamped eval/results/{config}_
    {timestamp}.json final output, which is written once a run's records are complete.
    """

    return Path("eval/results") / f".checkpoint_{config_name}_{split}.jsonl"


def _load_checkpoint(config_name: str, split: str) -> dict[str, "PerQuestionRecord"]:
    path = _checkpoint_path(config_name, split)
    if not path.exists():
        return {}

    done: dict[str, PerQuestionRecord] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            payload["retrieved_page_spans"] = [
                tuple(span) for span in payload["retrieved_page_spans"]
            ]
            done[payload["example_id"]] = PerQuestionRecord(**payload)
    return done


def _append_checkpoint(config_name: str, split: str, record: "PerQuestionRecord") -> None:
    path = _checkpoint_path(config_name, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record.__dict__) + "\n")


def _clear_checkpoint(config_name: str, split: str) -> None:
    path = _checkpoint_path(config_name, split)
    if path.exists():
        path.unlink()


async def run_config(
    config_name: str,
    examples: list[EvalExample],
    ragas_n: int,
    skip_ragas: bool,
    ragas_cache: dict[str, dict[str, float]],
    split: str,
    resume: bool,
) -> dict[str, Any]:
    pipeline_config = CONFIGS[config_name]
    app_config = get_config()

    done_records = _load_checkpoint(config_name, split) if resume else {}
    if done_records:
        logger.info(
            "resuming %s: %d of %d questions already checkpointed",
            config_name,
            len(done_records),
            len(examples),
        )

    pool = await create_pool(app_config.database_url)
    task_client = build_client(app_config.gemini_api_key_task)

    task_lm = None
    judge_lm = None
    compiled_rag_program = None
    if pipeline_config.use_dspy:
        configure_lm_cache()
        task_lm = build_lm(app_config.task_model, app_config.gemini_api_key_task)
        if app_config.gemini_api_key_judge:
            judge_lm = build_lm(app_config.judge_model, app_config.gemini_api_key_judge)
        if pipeline_config.use_compiled:
            import os

            from app.programs import RAGProgram

            if os.path.exists(app_config.compiled_program_path):
                compiled_rag_program = RAGProgram()
                compiled_rag_program.load(app_config.compiled_program_path)
            else:
                logger.warning(
                    "compiled program not found at %s, D_optimized will fall back to "
                    "the uncompiled program",
                    app_config.compiled_program_path,
                )

    try:
        records = []
        for example in examples:
            cached = done_records.get(example.id)
            if cached is not None:
                records.append(cached)
                continue

            record = await _run_one_question(
                example, pipeline_config, app_config, pool, task_client, task_lm, judge_lm,
                compiled_rag_program,
            )
            records.append(record)
            _append_checkpoint(config_name, split, record)
            logger.info(
                "%s: %s -> abstained=%s degraded=%s citations=%s",
                config_name,
                example.id,
                record.abstained,
                record.degraded,
                record.citations,
            )

        _clear_checkpoint(config_name, split)

        retrieval_records = [
            QuestionRecord(
                gold_pages=r.gold_pages,
                answerable=r.answerable,
                abstained=r.abstained,
                retrieved_page_spans=r.retrieved_page_spans,
                predicted_citations=r.citations,
            )
            for r in records
        ]
        group1 = compute_all(retrieval_records)

        group2: dict[str, float] = {}
        if not skip_ragas and app_config.gemini_api_key_judge:
            ragas_subset = [r for r in records if not r.abstained and r.answer][:ragas_n]
            share_key = _RETRIEVAL_SHARING.get(config_name, config_name)
            cached_context_scores = ragas_cache.get(share_key)
            include_context = cached_context_scores is None

            ragas_rows = [
                RagasRow(
                    question=r.question,
                    answer=r.answer,
                    contexts=r.context_texts or [""],
                    ground_truth=next(
                        (e.gold_answer for e in examples if e.id == r.example_id), ""
                    ),
                )
                for r in ragas_subset
            ]
            group2 = await compute_ragas_metrics(
                ragas_rows, app_config, include_context_metrics=include_context
            )
            if include_context:
                ragas_cache[share_key] = {
                    k: v
                    for k, v in group2.items()
                    if k in ("context_precision", "context_recall")
                }
            else:
                group2.update(cached_context_scores)

        return {
            "config": config_name,
            "n": len(records),
            "group1_retrieval": group1,
            "group2_ragas": group2,
            "records": [r.__dict__ for r in records],
        }
    finally:
        await pool.close()


def _markdown_table(results: list[dict[str, Any]]) -> str:
    metric_keys = ["recall_at_5", "recall_at_10", "mrr_at_10", "citation_validity_rate",
                   "abstention_accuracy", "false_abstention_rate", "answer_page_overlap"]
    ragas_keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    header = "| Config | " + " | ".join(metric_keys + ragas_keys) + " |"
    separator = "|" + "---|" * (len(metric_keys) + len(ragas_keys) + 1)
    rows = [header, separator]
    for result in results:
        g1 = result["group1_retrieval"]
        g2 = result["group2_ragas"]
        cells = [result["config"]]
        cells += [f"{g1.get(k, 0.0):.3f}" for k in metric_keys]
        cells += [f"{g2.get(k, 0.0):.3f}" if k in g2 else "TBD" for k in ragas_keys]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", required=True, help="comma separated config names")
    parser.add_argument("--split", default="dev", choices=["train", "dev"])
    parser.add_argument("--n", type=int, default=None, help="limit number of questions")
    parser.add_argument("--ragas-n", type=int, default=25)
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--trainset", default="data/trainset.csv")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip questions already checkpointed from a prior interrupted run of "
        "the same config and split",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    from dotenv import load_dotenv

    load_dotenv()

    config_names = [c.strip() for c in args.configs.split(",")]
    for name in config_names:
        if name not in CONFIGS:
            raise SystemExit(f"unknown config {name!r}, valid names: {sorted(CONFIGS)}")

    all_examples = load_trainset(args.trainset)
    examples = filter_split(all_examples, args.split)
    if args.n is not None:
        examples = examples[: args.n]
    if not examples:
        raise SystemExit(f"no examples found for split {args.split!r}")

    commit_hash = _git_commit_hash()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    app_config = get_config()
    results = []
    ragas_cache: dict[str, dict[str, float]] = {}
    for config_name in config_names:
        logger.info("running config %s on %d %s questions", config_name, len(examples), args.split)
        result = await run_config(
            config_name, examples, args.ragas_n, args.skip_ragas, ragas_cache,
            args.split, args.resume,
        )
        result["model_ids"] = {
            "task_model": app_config.task_model,
            "judge_model": app_config.judge_model,
            "embed_model": app_config.embed_model,
        }
        result["git_commit"] = commit_hash
        result["timestamp"] = timestamp
        results.append(result)

        out_path = Path("eval/results") / f"{config_name}_{timestamp}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("wrote %s", out_path)

    print()
    print(_markdown_table(results))


if __name__ == "__main__":
    asyncio.run(main())
