"""RAGAS metrics, expensive, see CLAUDE.md section 10 group 2.

Faithfulness and answer relevancy depend on the generated answer, so they are
computed per configuration. Context precision and context recall depend only on
retrieval, so eval/run_eval.py caches and reuses them across configs that share a
retrieval setup, per CLAUDE.md section 10: configs C and D share retrieval with B,
compute once. This module does not know about that caching, it just computes over
whatever rows it is given, run_eval.py owns the cache key.

Runs on JUDGE_MODEL with the judge API key, never the task key, see CLAUDE.md section 3.

Run standalone with: python -m eval.metrics_ragas
This runs RAGAS over a tiny synthetic sample so the wiring can be checked without
touching the real dataset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import Config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RagasRow:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str


async def compute_ragas_metrics(
    rows: list[RagasRow],
    config: Config,
    include_context_metrics: bool = True,
) -> dict[str, float]:
    """Computes faithfulness, answer relevancy, and optionally context precision and
    context recall, using JUDGE_MODEL and GEMINI_API_KEY_JUDGE.

    include_context_metrics is False when the caller already has cached context
    precision and recall for this retrieval configuration and only needs the
    generation dependent metrics recomputed.
    """

    if not rows:
        return {}
    if not config.gemini_api_key_judge:
        logger.warning("GEMINI_API_KEY_JUDGE is not set, skipping RAGAS metrics")
        return {}

    from datasets import Dataset
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.run_config import RunConfig

    dataset = Dataset.from_dict(
        {
            "question": [r.question for r in rows],
            "answer": [r.answer for r in rows],
            "contexts": [r.contexts for r in rows],
            "ground_truth": [r.ground_truth for r in rows],
        }
    )

    judge_llm = ChatGoogleGenerativeAI(
        model=config.judge_model, google_api_key=config.gemini_api_key_judge
    )
    # langchain_google_genai requires the models/ prefixed name, unlike
    # google.genai.Client used everywhere else in this project, see app/llm.py.
    judge_embeddings = GoogleGenerativeAIEmbeddings(
        model=f"models/{config.embed_model}", google_api_key=config.gemini_api_key_judge
    )

    metrics = [faithfulness, answer_relevancy]
    if include_context_metrics:
        metrics += [context_precision, context_recall]

    # ragas defaults to max_workers=16, which fires far more concurrent judge calls
    # than the Gemini free tier's 15 requests per minute ceiling can absorb. Left at
    # the default, every row's worth of sub-calls fires at once, all hit 429, and
    # ragas's own retry layer keeps retrying the same burst rather than backing off to
    # a rate the API accepts, so the run barely progresses. max_workers=2 keeps calls
    # trickling through at a pace the per minute quota tolerates.
    run_config = RunConfig(max_workers=2, max_retries=20, max_wait=90)

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
    )

    # EvaluationResult has no __contains__, and its __getitem__ takes a metric name
    # (not a row index), so `metric_name in result` silently falls back to Python's
    # iteration protocol and raises KeyError on the first row instead of behaving like
    # a dict membership check. result.scores is a list of per-row dicts; its first
    # row's keys are the metric names actually present.
    present_metrics = result.scores[0].keys() if result.scores else []
    scores: dict[str, float] = {}
    for metric_name in present_metrics:
        values = [row[metric_name] for row in result.scores if row.get(metric_name) is not None]
        scores[metric_name] = sum(values) / len(values) if values else 0.0
    return scores


if __name__ == "__main__":
    import asyncio

    from dotenv import load_dotenv

    from app.config import get_config

    async def _main() -> None:
        load_dotenv()
        cfg = get_config()
        sample_rows = [
            RagasRow(
                question="What is the GST rate for cotton textiles?",
                answer="The GST rate for cotton textiles is 5 percent (p. 42).",
                contexts=["[Page 42] The GST rate for cotton textiles is 5 percent."],
                ground_truth="5 percent",
            )
        ]
        scores = await compute_ragas_metrics(sample_rows, cfg)
        print(scores)

    asyncio.run(_main())
