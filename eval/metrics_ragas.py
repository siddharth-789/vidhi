"""RAGAS metrics: LLM-judged answer quality, on top of the deterministic retrieval
metrics in eval/metrics_retrieval.py.

Faithfulness and answer relevancy depend on the generated answer, so they are
computed per configuration. Context precision and context recall depend only on
retrieval, so eval/run_eval.py caches and reuses them across configs that share a
retrieval setup: configs C and D share retrieval with B, so it is computed once.
This module does not know about that caching, it just computes over whatever rows
it is given, run_eval.py owns the cache key.

Runs on JUDGE_MODEL with the judge API key, never the task key, so evaluation traffic
never competes with serving traffic for the same quota.

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
    """One question/answer/context/ground-truth tuple to score with RAGAS."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str


async def compute_ragas_metrics(
    rows: list[RagasRow],
    config: Config,
    include_context_metrics: bool = True,
) -> dict[str, float]:
    """Compute faithfulness, answer relevancy, and optionally context precision and
    context recall, using JUDGE_MODEL and GEMINI_API_KEY_JUDGE.

    include_context_metrics is False when the caller already has cached context
    precision and recall for this retrieval configuration and only needs the
    generation-dependent metrics recomputed.
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

    dataset = Dataset.from_dict(
        {
            "question": [r.question for r in rows],
            "answer": [r.answer for r in rows],
            "contexts": [r.contexts for r in rows],
            "ground_truth": [r.ground_truth for r in rows],
        }
    )

    result = evaluate(dataset, metrics=metrics, llm=judge_llm, embeddings=judge_embeddings)
    return {name: float(value) for name, value in result.items()}


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
