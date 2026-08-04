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
    seconds_between_calls: float = 5.0,
) -> dict[str, float]:
    """Computes faithfulness, answer relevancy, and optionally context precision and
    context recall, using JUDGE_MODEL and GEMINI_API_KEY_JUDGE.

    include_context_metrics is False when the caller already has cached context
    precision and recall for this retrieval configuration and only needs the
    generation dependent metrics recomputed.

    Runs one metric against one row per ragas.evaluate call, with an explicit sleep
    between calls, rather than handing ragas a batch to fan out internally. This was
    forced by observed behavior, not a style choice: ragas's RunConfig max_workers
    controls how many judge calls it fires concurrently, but the actual 429 recovery
    is handled by langchain_google_genai's own tenacity retry wrapper underneath,
    which retries after a short fixed delay regardless of the server's suggested
    retry_delay (sometimes 59s on this project's free tier). The result, verified by
    running this against the live API multiple times, is every combination of
    max_workers from 16 down to 1 either stalls indefinitely or makes only a few
    sub-evaluations of progress per 20 minutes, because the wrapper keeps re-triggering
    the same 429 before the quota window actually clears. Explicit sleeps here are
    outside that retry wrapper entirely and reliably let the per minute window clear.
    """

    if not rows:
        return {}
    if not config.gemini_api_key_judge:
        logger.warning("GEMINI_API_KEY_JUDGE is not set, skipping RAGAS metrics")
        return {}

    import asyncio

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

    run_config = RunConfig(max_workers=1, max_retries=3, max_wait=30)

    def _evaluate_one_row_one_metric(row: RagasRow, metric) -> float | None:
        # ragas.evaluate calls asyncio.run() internally (see ragas/executor.py), which
        # cannot be called from inside an already-running event loop, even with
        # nest_asyncio applied, observed to crash with "pop from an empty deque" on the
        # second call in this coroutine. Running it in a worker thread via
        # asyncio.to_thread below gives it a fresh thread with no running loop, which is
        # what actually made repeated calls reliable. time.sleep, not asyncio.sleep, is
        # used for pacing because this whole function runs inside that worker thread.
        dataset = Dataset.from_dict(
            {
                "question": [row.question],
                "answer": [row.answer],
                "contexts": [row.contexts],
                "ground_truth": [row.ground_truth],
            }
        )
        try:
            result = evaluate(
                dataset,
                metrics=[metric],
                llm=judge_llm,
                embeddings=judge_embeddings,
                run_config=run_config,
            )
        except Exception as exc:  # noqa: BLE001 - one bad row must not lose the rest
            logger.warning("ragas metric %s failed for one row, skipping: %s", metric.name, exc)
            return None

        if result.scores:
            return result.scores[0].get(metric.name)
        return None

    per_metric_scores: dict[str, list[float]] = {}
    for row in rows:
        for metric in metrics:
            value = await asyncio.to_thread(_evaluate_one_row_one_metric, row, metric)
            if value is not None:
                per_metric_scores.setdefault(metric.name, []).append(value)
            await asyncio.sleep(seconds_between_calls)

    return {
        name: sum(values) / len(values) for name, values in per_metric_scores.items() if values
    }


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
