"""Computes RAGAS metrics against already saved eval/run_eval.py result files,
without re-running the pipeline. Used when group 1 was run with --skip-ragas to save
task key quota, and RAGAS (judge key only) is added afterwards.

Reads the most recent eval/results/{config}_*.json for each requested config, builds
RagasRow objects from its saved per question records, computes RAGAS, and rewrites the
same file with group2_ragas filled in. Context precision and context recall are shared
across configs that reuse the same retrieval configuration, per CLAUDE.md section 10:
B, C, and D share retrieval, compute once.

Run standalone with:
    python -m eval.backfill_ragas --configs A_vanilla,B_hybrid,C_dspy,D_optimized --ragas-n 20
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import logging
from pathlib import Path

from app.config import get_config
from eval.dataset import load_trainset
from eval.metrics_ragas import RagasRow, compute_ragas_metrics

logger = logging.getLogger(__name__)

_RETRIEVAL_SHARING = {
    "A_vanilla": "dense_only",
    "B_hybrid": "hybrid_rerank",
    "C_dspy": "hybrid_rerank",
    "D_optimized": "hybrid_rerank",
}


def _latest_result_path(config_name: str) -> Path:
    candidates = sorted(glob.glob(f"eval/results/{config_name}_*.json"))
    if not candidates:
        raise SystemExit(f"no saved result file found for {config_name}")
    return Path(candidates[-1])


async def backfill_one(
    config_name: str,
    ragas_n: int,
    ragas_cache: dict[str, dict[str, float]],
    gold_answers: dict[str, str],
) -> None:
    app_config = get_config()
    path = _latest_result_path(config_name)
    data = json.loads(path.read_text(encoding="utf-8"))

    records = data["records"]
    ragas_subset = [r for r in records if not r["abstained"] and r["answer"]][:ragas_n]

    share_key = _RETRIEVAL_SHARING.get(config_name, config_name)
    cached_context_scores = ragas_cache.get(share_key)
    include_context = cached_context_scores is None

    ragas_rows = [
        RagasRow(
            question=r["question"],
            answer=r["answer"],
            contexts=r["context_texts"] or [""],
            ground_truth=gold_answers.get(r["example_id"], ""),
        )
        for r in ragas_subset
    ]

    group2 = await compute_ragas_metrics(
        ragas_rows, app_config, include_context_metrics=include_context
    )
    if include_context:
        ragas_cache[share_key] = {
            k: v for k, v in group2.items() if k in ("context_precision", "context_recall")
        }
    else:
        group2.update(cached_context_scores)

    data["group2_ragas"] = group2
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("%s: wrote RAGAS scores to %s: %s", config_name, path, group2)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", required=True, help="comma separated config names")
    parser.add_argument("--ragas-n", type=int, default=25)
    parser.add_argument("--trainset", default="data/trainset.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    from dotenv import load_dotenv

    load_dotenv()

    gold_answers = {e.id: e.gold_answer for e in load_trainset(args.trainset)}

    config_names = [c.strip() for c in args.configs.split(",")]
    ragas_cache: dict[str, dict[str, float]] = {}
    for config_name in config_names:
        await backfill_one(config_name, args.ragas_n, ragas_cache, gold_answers)


if __name__ == "__main__":
    asyncio.run(main())
