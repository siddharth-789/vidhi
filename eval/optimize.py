"""Runs the MIPROv2 optimizer against app.programs.AnswerFromManual.

MIPROv2 proposes new instructions and bootstraps few-shot demonstrations for the
answer predictor, then searches over the combinations with Bayesian optimization.
Both halves matter here because the grounding and abstention rules live in the
instruction text, not only in demonstrations. BootstrapFewShotWithRandomSearch, by
comparison, only searches over demonstration sets and cannot rewrite an instruction,
so its ceiling is lower on a task whose main failure mode is answering from
parametric knowledge instead of abstaining from the manual.

The metric is deliberately part deterministic:
    score = 0.6 * correctness + 0.4 * citation_validity
correctness comes from an LLM judge, citation_validity is purely arithmetic (fraction
of predicted pages that fall within the gold page set widened by plus or minus 2
pages). The deterministic 40 percent keeps the Bayesian search from chasing judge
noise, which is the usual reason a MIPROv2 run appears to accomplish nothing.

Run standalone with:
    python -m eval.optimize --auto light
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

import dspy

from app.config import get_config
from app.db import create_pool
from app.llm import build_client
from app.programs import AnswerFromManual, RAGProgram, build_lm, configure_lm_cache
from app.retrieve import retrieve
from app.llm import embed_query
from eval.dataset import filter_split, load_trainset, to_dspy_examples

logger = logging.getLogger(__name__)


def _build_metric(app_config, judge_lm: dspy.LM):
    """Build the composite metric: 0.6 correctness (LLM judge) plus 0.4
    citation_validity (purely arithmetic, no LLM call).
    """

    def metric(example, prediction, trace=None) -> float:
        gold_pages = set(example.gold_pages)
        predicted_pages = set(getattr(prediction, "citations", []) or [])
        widened = set()
        for page in gold_pages:
            widened.update(range(page - 2, page + 3))
        citation_validity = (
            len(predicted_pages & widened) / len(predicted_pages) if predicted_pages else 0.0
        )

        with dspy.context(lm=judge_lm):
            judge = dspy.Predict(
                dspy.Signature(
                    "question, gold_answer, predicted_answer -> correctness_label",
                    instructions=(
                        "Judge whether predicted_answer is correct given gold_answer for "
                        "this question about Indian GST law. correctness_label must be "
                        "exactly one of: 'correct', 'partially_correct', 'incorrect'."
                    ),
                )
            )
            judgment = judge(
                question=example.question,
                gold_answer=example.gold_answer,
                predicted_answer=getattr(prediction, "answer", ""),
            )

        label = getattr(judgment, "correctness_label", "incorrect").strip().lower()
        correctness = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}.get(label, 0.0)

        return 0.6 * correctness + 0.4 * citation_validity

    return metric


async def _make_retriever(pool, app_config, task_client):
    async def retriever(question: str) -> list[str]:
        embedding = await embed_query(
            task_client, question, app_config.embed_model, app_config.embed_dim
        )
        result = await retrieve(
            pool, question, embedding, use_hybrid=True, dense_k=30, sparse_k=30, fusion_k=12
        )
        blocks = []
        for c in result.candidates[:5]:
            page_label = (
                f"Page {c.page_start}"
                if c.page_start == c.page_end
                else f"Pages {c.page_start}-{c.page_end}"
            )
            blocks.append(f"[{page_label}] {c.content}")
        return blocks

    return retriever


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auto", default="light", choices=["light", "medium", "heavy"])
    parser.add_argument("--trainset", default="data/trainset.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    from dotenv import load_dotenv

    load_dotenv()

    all_examples = load_trainset(args.trainset)
    train_examples = filter_split(all_examples, "train")
    if not train_examples:
        raise SystemExit("no train split rows found, cannot run MIPROv2")

    app_config = get_config()
    if not app_config.gemini_api_key_task or not app_config.gemini_api_key_judge:
        raise SystemExit("both GEMINI_API_KEY_TASK and GEMINI_API_KEY_JUDGE are required")

    configure_lm_cache()
    task_lm = build_lm(app_config.task_model, app_config.gemini_api_key_task)
    prompt_lm = build_lm(app_config.prompt_model, app_config.gemini_api_key_task)
    judge_lm = build_lm(app_config.judge_model, app_config.gemini_api_key_judge)
    dspy.configure(lm=task_lm)

    pool = await create_pool(app_config.database_url)
    task_client = build_client(app_config.gemini_api_key_task)

    try:
        retriever = await _make_retriever(pool, app_config, task_client)

        async def build_context(question: str) -> str:
            blocks = await retriever(question)
            return "\n\n".join(blocks)

        dspy_examples = []
        for example in to_dspy_examples(train_examples):
            context = await build_context(example.question)
            dspy_examples.append(
                dspy.Example(
                    context=context,
                    question=example.question,
                    gold_answer=example.gold_answer,
                    gold_pages=example.gold_pages,
                ).with_inputs("context", "question")
            )

        # Compile a bare ChainOfThought(AnswerFromManual) in isolation, not the full
        # RAGProgram. RAGProgram.forward only ever calls self.answer, so self.route
        # gets no real training signal from this metric, MIPROv2 still perturbs every
        # named predictor it finds in a module, and a first attempt at compiling
        # RAGProgram directly spent its whole search budget "optimizing" the unused
        # route predictor while leaving the actually used answer predictor untouched.
        # Optimizing the answer predictor alone guarantees the search is scored on,
        # and can only improve, the predictor serving actually calls.
        program = dspy.ChainOfThought(AnswerFromManual)
        original_instruction = program.predict.signature.instructions
        metric = _build_metric(app_config, judge_lm)

        optimizer = dspy.MIPROv2(
            metric=metric, auto=args.auto, prompt_model=prompt_lm, task_model=task_lm
        )
        compiled_answer = optimizer.compile(program, trainset=dspy_examples)

        # Splice the optimized answer predictor into a fresh RAGProgram before saving,
        # so the artifact's predictor keys ("route", "answer.predict") match what
        # app/pipeline.py and app/api.py load into a RAGProgram() at request and
        # startup time. route keeps its uncompiled RouteQuery signature and zero demos,
        # which is correct since routing is exercised separately by app.pipeline, not
        # through RAGProgram.
        compiled = RAGProgram()
        compiled.answer = compiled_answer

        artifact_path = Path(app_config.compiled_program_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        compiled.save(str(artifact_path))
        print(f"saved compiled program to {artifact_path}")

        # trial_logs values hold deepcopy'd program objects, not JSON serializable, so
        # only the numeric score curve is pulled out here.
        score_curve = []
        for trial_num, entry in sorted(getattr(compiled_answer, "trial_logs", {}).items()):
            score_curve.append(
                {
                    "trial": trial_num,
                    "mb_score": entry.get("mb_score"),
                    "full_eval_score": entry.get("full_eval_score"),
                }
            )

        log_path = Path("eval/results/optimize_log.json")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_payload = {
            "auto": args.auto,
            "trainset_size": len(dspy_examples),
            "timestamp": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
            "original_instruction": original_instruction,
            "winning_instruction": compiled_answer.predict.signature.instructions,
            "demo_count": len(compiled_answer.predict.demos),
            "score_curve": score_curve,
        }
        log_path.write_text(json.dumps(log_payload, indent=2), encoding="utf-8")
        print(f"wrote optimizer log to {log_path}")

    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
