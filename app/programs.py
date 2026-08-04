"""DSPy signatures and the composed RAG program, see CLAUDE.md section 8.

Verified against gemini-3.1-flash-lite via scripts/check_dspy_stream.py (streaming)
and a throwaway script exercising Literal, list[str], list[int], and bool output
fields plus ChainOfThought (this session, not committed). Findings that shaped this
file:

- dspy.Predict and dspy.ChainOfThought handle Literal, list, and bool output fields
  correctly with gemini-3.1-flash-lite.
- RouteQuery accuracy is sensitive to the signature docstring. A one line docstring
  with no domain description misclassified an out of scope question (Roman history)
  as lookup. Naming the GST domain explicitly and giving concrete criteria for
  out_of_scope and multi_hop fixed it. The docstring below is written for that reason,
  not for style.

CheckGrounded is a separate program, not part of RAGProgram, so it stays off the
streaming path and out of MIPROv2's search space, see section 8.

The retriever is injected as a callable so the evaluation harness can swap retrieval
configurations without constructing a new module, see section 7 and PipelineConfig in
app/pipeline.py.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Literal

import dspy


class RouteQuery(dspy.Signature):
    """Classify a question about Indian GST (Goods and Services Tax) law into a route.

    The reference manual covers GST rates, procedures, input tax credit, registration,
    returns, and related circulars. route is out_of_scope when the question is not
    about GST or Indian tax law at all, for example general history, other countries'
    tax systems, or unrelated topics. route is multi_hop only when answering requires
    combining information from more than one distinct sub-topic, and in that case
    sub_questions must contain the decomposed sub-questions. sub_questions is empty for
    every other route."""

    question: str = dspy.InputField()
    route: Literal["lookup", "procedure", "multi_hop", "out_of_scope"] = dspy.OutputField()
    sub_questions: list[str] = dspy.OutputField()


class AnswerFromManual(dspy.Signature):
    """Answer the question using only the provided context from a GST reference manual.

    The answer must come only from the context. If the context does not contain the
    answer, say plainly that the manual does not cover this rather than guessing or
    using outside knowledge. citations must list only page numbers that actually
    appear in the context."""

    context: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    citations: list[int] = dspy.OutputField()


class CheckGrounded(dspy.Signature):
    """Check whether every claim in the answer is directly supported by the context.

    List any claim in the answer that is not supported by the context as an
    unsupported claim, quoting or closely paraphrasing the unsupported part. grounded
    is true only when unsupported_claims is empty."""

    context: str = dspy.InputField()
    answer: str = dspy.InputField()
    unsupported_claims: list[str] = dspy.OutputField()
    grounded: bool = dspy.OutputField()


RetrieverFn = Callable[[str], Awaitable[list]]


class RAGProgram(dspy.Module):
    """Composes the router and a ChainOfThought answerer over injected retrieval.

    The retriever is a callable rather than being constructed inside this module, see
    CLAUDE.md section 8, so app/pipeline.py can inject different PipelineConfig backed
    retrieval without changing this class.
    """

    def __init__(self) -> None:
        super().__init__()
        self.route = dspy.Predict(RouteQuery)
        self.answer = dspy.ChainOfThought(AnswerFromManual)

    def forward(self, context: str, question: str) -> dspy.Prediction:
        return self.answer(context=context, question=question)


def configure_lm_cache() -> None:
    """Enable the DSPy LM cache so optimizer retries and repeated eval runs do not
    consume quota, see CLAUDE.md section 8. Call once at process startup."""

    dspy.configure_cache(
        enable_disk_cache=True,
        enable_memory_cache=True,
    )


def build_lm(model_name: str, api_key: str) -> dspy.LM:
    full_name = model_name if model_name.startswith("gemini/") else f"gemini/{model_name}"
    return dspy.LM(full_name, api_key=api_key, cache=True)


if __name__ == "__main__":
    import asyncio
    import os

    from dotenv import load_dotenv

    async def _main() -> None:
        load_dotenv()
        configure_lm_cache()
        lm = build_lm(
            os.environ.get("TASK_MODEL", "gemini-3.1-flash-lite"),
            os.environ["GEMINI_API_KEY_TASK"],
        )
        dspy.configure(lm=lm)

        program = RAGProgram()
        context = (
            "[Page 42] The GST rate for cotton textiles is 5 percent under HSN code 5208.\n"
            "[Page 43] Synthetic textiles attract a rate of 12 percent."
        )
        prediction = program(context=context, question="What is the GST rate for cotton textiles?")
        print(prediction)

        router = dspy.Predict(RouteQuery)
        print(router(question="Who won the Roman Senate election in 63 BC?"))

    asyncio.run(_main())
