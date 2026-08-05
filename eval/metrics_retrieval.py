"""Deterministic retrieval and citation metrics: pure arithmetic, zero LLM calls, so
they always run on the full dev split.

Every function is pure, taking already computed per question records rather than
calling the pipeline itself, so it can be unit tested against small synthetic
examples and reused by eval/run_eval.py without duplicating pipeline invocation logic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuestionRecord:
    """One evaluated question, enough fields for every group 1 metric.

    retrieved_page_spans is a list of (page_start, page_end) tuples in the order the
    pipeline surfaced them post fusion or rerank, best first. predicted_citations is
    the list of page numbers the model actually cited, already validated against the
    retrieved set by app/guardrails.py.
    """

    gold_pages: list[int]
    answerable: bool
    abstained: bool
    retrieved_page_spans: list[tuple[int, int]]
    predicted_citations: list[int]


def _span_hits_gold(span: tuple[int, int], gold_pages: list[int]) -> bool:
    """Whether any gold page number falls inside a retrieved page span."""

    start, end = span
    return any(start <= page <= end for page in gold_pages)


def recall_at_k(records: list[QuestionRecord], k: int) -> float:
    """Fraction of answerable questions where any gold page appears in the top k spans."""

    answerable = [r for r in records if r.answerable]
    if not answerable:
        return 0.0
    hits = 0
    for r in answerable:
        if any(_span_hits_gold(span, r.gold_pages) for span in r.retrieved_page_spans[:k]):
            hits += 1
    return hits / len(answerable)


def mrr_at_k(records: list[QuestionRecord], k: int) -> float:
    """Mean reciprocal rank of the first retrieved span containing a gold page."""

    answerable = [r for r in records if r.answerable]
    if not answerable:
        return 0.0
    total = 0.0
    for r in answerable:
        for rank, span in enumerate(r.retrieved_page_spans[:k], start=1):
            if _span_hits_gold(span, r.gold_pages):
                total += 1.0 / rank
                break
    return total / len(answerable)


def citation_validity_rate(records: list[QuestionRecord]) -> float:
    """Fraction of answerable, non abstained questions whose citations survived
    app/guardrails.py validate_citations with none dropped.

    This metric reads predicted_citations post validation, so it measures how often
    the model cited cleanly rather than how often a citation had to be dropped.
    Compute the drop rate separately from the raw pre validation count if that
    distinction becomes useful.
    """

    scoped = [r for r in records if r.answerable and not r.abstained]
    if not scoped:
        return 0.0
    valid = sum(1 for r in scoped if r.predicted_citations)
    return valid / len(scoped)


def abstention_accuracy(records: list[QuestionRecord]) -> float:
    """Fraction of answerable=false questions where the pipeline correctly abstained."""

    unanswerable = [r for r in records if not r.answerable]
    if not unanswerable:
        return 0.0
    correct = sum(1 for r in unanswerable if r.abstained)
    return correct / len(unanswerable)


def false_abstention_rate(records: list[QuestionRecord]) -> float:
    """Fraction of answerable=true questions where the pipeline incorrectly abstained."""

    answerable = [r for r in records if r.answerable]
    if not answerable:
        return 0.0
    false_abstains = sum(1 for r in answerable if r.abstained)
    return false_abstains / len(answerable)


def answer_page_overlap(records: list[QuestionRecord]) -> float:
    """Mean fraction of gold pages that appear among the predicted citations, for
    answerable, non abstained questions."""

    scoped = [r for r in records if r.answerable and not r.abstained and r.gold_pages]
    if not scoped:
        return 0.0
    total = 0.0
    for r in scoped:
        gold_set = set(r.gold_pages)
        overlap = len(gold_set & set(r.predicted_citations))
        total += overlap / len(gold_set)
    return total / len(scoped)


def compute_all(records: list[QuestionRecord]) -> dict[str, float]:
    """Compute every group 1 metric over a set of evaluated questions."""

    return {
        "recall_at_5": recall_at_k(records, 5),
        "recall_at_10": recall_at_k(records, 10),
        "mrr_at_10": mrr_at_k(records, 10),
        "citation_validity_rate": citation_validity_rate(records),
        "abstention_accuracy": abstention_accuracy(records),
        "false_abstention_rate": false_abstention_rate(records),
        "answer_page_overlap": answer_page_overlap(records),
    }


if __name__ == "__main__":
    sample = [
        QuestionRecord(
            gold_pages=[412, 413],
            answerable=True,
            abstained=False,
            retrieved_page_spans=[(410, 411), (412, 414), (500, 502)],
            predicted_citations=[412],
        ),
        QuestionRecord(
            gold_pages=[],
            answerable=False,
            abstained=True,
            retrieved_page_spans=[],
            predicted_citations=[],
        ),
    ]
    for name, value in compute_all(sample).items():
        print(f"{name}: {value:.3f}")
