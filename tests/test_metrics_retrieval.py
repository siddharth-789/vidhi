"""Unit tests for eval/metrics_retrieval.py against hand computed examples.

Run with: python -m pytest tests/test_metrics_retrieval.py
"""

from __future__ import annotations

from eval.metrics_retrieval import (
    QuestionRecord,
    abstention_accuracy,
    answer_page_overlap,
    citation_validity_rate,
    false_abstention_rate,
    mrr_at_k,
    recall_at_k,
)


def test_recall_at_k_hand_computed():
    records = [
        QuestionRecord(
            gold_pages=[412],
            answerable=True,
            abstained=False,
            retrieved_page_spans=[(1, 2), (412, 413)],
            predicted_citations=[412],
        ),
        QuestionRecord(
            gold_pages=[900],
            answerable=True,
            abstained=False,
            retrieved_page_spans=[(1, 2), (3, 4)],
            predicted_citations=[],
        ),
    ]
    # One of two answerable questions has a gold page in its top spans.
    assert recall_at_k(records, 5) == 0.5


def test_mrr_at_k_hand_computed():
    records = [
        QuestionRecord(
            gold_pages=[412],
            answerable=True,
            abstained=False,
            retrieved_page_spans=[(1, 2), (412, 413), (500, 501)],
            predicted_citations=[412],
        ),
        QuestionRecord(
            gold_pages=[500],
            answerable=True,
            abstained=False,
            retrieved_page_spans=[(500, 501)],
            predicted_citations=[500],
        ),
    ]
    # First question's gold page hits at rank 2 -> 1/2. Second hits at rank 1 -> 1/1.
    expected = (0.5 + 1.0) / 2
    assert abs(mrr_at_k(records, 10) - expected) < 1e-9


def test_citation_validity_rate():
    records = [
        QuestionRecord(
            gold_pages=[1],
            answerable=True,
            abstained=False,
            retrieved_page_spans=[(1, 1)],
            predicted_citations=[1],
        ),
        QuestionRecord(
            gold_pages=[2],
            answerable=True,
            abstained=False,
            retrieved_page_spans=[(2, 2)],
            predicted_citations=[],
        ),
    ]
    assert citation_validity_rate(records) == 0.5


def test_abstention_accuracy_and_false_abstention_rate():
    records = [
        QuestionRecord(
            gold_pages=[],
            answerable=False,
            abstained=True,
            retrieved_page_spans=[],
            predicted_citations=[],
        ),
        QuestionRecord(
            gold_pages=[],
            answerable=False,
            abstained=False,
            retrieved_page_spans=[(1, 1)],
            predicted_citations=[1],
        ),
        QuestionRecord(
            gold_pages=[10],
            answerable=True,
            abstained=True,
            retrieved_page_spans=[],
            predicted_citations=[],
        ),
    ]
    assert abstention_accuracy(records) == 0.5
    assert false_abstention_rate(records) == 1.0


def test_answer_page_overlap():
    records = [
        QuestionRecord(
            gold_pages=[1, 2, 3],
            answerable=True,
            abstained=False,
            retrieved_page_spans=[(1, 3)],
            predicted_citations=[1, 2],
        )
    ]
    assert abs(answer_page_overlap(records) - (2 / 3)) < 1e-9
