"""Unit tests for app/guardrails.py against synthetic scores.

Run with: python -m pytest tests/test_guardrails.py
"""

from __future__ import annotations

import pytest

from app.errors import GuardrailAbstain
from app.guardrails import check_pre_generation, validate_citations
from app.retrieve import Candidate


def _candidate(page_start: int, page_end: int, rerank_score: float | None) -> Candidate:
    return Candidate(
        chunk_uid=f"uid-{page_start}",
        content="some content",
        heading_path=None,
        chapter=None,
        page_start=page_start,
        page_end=page_end,
        dense_rank=1,
        sparse_rank=1,
        fusion_score=0.03,
        rerank_score=rerank_score,
    )


def test_abstains_on_zero_candidates():
    with pytest.raises(GuardrailAbstain):
        check_pre_generation([], retrieval_floor=0.35)


def test_abstains_below_floor():
    candidates = [_candidate(10, 12, rerank_score=0.2)]
    with pytest.raises(GuardrailAbstain):
        check_pre_generation(candidates, retrieval_floor=0.35)


def test_passes_above_floor():
    candidates = [_candidate(10, 12, rerank_score=0.5)]
    check_pre_generation(candidates, retrieval_floor=0.35)


def test_passes_at_floor_boundary():
    candidates = [_candidate(10, 12, rerank_score=0.35)]
    check_pre_generation(candidates, retrieval_floor=0.35)


def test_passes_when_no_rerank_score_available():
    # Reranking was skipped or degraded to fusion order, nothing comparable to the
    # floor, so generation proceeds rather than abstaining on an incomparable value.
    candidates = [_candidate(10, 12, rerank_score=None)]
    check_pre_generation(candidates, retrieval_floor=0.35)


def test_validate_citations_keeps_pages_in_range():
    candidates = [_candidate(40, 45, rerank_score=0.6)]
    valid, any_dropped = validate_citations([42], candidates)
    assert valid == [42]
    assert any_dropped is False


def test_validate_citations_drops_out_of_range_pages():
    candidates = [_candidate(40, 45, rerank_score=0.6)]
    valid, any_dropped = validate_citations([42, 99], candidates)
    assert valid == [42]
    assert any_dropped is True


def test_validate_citations_across_multiple_candidates():
    candidates = [_candidate(10, 12, rerank_score=0.6), _candidate(50, 52, rerank_score=0.5)]
    valid, any_dropped = validate_citations([11, 51, 200], candidates)
    assert valid == [11, 51]
    assert any_dropped is True


def test_validate_citations_empty_input():
    candidates = [_candidate(10, 12, rerank_score=0.6)]
    valid, any_dropped = validate_citations([], candidates)
    assert valid == []
    assert any_dropped is False
