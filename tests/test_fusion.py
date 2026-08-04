"""Unit test for reciprocal rank fusion in app/retrieve.py.

Run with: python -m pytest tests/test_fusion.py
"""

from __future__ import annotations

from app.retrieve import fuse


def _row(chunk_uid: str) -> dict:
    return {
        "chunk_uid": chunk_uid,
        "content": f"content for {chunk_uid}",
        "heading_path": None,
        "chapter": None,
        "page_start": 1,
        "page_end": 1,
    }


def test_fuse_hand_computed_example():
    # Dense ranks: a=1, b=2, c=3. Sparse ranks: b=1, c=2, d=3.
    dense_rows = [_row("a"), _row("b"), _row("c")]
    sparse_rows = [_row("b"), _row("c"), _row("d")]

    fused = fuse(dense_rows, sparse_rows)

    k = 60
    expected_scores = {
        "a": 1 / (k + 1),
        "b": 1 / (k + 2) + 1 / (k + 1),
        "c": 1 / (k + 3) + 1 / (k + 2),
        "d": 1 / (k + 3),
    }

    scores_by_uid = {c.chunk_uid: c.fusion_score for c in fused}
    for uid, expected_score in expected_scores.items():
        assert abs(scores_by_uid[uid] - expected_score) < 1e-9

    # b appears at dense rank 2 and sparse rank 1, and scores highest overall.
    assert fused[0].chunk_uid == "b"
    assert fused[0].dense_rank == 2
    assert fused[0].sparse_rank == 1

    # a appears only in the dense list.
    a = next(c for c in fused if c.chunk_uid == "a")
    assert a.dense_rank == 1
    assert a.sparse_rank is None

    # d appears only in the sparse list.
    d = next(c for c in fused if c.chunk_uid == "d")
    assert d.dense_rank is None
    assert d.sparse_rank == 3

    # Fused order is sorted by score, descending.
    scores = [c.fusion_score for c in fused]
    assert scores == sorted(scores, reverse=True)


def test_fuse_empty_lists_returns_empty():
    assert fuse([], []) == []


def test_fuse_dense_only():
    dense_rows = [_row("a"), _row("b")]

    fused = fuse(dense_rows, [])

    assert len(fused) == 2
    assert all(c.sparse_rank is None for c in fused)
    assert fused[0].chunk_uid == "a"
    assert fused[0].dense_rank == 1
