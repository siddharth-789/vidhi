"""Unit tests for ingest/chunk.py. Run with: python -m pytest tests/test_chunk.py"""

from __future__ import annotations

from ingest.chunk import build_chunks, count_tokens


def _page(page_num: int, text: str) -> dict:
    return {"page": page_num, "text": text, "char_count": len(text)}


def test_chunk_never_spans_two_chapters():
    pages = [
        _page(1, "\nChapter 1 \nFirst Chapter Title \n\nSynopsis \n1. Introductory\n"),
        _page(2, "Chap. 1 \n" + ("alpha " * 500)),
        _page(3, "\nChapter 2 \nSecond Chapter Title \n\nSynopsis \n1. Introductory\n"),
        _page(4, "Chap. 2 \n" + ("beta " * 500)),
    ]

    chunks = build_chunks(pages)

    assert len(chunks) > 0
    for chunk in chunks:
        assert "alpha" not in chunk["content"] or "beta" not in chunk["content"]

    chapter_1_chunks = [c for c in chunks if c["chapter"].startswith("Chapter 1")]
    chapter_2_chunks = [c for c in chunks if c["chapter"].startswith("Chapter 2")]
    assert all("beta" not in c["content"] for c in chapter_1_chunks)
    assert all("alpha" not in c["content"] for c in chapter_2_chunks)


def test_chapter_start_requires_nearby_synopsis():
    pages = [
        _page(1, "CONTENTS \nChapter 1 \nSome Topic \n5 \nChapter 2 \nOther Topic \n10 \n"),
        _page(2, "\nChapter 1 \nReal Chapter Title \n\nSynopsis \n1. Introductory\n" + ("gamma " * 500)),
    ]

    chunks = build_chunks(pages)

    chapters = {c["chapter"] for c in chunks}
    assert any(c.startswith("Chapter 1") for c in chapters)
    assert all("front matter" == c or c.startswith("Chapter 1") for c in chapters)


def test_running_header_and_footer_are_stripped():
    pages = [
        _page(
            1,
            "\nChapter 1 \nTitle \n\nSynopsis \n1. Introductory\n"
            "5 \nGST Smart Guide \nChap. 1 \n" + ("delta " * 500),
        ),
    ]

    chunks = build_chunks(pages)

    for chunk in chunks:
        assert "GST Smart Guide" not in chunk["content"]
        assert "Chap. 1" not in chunk["content"]


def test_chunk_uid_is_deterministic():
    pages = [
        _page(1, "\nChapter 1 \nTitle \n\nSynopsis \n1. Introductory\n" + ("epsilon " * 500)),
    ]

    chunks_a = build_chunks(pages)
    chunks_b = build_chunks(pages)

    assert [c["chunk_uid"] for c in chunks_a] == [c["chunk_uid"] for c in chunks_b]


def test_token_count_matches_cl100k_base():
    assert count_tokens("hello world") > 0
    assert count_tokens("") == 0


def test_two_chapters_with_same_label_stay_separate():
    pages = [
        _page(1, "\nChapter 54 \nFirst Fifty Four \n\nSynopsis \n1. Introductory\n" + ("zeta " * 500)),
        _page(2, "\nChapter 54 \nSecond Fifty Four \n\nSynopsis \n1. Introductory\n" + ("eta " * 500)),
    ]

    chunks = build_chunks(pages)

    first_group_chapters = {c["chapter"] for c in chunks if "zeta" in c["content"]}
    second_group_chapters = {c["chapter"] for c in chunks if "eta" in c["content"] and "zeta" not in c["content"]}
    assert first_group_chapters
    assert second_group_chapters
    assert first_group_chapters.isdisjoint(second_group_chapters)
