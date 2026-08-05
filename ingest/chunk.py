"""Chunk extracted pages into token-bounded, single-chapter passages.

Run standalone with: python -m ingest.chunk [path/to/pages.jsonl]
Reads data/pages.jsonl, writes data/chunks.jsonl, prints a distribution summary.

Heading detection is tuned to this specific manual's structure, confirmed by reading
a sample of pages directly rather than assumed from a generic regex list. In this
document:

- A real chapter start is a standalone line "Chapter N" followed within a few lines
  by a standalone line "Synopsis". This distinguishes it from the table of contents,
  which lists many "Chapter N" lines with no nearby "Synopsis", and from the running
  page header and footer, which use the abbreviation "Chap. N" and never match
  "Chapter N" at all.
- The source book mislabels two different chapters as "Chapter 54". Page order, not
  the parsed chapter number, is therefore the source of truth for chapter boundaries.
- "N.N" and "N.N.N" numbering is real and common, but numbers sub-points under a
  synopsis topic rather than resetting per chapter, and does not reliably distinguish
  a heading line from a numbered sentence in running prose. It is used here only as a
  soft heading_path breadcrumb, never as a hard chunk boundary.
- "Section N" and "Rule N" appear only as inline citations to the CGST Act or Rules,
  not as document headings, and are not used for structure.
- All caps short lines are dominated by front matter and table column noise in this
  document and are not used for structure.
- The running header and footer text, for example "GST Smart Guide" and "Chap. N", is
  stripped from chunk content since it repeats on every page and carries no meaning.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")

_CHUNK_TOKEN_TARGET = 700
_CHUNK_TOKEN_OVERLAP = 120

_CHAPTER_START_RE = re.compile(r"^Chapter\s+(\d+)\s*$")
_SYNOPSIS_RE = re.compile(r"^Synopsis\s*$")
_CHAP_FOOTER_RE = re.compile(r"^Chap\.\s+\d+\s*$")
_APPX_FOOTER_RE = re.compile(r"^Appx\.\s+\d+\s*$")
_BOOK_TITLE_RE = re.compile(r"^GST Smart Guide\s*$")
_PAGE_NUMBER_LINE_RE = re.compile(r"^\d{1,4}\s*$")
_SUBHEADING_RE = re.compile(r"^(\d{1,3}\.\d{1,3}(?:\.\d{1,3})?)\s+(\S.{0,80})$")

# The book's main chapters (Chapter N, confirmed by a nearby standalone "Synopsis"
# line) are followed by an appendix section of reprinted circulars, each headed by a
# standalone "Appendix N" line with no "Synopsis" nearby. Both are treated as equally
# weighted chapter boundaries here, a chunk must never span either kind of boundary.
_APPENDIX_START_RE = re.compile(r"^Appendix\s+(\d+)\s*$")

_CHAPTER_START_LOOKAHEAD_LINES = 10


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken's cl100k_base encoding, purely as a size proxy."""

    return len(_ENCODING.encode(text))


def _strip_boilerplate_lines(text: str) -> str:
    """Remove running headers, footers, and bare page-number lines from a page's text."""

    lines = text.split("\n")
    kept = []
    for line in lines:
        stripped = line.strip()
        if _CHAP_FOOTER_RE.match(stripped) or _APPX_FOOTER_RE.match(stripped):
            continue
        if _BOOK_TITLE_RE.match(stripped):
            continue
        if _PAGE_NUMBER_LINE_RE.match(stripped):
            continue
        kept.append(line)
    return "\n".join(kept)


def _find_chapter_starts(pages: list[dict]) -> dict[int, tuple[int, str]]:
    """Map page number to (chapter_ordinal, chapter_label) for pages that start a
    chapter or an appendix entry.

    chapter_ordinal is the 1 based position among all detected starts, in document
    order, not the parsed chapter or appendix number, because the source document
    mislabels two different chapters both as Chapter 54.

    The book has two kinds of top level section, both treated as equally weighted
    boundaries: a numbered chapter, confirmed by a standalone "Chapter N" line
    followed within a few lines by a standalone "Synopsis" line, and an appendix
    entry reprinting a circular, confirmed by a standalone "Appendix N" line. The
    appendix section's own mini table of contents uses the word "Appendix" with no
    number, so it does not collide with this pattern.
    """
    starts: dict[int, tuple[int, str]] = {}
    ordinal = 0
    for page in pages:
        lines = [line.strip() for line in page["text"].split("\n")]
        for i, line in enumerate(lines):
            chapter_match = _CHAPTER_START_RE.match(line)
            appendix_match = _APPENDIX_START_RE.match(line)
            if chapter_match:
                lookahead = lines[i + 1 : i + 1 + _CHAPTER_START_LOOKAHEAD_LINES]
                if not any(_SYNOPSIS_RE.match(l) for l in lookahead):
                    continue
                title_lines = [l for l in lookahead if l and not _SYNOPSIS_RE.match(l)]
                title = " ".join(title_lines[:2]).strip()
                ordinal += 1
                starts[page["page"]] = (ordinal, f"Chapter {chapter_match.group(1)}: {title}")
                break
            if appendix_match:
                lookahead = lines[i + 1 : i + 1 + _CHAPTER_START_LOOKAHEAD_LINES]
                title_lines = [l for l in lookahead if l]
                title = " ".join(title_lines[:2]).strip()
                ordinal += 1
                label = f"Appendix {appendix_match.group(1)}: {title}" if title else f"Appendix {appendix_match.group(1)}"
                starts[page["page"]] = (ordinal, label)
                break
    return starts


def _assign_chapters(pages: list[dict]) -> list[dict]:
    starts = _find_chapter_starts(pages)
    current_chapter = "front matter"
    annotated = []
    for page in pages:
        if page["page"] in starts:
            _, label = starts[page["page"]]
            current_chapter = label
        annotated.append({**page, "chapter": current_chapter})
    return annotated


def _extract_heading_breadcrumb(text: str) -> str | None:
    for line in text.split("\n"):
        stripped = line.strip()
        match = _SUBHEADING_RE.match(stripped)
        if match:
            return f"{match.group(1)} {match.group(2)}".strip()
    return None


def _make_chunk_uid(chapter: str, page_start: int, content: str) -> str:
    """Deterministic id from chapter, starting page, and content, so a rerun upserts
    instead of duplicating rows."""

    basis = f"{chapter}|{page_start}|{content[:200]}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:32]


def build_chunks(pages: list[dict]) -> list[dict]:
    """Group pages into chapters, then pack each chapter into token-bounded chunks."""

    annotated_pages = _assign_chapters(pages)

    chunks: list[dict] = []
    chapter_groups: list[list[dict]] = []
    current_group: list[dict] = []
    current_chapter = None
    for page in annotated_pages:
        if page["chapter"] != current_chapter:
            if current_group:
                chapter_groups.append(current_group)
            current_group = []
            current_chapter = page["chapter"]
        current_group.append(page)
    if current_group:
        chapter_groups.append(current_group)

    for group in chapter_groups:
        chapter = group[0]["chapter"]
        heading_path = chapter

        page_texts = []
        for page in group:
            cleaned = _strip_boilerplate_lines(page["text"])
            breadcrumb = _extract_heading_breadcrumb(cleaned)
            if breadcrumb:
                heading_path = f"{chapter} > {breadcrumb}"
            page_texts.append((page["page"], cleaned, heading_path))

        chunks.extend(_pack_chapter(chapter, page_texts))

    return chunks


def _pack_chapter(chapter: str, page_texts: list[tuple[int, str, str]]) -> list[dict]:
    """Pack a single chapter's pages into token bounded chunks with overlap.

    page_texts is a list of (page_number, cleaned_text, heading_path_at_that_page).
    """
    tokens_with_page: list[tuple[int, str, str]] = []
    for page_num, text, heading_path in page_texts:
        words = text.split()
        for word in words:
            tokens_with_page.append((page_num, word, heading_path))

    chunks = []
    i = 0
    n = len(tokens_with_page)
    while i < n:
        window: list[tuple[int, str, str]] = []
        window_tokens = 0
        j = i
        while j < n and window_tokens < _CHUNK_TOKEN_TARGET:
            window.append(tokens_with_page[j])
            window_tokens += count_tokens(tokens_with_page[j][1] + " ")
            j += 1
        if not window:
            break

        content = " ".join(w[1] for w in window)
        page_start = window[0][0]
        page_end = window[-1][0]
        heading_path = window[0][2]
        token_count = count_tokens(content)

        chunks.append(
            {
                "chunk_uid": _make_chunk_uid(chapter, page_start, content),
                "content": content,
                "heading_path": heading_path,
                "chapter": chapter,
                "page_start": page_start,
                "page_end": page_end,
                "token_count": token_count,
            }
        )

        if j >= n:
            break

        overlap_tokens = 0
        back = j
        while back > i and overlap_tokens < _CHUNK_TOKEN_OVERLAP:
            back -= 1
            overlap_tokens += count_tokens(tokens_with_page[back][1] + " ")
        i = max(back, i + 1)

    return chunks


def write_chunks_jsonl(chunks: list[dict], out_path: Path) -> None:
    """Write one JSON object per chunk to out_path."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def print_summary(chunks: list[dict]) -> None:
    """Print chunk count, token count percentiles, and chunks per chapter."""

    token_counts = [c["token_count"] for c in chunks]
    per_chapter = Counter(c["chapter"] for c in chunks)

    print(f"total chunks: {len(chunks)}")
    if token_counts:
        quantiles = statistics.quantiles(token_counts, n=10)
        print(f"token count: min={min(token_counts)} p50={int(quantiles[4])} "
              f"p90={int(quantiles[8])} max={max(token_counts)}")
    print(f"chunks per chapter ({len(per_chapter)} chapters):")
    for chapter, count in sorted(per_chapter.items(), key=lambda kv: kv[0]):
        print(f"  {chapter}: {count}")


def load_pages(path: Path) -> list[dict]:
    """Read pages.jsonl into a list of dicts, one per line."""

    pages = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            pages.append(json.loads(line))
    return pages


def main(pages_path_arg: str | None = None, out_path_arg: str | None = None) -> None:
    """Chunk pages.jsonl into chunks.jsonl and print a distribution summary."""

    pages_path = Path(pages_path_arg or "data/pages.jsonl")
    out_path = Path(out_path_arg or "data/chunks.jsonl")

    pages = load_pages(pages_path)
    chunks = build_chunks(pages)
    write_chunks_jsonl(chunks, out_path)
    print_summary(chunks)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
