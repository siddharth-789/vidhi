"""Extract text per page from the manual PDF using pymupdf.

Run standalone with: python -m ingest.extract [path/to/manual.pdf]
Writes data/pages.jsonl, one JSON object per page: page, text, char_count.

A page with fewer than 100 characters of text is logged as a suspected scanned page.
If more than 5 percent of pages are suspect, a loud warning is printed telling the
user OCR may be needed. OCR itself is not implemented here.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import pymupdf

logger = logging.getLogger(__name__)

_SUSPECT_CHAR_THRESHOLD = 100
_SUSPECT_RATIO_WARNING = 0.05


def extract_pages(pdf_path: Path) -> list[dict]:
    """Extract raw text from every page of the PDF."""

    doc = pymupdf.open(pdf_path)
    pages = []
    try:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            text = page.get_text()
            pages.append(
                {
                    "page": page_index + 1,
                    "text": text,
                    "char_count": len(text),
                }
            )
    finally:
        doc.close()
    return pages


def write_pages_jsonl(pages: list[dict], out_path: Path) -> None:
    """Write one JSON object per page to out_path."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for page in pages:
            f.write(json.dumps(page, ensure_ascii=False) + "\n")


def summarize_suspected_scanned_pages(pages: list[dict]) -> int:
    """Print a count of pages that look scanned (very little extracted text) and warn
    loudly if more than 5 percent of the document is affected."""

    suspect_pages = [p["page"] for p in pages if p["char_count"] < _SUSPECT_CHAR_THRESHOLD]
    for page_num in suspect_pages:
        logger.info("page %d has fewer than %d characters, suspected scanned page",
                     page_num, _SUSPECT_CHAR_THRESHOLD)

    suspect_count = len(suspect_pages)
    total = len(pages)
    ratio = suspect_count / total if total else 0.0
    print(f"suspected scanned pages: {suspect_count} of {total} ({ratio:.1%})")
    if ratio > _SUSPECT_RATIO_WARNING:
        print(
            "WARNING: more than 5 percent of pages are suspected scanned pages. "
            "OCR may be needed before ingestion continues. This project does not "
            "implement OCR."
        )
    return suspect_count


def main(pdf_path_arg: str | None = None, out_path_arg: str | None = None) -> None:
    """Extract every page's text from the PDF and write it to pages.jsonl."""

    logging.basicConfig(level=logging.INFO)
    pdf_path = Path(pdf_path_arg or "data/manual.pdf")
    out_path = Path(out_path_arg or "data/pages.jsonl")

    if not pdf_path.exists():
        print(f"PDF not found at {pdf_path}", file=sys.stderr)
        sys.exit(1)

    start = time.perf_counter()
    pages = extract_pages(pdf_path)
    elapsed = time.perf_counter() - start

    write_pages_jsonl(pages, out_path)
    summarize_suspected_scanned_pages(pages)

    print(f"extracted {len(pages)} pages in {elapsed:.1f}s, wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
