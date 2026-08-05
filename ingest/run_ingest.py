"""Orchestrate extract, chunk, and embed, each skippable, with per stage timing.

Run standalone with: python -m ingest.run_ingest [--skip-extract] [--skip-chunk] [--skip-embed] [--pdf path/to/manual.pdf]
"""

from __future__ import annotations

import argparse
import asyncio
import time

from ingest import chunk as chunk_stage
from ingest import embed as embed_stage
from ingest import extract as extract_stage


def parse_args() -> argparse.Namespace:
    """Parse command-line flags for which ingestion stages to run and where files live."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", default="data/manual.pdf")
    parser.add_argument("--pages", default="data/pages.jsonl")
    parser.add_argument("--chunks", default="data/chunks.jsonl")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-chunk", action="store_true")
    parser.add_argument("--skip-embed", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run extract, chunk, and embed in order, skipping any stage that was requested
    to be skipped, and print how long each stage took."""

    args = parse_args()

    if not args.skip_extract:
        start = time.perf_counter()
        extract_stage.main(args.pdf, args.pages)
        print(f"extract stage took {time.perf_counter() - start:.1f}s")
    else:
        print("skipping extract stage")

    if not args.skip_chunk:
        start = time.perf_counter()
        chunk_stage.main(args.pages, args.chunks)
        print(f"chunk stage took {time.perf_counter() - start:.1f}s")
    else:
        print("skipping chunk stage")

    if not args.skip_embed:
        start = time.perf_counter()
        asyncio.run(embed_stage._main(args.chunks))
        print(f"embed stage took {time.perf_counter() - start:.1f}s")
    else:
        print("skipping embed stage")


if __name__ == "__main__":
    main()
