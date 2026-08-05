"""Two safety checks that bracket generation: one before it runs, one after.

Pre-generation: abstain without calling the generation model at all when the top
rerank score is below RETRIEVAL_FLOOR, or when zero chunks were retrieved. This
prevents the model from hallucinating an answer out of nothing.

Post-generation: validate citations against the retrieved page set, dropping any
citation that does not appear in it and marking the answer low confidence when any
were dropped. The groundedness check (app/programs.py CheckGrounded) is invoked by the
pipeline separately, after the stream completes, and is not part of this module.
"""

from __future__ import annotations

import logging

from app.errors import GuardrailAbstain
from app.retrieve import Candidate

logger = logging.getLogger(__name__)


def check_pre_generation(candidates: list[Candidate], retrieval_floor: float) -> None:
    """Raise GuardrailAbstain if generation must not proceed.

    Zero candidates always abstains, regardless of floor, so an empty context is never
    sent to the model.
    """

    if not candidates:
        raise GuardrailAbstain("no chunks retrieved")

    top_score = candidates[0].rerank_score
    if top_score is None:
        # No rerank score available, either reranking was skipped or degraded to
        # fusion order. Fusion scores are not on the same scale as rerank scores, so
        # there is nothing meaningful to compare against the floor, and the pipeline
        # proceeds to generation rather than abstaining on an incomparable value.
        return

    if top_score < retrieval_floor:
        raise GuardrailAbstain(
            f"top rerank score {top_score:.3f} below floor {retrieval_floor:.3f}"
        )


def validate_citations(citations: list[int], candidates: list[Candidate]) -> tuple[list[int], bool]:
    """Drop citations whose page is not covered by any retrieved candidate.

    Returns (valid_citations, any_dropped). any_dropped drives the low-confidence
    marking on the final answer.
    """

    retrieved_pages: set[int] = set()
    for c in candidates:
        retrieved_pages.update(range(c.page_start, c.page_end + 1))

    valid = [page for page in citations if page in retrieved_pages]
    any_dropped = len(valid) != len(citations)
    if any_dropped:
        dropped = [page for page in citations if page not in retrieved_pages]
        logger.warning("dropped citations not in retrieved pages: %s", dropped)

    return valid, any_dropped


if __name__ == "__main__":
    sample_candidates = [
        Candidate(
            chunk_uid="a",
            content="",
            heading_path=None,
            chapter=None,
            page_start=40,
            page_end=45,
            dense_rank=1,
            sparse_rank=1,
            fusion_score=0.03,
            rerank_score=0.6,
        )
    ]

    try:
        check_pre_generation(sample_candidates, retrieval_floor=0.35)
        print("pre generation guardrail passed")
    except GuardrailAbstain as exc:
        print(f"abstained: {exc}")

    try:
        check_pre_generation([], retrieval_floor=0.35)
    except GuardrailAbstain as exc:
        print(f"abstained on empty candidates: {exc}")

    valid, dropped = validate_citations([42, 99], sample_candidates)
    print(f"valid citations: {valid}, any dropped: {dropped}")
