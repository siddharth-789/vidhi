"""Loads and validates data/trainset.csv, see CLAUDE.md section 10.

trainset.csv is the single source of truth for evaluation. This module fails loudly
on malformed rows rather than silently skipping them, since a silently dropped row
would understate coverage without anyone noticing.

Run standalone with: python -m eval.dataset
This loads data/trainset.csv, validates every row, and prints a split and category
breakdown.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

_VALID_CATEGORIES = {"lookup", "rate", "procedure", "multi_hop", "out_of_scope"}
_VALID_SPLITS = {"train", "dev"}
_REQUIRED_COLUMNS = {
    "id",
    "question",
    "gold_answer",
    "gold_pages",
    "category",
    "answerable",
    "split",
}


class DatasetError(Exception):
    """A row in trainset.csv is malformed. Fails loudly, see CLAUDE.md section 10."""


@dataclass(frozen=True)
class EvalExample:
    id: str
    question: str
    gold_answer: str
    gold_pages: list[int]
    category: str
    answerable: bool
    split: str


def _parse_bool(raw: str, row_id: str) -> bool:
    normalized = raw.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise DatasetError(f"row {row_id}: answerable must be 'true' or 'false', got {raw!r}")


def _parse_gold_pages(raw: str, row_id: str, answerable: bool) -> list[int]:
    raw = raw.strip()
    if not raw:
        if answerable:
            raise DatasetError(f"row {row_id}: answerable rows must have non-empty gold_pages")
        return []
    try:
        return [int(p) for p in raw.split(";") if p.strip()]
    except ValueError as exc:
        raise DatasetError(f"row {row_id}: malformed gold_pages {raw!r}") from exc


def load_trainset(path: str | Path = "data/trainset.csv") -> list[EvalExample]:
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"trainset not found at {path}")

    examples: list[EvalExample] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = _REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise DatasetError(f"trainset is missing required columns: {sorted(missing)}")

        for row in reader:
            row_id = row["id"].strip()
            if not row_id:
                raise DatasetError("row with empty id")

            question = row["question"].strip()
            if not question:
                raise DatasetError(f"row {row_id}: empty question")

            category = row["category"].strip()
            if category not in _VALID_CATEGORIES:
                raise DatasetError(
                    f"row {row_id}: category must be one of {sorted(_VALID_CATEGORIES)}, "
                    f"got {category!r}"
                )

            split = row["split"].strip()
            if split not in _VALID_SPLITS:
                raise DatasetError(
                    f"row {row_id}: split must be one of {sorted(_VALID_SPLITS)}, got {split!r}"
                )

            answerable = _parse_bool(row["answerable"], row_id)
            gold_pages = _parse_gold_pages(row["gold_pages"], row_id, answerable)

            examples.append(
                EvalExample(
                    id=row_id,
                    question=question,
                    gold_answer=row["gold_answer"].strip(),
                    gold_pages=gold_pages,
                    category=category,
                    answerable=answerable,
                    split=split,
                )
            )

    return examples


def filter_split(examples: list[EvalExample], split: str) -> list[EvalExample]:
    return [e for e in examples if e.split == split]


def to_dspy_examples(examples: list[EvalExample]) -> list:
    """Converts train split rows into DSPy examples for the optimizer, see section 9.

    Imports dspy lazily so eval/dataset.py stays importable without dspy installed,
    for callers that only need CSV validation.
    """

    import dspy

    return [
        dspy.Example(
            question=e.question,
            gold_answer=e.gold_answer,
            gold_pages=e.gold_pages,
            category=e.category,
            answerable=e.answerable,
        ).with_inputs("question")
        for e in examples
    ]


if __name__ == "__main__":
    all_examples = load_trainset()
    by_split: dict[str, int] = {}
    by_category: dict[str, int] = {}
    for ex in all_examples:
        by_split[ex.split] = by_split.get(ex.split, 0) + 1
        by_category[ex.category] = by_category.get(ex.category, 0) + 1

    print(f"total rows: {len(all_examples)}")
    print(f"by split: {by_split}")
    print(f"by category: {by_category}")
