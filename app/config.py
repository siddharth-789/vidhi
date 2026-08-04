"""Single source of truth for environment configuration.

Every module that needs a model name, threshold, key, or connection string reads it
from here rather than from os.environ directly. See CLAUDE.md section 3.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache

from app.errors import ConfigError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    embed_model: str
    embed_dim: int
    task_model: str
    prompt_model: str
    judge_model: str

    gemini_api_key_task: str | None
    gemini_api_key_judge: str | None

    rerank_provider: str
    rerank_api_key: str | None

    database_url: str | None

    retrieval_floor: float
    compiled_program_path: str


def _load() -> Config:
    task_model = os.environ.get("TASK_MODEL", "gemini-3.1-flash-lite")

    gemini_api_key_task = os.environ.get("GEMINI_API_KEY_TASK")
    gemini_api_key_judge = os.environ.get("GEMINI_API_KEY_JUDGE")
    if not gemini_api_key_task:
        logger.warning("GEMINI_API_KEY_TASK is not set, serving and optimizing will fail")
    if not gemini_api_key_judge:
        logger.warning("GEMINI_API_KEY_JUDGE is not set, evaluation and judging will fail")

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.warning("DATABASE_URL is not set, database access will fail")

    return Config(
        embed_model=os.environ.get("EMBED_MODEL", "gemini-embedding-2"),
        embed_dim=int(os.environ.get("EMBED_DIM", "768")),
        task_model=task_model,
        prompt_model=os.environ.get("PROMPT_MODEL", task_model),
        judge_model=os.environ.get("JUDGE_MODEL", task_model),
        gemini_api_key_task=gemini_api_key_task,
        gemini_api_key_judge=gemini_api_key_judge,
        rerank_provider=os.environ.get("RERANK_PROVIDER", "llm"),
        rerank_api_key=os.environ.get("RERANK_API_KEY"),
        database_url=database_url,
        retrieval_floor=float(os.environ.get("RETRIEVAL_FLOOR", "0.35")),
        compiled_program_path=os.environ.get(
            "COMPILED_PROGRAM_PATH", "artifacts/compiled_program.json"
        ),
    )


@lru_cache(maxsize=1)
def get_config() -> Config:
    return _load()


def require_serving_config(config: Config) -> None:
    """Raise ConfigError if a value required to serve requests is missing.

    Called once at API startup, see CLAUDE.md section 12.1. Judging keys are not
    required here because evaluation is a separate process.
    """
    if not config.database_url:
        raise ConfigError("DATABASE_URL is required to serve requests")
    if not config.gemini_api_key_task:
        raise ConfigError("GEMINI_API_KEY_TASK is required to serve requests")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    cfg = get_config()
    secret_fields = {"gemini_api_key_task", "gemini_api_key_judge", "rerank_api_key", "database_url"}
    for field_name in cfg.__dataclass_fields__:
        value = getattr(cfg, field_name)
        if field_name in secret_fields and value:
            value = f"<set, {len(value)} chars>"
        print(f"{field_name}: {value}")
