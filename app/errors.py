"""Typed exceptions for the query and ingestion paths, see CLAUDE.md section 11.

Every handler that catches one of these must log and either degrade gracefully or
re raise. Never a bare except or an except Exception: pass.
"""

from __future__ import annotations


class ConfigError(Exception):
    """Required configuration is missing or malformed."""


class RetrievalError(Exception):
    """Both dense and sparse retrieval failed. Maps to HTTP 503."""


class RerankError(Exception):
    """Reranking failed or timed out. Callers should fall back to fusion order."""


class LLMError(Exception):
    """A generation or embedding call to the task or judge model failed after retries."""


class EmbeddingError(Exception):
    """An embedding call failed. Callers may fall back to sparse only retrieval."""


class DatabaseError(Exception):
    """A database operation failed."""


class GuardrailAbstain(Exception):
    """Raised internally to short circuit generation when a guardrail requires abstention.

    This is not an error in the ordinary sense, it is control flow for the case where
    the pipeline must not call the generation model at all, for example when the top
    rerank score is below RETRIEVAL_FLOOR or zero chunks were retrieved.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason
