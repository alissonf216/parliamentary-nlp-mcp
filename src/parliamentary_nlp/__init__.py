"""Parliamentary NLP — MCP server for discourse safety auditing."""

from parliamentary_nlp.model import (
    DEFAULT_MODEL_ID,
    ENTROPY_REVIEW_THRESHOLD,
    LABELS,
    AuditResult,
    ParliamentaryModel,
)

__all__ = [
    "DEFAULT_MODEL_ID",
    "ENTROPY_REVIEW_THRESHOLD",
    "LABELS",
    "AuditResult",
    "ParliamentaryModel",
]

__version__ = "0.1.0"
