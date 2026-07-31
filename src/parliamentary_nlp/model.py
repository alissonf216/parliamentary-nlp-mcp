"""Core inference engine for parliamentary discourse safety auditing.

Loads a Hugging Face sequence-classification model (default: the fine-tuned
parliamentary BERTimbau auditor) and returns structured predictions with
Shannon-entropy uncertainty scores suitable for human-in-the-loop review.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import TypedDict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Default production checkpoint. Override with PARLIAMENTARY_NLP_MODEL_ID
# or pass ``model_id`` to :class:`ParliamentaryModel`.
DEFAULT_MODEL_ID = "alissonf216/parliamentary-bertimbau-auditor"

# Canonical taxonomy for the custom parliamentary auditor (4-way single-label).
# Used when the loaded checkpoint has matching ``num_labels`` or generic LABEL_*
# names; otherwise the model's own ``id2label`` mapping is preferred.
LABELS: list[str] = [
    "NEUTRAL",
    "GENERIC_OFFENSE",
    "TARGETED_OFFENSE",
    "EXPLICIT_HATE_SPEECH",
]

# Predictions above this Shannon entropy (nats) are flagged for human review.
ENTROPY_REVIEW_THRESHOLD: float = 0.60

MAX_LENGTH: int = 512


class AuditResult(TypedDict):
    """Structured prediction returned by :meth:`ParliamentaryModel.predict`."""

    text: str
    classification: str
    confidence: float
    entropy_uncertainty: float
    class_probabilities: dict[str, float]
    requires_human_review: bool


def _resolve_device(device: str | None = None) -> torch.device:
    """Pick CUDA → Apple MPS → CPU unless an explicit device string is given."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def shannon_entropy(probabilities: list[float]) -> float:
    """Compute Shannon entropy ``H(X) = -Σ P(x) log P(x)`` in nats.

    Zero-probability bins are skipped to avoid ``log(0)``.
    """
    return -sum(p * math.log(p) for p in probabilities if p > 0.0)


def _resolve_labels(model: AutoModelForSequenceClassification) -> list[str]:
    """Prefer the research taxonomy when compatible; else use ``id2label``."""
    num_labels = int(model.config.num_labels)
    raw_id2label = getattr(model.config, "id2label", None) or {}
    # HF configs may store keys as str ("0") or int (0).
    id2label = {int(k): str(v) for k, v in raw_id2label.items()}

    config_labels = [
        id2label.get(i, f"LABEL_{i}") for i in range(num_labels)
    ]

    generic = all(label.startswith("LABEL_") for label in config_labels)
    if num_labels == len(LABELS) and (generic or config_labels == LABELS):
        return list(LABELS)
    return config_labels


class ParliamentaryModel:
    """Hugging Face wrapper for offensive / hate-speech classification.

    Parameters
    ----------
    model_id:
        Hugging Face model id or local path. Defaults to ``DEFAULT_MODEL_ID``,
        overridable via the ``PARLIAMENTARY_NLP_MODEL_ID`` environment variable.
    device:
        Optional torch device string (``\"cpu\"``, ``\"cuda\"``, ``\"mps\"``).
    entropy_threshold:
        Shannon-entropy cutoff for ``requires_human_review``.
    """

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        entropy_threshold: float = ENTROPY_REVIEW_THRESHOLD,
    ) -> None:
        self.model_id = (
            model_id
            or os.getenv("PARLIAMENTARY_NLP_MODEL_ID")
            or DEFAULT_MODEL_ID
        )
        self.device = _resolve_device(device)
        self.entropy_threshold = entropy_threshold

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()

        self.labels = _resolve_labels(self.model)

    def predict(self, text: str) -> AuditResult:
        """Classify ``text`` and return a structured audit payload.

        Pipeline: tokenize → forward pass (no grad) → softmax → Shannon entropy.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            logits = self.model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1).squeeze(0)

        probs_list = [float(p) for p in probabilities.tolist()]
        if len(probs_list) != len(self.labels):
            raise RuntimeError(
                f"Label/logit mismatch: got {len(probs_list)} probabilities "
                f"for {len(self.labels)} labels"
            )

        top_index = int(probabilities.argmax().item())
        confidence = probs_list[top_index]
        entropy = round(shannon_entropy(probs_list), 4)

        class_probabilities = {
            label: round(prob, 6) for label, prob in zip(self.labels, probs_list)
        }

        return AuditResult(
            text=text,
            classification=self.labels[top_index],
            confidence=round(confidence, 6),
            entropy_uncertainty=entropy,
            class_probabilities=class_probabilities,
            requires_human_review=entropy > self.entropy_threshold,
        )


@lru_cache(maxsize=1)
def get_model() -> ParliamentaryModel:
    """Return a process-wide singleton (lazy load on first MCP tool call)."""
    return ParliamentaryModel()
