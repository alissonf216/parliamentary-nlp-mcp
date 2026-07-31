"""Unit tests for prediction schema and entropy helpers (no model download)."""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from parliamentary_nlp.model import (
    ENTROPY_REVIEW_THRESHOLD,
    LABELS,
    ParliamentaryModel,
    shannon_entropy,
)


def test_shannon_entropy_uniform_four_classes() -> None:
    """Maximum entropy for 4 equiprobable classes is ln(4)."""
    probs = [0.25, 0.25, 0.25, 0.25]
    assert shannon_entropy(probs) == pytest.approx(math.log(4), abs=1e-6)


def test_shannon_entropy_deterministic() -> None:
    """A one-hot distribution has zero uncertainty."""
    assert shannon_entropy([1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.0)


def test_predict_schema_with_mocked_forward() -> None:
    """``predict`` must return the documented AuditResult keys and types."""
    logits = torch.tensor([[4.0, 1.0, 0.5, 0.1]])  # NEUTRAL dominates

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[101, 102]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }

    mock_hf_model = MagicMock()
    mock_hf_model.config.num_labels = 4
    mock_hf_model.config.id2label = {i: label for i, label in enumerate(LABELS)}
    mock_output = MagicMock()
    mock_output.logits = logits
    mock_hf_model.return_value = mock_output
    mock_hf_model.to.return_value = mock_hf_model

    with (
        patch(
            "parliamentary_nlp.model.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ),
        patch(
            "parliamentary_nlp.model.AutoModelForSequenceClassification.from_pretrained",
            return_value=mock_hf_model,
        ),
    ):
        engine = ParliamentaryModel(model_id="mock/model", device="cpu")

    result = engine.predict("Discurso parlamentar de teste.")

    assert set(result.keys()) == {
        "text",
        "classification",
        "confidence",
        "entropy_uncertainty",
        "class_probabilities",
        "requires_human_review",
    }
    assert result["text"] == "Discurso parlamentar de teste."
    assert result["classification"] == "NEUTRAL"
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["entropy_uncertainty"], float)
    assert result["entropy_uncertainty"] == round(result["entropy_uncertainty"], 4)
    assert set(result["class_probabilities"].keys()) == set(LABELS)
    assert abs(sum(result["class_probabilities"].values()) - 1.0) < 1e-5
    assert isinstance(result["requires_human_review"], bool)
    assert result["requires_human_review"] is (
        result["entropy_uncertainty"] > ENTROPY_REVIEW_THRESHOLD
    )


def test_predict_rejects_empty_text() -> None:
    engine = ParliamentaryModel.__new__(ParliamentaryModel)
    engine.labels = list(LABELS)
    engine.entropy_threshold = ENTROPY_REVIEW_THRESHOLD
    with pytest.raises(ValueError, match="non-empty"):
        engine.predict("   ")


def test_high_entropy_triggers_human_review() -> None:
    """Near-uniform logits should flag ``requires_human_review``."""
    logits = torch.tensor([[0.1, 0.1, 0.1, 0.1]])

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[101]]),
        "attention_mask": torch.tensor([[1]]),
    }
    mock_hf_model = MagicMock()
    mock_hf_model.config.num_labels = 4
    mock_hf_model.config.id2label = {i: label for i, label in enumerate(LABELS)}
    mock_output = MagicMock()
    mock_output.logits = logits
    mock_hf_model.return_value = mock_output
    mock_hf_model.to.return_value = mock_hf_model

    with (
        patch(
            "parliamentary_nlp.model.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ),
        patch(
            "parliamentary_nlp.model.AutoModelForSequenceClassification.from_pretrained",
            return_value=mock_hf_model,
        ),
    ):
        engine = ParliamentaryModel(model_id="mock/model", device="cpu")

    result: dict[str, Any] = engine.predict("Texto ambíguo.")
    assert result["entropy_uncertainty"] > ENTROPY_REVIEW_THRESHOLD
    assert result["requires_human_review"] is True
