"""Unit and integration tests for production Banking77Predictor."""

import numpy as np
import pytest
import torch

from src.inference import Banking77Predictor, InferenceOutput


@pytest.fixture(scope="module")
def predictor():
    """Load predictor once for inference test suite."""
    return Banking77Predictor()


def test_predictor_initialization(predictor):
    """Verify model and tokenizer initialization, device selection, and eval mode."""
    assert predictor.model is not None
    assert predictor.tokenizer is not None
    assert predictor.model.training is False
    assert all(not p.requires_grad for p in predictor.model.parameters())
    assert len(predictor.id2label) == 77
    assert len(predictor.label2id) == 77
    assert predictor.initialization_time_ms > 0.0


def test_single_prediction_format(predictor):
    """Verify single query prediction schema, confidence bounds, and candidate ranking."""
    query = "My card was stolen"
    result = predictor.predict(query, top_k=3, return_all_probs=True)

    assert isinstance(result, InferenceOutput)
    assert result.query == query
    assert result.predicted_intent in predictor.label2id
    assert 0 <= result.predicted_class_id < 77
    assert 0.0 <= result.confidence <= 1.0
    assert result.latency_ms > 0.0

    # Top-3 checks
    assert len(result.top_k_predictions) == 3
    probs = [c.probability for c in result.top_k_predictions]
    assert probs == sorted(probs, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in probs)

    # All probabilities sum check
    assert result.all_probabilities is not None
    assert len(result.all_probabilities) == 77
    total_prob = sum(result.all_probabilities.values())
    assert np.isclose(total_prob, 1.0, atol=1e-4)


def test_five_target_queries(predictor):
    """Verify inference runs cleanly and returns valid intents for the 5 requested example queries."""
    target_queries = [
        "My card was stolen",
        "Why do I need to verify my identity?",
        "My transfer is still pending",
        "I cannot use my virtual card",
        "How long will an international transfer take?",
    ]

    for q in target_queries:
        out = predictor.predict(q, top_k=3)
        assert out.query == q
        assert out.predicted_intent in predictor.label2id
        assert 0.0 <= out.confidence <= 1.0
        assert len(out.top_k_predictions) == 3
        assert out.latency_ms > 0.0


def test_batch_prediction_consistency(predictor):
    """Verify batch predictions match single-query prediction outputs."""
    queries = [
        "I need a new pin",
        "Where is my transfer?",
        "How do I activate my card?",
    ]
    batch_results = predictor.batch_predict(queries, top_k=3)
    assert len(batch_results) == len(queries)

    for q, b_res in zip(queries, batch_results):
        s_res = predictor.predict(q, top_k=3)
        assert b_res.predicted_intent == s_res.predicted_intent
        assert b_res.predicted_class_id == s_res.predicted_class_id
        assert np.isclose(b_res.confidence, s_res.confidence, atol=1e-3)


def test_input_validation(predictor):
    """Verify predictor rejects empty or non-string queries."""
    with pytest.raises(ValueError):
        predictor.predict("")

    with pytest.raises(ValueError):
        predictor.predict("   ")

    with pytest.raises(TypeError):
        predictor.predict(12345)  # type: ignore


def test_internal_logits_and_inference_softmax(predictor):
    """Verify model produces raw unnormalized logits and softmax is applied strictly at inference time."""
    inputs = predictor.tokenizer("Card declined", return_tensors="pt")
    inputs = {k: v.to(predictor.device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = predictor.model(**inputs)
        raw_logits = outputs.logits.cpu().numpy()[0]

    # Raw logits can be negative and do not sum to 1
    assert not np.isclose(np.sum(raw_logits), 1.0, atol=0.1)

    # Softmax normalizes them to sum to 1
    probs = torch.softmax(torch.tensor(raw_logits), dim=-1).numpy()
    assert np.isclose(np.sum(probs), 1.0, atol=1e-4)
