"""Unit and integration tests for BERT-base tokenizer, model, inference, and reload."""

from pathlib import Path
import numpy as np
import pytest
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.bert.config import BertConfig
from src.bert.dataset import BankingBertDataset, DynamicPaddingBertCollator
from src.bert.predictor import BertPredictor


@pytest.fixture(scope="module")
def bert_components():
    """Load tokenizer and fresh model initialized for 77 classes once for testing."""
    config = BertConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    id2label = {i: f"intent_{i}" for i in range(77)}
    label2id = {v: k for k, v in id2label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=77,
        id2label=id2label,
        label2id=label2id,
    )
    return tokenizer, model, id2label, label2id


def test_bert_tokenizer_model_compatibility(bert_components):
    """Verify BERT tokenizer outputs compatible tensors (including token_type_ids) for model forward pass."""
    tokenizer, model, _, _ = bert_components
    texts = ["I lost my credit card", "How do I top up my balance?"]

    inputs = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    assert "input_ids" in inputs
    assert "attention_mask" in inputs
    assert "token_type_ids" in inputs

    with torch.no_grad():
        outputs = model(**inputs)

    assert outputs.logits.shape == (2, 77)


def test_bert_label_mapping_integrity(bert_components):
    """Verify bidirectional 77-class label mapping for BERT."""
    _, model, id2label, label2id = bert_components

    assert len(id2label) == 77
    assert len(label2id) == 77

    for i in range(77):
        assert i in id2label
        name = id2label[i]
        assert label2id[name] == i

    assert model.config.num_labels == 77


def test_bert_dynamic_padding_collator(bert_components):
    """Verify BERT dynamic padding collator adjusts batch to max length and pads token_type_ids."""
    tokenizer, _, _, _ = bert_components
    collator = DynamicPaddingBertCollator(tokenizer=tokenizer)

    batch_items = [
        {
            "input_ids": [101, 2054, 102],
            "attention_mask": [1, 1, 1],
            "token_type_ids": [0, 0, 0],
            "label": 0,
            "text": "What",
        },
        {
            "input_ids": [101, 2054, 2003, 1037, 102],
            "attention_mask": [1, 1, 1, 1, 1],
            "token_type_ids": [0, 0, 0, 0, 0],
            "label": 1,
            "text": "What is a",
        },
    ]

    padded = collator(batch_items)
    assert padded["input_ids"].shape == (2, 5)
    assert padded["attention_mask"].shape == (2, 5)
    assert padded["token_type_ids"].shape == (2, 5)
    assert padded["labels"].shape == (2,)
    assert len(padded["texts"]) == 2


def test_bert_inference_output_format(bert_components):
    """Verify BertPredictor output structure and probability constraints."""
    tokenizer, model, _, _ = bert_components
    predictor = BertPredictor(model=model, tokenizer=tokenizer, device=torch.device("cpu"))

    query = "My card payment was declined yesterday"
    result = predictor.predict(query, top_k=3, return_all_probs=True)

    assert result.input_text == query
    assert isinstance(result.predicted_intent, str)
    assert 0 <= result.predicted_class_id < 77
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.top_3_predictions) == 3

    probs = [c.probability for c in result.top_3_predictions]
    assert probs == sorted(probs, reverse=True)

    assert len(result.all_probabilities) == 77
    prob_sum = sum(result.all_probabilities.values())
    assert np.isclose(prob_sum, 1.0, atol=1e-4)


def test_bert_saved_model_reload_and_prediction(bert_components, tmp_path):
    """Verify saved BERT checkpoint can be reloaded from disk and run without errors."""
    tokenizer, model, _, _ = bert_components
    save_dir = tmp_path / "bert_test_save"
    save_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    reloaded_predictor = BertPredictor.from_pretrained(save_dir, device="cpu")

    query = "Can I activate my card through the mobile app?"
    result = reloaded_predictor.predict(query, top_k=3)

    assert result.input_text == query
    assert isinstance(result.predicted_intent, str)
    assert len(result.top_3_predictions) == 3


def test_bert_batch_prediction(bert_components):
    """Verify batch prediction consistency for BERT."""
    tokenizer, model, _, _ = bert_components
    predictor = BertPredictor(model=model, tokenizer=tokenizer, device=torch.device("cpu"))

    queries = [
        "Card was stolen while on holiday",
        "Why is there an extra fee on my statement?",
        "Can I exchange currencies via the mobile app?",
    ]
    results = predictor.batch_predict(queries, top_k=3)

    assert len(results) == len(queries)
    for q, r in zip(queries, results):
        assert r.input_text == q
        assert len(r.top_3_predictions) == 3
        assert r.confidence > 0.0
