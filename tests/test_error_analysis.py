"""Unit and integration tests for Banking77 error analysis engine."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.analysis.categorizer import build_qualitative_error_taxonomy
from src.analysis.collector import ValidationPredictionRow, evaluate_probabilities
from src.analysis.confidence import extract_confidence_analysis
from src.analysis.confusion import (
    export_confusion_pairs_csv,
    extract_top_confusion_pairs,
)
from src.analysis.focused import (
    assign_priority_tier,
    extract_focused_intent_examples,
)


@pytest.fixture
def mock_validation_rows():
    """Create synthetic validation rows covering various error combinations."""
    rows = []
    # Class 0: virtual_card_not_working
    # Sample 0: both wrong
    rows.append(
        ValidationPredictionRow(
            sample_id=0,
            text="My virtual card is broken",
            true_label_id=0,
            true_intent="virtual_card_not_working",
            distilbert_pred_id=1,
            distilbert_pred_intent="getting_virtual_card",
            distilbert_confidence=0.88,
            distilbert_second_intent="card_not_working",
            distilbert_second_prob=0.08,
            distilbert_margin=0.80,
            distilbert_correct=False,
            bert_pred_id=1,
            bert_pred_intent="getting_virtual_card",
            bert_confidence=0.82,
            bert_second_intent="card_not_working",
            bert_second_prob=0.10,
            bert_margin=0.72,
            bert_correct=False,
        )
    )
    # Sample 1: DistilBERT wrong, BERT correct
    rows.append(
        ValidationPredictionRow(
            sample_id=1,
            text="Can't use my virtual card online",
            true_label_id=0,
            true_intent="virtual_card_not_working",
            distilbert_pred_id=1,
            distilbert_pred_intent="getting_virtual_card",
            distilbert_confidence=0.76,
            distilbert_second_intent="virtual_card_not_working",
            distilbert_second_prob=0.20,
            distilbert_margin=0.56,
            distilbert_correct=False,
            bert_pred_id=0,
            bert_pred_intent="virtual_card_not_working",
            bert_confidence=0.65,
            bert_second_intent="getting_virtual_card",
            bert_second_prob=0.30,
            bert_margin=0.35,
            bert_correct=True,
        )
    )
    # Sample 2: BERT wrong, DistilBERT correct
    rows.append(
        ValidationPredictionRow(
            sample_id=2,
            text="My physical card is chipped and not working",
            true_label_id=2,
            true_intent="card_not_working",
            distilbert_pred_id=2,
            distilbert_pred_intent="card_not_working",
            distilbert_confidence=0.48,  # low confidence correct
            distilbert_second_intent="contactless_not_working",
            distilbert_second_prob=0.44,
            distilbert_margin=0.04,  # semantic margin uncertainty
            distilbert_correct=True,
            bert_pred_id=3,
            bert_pred_intent="contactless_not_working",
            bert_confidence=0.55,
            bert_second_intent="card_not_working",
            bert_second_prob=0.40,
            bert_margin=0.15,
            bert_correct=False,
        )
    )
    # Sample 3: Both correct
    rows.append(
        ValidationPredictionRow(
            sample_id=3,
            text="What is the exchange rate for USD to EUR?",
            true_label_id=4,
            true_intent="exchange_rate",
            distilbert_pred_id=4,
            distilbert_pred_intent="exchange_rate",
            distilbert_confidence=0.95,
            distilbert_second_intent="card_payment_wrong_exchange_rate",
            distilbert_second_prob=0.03,
            distilbert_margin=0.92,
            distilbert_correct=True,
            bert_pred_id=4,
            bert_pred_intent="exchange_rate",
            bert_confidence=0.98,
            bert_second_intent="card_payment_wrong_exchange_rate",
            bert_second_prob=0.01,
            bert_margin=0.97,
            bert_correct=True,
        )
    )
    return rows


def test_confusion_pairs_percentage_calculation():
    """Verify calculation of percent of true class affected in confusion pairs."""
    id_to_label = {0: "card_error", 1: "transfer_error", 2: "login_error"}
    # Class 0 support = 10 (4 correct, 6 confused with class 1)
    cm = np.array([
        [4, 6, 0],
        [1, 9, 0],
        [0, 0, 10],
    ])
    pairs = extract_top_confusion_pairs(cm, id_to_label, "TestModel", top_n=5)
    assert len(pairs) == 2
    top_pair = pairs[0]
    assert top_pair.true_intent == "card_error"
    assert top_pair.predicted_intent == "transfer_error"
    assert top_pair.error_count == 6
    assert top_pair.class_support == 10
    assert np.isclose(top_pair.percent_true_class_affected, 60.0)


def test_priority_tier_assignment():
    """Verify 4-tier logic for focused intent comparisons."""
    assert assign_priority_tier(False, False) == (1, "both_wrong")
    assert assign_priority_tier(False, True) == (2, "distilbert_wrong_bert_correct")
    assert assign_priority_tier(True, False) == (3, "bert_wrong_distilbert_correct")
    assert assign_priority_tier(True, True) == (4, "both_correct")


def test_focused_intent_prioritization(mock_validation_rows):
    """Verify focused intent examples are prioritized by error severity tiers."""
    records = extract_focused_intent_examples(
        rows=mock_validation_rows,
        target_intents=["virtual_card_not_working"],
        min_examples_per_intent=5,
    )
    assert len(records) == 2
    # Tier 1 (both wrong) should precede Tier 2 (DistilBERT wrong, BERT correct)
    assert records[0].priority_tier == 1
    assert records[0].comparison_category == "both_wrong"
    assert records[1].priority_tier == 2
    assert records[1].comparison_category == "distilbert_wrong_bert_correct"


def test_confidence_analysis_filtering(mock_validation_rows):
    """Verify high-confidence error, low-confidence correct, and margin uncertainty filtering."""
    conf = extract_confidence_analysis(
        rows=mock_validation_rows,
        high_conf_threshold=0.75,
        low_conf_threshold=0.50,
        margin_uncertainty_threshold=0.15,
    )

    # DistilBERT has 2 errors: sample 0 (conf=0.88), sample 1 (conf=0.76) - both >= 0.75
    assert len(conf["distilbert_high_confidence_errors"]) == 2
    assert conf["distilbert_high_confidence_errors"][0].sample_id == 0

    # DistilBERT low-confidence correct: sample 2 (conf=0.48 < 0.50)
    assert len(conf["distilbert_low_confidence_correct"]) == 1
    assert conf["distilbert_low_confidence_correct"][0].sample_id == 2

    # DistilBERT margin uncertainty (sample 2 has margin 0.04 <= 0.15)
    assert len(conf["distilbert_semantic_uncertain"]) >= 1


def test_export_csvs(mock_validation_rows, tmp_path):
    """Verify CSV export functionality."""
    cm = np.zeros((3, 3), dtype=int)
    cm[0, 1] = 2
    cm[0, 0] = 5
    pairs = extract_top_confusion_pairs(cm, {0: "a", 1: "b", 2: "c"}, "TestModel")

    csv_path = tmp_path / "test_confusion.csv"
    df = export_confusion_pairs_csv(pairs, pairs, csv_path)
    assert csv_path.exists()
    assert "percent_true_class_affected" in df.columns
    assert len(df) == 2


def test_taxonomy_extraction(mock_validation_rows):
    """Verify qualitative taxonomy attaches valid evidence."""
    tax = build_qualitative_error_taxonomy(mock_validation_rows)
    assert len(tax) > 0
    categories = [t.root_cause_category for t in tax]
    assert "insufficient distinction between “how to obtain” and “not working”" in categories
