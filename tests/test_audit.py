"""Unit tests for audit metrics and quality diagnostics."""

import numpy as np
import pandas as pd
import pytest
from src.audit import (
    compute_length_stats,
    compute_class_distribution,
    perform_quality_diagnostics,
    audit_dataset,
)


def test_compute_length_stats():
    """Verify statistical summary calculations."""
    series = pd.Series([10, 20, 30, 40, 50])
    stats = compute_length_stats(series)

    assert stats.min == 10
    assert stats.max == 50
    assert stats.mean == 30.0
    assert stats.median == 30.0
    assert round(stats.std, 2) == 15.81

    # Empty series edge case
    empty_stats = compute_length_stats(pd.Series([], dtype=float))
    assert empty_stats.min == 0
    assert empty_stats.max == 0
    assert empty_stats.mean == 0.0


def test_compute_class_distribution():
    """Verify class frequency distribution calculations."""
    df = pd.DataFrame({
        "intent_name": ["card_lost", "card_lost", "card_lost", "transfer_fee", "transfer_fee"]
    })
    dist = compute_class_distribution(df, label_col="intent_name")

    assert dist.num_classes == 2
    assert dist.counts_per_class["card_lost"] == 3
    assert dist.counts_per_class["transfer_fee"] == 2
    assert dist.percentages_per_class["card_lost"] == 60.0
    assert dist.percentages_per_class["transfer_fee"] == 40.0
    assert dist.min_count == 2
    assert dist.max_count == 3


def test_quality_diagnostics_detects_anomalies():
    """Verify diagnostic detection of nulls, empty strings, HTML, control chars, and non-ASCII."""
    df_synthetic = pd.DataFrame({
        "text": [
            "Valid banking query regarding pin change",
            None,                       # Null
            "",                         # Empty string
            "   \t  ",                  # Whitespace-only
            "I want a refund <br> please",  # HTML tag
            "Query &amp; support please",    # HTML entity
            "Text with \u0000 null byte",    # Control char
            "Payment of £50 and €100",       # Non-ASCII currency
        ]
    })

    diag = perform_quality_diagnostics(df_synthetic, text_col="text")

    assert diag.null_count == 1
    assert diag.empty_count == 1
    assert diag.whitespace_only_count == 1
    assert diag.html_markup_text_count == 2  # <br> and &amp;
    assert diag.control_char_text_count == 1  # \u0000
    assert diag.non_ascii_text_count == 1    # £ and €
    assert "£" in diag.non_ascii_char_details
    assert "€" in diag.non_ascii_char_details
    assert diag.non_ascii_char_details["£"]["unicode_name"] == "POUND SIGN"
    assert diag.non_ascii_char_details["€"]["unicode_name"] == "EURO SIGN"


def test_audit_dataset_cross_split_duplicates():
    """Verify detection of overlapping records across train and test splits."""
    train_df = pd.DataFrame({
        "text": ["Where is my card?", "How to top up?", "Common query"],
        "intent_name": ["card_arrival", "top_up", "general"],
    })
    test_df = pd.DataFrame({
        "text": ["Different query", "Common query"],
        "intent_name": ["general", "general"],
    })

    audit_res = audit_dataset(
        train_df=train_df,
        test_df=test_df,
        intent_names=["card_arrival", "top_up", "general"],
        dataset_name="mock_dataset",
    )

    assert audit_res.cross_split_duplicate_count == 1
    assert "Common query" in audit_res.cross_split_duplicate_examples
