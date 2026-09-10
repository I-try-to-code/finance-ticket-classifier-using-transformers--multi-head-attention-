"""Unit tests for dataset loading and schema verification."""

import pytest
import pandas as pd
from src.loader import load_banking77, LoadedDataset
from src.config import DATASET_NAME, EXPECTED_NUM_INTENTS


def test_load_banking77_schema_and_types():
    """Verify official dataset loads with expected shapes, columns, and types."""
    loaded = load_banking77(DATASET_NAME)

    assert isinstance(loaded, LoadedDataset)
    assert isinstance(loaded.train_df, pd.DataFrame)
    assert isinstance(loaded.test_df, pd.DataFrame)

    # Check row counts match official PolyAI/BANKING77 ground truth
    assert len(loaded.train_df) == 10003
    assert len(loaded.test_df) == 3080

    # Required columns
    for col in ["text", "label", "intent_name"]:
        assert col in loaded.train_df.columns
        assert col in loaded.test_df.columns

    # Label space
    assert len(loaded.label_names) == EXPECTED_NUM_INTENTS
    assert loaded.train_df["label"].nunique() == EXPECTED_NUM_INTENTS
    assert loaded.test_df["label"].nunique() == EXPECTED_NUM_INTENTS

    # Type verification
    assert loaded.train_df["text"].dtype == object
    assert pd.api.types.is_integer_dtype(loaded.train_df["label"])
    assert loaded.train_df["intent_name"].dtype == object


def test_load_banking77_invalid_dataset_raises():
    """Verify loader raises descriptive error when dataset identifier is nonexistent."""
    with pytest.raises(RuntimeError):
        load_banking77("nonexistent_dummy_banking_dataset_12345")
