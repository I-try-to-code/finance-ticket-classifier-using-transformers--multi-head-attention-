"""Unit tests for reproducible stratified splitting and integrity verification."""

import json
from pathlib import Path
import pandas as pd
import pytest

from src.loader import load_banking77
from src.splitter import (
    create_reproducible_split,
    save_processed_splits,
    verify_split_integrity,
    SplitResult,
)


@pytest.fixture(scope="module")
def banking77_data():
    """Load dataset once for module tests."""
    return load_banking77()


def test_splitter_reproducibility(banking77_data):
    """Verify that using the same random seed yields identical splits."""
    split_1 = create_reproducible_split(
        train_df=banking77_data.train_df,
        test_df=banking77_data.test_df,
        val_size=0.20,
        random_seed=42,
    )
    split_2 = create_reproducible_split(
        train_df=banking77_data.train_df,
        test_df=banking77_data.test_df,
        val_size=0.20,
        random_seed=42,
    )

    # Identical index selection and row values
    pd.testing.assert_frame_equal(split_1.train_df, split_2.train_df)
    pd.testing.assert_frame_equal(split_1.val_df, split_2.val_df)


def test_splitter_different_seeds(banking77_data):
    """Verify different seeds produce different sample assignments."""
    split_a = create_reproducible_split(
        train_df=banking77_data.train_df,
        test_df=banking77_data.test_df,
        val_size=0.20,
        random_seed=42,
    )
    split_b = create_reproducible_split(
        train_df=banking77_data.train_df,
        test_df=banking77_data.test_df,
        val_size=0.20,
        random_seed=123,
    )

    # They should not have identical indices
    assert not split_a.train_df.index.equals(split_b.train_df.index)


def test_stratification_and_class_coverage(banking77_data):
    """Verify all 77 classes exist in both splits with preserved proportions."""
    split = create_reproducible_split(
        train_df=banking77_data.train_df,
        test_df=banking77_data.test_df,
        val_size=0.20,
        random_seed=42,
    )

    # 1. Total row split matches 80/20 of 10,003
    assert len(split.train_df) == 8002
    assert len(split.val_df) == 2001
    assert len(split.train_df) + len(split.val_df) == 10003

    # 2. Every intent exists in both train and validation
    assert split.train_df["label"].nunique() == 77
    assert split.val_df["label"].nunique() == 77

    # 3. Class proportions preserved within tolerance < 0.001
    assert split.verification.max_prop_diff_train < 0.001
    assert split.verification.max_prop_diff_val < 0.001


def test_disjoint_partition_zero_leakage(banking77_data):
    """Verify no accidental overlap in index or text between train and validation."""
    split = create_reproducible_split(
        train_df=banking77_data.train_df,
        test_df=banking77_data.test_df,
        val_size=0.20,
        random_seed=42,
    )

    train_indices = set(split.train_df.index)
    val_indices = set(split.val_df.index)
    assert len(train_indices.intersection(val_indices)) == 0

    train_texts = set(split.train_df["text"])
    val_texts = set(split.val_df["text"])
    assert len(train_texts.intersection(val_texts)) == 0


def test_official_test_split_untouched(banking77_data):
    """Verify official test set remains completely identical."""
    split = create_reproducible_split(
        train_df=banking77_data.train_df,
        test_df=banking77_data.test_df,
        val_size=0.20,
        random_seed=42,
    )

    pd.testing.assert_frame_equal(split.test_df, banking77_data.test_df)
    assert len(split.test_df) == 3080


def test_save_processed_splits(banking77_data, tmp_path):
    """Verify artifact export to CSV, Parquet, and JSON metadata."""
    split = create_reproducible_split(
        train_df=banking77_data.train_df,
        test_df=banking77_data.test_df,
        val_size=0.20,
        random_seed=42,
    )

    paths = save_processed_splits(split, output_dir=tmp_path)

    # Verify CSV files
    assert paths["train"].exists()
    assert paths["val"].exists()
    assert paths["test"].exists()
    assert paths["metadata"].exists()

    # Verify Parquet files
    assert (tmp_path / "train.parquet").exists()
    assert (tmp_path / "val.parquet").exists()
    assert (tmp_path / "test.parquet").exists()

    # Read back and check row counts
    df_tr = pd.read_parquet(tmp_path / "train.parquet")
    df_va = pd.read_parquet(tmp_path / "val.parquet")
    df_te = pd.read_parquet(tmp_path / "test.parquet")
    assert len(df_tr) == 8002
    assert len(df_va) == 2001
    assert len(df_te) == 3080

    # Read metadata
    with open(paths["metadata"], "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["random_seed"] == 42
    assert meta["train_rows"] == 8002
    assert meta["val_rows"] == 2001
    assert meta["test_rows"] == 3080
    assert meta["classes_in_train"] == 77
    assert meta["classes_in_val"] == 77
    assert meta["verification_passed"] is True
