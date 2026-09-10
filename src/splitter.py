"""Reproducible stratified train/validation splitting and integrity verification."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Set
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DATA_DIR, RANDOM_SEED, STRATIFY_COLUMN, VAL_SIZE


@dataclass(frozen=True)
class SplitVerificationReport:
    """Results of post-split integrity and stratification checks."""
    passed: bool
    num_train: int
    num_val: int
    num_test: int
    train_ratio: float
    val_ratio: float
    classes_in_train: int
    classes_in_val: int
    missing_in_train: Set[Any]
    missing_in_val: Set[Any]
    max_prop_diff_train: float
    max_prop_diff_val: float
    index_overlap_count: int
    text_overlap_count: int
    test_untouched: bool


@dataclass(frozen=True)
class SplitResult:
    """Contains partitioned DataFrames and verification metadata."""
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    random_seed: int
    val_size: float
    verification: SplitVerificationReport


def verify_split_integrity(
    train_split: pd.DataFrame,
    val_split: pd.DataFrame,
    orig_train: pd.DataFrame,
    orig_test: pd.DataFrame,
    test_split: pd.DataFrame,
    stratify_col: str = STRATIFY_COLUMN,
    text_col: str = "text",
    max_allowed_prop_diff: float = 0.001,
) -> SplitVerificationReport:
    """Run strict verification assertions on partitioned datasets."""
    expected_classes = set(orig_train[stratify_col].unique())
    train_classes = set(train_split[stratify_col].unique())
    val_classes = set(val_split[stratify_col].unique())

    missing_in_train = expected_classes - train_classes
    missing_in_val = expected_classes - val_classes

    # Class proportion preservation check
    orig_props = orig_train[stratify_col].value_counts(normalize=True).sort_index()
    train_props = train_split[stratify_col].value_counts(normalize=True).sort_index()
    val_props = val_split[stratify_col].value_counts(normalize=True).sort_index()

    max_diff_train = float((train_props - orig_props).abs().max())
    max_diff_val = float((val_props - orig_props).abs().max())

    # Overlap checks
    train_indices = set(train_split.index)
    val_indices = set(val_split.index)
    index_overlap = len(train_indices.intersection(val_indices))

    train_texts = set(train_split[text_col])
    val_texts = set(val_split[text_col])
    text_overlap = len(train_texts.intersection(val_texts))

    # Test immutability check
    test_untouched = (
        len(test_split) == len(orig_test) and
        test_split[text_col].equals(orig_test[text_col]) and
        test_split[stratify_col].equals(orig_test[stratify_col])
    )

    passed = (
        len(missing_in_train) == 0 and
        len(missing_in_val) == 0 and
        max_diff_train <= max_allowed_prop_diff and
        max_diff_val <= max_allowed_prop_diff and
        index_overlap == 0 and
        text_overlap == 0 and
        test_untouched and
        (len(train_split) + len(val_split) == len(orig_train))
    )

    total_orig_train = len(orig_train)
    return SplitVerificationReport(
        passed=passed,
        num_train=len(train_split),
        num_val=len(val_split),
        num_test=len(test_split),
        train_ratio=round(len(train_split) / total_orig_train, 4),
        val_ratio=round(len(val_split) / total_orig_train, 4),
        classes_in_train=len(train_classes),
        classes_in_val=len(val_classes),
        missing_in_train=missing_in_train,
        missing_in_val=missing_in_val,
        max_prop_diff_train=round(max_diff_train, 6),
        max_prop_diff_val=round(max_diff_val, 6),
        index_overlap_count=index_overlap,
        text_overlap_count=text_overlap,
        test_untouched=test_untouched,
    )


def create_reproducible_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_size: float = VAL_SIZE,
    random_seed: int = RANDOM_SEED,
    stratify_col: str = STRATIFY_COLUMN,
    text_col: str = "text",
) -> SplitResult:
    """Create a stratified train/validation partition from the official train split.
    
    The official test split remains strictly untouched.
    """
    if stratify_col not in train_df.columns:
        raise KeyError(f"Stratify column '{stratify_col}' not found in train DataFrame.")

    # Preserve original indices for absolute traceability
    train_work_df = train_df.copy()
    if "orig_train_index" not in train_work_df.columns:
        train_work_df["orig_train_index"] = train_work_df.index

    # Stratified split using explicit seed
    train_sub, val_sub = train_test_split(
        train_work_df,
        test_size=val_size,
        random_state=random_seed,
        stratify=train_work_df[stratify_col],
    )

    # Untouched test split (defensive copy)
    test_split = test_df.copy()

    verification = verify_split_integrity(
        train_split=train_sub,
        val_split=val_sub,
        orig_train=train_df,
        orig_test=test_df,
        test_split=test_split,
        stratify_col=stratify_col,
        text_col=text_col,
    )

    if not verification.passed:
        raise ValueError(f"Split verification failed integrity checks: {verification}")

    return SplitResult(
        train_df=train_sub,
        val_df=val_sub,
        test_df=test_split,
        random_seed=random_seed,
        val_size=val_size,
        verification=verification,
    )


def save_processed_splits(
    split_result: SplitResult,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> Dict[str, Path]:
    """Save train, validation, and test splits along with JSON reproducibility metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {
        "train": output_dir / "train.csv",
        "val": output_dir / "val.csv",
        "test": output_dir / "test.csv",
        "metadata": output_dir / "split_metadata.json",
    }

    # Save CSVs
    split_result.train_df.to_csv(paths["train"], index=False)
    split_result.val_df.to_csv(paths["val"], index=False)
    split_result.test_df.to_csv(paths["test"], index=False)

    # Save Parquet formats as well for optimal IO in downstream training
    split_result.train_df.to_parquet(output_dir / "train.parquet", index=False)
    split_result.val_df.to_parquet(output_dir / "val.parquet", index=False)
    split_result.test_df.to_parquet(output_dir / "test.parquet", index=False)

    metadata: Dict[str, Any] = {
        "random_seed": split_result.random_seed,
        "val_size": split_result.val_size,
        "train_rows": len(split_result.train_df),
        "val_rows": len(split_result.val_df),
        "test_rows": len(split_result.test_df),
        "verification_passed": split_result.verification.passed,
        "classes_in_train": split_result.verification.classes_in_train,
        "classes_in_val": split_result.verification.classes_in_val,
        "max_prop_diff_train": split_result.verification.max_prop_diff_train,
        "max_prop_diff_val": split_result.verification.max_prop_diff_val,
        "index_overlap_count": split_result.verification.index_overlap_count,
        "text_overlap_count": split_result.verification.text_overlap_count,
        "test_untouched": split_result.verification.test_untouched,
    }

    with open(paths["metadata"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return paths
