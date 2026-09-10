"""Reproducible dataset loading and schema verification for BANKING77."""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datasets import DatasetDict, load_dataset
import pandas as pd

from src.config import DATASET_NAME, EXPECTED_NUM_INTENTS


@dataclass(frozen=True)
class LoadedDataset:
    """Encapsulates loaded DataFrames and label metadata for BANKING77."""
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    label_names: List[str]
    label_to_id: Dict[str, int]
    id_to_label: Dict[int, str]
    dataset_version: Optional[str] = None
    citation: Optional[str] = None


def load_banking77(dataset_name: str = DATASET_NAME) -> LoadedDataset:
    """Load the official BANKING77 dataset from Hugging Face Datasets and convert to DataFrames.
    
    Args:
        dataset_name: Hugging Face dataset identifier.
        
    Returns:
        LoadedDataset containing train/test DataFrames and label mappings.
        
    Raises:
        ValueError: If expected splits or columns are missing, or label count is unexpected.
        RuntimeError: If dataset download or extraction fails.
    """
    try:
        raw_dataset: DatasetDict = load_dataset(dataset_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load dataset '{dataset_name}' from Hugging Face Datasets. Error: {exc}"
        ) from exc

    # Validate split presence
    if "train" not in raw_dataset or "test" not in raw_dataset:
        raise ValueError(
            f"Dataset '{dataset_name}' must contain both 'train' and 'test' splits. Found: {list(raw_dataset.keys())}"
        )

    # Extract label names
    train_features = raw_dataset["train"].features
    if "label" not in train_features or not hasattr(train_features["label"], "names"):
        raise ValueError("Missing 'label' feature with ClassLabel metadata in dataset.")

    label_names: List[str] = list(train_features["label"].names)
    num_labels = len(label_names)
    if num_labels != EXPECTED_NUM_INTENTS:
        raise ValueError(
            f"Expected {EXPECTED_NUM_INTENTS} intent labels, but found {num_labels}."
        )

    label_to_id: Dict[str, int] = {name: idx for idx, name in enumerate(label_names)}
    id_to_label: Dict[int, str] = {idx: name for idx, name in enumerate(label_names)}

    # Convert to DataFrames
    train_df = raw_dataset["train"].to_pandas()
    test_df = raw_dataset["test"].to_pandas()

    for split_name, df in [("train", train_df), ("test", test_df)]:
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError(f"Split '{split_name}' missing required columns ('text', 'label').")
        
        # Enforce consistent types
        df["text"] = df["text"].astype(str)
        df["label"] = df["label"].astype(int)
        df["intent_name"] = df["label"].map(id_to_label).astype(str)

    # Dataset metadata
    info = raw_dataset["train"].info
    dataset_version = str(info.version) if info and info.version else None
    citation = str(info.citation) if info and info.citation else None

    return LoadedDataset(
        train_df=train_df,
        test_df=test_df,
        label_names=label_names,
        label_to_id=label_to_id,
        id_to_label=id_to_label,
        dataset_version=dataset_version,
        citation=citation,
    )
