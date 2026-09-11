"""PyTorch Dataset and dynamic collator for BERT-base fine-tuning."""

from typing import Any, Dict, List, Tuple
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase


class BankingBertDataset(Dataset):
    """PyTorch Dataset for banking customer queries using BERT tokenization."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
    ) -> None:
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

        encodings = tokenizer(
            self.texts,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        self.input_ids = encodings["input_ids"]
        self.attention_masks = encodings["attention_mask"]
        self.token_type_ids = encodings.get("token_type_ids", [[0] * len(ids) for ids in self.input_ids])

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "token_type_ids": self.token_type_ids[idx],
            "label": self.labels[idx],
            "text": self.texts[idx],
        }


class DynamicPaddingBertCollator:
    """Collator that dynamically pads input_ids, attention_mask, and token_type_ids to the batch max length."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        texts = [item["text"] for item in batch]

        features = [
            {
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"],
                "token_type_ids": item["token_type_ids"],
            }
            for item in batch
        ]

        padded = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

        padded["labels"] = labels
        padded["texts"] = texts
        return padded


def create_bert_data_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 16,
    max_length: int = 128,
) -> Tuple[DataLoader, DataLoader]:
    """Construct PyTorch DataLoaders with dynamic batch padding for BERT."""
    train_ds = BankingBertDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )

    val_ds = BankingBertDataset(
        texts=val_df["text"].tolist(),
        labels=val_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )

    collator = DynamicPaddingBertCollator(tokenizer=tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        drop_last=False,
    )

    return train_loader, val_loader
