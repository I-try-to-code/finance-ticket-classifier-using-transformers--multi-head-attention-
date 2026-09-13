"""Prediction and evaluation metric collection across DistilBERT V2 and BERT V3."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from src.config import PROCESSED_DATA_DIR, PROJECT_ROOT


@dataclass
class ValidationPredictionRow:
    """Detailed per-sample validation inference record comparing both models."""
    sample_id: int
    text: str
    true_label_id: int
    true_intent: str

    # DistilBERT predictions
    distilbert_pred_id: int
    distilbert_pred_intent: str
    distilbert_confidence: float
    distilbert_second_intent: str
    distilbert_second_prob: float
    distilbert_margin: float
    distilbert_correct: bool

    # BERT-base predictions
    bert_pred_id: int
    bert_pred_intent: str
    bert_confidence: float
    bert_second_intent: str
    bert_second_prob: float
    bert_margin: float
    bert_correct: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "text": self.text,
            "true_label_id": self.true_label_id,
            "true_intent": self.true_intent,
            "distilbert_pred_id": self.distilbert_pred_id,
            "distilbert_pred_intent": self.distilbert_pred_intent,
            "distilbert_confidence": round(self.distilbert_confidence, 4),
            "distilbert_second_intent": self.distilbert_second_intent,
            "distilbert_second_prob": round(self.distilbert_second_prob, 4),
            "distilbert_margin": round(self.distilbert_margin, 4),
            "distilbert_correct": self.distilbert_correct,
            "bert_pred_id": self.bert_pred_id,
            "bert_pred_intent": self.bert_pred_intent,
            "bert_confidence": round(self.bert_confidence, 4),
            "bert_second_intent": self.bert_second_intent,
            "bert_second_prob": round(self.bert_second_prob, 4),
            "bert_margin": round(self.bert_margin, 4),
            "bert_correct": self.bert_correct,
        }


@dataclass
class ModelMetricSummary:
    """Summary of model validation metrics and per-class performance."""
    model_name: str
    total_samples: int
    correct_predictions: int
    incorrect_predictions: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: Dict[str, Dict[str, float]]
    confusion_matrix: np.ndarray
    all_probabilities: np.ndarray
    all_predictions: np.ndarray


class SimpleInferenceDataset(Dataset):
    """Tokenized text dataset for batch inference."""
    def __init__(self, texts: List[str], tokenizer: PreTrainedTokenizerBase, max_length: int = 128) -> None:
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


def create_inference_collator(tokenizer: PreTrainedTokenizerBase, max_length: int = 128):
    """Dynamic padding collator for inference texts."""
    def collate_fn(batch_texts: List[str]) -> Dict[str, torch.Tensor]:
        return tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    return collate_fn


def run_model_inference_on_validation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: List[str],
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    fp16: bool = True,
) -> np.ndarray:
    """Compute softmax probabilities for all validation texts in batches."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    dataset = SimpleInferenceDataset(texts, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=create_inference_collator(tokenizer),
    )

    use_cuda_amp = fp16 and device.type == "cuda"
    logits_list: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_cuda_amp):
                outputs = model(**inputs)
                logits = outputs.logits
            logits_list.append(logits.cpu().numpy())

    all_logits = np.concatenate(logits_list, axis=0)
    exp_logits = np.exp(all_logits - np.max(all_logits, axis=-1, keepdims=True))
    all_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    return all_probs


def evaluate_probabilities(
    probs: np.ndarray,
    y_true: np.ndarray,
    id_to_label: Dict[int, str],
    model_name: str,
) -> ModelMetricSummary:
    """Compute global and per-class metrics from predicted probabilities."""
    preds = np.argmax(probs, axis=-1)
    total = len(y_true)
    correct = int(np.sum(preds == y_true))
    incorrect = total - correct

    acc = float(accuracy_score(y_true, preds))
    macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, preds, average="weighted", zero_division=0))

    labels = sorted(list(id_to_label.keys()))
    prec, rec, f1s, supp = precision_recall_fscore_support(
        y_true, preds, labels=labels, zero_division=0
    )

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, preds, labels=labels)

    per_class: Dict[str, Dict[str, float]] = {}
    for cid in labels:
        name = id_to_label[cid]
        per_class[name] = {
            "class_id": cid,
            "precision": round(float(prec[cid]), 4),
            "recall": round(float(rec[cid]), 4),
            "f1": round(float(f1s[cid]), 4),
            "support": int(supp[cid]),
        }

    return ModelMetricSummary(
        model_name=model_name,
        total_samples=total,
        correct_predictions=correct,
        incorrect_predictions=incorrect,
        accuracy=round(acc, 4),
        macro_f1=round(macro_f1, 4),
        weighted_f1=round(weighted_f1, 4),
        per_class=per_class,
        confusion_matrix=cm,
        all_probabilities=probs,
        all_predictions=preds,
    )


def collect_validation_predictions(
    data_dir: Path = PROCESSED_DATA_DIR,
    distilbert_dir: Path = PROJECT_ROOT / "models" / "distilbert_banking77",
    bert_dir: Path = PROJECT_ROOT / "models" / "bert_banking77",
    device: Optional[torch.device] = None,
    batch_size: int = 32,
) -> Tuple[List[ValidationPredictionRow], ModelMetricSummary, ModelMetricSummary, pd.DataFrame, Dict[int, str]]:
    """Execute complete validation inference across DistilBERT and BERT-base, returning aligned comparison rows."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load validation dataset
    val_parquet = data_dir / "val.parquet"
    val_csv = data_dir / "val.csv"
    val_df = pd.read_parquet(val_parquet) if val_parquet.exists() else pd.read_csv(val_csv)

    # 2. Build id_to_label mapping
    train_parquet = data_dir / "train.parquet"
    train_csv = data_dir / "train.csv"
    train_df = pd.read_parquet(train_parquet) if train_parquet.exists() else pd.read_csv(train_csv)
    id_to_label = dict(zip(train_df["label"].astype(int), train_df["intent_name"].astype(str)))

    texts = val_df["text"].tolist()
    y_true = val_df["label"].to_numpy().astype(int)

    # 3. DistilBERT Inference
    print(f"      [Collector] Running inference for DistilBERT on {len(texts)} validation samples...")
    distilbert_tokenizer = AutoTokenizer.from_pretrained(distilbert_dir)
    distilbert_model = AutoModelForSequenceClassification.from_pretrained(distilbert_dir)
    distilbert_probs = run_model_inference_on_validation(
        model=distilbert_model,
        tokenizer=distilbert_tokenizer,
        texts=texts,
        batch_size=batch_size,
        device=device,
        fp16=True,
    )
    distilbert_summary = evaluate_probabilities(
        probs=distilbert_probs,
        y_true=y_true,
        id_to_label=id_to_label,
        model_name="DistilBERT (V2)",
    )
    del distilbert_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 4. BERT-base Inference
    print(f"      [Collector] Running inference for BERT-base on {len(texts)} validation samples...")
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_dir)
    bert_model = AutoModelForSequenceClassification.from_pretrained(bert_dir)
    bert_probs = run_model_inference_on_validation(
        model=bert_model,
        tokenizer=bert_tokenizer,
        texts=texts,
        batch_size=batch_size,
        device=device,
        fp16=True,
    )
    bert_summary = evaluate_probabilities(
        probs=bert_probs,
        y_true=y_true,
        id_to_label=id_to_label,
        model_name="BERT-base (V3)",
    )
    del bert_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 5. Build aligned per-sample rows
    aligned_rows: List[ValidationPredictionRow] = []
    for idx in range(len(texts)):
        true_lbl = int(y_true[idx])
        true_name = id_to_label[true_lbl]

        # DistilBERT stats
        db_p = distilbert_probs[idx]
        db_sorted_indices = np.argsort(db_p)[::-1]
        db_top1_id = int(db_sorted_indices[0])
        db_top2_id = int(db_sorted_indices[1])
        db_top1_prob = float(db_p[db_top1_id])
        db_top2_prob = float(db_p[db_top2_id])
        db_margin = db_top1_prob - db_top2_prob
        db_correct = bool(db_top1_id == true_lbl)

        # BERT stats
        b_p = bert_probs[idx]
        b_sorted_indices = np.argsort(b_p)[::-1]
        b_top1_id = int(b_sorted_indices[0])
        b_top2_id = int(b_sorted_indices[1])
        b_top1_prob = float(b_p[b_top1_id])
        b_top2_prob = float(b_p[b_top2_id])
        b_margin = b_top1_prob - b_top2_prob
        b_correct = bool(b_top1_id == true_lbl)

        row = ValidationPredictionRow(
            sample_id=idx,
            text=texts[idx],
            true_label_id=true_lbl,
            true_intent=true_name,
            distilbert_pred_id=db_top1_id,
            distilbert_pred_intent=id_to_label[db_top1_id],
            distilbert_confidence=db_top1_prob,
            distilbert_second_intent=id_to_label[db_top2_id],
            distilbert_second_prob=db_top2_prob,
            distilbert_margin=db_margin,
            distilbert_correct=db_correct,
            bert_pred_id=b_top1_id,
            bert_pred_intent=id_to_label[b_top1_id],
            bert_confidence=b_top1_prob,
            bert_second_intent=id_to_label[b_top2_id],
            bert_second_prob=b_top2_prob,
            bert_margin=b_margin,
            bert_correct=b_correct,
        )
        aligned_rows.append(row)

    return aligned_rows, distilbert_summary, bert_summary, val_df, id_to_label
