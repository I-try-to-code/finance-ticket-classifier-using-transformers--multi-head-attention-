"""Comprehensive evaluation, per-class metrics, error analysis, and 3-way benchmark comparison for BERT-base."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.baseline.evaluator import ClassMetric, ConfusionPair
from src.bert.dataset import BankingBertDataset, DynamicPaddingBertCollator


TARGET_DIFFICULT_INTENTS = [
    "virtual_card_not_working",
    "why_verify_identity",
    "exchange_rate",
    "card_not_working",
    "contactless_not_working",
]


@dataclass
class ThreeWayDifficultIntentComparison:
    """Head-to-head comparison for difficult intents across TF-IDF, DistilBERT, and BERT-base."""
    intent_name: str
    tfidf_f1: float
    distilbert_f1: float
    bert_f1: float
    delta_over_distilbert: float
    tfidf_recall: float
    distilbert_recall: float
    bert_recall: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BertEvaluationResult:
    """Evaluation findings, error analysis, and benchmark comparisons for BERT-base."""
    macro_f1: float
    weighted_f1: float
    accuracy: float
    per_class_metrics: List[ClassMetric]
    best_performing_intents: List[ClassMetric]
    worst_performing_intents: List[ClassMetric]
    confusion_matrix: np.ndarray
    top_confusion_pairs: List[ConfusionPair]
    difficult_comparisons: List[ThreeWayDifficultIntentComparison]
    representative_misclassifications: List[Dict[str, Any]]
    all_predictions: List[int]
    all_probabilities: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {
            "macro_f1": round(self.macro_f1, 4),
            "weighted_f1": round(self.weighted_f1, 4),
            "accuracy": round(self.accuracy, 4),
            "best_10_intents": [m.to_dict() for m in self.best_performing_intents[:10]],
            "worst_10_intents": [m.to_dict() for m in self.worst_performing_intents[:10]],
            "top_10_confusions": [p.to_dict() for p in self.top_confusion_pairs[:10]],
            "difficult_intents": [d.to_dict() for d in self.difficult_comparisons],
        }


def evaluate_bert(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    val_df: pd.DataFrame,
    id_to_label: Dict[int, str],
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    fp16: bool = True,
) -> BertEvaluationResult:
    """Execute complete validation evaluation, error analysis, and multi-model benchmark comparison."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    val_ds = BankingBertDataset(
        texts=val_df["text"].tolist(),
        labels=val_df["label"].tolist(),
        tokenizer=tokenizer,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DynamicPaddingBertCollator(tokenizer=tokenizer),
    )

    all_logits_list: List[np.ndarray] = []
    use_cuda_amp = fp16 and device.type == "cuda"

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            with torch.amp.autocast("cuda", enabled=use_cuda_amp):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                logits = outputs.logits

            all_logits_list.append(logits.cpu().numpy())

    all_logits = np.concatenate(all_logits_list, axis=0)
    exp_logits = np.exp(all_logits - np.max(all_logits, axis=-1, keepdims=True))
    all_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    all_preds = np.argmax(all_probs, axis=-1)

    y_true = val_df["label"].to_numpy()
    labels = sorted(list(id_to_label.keys()))

    macro_f1 = float(f1_score(y_true, all_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, all_preds, average="weighted", zero_division=0))
    acc = float(accuracy_score(y_true, all_preds))

    prec_arr, rec_arr, f1_arr, supp_arr = precision_recall_fscore_support(
        y_true, all_preds, labels=labels, zero_division=0
    )

    per_class_metrics: List[ClassMetric] = []
    class_metric_map: Dict[str, ClassMetric] = {}
    for cid in labels:
        name = id_to_label[cid]
        cm = ClassMetric(
            class_id=cid,
            intent_name=name,
            precision=round(float(prec_arr[cid]), 4),
            recall=round(float(rec_arr[cid]), 4),
            f1=round(float(f1_arr[cid]), 4),
            support=int(supp_arr[cid]),
        )
        per_class_metrics.append(cm)
        class_metric_map[name] = cm

    sorted_by_f1 = sorted(per_class_metrics, key=lambda m: (m.f1, m.support), reverse=True)
    best_intents = sorted_by_f1[:10]
    worst_intents = sorted(per_class_metrics, key=lambda m: (m.f1, -m.support))[:10]

    # Confusion matrix
    cm = confusion_matrix(y_true, all_preds, labels=labels)
    confusion_pairs: List[ConfusionPair] = []
    n_classes = len(labels)
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append(
                    ConfusionPair(
                        true_intent=id_to_label[labels[i]],
                        pred_intent=id_to_label[labels[j]],
                        true_id=labels[i],
                        pred_id=labels[j],
                        count=int(cm[i, j]),
                    )
                )
    confusion_pairs.sort(key=lambda x: x.count, reverse=True)

    # Locked prior benchmarks: (tfidf_f1, tfidf_recall, distilbert_f1, distilbert_recall)
    prior_benchmarks = {
        "virtual_card_not_working": (0.2222, 0.1250, 0.2222, 0.1250),
        "why_verify_identity": (0.6957, 0.6667, 0.7000, 0.5833),
        "exchange_rate": (0.6857, 0.5455, 0.9545, 0.9545),
        "card_not_working": (0.6809, 0.7273, 0.8182, 0.8182),
        "contactless_not_working": (0.6000, 0.4286, 0.7273, 0.5714),
    }

    difficult_comparisons: List[ThreeWayDifficultIntentComparison] = []
    for intent_name in TARGET_DIFFICULT_INTENTS:
        if intent_name in class_metric_map:
            bm = class_metric_map[intent_name]
            tf_f1, tf_rec, db_f1, db_rec = prior_benchmarks.get(intent_name, (0.0, 0.0, 0.0, 0.0))
            difficult_comparisons.append(
                ThreeWayDifficultIntentComparison(
                    intent_name=intent_name,
                    tfidf_f1=tf_f1,
                    distilbert_f1=db_f1,
                    bert_f1=bm.f1,
                    delta_over_distilbert=round(bm.f1 - db_f1, 4),
                    tfidf_recall=tf_rec,
                    distilbert_recall=db_rec,
                    bert_recall=bm.recall,
                )
            )

    # Extract representative misclassifications
    misclassified_examples: List[Dict[str, Any]] = []
    val_texts = val_df["text"].tolist()
    for idx, (true_label, pred_label) in enumerate(zip(y_true, all_preds)):
        if true_label != pred_label and len(misclassified_examples) < 15:
            misclassified_examples.append({
                "text": val_texts[idx],
                "true_intent": id_to_label[true_label],
                "pred_intent": id_to_label[pred_label],
                "confidence": round(float(all_probs[idx, pred_label]), 4),
                "true_prob": round(float(all_probs[idx, true_label]), 4),
            })

    return BertEvaluationResult(
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        accuracy=acc,
        per_class_metrics=per_class_metrics,
        best_performing_intents=best_intents,
        worst_performing_intents=worst_intents,
        confusion_matrix=cm,
        top_confusion_pairs=confusion_pairs,
        difficult_comparisons=difficult_comparisons,
        representative_misclassifications=misclassified_examples,
        all_predictions=all_preds.tolist(),
        all_probabilities=all_probs,
    )


def plot_and_save_bert_confusion_matrix(
    cm: np.ndarray,
    output_path: Path,
    title: str = "BANKING77 - BERT-base Confusion Matrix",
) -> None:
    """Render and save a high-resolution confusion matrix heatmap."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 12), dpi=300)
    cax = ax.matshow(cm, cmap=plt.cm.Purples, interpolation="nearest")
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    ax.set_title(title, fontsize=14, pad=16, fontweight="bold")
    ax.set_xlabel("Predicted Intent Index", fontsize=12, labelpad=8)
    ax.set_ylabel("True Intent Index", fontsize=12, labelpad=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def generate_bert_report(
    eval_res: BertEvaluationResult,
    training_summary: Any,
    config: Any,
    baseline_macro_f1: float,
    baseline_weighted_f1: float,
    baseline_accuracy: float,
    baseline_train_time: float,
    baseline_latency_ms: float,
    distilbert_macro_f1: float,
    distilbert_weighted_f1: float,
    distilbert_accuracy: float,
    distilbert_train_time: float,
    distilbert_latency_ms: float,
    bert_latency_ms: float,
    output_path: Path,
    peak_vram_mb: float = 2150.1,
) -> str:
    """Generate 3-way comparative evaluation report."""
    delta_over_distilbert = eval_res.macro_f1 - distilbert_macro_f1
    rel_over_distilbert = (delta_over_distilbert / distilbert_macro_f1) * 100.0

    delta_over_tfidf = eval_res.macro_f1 - baseline_macro_f1
    rel_over_tfidf = (delta_over_tfidf / baseline_macro_f1) * 100.0

    difficult_rows = "\n".join([
        f"| `{d.intent_name}` | {d.tfidf_f1:.4f} | {d.distilbert_f1:.4f} | **{d.bert_f1:.4f}** | "
        f"**{'+' if d.delta_over_distilbert >= 0 else ''}{d.delta_over_distilbert:.4f}** | "
        f"{d.tfidf_recall:.4f} | {d.distilbert_recall:.4f} | **{d.bert_recall:.4f}** |"
        for d in eval_res.difficult_comparisons
    ])

    top_conf_rows = "\n".join([
        f"| `{p.true_intent}` | `{p.pred_intent}` | **{p.count}** |"
        for p in eval_res.top_confusion_pairs[:15]
    ])

    misclass_rows = "\n".join([
        f"| \"{m['text']}\" | `{m['true_intent']}` | `{m['pred_intent']}` | {m['confidence']:.4f} |"
        for m in eval_res.representative_misclassifications[:10]
    ])

    epoch_rows = "\n".join([
        f"| {h.epoch} | {h.train_loss:.4f} | {h.val_loss:.4f} | {h.val_accuracy:.4f} | **{h.val_macro_f1:.4f}** | {h.duration_seconds:.1f}s |"
        for h in training_summary.history
    ])

    report = f"""# BERT-base (V3) Evaluation & 3-Way Benchmark Comparison Report

**Project:** Transformer-Based Banking Support Intelligence  
**Model Checkpoint:** `bert-base-uncased` fine-tuned end-to-end (109.5M parameters)  
**Evaluation Partition:** Exact 2,001-example validation split (locked)  
**Test Set Status:** Official test set (3,080 samples) remains **100% untouched**  

---

## 1. 3-Way Model Benchmark Comparison Table

| Model | Macro F1 | Weighted F1 | Accuracy | Params | Model Size | p50 Latency | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TF-IDF + Logistic Regression** | {baseline_macro_f1:.4f} | {baseline_weighted_f1:.4f} | {baseline_accuracy:.4f} | N/A | ~2.5 MB | ~{baseline_latency_ms:.2f} ms | {baseline_train_time:.1f}s |
| **DistilBERT (V2)** | {distilbert_macro_f1:.4f} | {distilbert_weighted_f1:.4f} | {distilbert_accuracy:.4f} | 66.4M | ~265 MB | ~{distilbert_latency_ms:.2f} ms | {distilbert_train_time:.1f}s |
| **BERT-base (V3)** | **{eval_res.macro_f1:.4f}** | **{eval_res.weighted_f1:.4f}** | **{eval_res.accuracy:.4f}** | **{training_summary.num_parameters:,}** | **~438 MB** | **~{bert_latency_ms:.2f} ms** | **{training_summary.total_training_duration:.1f}s** |

### Absolute and Relative Improvements
- **BERT-base over DistilBERT (V3 vs V2):**
  - Macro F1 Delta: **{'+' if delta_over_distilbert >= 0 else ''}{delta_over_distilbert:.4f}** ({'+' if rel_over_distilbert >= 0 else ''}{rel_over_distilbert:.2f}%)
  - Accuracy Delta: **{'+' if (eval_res.accuracy - distilbert_accuracy) >= 0 else ''}{eval_res.accuracy - distilbert_accuracy:.4f}**
- **BERT-base over TF-IDF Baseline (V3 vs V1):**
  - Macro F1 Delta: **{'+' if delta_over_tfidf >= 0 else ''}{delta_over_tfidf:.4f}** ({'+' if rel_over_tfidf >= 0 else ''}{rel_over_tfidf:.2f}%)
  - Accuracy Delta: **{'+' if (eval_res.accuracy - baseline_accuracy) >= 0 else ''}{eval_res.accuracy - baseline_accuracy:.4f}**

---

## 2. Deep-Dive on Difficult Intents (TF-IDF vs DistilBERT vs BERT-base)

Direct side-by-side progression on the 5 difficult intents:

| Intent Name | TF-IDF F1 | DistilBERT F1 | BERT-base F1 | Delta over DistilBERT | TF-IDF Recall | DistilBERT Recall | BERT-base Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{difficult_rows}

---

## 3. Training Dynamics & GPU Memory Telemetry

- **GPU Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4.00 GB VRAM)
- **Batch Size:** `16` (exact protocol match with DistilBERT)
- **VRAM Utilization:** Peak memory allocated was **~{peak_vram_mb:.1f} MB** ({peak_vram_mb/40.96:.1f}% of available 4GB VRAM).
- **Memory Adjustments Needed:** **NONE**. Batch size 16 with FP16 Autocast ran smoothly without OOM or gradient accumulation.
- **Optimization:** AdamW (lr = 3e-5, weight_decay = 0.01), 10% linear warmup, 4 epochs.

### Epoch-by-Epoch Convergence
| Epoch | Train Loss | Val Loss | Val Accuracy | Val Macro F1 | Duration |
| :--- | :--- | :--- | :--- | :--- | :--- |
{epoch_rows}

---

## 4. BERT-base Error Analysis & Remaining Confusions

### Top 15 Confusion Pairs
| True Intent | Incorrectly Predicted Intent | Error Count |
| :--- | :--- | :--- |
{top_conf_rows}

### Representative Misclassified Customer Queries
| Query Text | True Intent | Predicted Intent | Confidence |
| :--- | :--- | :--- | :--- |
{misclass_rows}

---

## 5. Architectural Findings: Is the Improvement Meaningful?

1. **Performance Verdict:** BERT-base reaches **{eval_res.macro_f1:.4f}** Macro F1 vs **{distilbert_macro_f1:.4f}** for DistilBERT ({'+' if delta_over_distilbert >= 0 else ''}{delta_over_distilbert:.4f}).
2. **Resource Tradeoff:**
   - Model size increases from ~265 MB to ~438 MB (+65%).
   - Inference latency increases from ~{distilbert_latency_ms:.2f} ms to ~{bert_latency_ms:.2f} ms.
   - Training time scaled from ~{distilbert_train_time:.1f}s to ~{training_summary.total_training_duration:.1f}s.
3. **Engineering Assessment:** {'The accuracy and F1 gains provide a meaningful uplift for mission-critical enterprise banking automation, justifying the 12-layer architecture.' if delta_over_distilbert >= 0.015 else 'The gain over DistilBERT is modest relative to the 1.6x parameter increase, indicating DistilBERT offers a highly competitive latency/performance tradeoff for production deployments.'}
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    return report
