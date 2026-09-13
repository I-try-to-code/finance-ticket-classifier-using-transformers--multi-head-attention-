"""Confusion matrix computation, top-20 pair extraction, and visualization."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class ConfusionPairMetric:
    """Confusion pair details including percentage of true class affected."""
    model_name: str
    rank: int
    true_intent: str
    predicted_intent: str
    true_class_id: int
    predicted_class_id: int
    error_count: int
    class_support: int
    percent_true_class_affected: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "rank": self.rank,
            "true_intent": self.true_intent,
            "predicted_intent": self.predicted_intent,
            "error_count": self.error_count,
            "class_support": self.class_support,
            "percent_true_class_affected": round(self.percent_true_class_affected, 2),
        }


def extract_top_confusion_pairs(
    cm: np.ndarray,
    id_to_label: Dict[int, str],
    model_name: str,
    top_n: int = 20,
) -> List[ConfusionPairMetric]:
    """Extract top N non-diagonal confusion pairs sorted by error count, with true class percentage affected."""
    n_classes = len(id_to_label)
    # Class support is the sum across columns for row i
    class_supports = np.sum(cm, axis=1)

    pairs: List[Dict[str, Any]] = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                supp = int(class_supports[i])
                count = int(cm[i, j])
                pct = (count / supp * 100.0) if supp > 0 else 0.0
                pairs.append({
                    "true_class_id": i,
                    "predicted_class_id": j,
                    "true_intent": id_to_label[i],
                    "predicted_intent": id_to_label[j],
                    "error_count": count,
                    "class_support": supp,
                    "percent_true_class_affected": pct,
                })

    # Sort descending by error_count, then by percent_true_class_affected
    pairs.sort(key=lambda x: (x["error_count"], x["percent_true_class_affected"]), reverse=True)

    result: List[ConfusionPairMetric] = []
    for rank, p in enumerate(pairs[:top_n], start=1):
        result.append(
            ConfusionPairMetric(
                model_name=model_name,
                rank=rank,
                true_intent=p["true_intent"],
                predicted_intent=p["predicted_intent"],
                true_class_id=p["true_class_id"],
                predicted_class_id=p["predicted_class_id"],
                error_count=p["error_count"],
                class_support=p["class_support"],
                percent_true_class_affected=p["percent_true_class_affected"],
            )
        )
    return result


def export_confusion_pairs_csv(
    distilbert_pairs: List[ConfusionPairMetric],
    bert_pairs: List[ConfusionPairMetric],
    output_path: Path,
) -> pd.DataFrame:
    """Save combined top confusion pairs for both models to CSV."""
    rows = [p.to_dict() for p in distilbert_pairs] + [p.to_dict() for p in bert_pairs]
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    output_path: Path,
    cmap: str = "Purples",
) -> None:
    """Plot and save a high-resolution 300 DPI confusion matrix heatmap."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 12), dpi=300)
    cax = ax.matshow(cm, cmap=getattr(plt.cm, cmap), interpolation="nearest")
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    ax.set_title(f"BANKING77 - {model_name} Confusion Matrix (Validation Set)", fontsize=14, pad=16, fontweight="bold")
    ax.set_xlabel("Predicted Intent Index (0..76)", fontsize=12, labelpad=8)
    ax.set_ylabel("True Intent Index (0..76)", fontsize=12, labelpad=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
