"""Configuration constants and hyperparameter dataclasses for V3 BERT-base."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

from src.config import PROJECT_ROOT, PROCESSED_DATA_DIR, REPORTS_DIR, RANDOM_SEED


@dataclass
class BertConfig:
    """Hyperparameters and artifact paths for BERT-base fine-tuning."""

    # Model architecture
    model_name: str = "bert-base-uncased"
    num_labels: int = 77
    max_length: int = 128

    # Optimization
    learning_rate: float = 3e-5
    batch_size: int = 16
    epochs: int = 4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    seed: int = RANDOM_SEED
    fp16: bool = True
    gradient_clip_val: float = 1.0

    # Paths
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROCESSED_DATA_DIR
    models_dir: Path = PROJECT_ROOT / "models"
    experiments_dir: Path = PROJECT_ROOT / "experiments"
    reports_dir: Path = REPORTS_DIR

    model_output_dir: Path = PROJECT_ROOT / "models" / "bert_banking77"
    experiment_log_path: Path = PROJECT_ROOT / "experiments" / "bert_v3_run.json"
    evaluation_report_path: Path = REPORTS_DIR / "bert_v3_evaluation_report.md"
    confusion_matrix_png: Path = REPORTS_DIR / "bert_confusion_matrix.png"
    top_confusions_csv: Path = REPORTS_DIR / "bert_top_confusions.csv"
    difficult_intents_csv: Path = REPORTS_DIR / "bert_vs_distilbert_difficult_intents.csv"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "max_length": self.max_length,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "seed": self.seed,
            "fp16": self.fp16,
        }

    def ensure_directories(self) -> None:
        """Create necessary directories."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
