"""Hyperparameter configuration and file paths for V1 classical baseline."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.config import PROJECT_ROOT, PROCESSED_DATA_DIR, REPORTS_DIR, RANDOM_SEED


@dataclass(frozen=True)
class TfidfConfig:
    """TF-IDF Vectorizer configuration."""
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    sublinear_tf: bool = True
    lowercase: bool = True
    max_features: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LogRegConfig:
    """Multiclass Logistic Regression configuration."""
    solver: str = "lbfgs"
    C: float = 1.0
    max_iter: int = 1000
    random_state: int = RANDOM_SEED

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BaselineConfig:
    """Composite configuration for baseline training, evaluation, and serialization."""
    tfidf: TfidfConfig = field(default_factory=TfidfConfig)
    logreg: LogRegConfig = field(default_factory=LogRegConfig)

    # Directories
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROCESSED_DATA_DIR
    models_dir: Path = PROJECT_ROOT / "models"
    experiments_dir: Path = PROJECT_ROOT / "experiments"
    reports_dir: Path = REPORTS_DIR

    # Artifact paths
    model_artifact_path: Path = PROJECT_ROOT / "models" / "baseline_tfidf_logreg.joblib"
    experiment_log_path: Path = PROJECT_ROOT / "experiments" / "baseline_v1_run.json"
    evaluation_report_path: Path = REPORTS_DIR / "baseline_v1_evaluation_report.md"
    confusion_matrix_png: Path = REPORTS_DIR / "baseline_confusion_matrix.png"
    top_confusions_csv: Path = REPORTS_DIR / "baseline_top_confusions.csv"

    def ensure_directories(self) -> None:
        """Create output directories if they do not exist."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
