"""V1 Baseline Package: TF-IDF + Logistic Regression."""

from src.baseline.config import BaselineConfig, LogRegConfig, TfidfConfig
from src.baseline.evaluator import BaselineEvaluationResult, evaluate_baseline
from src.baseline.predictor import BaselinePredictor, PredictionResult, save_baseline_artifact
from src.baseline.trainer import build_baseline_pipeline, train_baseline

__all__ = [
    "BaselineConfig",
    "TfidfConfig",
    "LogRegConfig",
    "build_baseline_pipeline",
    "train_baseline",
    "evaluate_baseline",
    "BaselineEvaluationResult",
    "BaselinePredictor",
    "PredictionResult",
    "save_baseline_artifact",
]
