"""Local structured experiment tracking for baseline models."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class ExperimentRecord:
    """Standardized experiment record schema for local run logging."""
    experiment_name: str
    run_id: str
    timestamp: str
    dataset_info: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    training_info: Dict[str, Any]
    validation_metrics: Dict[str, Any]
    artifacts: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def create_experiment_record(
    experiment_name: str,
    dataset_name: str,
    train_count: int,
    val_count: int,
    test_count: int,
    num_classes: int,
    random_seed: int,
    tfidf_params: Dict[str, Any],
    logreg_params: Dict[str, Any],
    training_duration: float,
    vocab_size: int,
    val_metrics: Dict[str, Any],
    artifacts: Dict[str, str],
) -> ExperimentRecord:
    """Build a comprehensive experiment tracking record."""
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    now_iso = datetime.now(timezone.utc).isoformat()

    return ExperimentRecord(
        experiment_name=experiment_name,
        run_id=run_id,
        timestamp=now_iso,
        dataset_info={
            "dataset_name": dataset_name,
            "train_samples": train_count,
            "val_samples": val_count,
            "test_samples_untouched": test_count,
            "num_classes": num_classes,
        },
        hyperparameters={
            "random_seed": random_seed,
            "tfidf": tfidf_params,
            "logistic_regression": logreg_params,
        },
        training_info={
            "duration_seconds": round(training_duration, 4),
            "vocabulary_size": vocab_size,
        },
        validation_metrics=val_metrics,
        artifacts=artifacts,
    )


def save_experiment_record(record: ExperimentRecord, output_path: Path) -> Path:
    """Persist structured experiment record to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(record.to_dict(), f, indent=2)
    return output_path
