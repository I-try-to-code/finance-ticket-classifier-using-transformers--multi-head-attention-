"""Inference engine and artifact serialization for the baseline classifier."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class CandidatePrediction:
    """Represents an alternative candidate intent and its predicted probability."""
    intent_name: str
    class_id: int
    probability: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionResult:
    """Output of inference for a customer support query."""
    input_text: str
    predicted_intent: str
    predicted_class_id: int
    confidence: float
    top_k_predictions: List[CandidatePrediction]
    all_probabilities: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "input_text": self.input_text,
            "predicted_intent": self.predicted_intent,
            "predicted_class_id": self.predicted_class_id,
            "confidence": round(self.confidence, 4),
            "top_k_predictions": [c.to_dict() for c in self.top_k_predictions],
        }
        if self.all_probabilities is not None:
            res["all_probabilities"] = {k: round(v, 6) for k, v in self.all_probabilities.items()}
        return res


class BaselinePredictor:
    """Production inference wrapper using the exact fitted TF-IDF + Logistic Regression pipeline."""

    def __init__(
        self,
        pipeline: Pipeline,
        id_to_label: Dict[int, str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.pipeline = pipeline
        self.id_to_label = {int(k): str(v) for k, v in id_to_label.items()}
        self.label_to_id = {v: k for k, v in self.id_to_label.items()}
        self.metadata = metadata or {}

    @classmethod
    def from_artifact(cls, artifact_path: Union[str, Path]) -> "BaselinePredictor":
        """Load fitted pipeline and intent mappings from serialized joblib artifact."""
        path = Path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {path}")

        payload = joblib.load(path)
        if not isinstance(payload, dict) or "pipeline" not in payload or "id_to_label" not in payload:
            raise ValueError(f"Invalid artifact payload structure in {path}")

        return cls(
            pipeline=payload["pipeline"],
            id_to_label=payload["id_to_label"],
            metadata=payload.get("metadata", {}),
        )

    def predict(
        self,
        text: str,
        top_k: int = 5,
        return_all_probs: bool = False,
    ) -> PredictionResult:
        """Run single-query intent classification."""
        if not isinstance(text, str):
            raise TypeError(f"Expected string input, received {type(text)}")

        probs = self.pipeline.predict_proba([text])[0]
        pred_id = int(np.argmax(probs))
        confidence = float(probs[pred_id])
        pred_intent = self.id_to_label[pred_id]

        # Rank top-k predictions
        top_k_indices = np.argsort(probs)[::-1][:top_k]
        top_candidates = [
            CandidatePrediction(
                intent_name=self.id_to_label[idx],
                class_id=int(idx),
                probability=round(float(probs[idx]), 4),
            )
            for idx in top_k_indices
        ]

        all_probs_dict = None
        if return_all_probs:
            all_probs_dict = {
                self.id_to_label[idx]: float(prob) for idx, prob in enumerate(probs)
            }

        return PredictionResult(
            input_text=text,
            predicted_intent=pred_intent,
            predicted_class_id=pred_id,
            confidence=confidence,
            top_k_predictions=top_candidates,
            all_probabilities=all_probs_dict,
        )

    def batch_predict(
        self,
        texts: List[str],
        top_k: int = 5,
    ) -> List[PredictionResult]:
        """Run batch intent classification across multiple queries."""
        if not texts:
            return []

        all_probs = self.pipeline.predict_proba(texts)
        results: List[PredictionResult] = []

        for text, probs in zip(texts, all_probs):
            pred_id = int(np.argmax(probs))
            confidence = float(probs[pred_id])
            pred_intent = self.id_to_label[pred_id]

            top_k_indices = np.argsort(probs)[::-1][:top_k]
            top_candidates = [
                CandidatePrediction(
                    intent_name=self.id_to_label[idx],
                    class_id=int(idx),
                    probability=round(float(probs[idx]), 4),
                )
                for idx in top_k_indices
            ]

            results.append(
                PredictionResult(
                    input_text=text,
                    predicted_intent=pred_intent,
                    predicted_class_id=pred_id,
                    confidence=confidence,
                    top_k_predictions=top_candidates,
                )
            )

        return results


def save_baseline_artifact(
    pipeline: Pipeline,
    id_to_label: Dict[int, str],
    output_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Serialize pipeline and intent dictionary into joblib artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline": pipeline,
        "id_to_label": id_to_label,
        "label_to_id": {v: k for k, v in id_to_label.items()},
        "metadata": metadata or {},
    }
    joblib.dump(payload, output_path, compress=3)
    return output_path
