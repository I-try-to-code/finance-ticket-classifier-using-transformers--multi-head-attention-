"""Production inference module for fine-tuned DistilBERT on Banking77 intent classification."""

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from src.config import PROJECT_ROOT

DEFAULT_DISTILBERT_CHECKPOINT = PROJECT_ROOT / "models" / "distilbert_banking77"


@dataclass(frozen=True)
class IntentPrediction:
    """Individual candidate prediction with probability."""
    intent_name: str
    class_id: int
    probability: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent_name": self.intent_name,
            "class_id": self.class_id,
            "probability": round(self.probability, 4),
        }


@dataclass(frozen=True)
class InferenceOutput:
    """Structured inference response for a customer query."""
    query: str
    predicted_intent: str
    predicted_class_id: int
    confidence: float
    top_k_predictions: List[IntentPrediction]
    latency_ms: float
    device: str
    all_probabilities: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            "query": self.query,
            "predicted_intent": self.predicted_intent,
            "predicted_class_id": self.predicted_class_id,
            "confidence": round(self.confidence, 4),
            "top_k_predictions": [c.to_dict() for c in self.top_k_predictions],
            "latency_ms": round(self.latency_ms, 2),
            "device": self.device,
        }
        if self.all_probabilities is not None:
            res["all_probabilities"] = {k: round(v, 6) for k, v in self.all_probabilities.items()}
        return res


class Banking77Predictor:
    """Production predictor wrapping fine-tuned DistilBERT checkpoint and tokenizer."""

    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        max_length: int = 128,
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_DISTILBERT_CHECKPOINT
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"DistilBERT checkpoint not found at: {self.model_dir}. "
                "Ensure the model checkpoint has been saved to models/distilbert_banking77."
            )

        # 1. Device resolution: auto GPU if available, else CPU
        if device is not None:
            self.device = torch.device(device) if isinstance(device, str) else device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 2. Load tokenizer and model once during initialization
        load_start = time.perf_counter()
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(self.model_dir)
        self.model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(self.model_dir)

        # 3. Model setup: evaluation mode, frozen parameters, device transfer
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # 4. Warmup CUDA kernels during initialization so initial user queries have low latency
        if self.device.type == "cuda":
            warmup_inputs = self.tokenizer("warmup", return_tensors="pt")
            warmup_inputs = {k: v.to(self.device) for k, v in warmup_inputs.items()}
            with torch.inference_mode():
                self.model(**warmup_inputs)

        self.initialization_time_ms = (time.perf_counter() - load_start) * 1000.0
        self.max_length = max_length

        # 4. Extract and validate label mappings (77 classes)
        self.id2label: Dict[int, str] = {int(k): str(v) for k, v in self.model.config.id2label.items()}
        self.label2id: Dict[str, int] = {v: k for k, v in self.id2label.items()}

        if len(self.id2label) != 77 or len(self.label2id) != 77:
            raise ValueError(
                f"Expected 77 distinct Banking77 classes, but loaded {len(self.id2label)} id2label mappings."
            )

    def predict(
        self,
        query: str,
        top_k: int = 3,
        return_all_probs: bool = False,
    ) -> InferenceOutput:
        """Run single-query inference with high-precision latency measurement.

        Softmax is applied strictly at inference time on raw model logits.
        """
        if not isinstance(query, str):
            raise TypeError(f"Expected query of type str, received {type(query).__name__}")
        if not query.strip():
            raise ValueError("Query string cannot be empty or whitespace only.")

        # Latency starts here, isolated from model initialization time
        t0 = time.perf_counter()

        # Tokenization settings match validation policy
        inputs = self.tokenizer(
            query,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)
            raw_logits = outputs.logits
            # Softmax computed only here at inference time
            probs = torch.softmax(raw_logits, dim=-1)[0].cpu().numpy()

        latency_ms = (time.perf_counter() - t0) * 1000.0

        pred_id = int(np.argmax(probs))
        confidence = float(probs[pred_id])
        pred_intent = self.id2label[pred_id]

        top_k = max(1, min(top_k, len(self.id2label)))
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_candidates = [
            IntentPrediction(
                intent_name=self.id2label[int(idx)],
                class_id=int(idx),
                probability=float(probs[idx]),
            )
            for idx in top_indices
        ]

        all_probs_dict = None
        if return_all_probs:
            all_probs_dict = {self.id2label[int(idx)]: float(prob) for idx, prob in enumerate(probs)}

        return InferenceOutput(
            query=query,
            predicted_intent=pred_intent,
            predicted_class_id=pred_id,
            confidence=confidence,
            top_k_predictions=top_candidates,
            latency_ms=latency_ms,
            device=str(self.device),
            all_probabilities=all_probs_dict,
        )

    def batch_predict(
        self,
        queries: List[str],
        top_k: int = 3,
    ) -> List[InferenceOutput]:
        """Classify a batch of customer queries using dynamic batch padding."""
        if not isinstance(queries, list):
            raise TypeError(f"Expected queries of type list, received {type(queries).__name__}")
        if not queries:
            return []

        for q in queries:
            if not isinstance(q, str) or not q.strip():
                raise ValueError("All queries in batch must be non-empty strings.")

        t0 = time.perf_counter()

        inputs = self.tokenizer(
            queries,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)
            raw_logits = outputs.logits
            all_probs = torch.softmax(raw_logits, dim=-1).cpu().numpy()

        total_latency_ms = (time.perf_counter() - t0) * 1000.0
        avg_latency_ms = total_latency_ms / len(queries)

        results: List[InferenceOutput] = []
        top_k = max(1, min(top_k, len(self.id2label)))

        for query, probs in zip(queries, all_probs):
            pred_id = int(np.argmax(probs))
            confidence = float(probs[pred_id])
            pred_intent = self.id2label[pred_id]

            top_indices = np.argsort(probs)[::-1][:top_k]
            top_candidates = [
                IntentPrediction(
                    intent_name=self.id2label[int(idx)],
                    class_id=int(idx),
                    probability=float(probs[idx]),
                )
                for idx in top_indices
            ]

            results.append(
                InferenceOutput(
                    query=query,
                    predicted_intent=pred_intent,
                    predicted_class_id=pred_id,
                    confidence=confidence,
                    top_k_predictions=top_candidates,
                    latency_ms=avg_latency_ms,
                    device=str(self.device),
                )
            )

        return results
