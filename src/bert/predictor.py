"""Production inference engine and latency benchmark for fine-tuned BERT-base."""

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from src.distilbert.predictor import CandidatePrediction, PredictionResult


class BertPredictor:
    """Inference predictor utilizing fine-tuned BERT-base weights and WordPiece tokenizer."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer

        # Extract label mappings from model configuration
        self.id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}
        self.label2id = {v: k for k, v in self.id2label.items()}

    @classmethod
    def from_pretrained(
        cls,
        model_dir: Union[str, Path],
        device: Optional[str] = None,
    ) -> "BertPredictor":
        """Load fine-tuned model and tokenizer from local disk directory."""
        path = Path(model_dir)
        if not path.exists():
            raise FileNotFoundError(f"Model directory not found at: {path}")

        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)

        dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return cls(model=model, tokenizer=tokenizer, device=dev)

    def predict(
        self,
        text: str,
        top_k: int = 3,
        return_all_probs: bool = False,
    ) -> PredictionResult:
        """Run intent classification for a single customer query."""
        if not isinstance(text, str):
            raise TypeError(f"Expected text input of type str, received {type(text)}")

        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=128,
            padding=False,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        pred_id = int(np.argmax(probs))
        confidence = float(probs[pred_id])
        pred_intent = self.id2label[pred_id]

        # Top-k candidates
        top_k_indices = np.argsort(probs)[::-1][:top_k]
        top_candidates = [
            CandidatePrediction(
                intent_name=self.id2label[idx],
                class_id=int(idx),
                probability=round(float(probs[idx]), 4),
            )
            for idx in top_k_indices
        ]

        all_probs_dict = None
        if return_all_probs:
            all_probs_dict = {self.id2label[idx]: float(prob) for idx, prob in enumerate(probs)}

        return PredictionResult(
            input_text=text,
            predicted_intent=pred_intent,
            predicted_class_id=pred_id,
            confidence=confidence,
            top_3_predictions=top_candidates,
            all_probabilities=all_probs_dict,
        )

    def batch_predict(
        self,
        texts: List[str],
        top_k: int = 3,
    ) -> List[PredictionResult]:
        """Classify a batch of customer queries."""
        if not texts:
            return []

        inputs = self.tokenizer(
            texts,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            all_probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        results: List[PredictionResult] = []
        for text, probs in zip(texts, all_probs):
            pred_id = int(np.argmax(probs))
            confidence = float(probs[pred_id])
            pred_intent = self.id2label[pred_id]

            top_k_indices = np.argsort(probs)[::-1][:top_k]
            top_candidates = [
                CandidatePrediction(
                    intent_name=self.id2label[idx],
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
                    top_3_predictions=top_candidates,
                )
            )

        return results

    def measure_latency_ms(
        self,
        sample_query: str = "My card payment was declined yesterday",
        num_warmup: int = 10,
        num_runs: int = 50,
    ) -> float:
        """Measure median (p50) single-query inference latency in milliseconds."""
        for _ in range(num_warmup):
            self.predict(sample_query)

        latencies = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            self.predict(sample_query)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        return float(np.median(latencies))
