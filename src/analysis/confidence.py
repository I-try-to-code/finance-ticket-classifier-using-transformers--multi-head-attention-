"""Confidence calibration, high-confidence errors, and semantic ambiguity analysis."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from src.analysis.collector import ValidationPredictionRow


@dataclass
class HighConfidenceErrorRecord:
    """Record of a confident misclassification."""
    model_name: str
    sample_id: int
    text: str
    true_intent: str
    predicted_intent: str
    confidence: float
    second_intent: str
    second_prob: float
    margin: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "sample_id": self.sample_id,
            "text": self.text,
            "true_intent": self.true_intent,
            "predicted_intent": self.predicted_intent,
            "confidence": round(self.confidence, 4),
            "second_intent": self.second_intent,
            "second_prob": round(self.second_prob, 4),
            "margin": round(self.margin, 4),
        }


@dataclass
class LowConfidenceCorrectRecord:
    """Record of a fragile correct prediction."""
    model_name: str
    sample_id: int
    text: str
    true_intent: str
    confidence: float
    runner_up_intent: str
    runner_up_prob: float
    margin: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "sample_id": self.sample_id,
            "text": self.text,
            "true_intent": self.true_intent,
            "confidence": round(self.confidence, 4),
            "runner_up_intent": self.runner_up_intent,
            "runner_up_prob": round(self.runner_up_prob, 4),
            "margin": round(self.margin, 4),
        }


@dataclass
class SemanticUncertaintyRecord:
    """Record where the model hesitated between two closely matched intents."""
    model_name: str
    sample_id: int
    text: str
    true_intent: str
    top1_intent: str
    top1_prob: float
    top2_intent: str
    top2_prob: float
    margin: float
    is_correct: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "sample_id": self.sample_id,
            "text": self.text,
            "true_intent": self.true_intent,
            "top1_intent": self.top1_intent,
            "top1_prob": round(self.top1_prob, 4),
            "top2_intent": self.top2_intent,
            "top2_prob": round(self.top2_prob, 4),
            "margin": round(self.margin, 4),
            "is_correct": self.is_correct,
        }


def extract_confidence_analysis(
    rows: List[ValidationPredictionRow],
    high_conf_threshold: float = 0.75,
    low_conf_threshold: float = 0.50,
    margin_uncertainty_threshold: float = 0.15,
) -> Dict[str, Any]:
    """Identify high-confidence errors, low-confidence correct predictions, and semantic ambiguity."""
    db_high_conf_errors: List[HighConfidenceErrorRecord] = []
    b_high_conf_errors: List[HighConfidenceErrorRecord] = []

    db_low_conf_correct: List[LowConfidenceCorrectRecord] = []
    b_low_conf_correct: List[LowConfidenceCorrectRecord] = []

    db_semantic_uncertain: List[SemanticUncertaintyRecord] = []
    b_semantic_uncertain: List[SemanticUncertaintyRecord] = []

    for r in rows:
        # 1. DistilBERT High-Confidence Errors
        if not r.distilbert_correct and r.distilbert_confidence >= high_conf_threshold:
            db_high_conf_errors.append(
                HighConfidenceErrorRecord(
                    model_name="DistilBERT",
                    sample_id=r.sample_id,
                    text=r.text,
                    true_intent=r.true_intent,
                    predicted_intent=r.distilbert_pred_intent,
                    confidence=r.distilbert_confidence,
                    second_intent=r.distilbert_second_intent,
                    second_prob=r.distilbert_second_prob,
                    margin=r.distilbert_margin,
                )
            )

        # 2. BERT High-Confidence Errors
        if not r.bert_correct and r.bert_confidence >= high_conf_threshold:
            b_high_conf_errors.append(
                HighConfidenceErrorRecord(
                    model_name="BERT-base",
                    sample_id=r.sample_id,
                    text=r.text,
                    true_intent=r.true_intent,
                    predicted_intent=r.bert_pred_intent,
                    confidence=r.bert_confidence,
                    second_intent=r.bert_second_intent,
                    second_prob=r.bert_second_prob,
                    margin=r.bert_margin,
                )
            )

        # 3. Low-Confidence Correct Predictions
        if r.distilbert_correct and r.distilbert_confidence < low_conf_threshold:
            db_low_conf_correct.append(
                LowConfidenceCorrectRecord(
                    model_name="DistilBERT",
                    sample_id=r.sample_id,
                    text=r.text,
                    true_intent=r.true_intent,
                    confidence=r.distilbert_confidence,
                    runner_up_intent=r.distilbert_second_intent,
                    runner_up_prob=r.distilbert_second_prob,
                    margin=r.distilbert_margin,
                )
            )

        if r.bert_correct and r.bert_confidence < low_conf_threshold:
            b_low_conf_correct.append(
                LowConfidenceCorrectRecord(
                    model_name="BERT-base",
                    sample_id=r.sample_id,
                    text=r.text,
                    true_intent=r.true_intent,
                    confidence=r.bert_confidence,
                    runner_up_intent=r.bert_second_intent,
                    runner_up_prob=r.bert_second_prob,
                    margin=r.bert_margin,
                )
            )

        # 4. Semantic Ambiguity / Margin Uncertainty (p1 - p2 <= 0.15)
        if r.distilbert_margin <= margin_uncertainty_threshold:
            db_semantic_uncertain.append(
                SemanticUncertaintyRecord(
                    model_name="DistilBERT",
                    sample_id=r.sample_id,
                    text=r.text,
                    true_intent=r.true_intent,
                    top1_intent=r.distilbert_pred_intent,
                    top1_prob=r.distilbert_confidence,
                    top2_intent=r.distilbert_second_intent,
                    top2_prob=r.distilbert_second_prob,
                    margin=r.distilbert_margin,
                    is_correct=r.distilbert_correct,
                )
            )

        if r.bert_margin <= margin_uncertainty_threshold:
            b_semantic_uncertain.append(
                SemanticUncertaintyRecord(
                    model_name="BERT-base",
                    sample_id=r.sample_id,
                    text=r.text,
                    true_intent=r.true_intent,
                    top1_intent=r.bert_pred_intent,
                    top1_prob=r.bert_confidence,
                    top2_intent=r.bert_second_intent,
                    top2_prob=r.bert_second_prob,
                    margin=r.bert_margin,
                    is_correct=r.bert_correct,
                )
            )

    # Sort high-confidence errors descending by confidence
    db_high_conf_errors.sort(key=lambda x: x.confidence, reverse=True)
    b_high_conf_errors.sort(key=lambda x: x.confidence, reverse=True)

    # Sort low-confidence correct ascending by confidence
    db_low_conf_correct.sort(key=lambda x: x.confidence)
    b_low_conf_correct.sort(key=lambda x: x.confidence)

    # Sort uncertain cases ascending by margin
    db_semantic_uncertain.sort(key=lambda x: x.margin)
    b_semantic_uncertain.sort(key=lambda x: x.margin)

    return {
        "distilbert_high_confidence_errors": db_high_conf_errors,
        "bert_high_confidence_errors": b_high_conf_errors,
        "distilbert_low_confidence_correct": db_low_conf_correct,
        "bert_low_confidence_correct": b_low_conf_correct,
        "distilbert_semantic_uncertain": db_semantic_uncertain,
        "bert_semantic_uncertain": b_semantic_uncertain,
    }


def export_high_confidence_errors_csv(
    db_errors: List[HighConfidenceErrorRecord],
    bert_errors: List[HighConfidenceErrorRecord],
    output_path: Path,
) -> pd.DataFrame:
    """Save high-confidence errors for both models to CSV."""
    rows = [r.to_dict() for r in db_errors] + [r.to_dict() for r in bert_errors]
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
