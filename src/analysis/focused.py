"""Focused intent comparative analysis and prioritized representative example selection."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd

from src.analysis.collector import ValidationPredictionRow

FOCUSED_INTENTS = [
    "virtual_card_not_working",
    "getting_virtual_card",
    "why_verify_identity",
    "unable_to_verify_identity",
    "verify_my_identity",
    "exchange_rate",
    "card_payment_wrong_exchange_rate",
    "card_not_working",
    "contactless_not_working",
]


@dataclass
class FocusedExampleRecord:
    """Prioritized comparative example for focused intent analysis."""
    focused_intent: str
    priority_tier: int
    comparison_category: str
    sample_id: int
    input_text: str
    true_label: str
    distilbert_prediction: str
    distilbert_confidence: float
    distilbert_correct: bool
    bert_prediction: str
    bert_confidence: float
    bert_correct: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "focused_intent": self.focused_intent,
            "priority_tier": self.priority_tier,
            "comparison_category": self.comparison_category,
            "sample_id": self.sample_id,
            "input_text": self.input_text,
            "true_label": self.true_label,
            "distilbert_prediction": self.distilbert_prediction,
            "distilbert_confidence": round(self.distilbert_confidence, 4),
            "distilbert_correct": self.distilbert_correct,
            "bert_prediction": self.bert_prediction,
            "bert_confidence": round(self.bert_confidence, 4),
            "bert_correct": self.bert_correct,
        }


def assign_priority_tier(db_correct: bool, b_correct: bool) -> Tuple[int, str]:
    """Assign tier and category based on model outcome combinations:
    Tier 1: Both models incorrect
    Tier 2: DistilBERT wrong, BERT correct (BERT improvement)
    Tier 3: BERT wrong, DistilBERT correct (BERT regression)
    Tier 4: Both models correct
    """
    if not db_correct and not b_correct:
        return 1, "both_wrong"
    elif not db_correct and b_correct:
        return 2, "distilbert_wrong_bert_correct"
    elif db_correct and not b_correct:
        return 3, "bert_wrong_distilbert_correct"
    else:
        return 4, "both_correct"


def extract_focused_intent_examples(
    rows: List[ValidationPredictionRow],
    target_intents: List[str] = FOCUSED_INTENTS,
    min_examples_per_intent: int = 10,
) -> List[FocusedExampleRecord]:
    """Extract prioritized validation examples for each targeted difficult intent."""
    by_intent: Dict[str, List[ValidationPredictionRow]] = {intent: [] for intent in target_intents}

    for row in rows:
        if row.true_intent in by_intent:
            by_intent[row.true_intent].append(row)

    selected_records: List[FocusedExampleRecord] = []

    for intent in target_intents:
        candidates = by_intent[intent]
        # Classify each candidate into priority tier
        tiered_candidates: List[Tuple[int, str, ValidationPredictionRow]] = []
        for cand in candidates:
            tier, cat = assign_priority_tier(cand.distilbert_correct, cand.bert_correct)
            tiered_candidates.append((tier, cat, cand))

        # Sort primarily by tier (1 ascending), then by max error confidence (descending)
        tiered_candidates.sort(
            key=lambda x: (
                x[0],  # priority tier 1, 2, 3, 4
                -max(
                    x[2].distilbert_confidence if not x[2].distilbert_correct else 0.0,
                    x[2].bert_confidence if not x[2].bert_correct else 0.0,
                ),
            )
        )

        limit = max(min_examples_per_intent, len(candidates)) if len(candidates) < min_examples_per_intent else min_examples_per_intent
        selected_for_intent = tiered_candidates[:limit]

        for tier, cat, cand in selected_for_intent:
            record = FocusedExampleRecord(
                focused_intent=intent,
                priority_tier=tier,
                comparison_category=cat,
                sample_id=cand.sample_id,
                input_text=cand.text,
                true_label=cand.true_intent,
                distilbert_prediction=cand.distilbert_pred_intent,
                distilbert_confidence=cand.distilbert_confidence,
                distilbert_correct=cand.distilbert_correct,
                bert_prediction=cand.bert_pred_intent,
                bert_confidence=cand.bert_confidence,
                bert_correct=cand.bert_correct,
            )
            selected_records.append(record)

    return selected_records


def export_focused_examples_csv(
    records: List[FocusedExampleRecord],
    output_path: Path,
) -> pd.DataFrame:
    """Export focused intent records to CSV."""
    df = pd.DataFrame([r.to_dict() for r in records])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
