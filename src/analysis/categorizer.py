"""Qualitative error categorization and root-cause taxonomy with concrete validation evidence."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from src.analysis.collector import ValidationPredictionRow


@dataclass
class CategorizedConfusionEvidence:
    """A documented error pattern with root cause and supporting validation examples."""
    pair_name: str
    true_intent: str
    predicted_intent: str
    root_cause_category: str
    is_hypothesis: bool
    explanation: str
    supporting_examples: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_name": self.pair_name,
            "true_intent": self.true_intent,
            "predicted_intent": self.predicted_intent,
            "root_cause_category": self.root_cause_category,
            "is_hypothesis": self.is_hypothesis,
            "explanation": self.explanation,
            "supporting_examples": self.supporting_examples,
        }


def build_qualitative_error_taxonomy(
    rows: List[ValidationPredictionRow],
) -> List[CategorizedConfusionEvidence]:
    """Analyze validation errors and map major confusion clusters to the standardized error taxonomy."""
    # Group errors by (true_intent, pred_intent)
    pair_examples_db: Dict[str, List[ValidationPredictionRow]] = {}
    pair_examples_b: Dict[str, List[ValidationPredictionRow]] = {}

    for r in rows:
        if not r.distilbert_correct:
            key = f"{r.true_intent} -> {r.distilbert_pred_intent}"
            pair_examples_db.setdefault(key, []).append(r)
        if not r.bert_correct:
            key = f"{r.true_intent} -> {r.bert_pred_intent}"
            pair_examples_b.setdefault(key, []).append(r)

    results: List[CategorizedConfusionEvidence] = []

    # 1. How to obtain vs not working
    key1 = "virtual_card_not_working -> getting_virtual_card"
    examples1 = pair_examples_db.get(key1, []) + pair_examples_b.get(key1, [])
    # deduplicate by sample_id
    seen = set()
    dedup_ex1 = []
    for ex in examples1:
        if ex.sample_id not in seen:
            seen.add(ex.sample_id)
            dedup_ex1.append({
                "sample_id": ex.sample_id,
                "text": ex.text,
                "distilbert_pred": ex.distilbert_pred_intent,
                "distilbert_conf": round(ex.distilbert_confidence, 4),
                "bert_pred": ex.bert_pred_intent,
                "bert_conf": round(ex.bert_confidence, 4),
            })

    results.append(
        CategorizedConfusionEvidence(
            pair_name=key1,
            true_intent="virtual_card_not_working",
            predicted_intent="getting_virtual_card",
            root_cause_category="insufficient distinction between “how to obtain” and “not working”",
            is_hypothesis=False,
            explanation=(
                "Users asking how to make their virtual card operational frequently use action verbs like 'get ... to work' "
                "or 'set up', which models associate with requisitioning/provisioning a new virtual card ('getting_virtual_card') "
                "rather than diagnosing a defective virtual card."
            ),
            supporting_examples=dedup_ex1[:5],
        )
    )

    # 2. Why verification is needed vs verification failed / how to verify
    key2a = "why_verify_identity -> unable_to_verify_identity"
    examples2a = pair_examples_db.get(key2a, []) + pair_examples_b.get(key2a, [])
    seen = set()
    dedup_ex2a = []
    for ex in examples2a:
        if ex.sample_id not in seen:
            seen.add(ex.sample_id)
            dedup_ex2a.append({
                "sample_id": ex.sample_id,
                "text": ex.text,
                "distilbert_pred": ex.distilbert_pred_intent,
                "distilbert_conf": round(ex.distilbert_confidence, 4),
                "bert_pred": ex.bert_pred_intent,
                "bert_conf": round(ex.bert_confidence, 4),
            })

    results.append(
        CategorizedConfusionEvidence(
            pair_name=key2a,
            true_intent="why_verify_identity",
            predicted_intent="unable_to_verify_identity",
            root_cause_category="insufficient distinction between “why verification is needed” and “verification failed”",
            is_hypothesis=False,
            explanation=(
                "When customer prompts challenge why the bank demands identity documents, questions referencing an active "
                "verification hurdle (e.g. 'I do not want to upload my passport') are misconstrued as an operational failure "
                "to complete verification."
            ),
            supporting_examples=dedup_ex2a[:5],
        )
    )

    # 3. Generic card failure vs contactless failure
    key3 = "card_not_working -> contactless_not_working"
    key3_alt = "contactless_not_working -> card_not_working"
    examples3 = pair_examples_db.get(key3, []) + pair_examples_b.get(key3, []) + pair_examples_db.get(key3_alt, []) + pair_examples_b.get(key3_alt, [])
    seen = set()
    dedup_ex3 = []
    for ex in examples3:
        if ex.sample_id not in seen:
            seen.add(ex.sample_id)
            dedup_ex3.append({
                "sample_id": ex.sample_id,
                "text": ex.text,
                "true_intent": ex.true_intent,
                "distilbert_pred": ex.distilbert_pred_intent,
                "distilbert_conf": round(ex.distilbert_confidence, 4),
                "bert_pred": ex.bert_pred_intent,
                "bert_conf": round(ex.bert_confidence, 4),
            })

    results.append(
        CategorizedConfusionEvidence(
            pair_name="card_not_working <-> contactless_not_working",
            true_intent="card_not_working / contactless_not_working",
            predicted_intent="contactless_not_working / card_not_working",
            root_cause_category="insufficient distinction between generic card failure and contactless failure",
            is_hypothesis=False,
            explanation=(
                "Failure at a physical payment terminal is described generically ('my card did not work at the register') "
                "without explicitly distinguishing whether the magnetic stripe, chip-and-PIN, or NFC tap was attempted."
            ),
            supporting_examples=dedup_ex3[:5],
        )
    )

    # 4. Lexical Overlap: card_delivery_estimate vs card_arrival
    key4 = "card_delivery_estimate -> card_arrival"
    examples4 = pair_examples_db.get(key4, []) + pair_examples_b.get(key4, [])
    seen = set()
    dedup_ex4 = []
    for ex in examples4:
        if ex.sample_id not in seen:
            seen.add(ex.sample_id)
            dedup_ex4.append({
                "sample_id": ex.sample_id,
                "text": ex.text,
                "distilbert_pred": ex.distilbert_pred_intent,
                "distilbert_conf": round(ex.distilbert_confidence, 4),
                "bert_pred": ex.bert_pred_intent,
                "bert_conf": round(ex.bert_confidence, 4),
            })

    results.append(
        CategorizedConfusionEvidence(
            pair_name=key4,
            true_intent="card_delivery_estimate",
            predicted_intent="card_arrival",
            root_cause_category="lexical overlap",
            is_hypothesis=False,
            explanation=(
                "Both intents share near-identical lexical tokens ('when will card arrive', 'how many days for delivery', "
                "'where is my card'). The distinction between requesting an ETA versus reporting that a card has not arrived "
                "hinges on subtle temporal phrasing."
            ),
            supporting_examples=dedup_ex4[:5],
        )
    )

    # 5. Lexical Overlap: top_up_reverted vs top_up_failed
    key5 = "top_up_reverted -> top_up_failed"
    examples5 = pair_examples_db.get(key5, []) + pair_examples_b.get(key5, [])
    seen = set()
    dedup_ex5 = []
    for ex in examples5:
        if ex.sample_id not in seen:
            seen.add(ex.sample_id)
            dedup_ex5.append({
                "sample_id": ex.sample_id,
                "text": ex.text,
                "distilbert_pred": ex.distilbert_pred_intent,
                "distilbert_conf": round(ex.distilbert_confidence, 4),
                "bert_pred": ex.bert_pred_intent,
                "bert_conf": round(ex.bert_confidence, 4),
            })

    results.append(
        CategorizedConfusionEvidence(
            pair_name=key5,
            true_intent="top_up_reverted",
            predicted_intent="top_up_failed",
            root_cause_category="lexical overlap",
            is_hypothesis=False,
            explanation=(
                "From a customer perspective, a reverted deposit and an outright failed deposit manifest identically: "
                "money was deducted or attempted but not credited to the card balance."
            ),
            supporting_examples=dedup_ex5[:5],
        )
    )

    # 6. Missing Contextual Cue: pending_transfer vs transfer_timing
    key6 = "pending_transfer -> transfer_timing"
    examples6 = pair_examples_db.get(key6, []) + pair_examples_b.get(key6, [])
    seen = set()
    dedup_ex6 = []
    for ex in examples6:
        if ex.sample_id not in seen:
            seen.add(ex.sample_id)
            dedup_ex6.append({
                "sample_id": ex.sample_id,
                "text": ex.text,
                "distilbert_pred": ex.distilbert_pred_intent,
                "distilbert_conf": round(ex.distilbert_confidence, 4),
                "bert_pred": ex.bert_pred_intent,
                "bert_conf": round(ex.bert_confidence, 4),
            })

    results.append(
        CategorizedConfusionEvidence(
            pair_name=key6,
            true_intent="pending_transfer",
            predicted_intent="transfer_timing",
            root_cause_category="missing contextual cue",
            is_hypothesis=True,
            explanation=(
                "[Hypothesis] Inquiries such as 'How long does a transfer take?' lack the contextual cue of whether an actual "
                "transfer is currently pending in the user's account or if they are asking about general SLA transfer timelines."
            ),
            supporting_examples=dedup_ex6[:5],
        )
    )

    # 7. Ambiguous User Wording: Short queries
    key7 = "get_disposable_virtual_card -> card_not_working / order_physical_card"
    examples7 = [r for r in rows if len(r.text.split()) <= 6 and (not r.distilbert_correct or not r.bert_correct)]
    dedup_ex7 = []
    for ex in examples7[:5]:
        dedup_ex7.append({
            "sample_id": ex.sample_id,
            "text": ex.text,
            "true_intent": ex.true_intent,
            "distilbert_pred": ex.distilbert_pred_intent,
            "bert_pred": ex.bert_pred_intent,
        })

    results.append(
        CategorizedConfusionEvidence(
            pair_name="low_token_count_ambiguity",
            true_intent="various short customer queries",
            predicted_intent="multiple",
            root_cause_category="ambiguous user wording",
            is_hypothesis=True,
            explanation=(
                "[Hypothesis] Short, terse queries (e.g. 'how do the cards work?', 'deposit issue') omit grammatical arguments, "
                "forcing the Transformer heads to distribute attention across multiple plausibly relevant product categories."
            ),
            supporting_examples=dedup_ex7,
        )
    )

    # 8. Possible Annotation Ambiguity: direct_debit_payment_not_recognised vs card_payment_not_recognised
    key8 = "direct_debit_payment_not_recognised -> card_payment_not_recognised"
    examples8 = pair_examples_db.get(key8, []) + pair_examples_b.get(key8, [])
    seen = set()
    dedup_ex8 = []
    for ex in examples8:
        if ex.sample_id not in seen:
            seen.add(ex.sample_id)
            dedup_ex8.append({
                "sample_id": ex.sample_id,
                "text": ex.text,
                "distilbert_pred": ex.distilbert_pred_intent,
                "distilbert_conf": round(ex.distilbert_confidence, 4),
                "bert_pred": ex.bert_pred_intent,
                "bert_conf": round(ex.bert_confidence, 4),
            })

    results.append(
        CategorizedConfusionEvidence(
            pair_name=key8,
            true_intent="direct_debit_payment_not_recognised",
            predicted_intent="card_payment_not_recognised",
            root_cause_category="possible annotation ambiguity",
            is_hypothesis=True,
            explanation=(
                "[Hypothesis] Customer statements like 'There is an unfamiliar transaction on my account' often omit whether the "
                "transaction was a direct debit, card swipe, or standing order. Human annotators may have labeled these based "
                "on synthetic assumptions not apparent from the text itself."
            ),
            supporting_examples=dedup_ex8[:5],
        )
    )

    return results
